"""Shared preprocessing for training, inference, and Grad-CAM."""
import io
import struct
import ast
import gzip
import zipfile
from pathlib import PurePosixPath
import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydicom
except ImportError:  # Training-only environments may omit direct DICOM support.
    pydicom = None

try:
    import nibabel as nib
except ImportError:  # NIfTI support is optional until dependencies are installed.
    nib = None

class InputError(ValueError):
    pass


def input_config(cfg):
    """Accept both legacy flat intake settings and config.yaml's nested settings."""
    nested = cfg.get('preprocessing', {}) if isinstance(cfg, dict) else {}
    return {**cfg, **nested} if nested else cfg


def validate_volume(v, cfg):
    cfg = input_config(cfg)
    if not isinstance(v,np.ndarray) or v.ndim!=3:
        raise InputError('Each MRI sequence must be a 3D NumPy array: slices × height × width.')
    if v.dtype.kind not in 'uif' or not np.isfinite(v).all():
        raise InputError('MRI pixels must be finite numeric values; object arrays and NaNs are not supported.')
    if not 1<=v.shape[0]<=cfg['max_slices'] or min(v.shape[1:])<16 or max(v.shape[1:])>cfg['max_dimension']:
        raise InputError('MRI dimensions are outside supported limits (1–512 slices; image dimensions 16–1024).')
    if float(v.max())<=float(v.min()):
        raise InputError('This sequence has no intensity variation. Upload a valid MRI stack.')
    return v

def safe_npy(data, cfg):
    cfg = input_config(cfg)
    if len(data)>cfg['max_upload_mb']*1024**2:
        raise InputError('The file exceeds the configured upload limit.')
    try:
        f=io.BytesIO(data)
        if f.read(6)!=b'\x93NUMPY': raise InputError('This is not a valid NumPy MRI file.')
        version=f.read(2)
        if len(version)!=2 or version[0] not in (1,2,3): raise InputError('Unsupported NumPy format.')
        size_len=2 if version[0]==1 else 4
        header_len=struct.unpack('<H' if size_len==2 else '<I',f.read(size_len))[0]
        if header_len>10000: raise InputError('The array header is too large.')
        header=ast.literal_eval(f.read(header_len).decode('utf-8' if version[0]==3 else 'latin1').strip())
        shape=header['shape']; dtype=np.dtype(header['descr'])
        if dtype.hasobject or dtype.kind not in 'uif' or len(shape)!=3 or any(type(n)!=int or n<1 for n in shape):
            raise InputError('Expected a numeric, three-dimensional MRI array.')
        count=1
        for n in shape: count*=n
        if count*dtype.itemsize>cfg['max_upload_mb']*1024**2 or count*dtype.itemsize!=len(data)-f.tell():
            raise InputError('The array size is invalid or exceeds the upload limit.')
        return validate_volume(np.load(io.BytesIO(data),allow_pickle=False),cfg)
    except InputError: raise
    except Exception as e: raise InputError('The MRI file could not be read. Use a valid .npy slice stack.') from e

def _plane_from_path(name, planes, override=None, allow_unknown=False):
    path=PurePosixPath(name.replace('\\','/'))
    if path.is_absolute() or '..' in path.parts:
        raise InputError('Unsafe paths are not accepted in study archives.')
    if override in planes:
        return path,override
    matches=[p for p in planes if p==path.stem.lower() or p in [x.lower() for x in path.parts[:-1]]]
    if len(matches)!=1:
        if allow_unknown:
            return path,None
        raise InputError('Name each sequence axial, coronal, or sagittal, or place it in that plane folder.')
    return path,matches[0]


def _upload_identifier(name, index):
    """Return a stable per-upload identifier even when browser filenames repeat."""
    return f"upload-{int(index)}::{name}"

def _nifti_volume(data, cfg, compressed=False):
    """Read a NIfTI study in memory and derive only a cautious plane suggestion."""
    cfg = input_config(cfg)
    if nib is None:
        raise InputError('NIfTI support is unavailable because nibabel is not installed. Re-run Install.cmd, then restart KneeAssist AI.')
    if len(data)>cfg['max_upload_mb']*1024**2:
        raise InputError('The file exceeds the configured upload limit.')
    try:
        raw=gzip.decompress(data) if compressed else data
        image=nib.Nifti1Image.from_bytes(raw)
        values=np.asarray(image.dataobj,dtype=np.float32)
    except Exception as e:
        raise InputError('The NIfTI file could not be read. Upload a valid .nii or .nii.gz MRI volume.') from e
    if values.ndim==4 and values.shape[-1]==1:
        values=values[...,0]
    if values.ndim!=3:
        raise InputError('A NIfTI MRI study must contain one three-dimensional volume. Four-dimensional time series are not supported.')
    try:
        codes=nib.orientations.aff2axcodes(image.affine)
        zooms=np.asarray(image.header.get_zooms()[:3],dtype=float)
        axis=int(np.argmax(zooms))
        plane_by_code={'R':'sagittal','L':'sagittal','A':'coronal','P':'coronal','S':'axial','I':'axial'}
        suggested=plane_by_code.get(codes[axis])
        ordered=np.moveaxis(values,axis,0)
        ordered=validate_volume(ordered,cfg)
        ordered_zooms=np.sort(zooms)
        ratio=float(ordered_zooms[-1]/max(ordered_zooms[-2],1e-6))
        confidence=0.8 if ratio>=1.5 else 0.45
        return ordered,{'plane':suggested,'confidence':confidence,'method':'NIfTI affine and voxel spacing','orientation_codes':list(codes),'zooms':zooms.tolist()}
    except InputError:
        raise
    except Exception as e:
        raise InputError('The NIfTI orientation or pixel data could not be validated.') from e

def _dicom_plane(dataset):
    """Derive acquisition plane from DICOM direction cosines, when available."""
    try:
        orientation=np.asarray(dataset.ImageOrientationPatient,dtype=float)
        if orientation.shape != (6,): return None
        axis=int(np.argmax(np.abs(np.cross(orientation[:3],orientation[3:]))))
        return ('sagittal','coronal','axial')[axis]
    except (AttributeError,TypeError,ValueError):
        return None

def _dicom_sort_key(dataset):
    try:
        orientation=np.asarray(dataset.ImageOrientationPatient,dtype=float)
        position=np.asarray(dataset.ImagePositionPatient,dtype=float)
        return (0,float(np.dot(position,np.cross(orientation[:3],orientation[3:]))))
    except (AttributeError,TypeError,ValueError):
        try: return (1,float(dataset.InstanceNumber))
        except (AttributeError,TypeError,ValueError): return (2,0.0)

def _dicom_pixels(dataset):
    try:
        pixels=np.asarray(dataset.pixel_array,dtype=np.float32)
    except Exception as e:
        raise InputError('A DICOM slice could not be decoded. Use uncompressed DICOM, or install its required pixel decoder.') from e
    if pixels.ndim!=2 or not np.isfinite(pixels).all():
        raise InputError('Each DICOM file must contain one finite two-dimensional MRI image.')
    pixels=pixels*float(getattr(dataset,'RescaleSlope',1.0))+float(getattr(dataset,'RescaleIntercept',0.0))
    if str(getattr(dataset,'PhotometricInterpretation','')).upper()=='MONOCHROME1':
        pixels=pixels.max()+pixels.min()-pixels
    return pixels

def _dicom_text_plane(dataset, planes):
    text=' '.join(str(getattr(dataset,field,'')) for field in ('SeriesDescription','ProtocolName','SequenceName')).lower()
    matches=[plane for plane in planes if plane in text]
    return matches[0] if len(matches)==1 else None

def safe_dicom_study(entries, cfg, overrides=None, require_declared=False):
    """Read DICOM slices in memory using geometry before names and metadata hints."""
    cfg = input_config(cfg)
    if pydicom is None:
        raise InputError('DICOM support is unavailable because pydicom is not installed. Re-run Install.cmd, then restart KneeAssist AI.')
    groups={}
    overrides=overrides or {}
    for item in entries:
        name,raw=item[:2]
        key=item[2] if len(item)>2 else name
        path,declared=_plane_from_path(name,cfg['planes'],overrides.get(key) or overrides.get(name),allow_unknown=True)
        try: dataset=pydicom.dcmread(io.BytesIO(raw),force=False)
        except Exception as e: raise InputError('A ZIP entry is not a valid DICOM file.') from e
        actual=_dicom_plane(dataset)
        if actual is not None and declared is not None and actual!=declared:
            raise InputError(f'DICOM geometry for {path.name} is {actual}, but its folder declares {declared}. Correct the plane folders and try again.')
        if require_declared and declared is None:
            raise InputError('Place DICOM slices in axial, coronal, or sagittal folders.')
        plane=actual or declared or _dicom_text_plane(dataset,cfg['planes'])
        if plane is None:
            raise InputError(f'The DICOM series {path.name} has no usable orientation metadata. Confirm its axial, coronal, or sagittal plane before analysis.')
        series=str(getattr(dataset,'SeriesInstanceUID',''))
        if not series: raise InputError('Each DICOM slice must include a Series Instance UID.')
        groups.setdefault(plane,{}).setdefault(series,[]).append(dataset)
    volumes={}
    for plane,series_groups in groups.items():
        if len(series_groups)!=1:
            raise InputError(f'More than one DICOM series was found for {plane}. Create a ZIP with exactly one MRI series per plane.')
        slices=next(iter(series_groups.values()))
        if not 1<=len(slices)<=cfg.get('max_dicom_series_slices',cfg['max_slices']):
            raise InputError(f'The {plane} DICOM series has an unsupported number of slices.')
        pixels=[_dicom_pixels(dataset) for dataset in sorted(slices,key=_dicom_sort_key)]
        shape=pixels[0].shape
        if any(pixel.shape!=shape for pixel in pixels):
            raise InputError(f'The {plane} DICOM series has inconsistent slice dimensions.')
        volumes[plane]=validate_volume(np.stack(pixels,axis=0),cfg)
    return volumes

def safe_dicom_series(entries, cfg):
    """Backwards-compatible strict reader for the documented plane-labelled DICOM ZIP route."""
    return safe_dicom_study(entries,cfg,require_declared=True)

def _file_kind(name):
    lowered=name.lower()
    if lowered.endswith('.nii.gz'): return 'nifti_gz'
    if lowered.endswith('.nii'): return 'nifti'
    return PurePosixPath(lowered).suffix.lstrip('.')

def load_uploads(files, cfg, plane_overrides=None):
    """Read one MRI study in memory. Metadata takes precedence over filename hints."""
    cfg = input_config(cfg)
    planes=cfg['planes'];volumes={};overrides=plane_overrides or {};dicom_entries=[]
    def add(name, raw, key=None):
        _,plane=_plane_from_path(name,planes,overrides.get(key or name) or overrides.get(name),allow_unknown=True)
        if plane is None:
            raise InputError(f'Plane for {PurePosixPath(name).name} could not be determined from metadata. Confirm axial, coronal, or sagittal in the intake panel.')
        if plane in volumes: raise InputError(f'More than one {plane} stack was supplied. Upload one study at a time.')
        volumes[plane]=safe_npy(raw,cfg)
    def add_nifti(name,raw,key=None):
        volume,details=_nifti_volume(raw,cfg,compressed=_file_kind(name)=='nifti_gz')
        _,declared=_plane_from_path(name,planes,overrides.get(key or name) or overrides.get(name),allow_unknown=True)
        metadata_plane=details['plane']
        if (metadata_plane is not None and declared is not None
                and details['confidence'] >= 0.6 and metadata_plane != declared):
            raise InputError(
                f"NIfTI orientation for {PurePosixPath(name).name} is {metadata_plane}, but its filename, folder, "
                f"or supplied plane declares {declared}. Correct the conflicting plane information and try again."
            )
        plane=metadata_plane or declared
        if plane is None or (details['confidence']<0.6 and declared is None):
            raise InputError(f'Plane for {PurePosixPath(name).name} is uncertain from NIfTI orientation. Confirm it in the intake panel.')
        if plane in volumes: raise InputError(f'More than one {plane} stack was supplied. Upload one study at a time.')
        volumes[plane]=volume
    for upload_index,(name,raw) in enumerate(files):
        upload_key=_upload_identifier(name,upload_index)
        kind=_file_kind(name)
        if kind=='zip':
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    items=[i for i in z.infolist() if not i.is_dir()]
                    if not items: raise InputError('The study ZIP is empty.')
                    if sum(i.file_size for i in items)>cfg['max_upload_mb']*1024**2:
                        raise InputError('The uncompressed study ZIP exceeds the configured upload limit.')
                    kinds={_file_kind(i.filename) for i in items}
                    if kinds=={'npy'}:
                        if len(items)>3: raise InputError('A NumPy study ZIP must contain at most three MRI stacks.')
                        for i in items:add(i.filename,z.read(i),f'{upload_key}::{i.filename}')
                    elif kinds=={'dcm'}:
                        if len(items)>cfg.get('max_dicom_files',768): raise InputError('The DICOM ZIP contains too many files.')
                        if volumes or dicom_entries: raise InputError('Upload one MRI study at a time; do not mix DICOM and processed arrays.')
                        dicom_entries.extend((i.filename,z.read(i),f'{upload_key}::{i.filename}') for i in items)
                    elif kinds.issubset({'nifti','nifti_gz'}):
                        if len(items)>3: raise InputError('A NIfTI study ZIP must contain at most three MRI volumes.')
                        for i in items:add_nifti(i.filename,z.read(i),f'{upload_key}::{i.filename}')
                    else:
                        raise InputError('A study ZIP must contain only one supported MRI type: .npy, .nii/.nii.gz, or .dcm slices.')
            except (zipfile.BadZipFile,RuntimeError) as e: raise InputError('The study ZIP is corrupt or encrypted.') from e
        elif kind=='npy': add(name,raw,upload_key)
        elif kind in ('nifti','nifti_gz'): add_nifti(name,raw,upload_key)
        elif kind=='dcm': dicom_entries.append((name,raw,upload_key))
        else: raise InputError('Supported inputs are .npy, .nii, .nii.gz, .dcm, or a ZIP containing one supported MRI study. JPEG/PNG are not supported.')
    if dicom_entries:
        if volumes: raise InputError('Upload one MRI study at a time; do not mix DICOM and processed arrays.')
        volumes.update(safe_dicom_study(dicom_entries,cfg,overrides=overrides))
    if not volumes: raise InputError('Upload at least one MRI sequence before analysing the case.')
    return volumes

def prepare(v, cfg):
    v=validate_volume(v,cfg)
    indices=np.unique(np.rint(np.linspace(0,len(v)-1,min(len(v),cfg['slices_per_plane']))).astype(int))
    # Percentiles are calculated within this sequence, never across cases/splits.
    low,high=np.percentile(v,[cfg['percentile_low'],cfg['percentile_high']])
    if high<=low: low,high=float(v.min()),float(v.max())
    x=np.clip((v[indices].astype(np.float32)-low)/max(high-low,1e-6),0,1).astype(np.float32)
    x=torch.from_numpy(x).unsqueeze(1)
    x=F.interpolate(x,size=(cfg['image_size'],cfg['image_size']),mode='bilinear',align_corners=False,antialias=True)
    original=x[:,0].clone()
    x=x.repeat(1,3,1,1)
    x=(x-torch.tensor([.485,.456,.406]).view(1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,3,1,1)
    return x, original.numpy(), indices.tolist()
