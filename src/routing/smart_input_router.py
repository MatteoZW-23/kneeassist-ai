"""Read-only study intake inspection. It does not make diagnostic predictions."""
import io
import zipfile
from pathlib import PurePosixPath

import numpy as np

from src.data.preprocessing import (
    InputError, _dicom_plane, _dicom_text_plane, _file_kind, _nifti_volume,
    _plane_from_path, pydicom, safe_npy,
)


def _array_descriptor(identifier, display_name, values, file_format, plane=None,
                      confidence=None, method='No anatomical metadata'):
    return {
        'id': identifier,
        'file': display_name,
        'format': file_format,
        'shape': [int(x) for x in values.shape],
        'slices': int(values.shape[0]),
        'height': int(values.shape[1]),
        'width': int(values.shape[2]),
        'dtype': str(values.dtype),
        'intensity_min': float(np.min(values)),
        'intensity_max': float(np.max(values)),
        'has_invalid_values': bool(not np.isfinite(values).all()),
        'detected_plane': plane,
        'plane_confidence': confidence,
        'plane_method': method,
        'requires_confirmation': plane is None or (confidence is not None and confidence < 0.6),
        'status': 'Needs plane confirmation' if plane is None or (confidence is not None and confidence < 0.6) else 'Accepted for model compatibility check',
    }


def _npy_descriptor(identifier, name, raw, cfg):
    values=safe_npy(raw,cfg)
    _,hint=_plane_from_path(name,cfg['planes'],allow_unknown=True)
    return _array_descriptor(identifier,name,values,'NumPy MRI stack',hint,
                             None,'Filename hint only; no anatomical metadata')


def _nifti_descriptor(identifier, name, raw, cfg):
    values,details=_nifti_volume(raw,cfg,compressed=_file_kind(name)=='nifti_gz')
    return _array_descriptor(identifier,name,values,'NIfTI MRI volume',details['plane'],
                             details['confidence'],details['method'])


def _dicom_descriptor(identifier, name, raw, cfg):
    if pydicom is None:
        raise InputError('DICOM support is unavailable because pydicom is not installed.')
    try:
        dataset=pydicom.dcmread(io.BytesIO(raw),stop_before_pixels=True,force=False)
    except Exception as e:
        raise InputError(f'{PurePosixPath(name).name} is not a readable DICOM file.') from e
    rows=int(getattr(dataset,'Rows',0) or 0);columns=int(getattr(dataset,'Columns',0) or 0)
    actual=_dicom_plane(dataset) or _dicom_text_plane(dataset,cfg['planes'])
    _,hint=_plane_from_path(name,cfg['planes'],allow_unknown=True)
    plane=actual or hint
    return {
        'id':identifier,'file':name,'format':'DICOM MRI slice','shape':None,'slices':None,
        'height':rows or None,'width':columns or None,'dtype':str(getattr(dataset,'BitsAllocated','unknown'))+'-bit',
        'intensity_min':None,'intensity_max':None,'has_invalid_values':False,'detected_plane':plane,
        'plane_confidence':1.0 if actual else (None if hint else 0.0),
        'plane_method':'DICOM orientation/series metadata' if actual else 'Filename hint only; no usable DICOM orientation metadata',
        'requires_confirmation':plane is None,'status':'Needs plane confirmation' if plane is None else 'Accepted for model compatibility check',
    }


def inspect_uploads(files, cfg):
    """Return file facts and cautious plane evidence without running a model."""
    descriptors=[]
    for name,raw in files:
        kind=_file_kind(name)
        if kind=='npy': descriptors.append(_npy_descriptor(name,name,raw,cfg))
        elif kind in ('nifti','nifti_gz'): descriptors.append(_nifti_descriptor(name,name,raw,cfg))
        elif kind=='dcm': descriptors.append(_dicom_descriptor(name,name,raw,cfg))
        elif kind=='zip':
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    items=[item for item in archive.infolist() if not item.is_dir()]
                    if not items: raise InputError('The study ZIP is empty.')
                    if sum(item.file_size for item in items)>cfg['max_upload_mb']*1024**2:
                        raise InputError('The uncompressed study ZIP exceeds the configured upload limit.')
                    for item in items:
                        identifier=f'{name}::{item.filename}';entry=archive.read(item);entry_kind=_file_kind(item.filename)
                        if entry_kind=='npy': descriptors.append(_npy_descriptor(identifier,item.filename,entry,cfg))
                        elif entry_kind in ('nifti','nifti_gz'): descriptors.append(_nifti_descriptor(identifier,item.filename,entry,cfg))
                        elif entry_kind=='dcm': descriptors.append(_dicom_descriptor(identifier,item.filename,entry,cfg))
                        else: raise InputError('The study ZIP contains an unsupported file type.')
            except (zipfile.BadZipFile,RuntimeError) as e:
                raise InputError('The study ZIP is corrupt or encrypted.') from e
        else:
            raise InputError('Supported inputs are .npy, .nii, .nii.gz, .dcm, or a ZIP containing one supported MRI study.')
    if not descriptors: raise InputError('Upload at least one MRI sequence before analysing the case.')
    detected={item['detected_plane'] for item in descriptors if item['detected_plane']}
    return {'files':descriptors,'available_planes':sorted(detected),'requires_confirmation':[item for item in descriptors if item['requires_confirmation']]}
