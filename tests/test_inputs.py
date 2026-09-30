import io
import gzip
import zipfile
import numpy as np
import pytest
from src.utils import config
from src.data.preprocessing import load_uploads,prepare,InputError
from src.routing.smart_input_router import inspect_uploads
from src.selection.model_selector import select_models

def npy(a):
    f=io.BytesIO();np.save(f,a);return f.getvalue()

def test_valid_and_missing_plane():
    cfg=config()['preprocessing'];a=np.arange(4*32*32,dtype=np.float32).reshape(4,32,32)
    volumes=load_uploads([('axial.npy',npy(a))],cfg)
    x,original,idx=prepare(volumes['axial'],cfg)
    assert x.shape==(4,3,224,224) and idx==list(range(4)) and np.isfinite(original).all()

@pytest.mark.parametrize('value',[np.zeros((32,32)),np.zeros((2,32,32)),np.full((2,32,32),np.nan),np.array([{'x':1}],dtype=object)])
def test_invalid_array(value):
    with pytest.raises(InputError):load_uploads([('axial.npy',npy(value))],config()['preprocessing'])

def test_no_input():
    with pytest.raises(InputError,match='Upload'):load_uploads([],config()['preprocessing'])

def test_unnamed_numpy_stack_requires_and_accepts_manual_plane_confirmation():
    cfg=config()['preprocessing'];a=np.arange(4*32*32,dtype=np.float32).reshape(4,32,32)
    report=inspect_uploads([('0000.npy',npy(a))],cfg)
    assert report['requires_confirmation'][0]['id']=='0000.npy'
    with pytest.raises(InputError,match='Confirm'):
        load_uploads([('0000.npy',npy(a))],cfg)
    volumes=load_uploads([('0000.npy',npy(a))],cfg,{'0000.npy':'sagittal'})
    assert list(volumes)==['sagittal'] and volumes['sagittal'].shape==(4,32,32)

def test_nifti_input_uses_affine_plane_when_spacing_is_informative():
    nib=pytest.importorskip('nibabel');cfg=config()['preprocessing']
    values=np.arange(32*32*4,dtype=np.float32).reshape(32,32,4)
    image=nib.Nifti1Image(values,np.diag([1.,1.,3.,1.]))
    raw=image.to_bytes()
    report=inspect_uploads([('0000.nii',raw)],cfg)
    assert report['files'][0]['detected_plane']=='axial'
    volumes=load_uploads([('0000.nii',raw)],cfg)
    assert list(volumes)==['axial'] and volumes['axial'].shape==(4,32,32)

def test_compressed_nifti_and_direct_dicom_study_are_supported():
    nib=pytest.importorskip('nibabel');cfg=config()['preprocessing']
    values=np.arange(32*32*4,dtype=np.float32).reshape(32,32,4)
    raw=gzip.compress(nib.Nifti1Image(values,np.diag([1.,1.,3.,1.])).to_bytes())
    nifti=load_uploads([('scan.nii.gz',raw)],cfg)
    dicom=load_uploads([('0001.dcm',dicom_slice('coronal',1,10)),('0002.dcm',dicom_slice('coronal',2,20))],cfg)
    assert list(nifti)==['axial'] and dicom['coronal'].shape==(2,32,32)

def test_registry_selects_existing_checkpoint_for_partial_study():
    selection=select_models(['sagittal'],['abnormal','acl','meniscus'])
    assert len(selection['compatible_models'])==1
    assert {item['architecture'] for item in selection['selections']}=={'efficientnet_b0'}

def test_bad_zip_and_duplicate():
    a=npy(np.arange(2048,dtype=np.float32).reshape(2,32,32));cfg=config()['preprocessing']
    with pytest.raises(InputError):load_uploads([('bad.zip',b'not a zip')],cfg)
    with pytest.raises(InputError,match='More than one'):load_uploads([('axial.npy',a),('axial.npy',a)],cfg)
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w') as f:f.writestr('../axial.npy',a)
    with pytest.raises(InputError,match='Unsafe'):load_uploads([('study.zip',z.getvalue())],cfg)

def test_deterministic_normalization_and_sampling():
    a=np.random.default_rng(42).integers(0,255,(20,32,32),dtype=np.uint8);cfg=config()['preprocessing']
    first=prepare(a,cfg);second=prepare(a,cfg)
    assert np.array_equal(first[0].numpy(),second[0].numpy()) and len(first[2])==12

def dicom_slice(plane, instance, value, series='1.2.826.0.1.3680043.8.498.999'):
    pydicom=pytest.importorskip('pydicom')
    from pydicom.dataset import Dataset,FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,MRImageStorage,generate_uid
    d=Dataset();d.file_meta=FileMetaDataset();d.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
    d.file_meta.MediaStorageSOPClassUID=MRImageStorage;d.file_meta.MediaStorageSOPInstanceUID=generate_uid()
    d.SOPClassUID=MRImageStorage;d.SOPInstanceUID=d.file_meta.MediaStorageSOPInstanceUID;d.SeriesInstanceUID=series
    d.Rows=d.Columns=32;d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME2';d.BitsAllocated=16;d.BitsStored=16;d.HighBit=15;d.PixelRepresentation=0;d.InstanceNumber=instance
    orientations={'axial':[1,0,0,0,1,0],'coronal':[1,0,0,0,0,1],'sagittal':[0,1,0,0,0,1]};d.ImageOrientationPatient=orientations[plane];d.ImagePositionPatient=[0,0,float(instance)]
    d.PixelData=np.full((32,32),value,dtype=np.uint16).tobytes();f=io.BytesIO();pydicom.dcmwrite(f,d,enforce_file_format=True);return f.getvalue()

def test_plane_labelled_dicom_zip_loads_in_memory():
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w') as archive:
        for plane in ('axial','coronal','sagittal'):
            for instance,value in ((1,10),(2,20)):
                archive.writestr(f'{plane}/slice_{instance}.dcm',dicom_slice(plane,instance,value,series=f'1.2.3.{plane.__len__()}'))
    volumes=load_uploads([('study.zip',z.getvalue())],config()['preprocessing'])
    assert set(volumes)=={'axial','coronal','sagittal'} and volumes['axial'].shape==(2,32,32)

def test_dicom_geometry_and_series_are_checked():
    z=io.BytesIO()
    with zipfile.ZipFile(z,'w') as archive: archive.writestr('axial/slice.dcm',dicom_slice('coronal',1,10))
    with pytest.raises(InputError,match='geometry'):load_uploads([('study.zip',z.getvalue())],config()['preprocessing'])
