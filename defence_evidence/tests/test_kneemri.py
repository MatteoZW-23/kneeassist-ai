import io,pickle
import numpy as np
import pytest
from src.data.kneemri import NumericUnpickler,load_kneemri
from src.utils import config

def test_original_numeric_pickle(tmp_path):
    v=np.arange(2*32*32,dtype=np.uint16).reshape(2,32,32)
    p=tmp_path/'volume.pck';p.write_bytes(pickle.dumps(v,protocol=2))
    assert np.array_equal(load_kneemri(p,config()['preprocessing']),v)

def test_reject_executable_pickle():
    class Untrusted:
        def __reduce__(self):return (eval,('1+1',))
    with pytest.raises(ValueError,match='Unsupported'):
        NumericUnpickler(io.BytesIO(pickle.dumps(Untrusted()))).load()

def test_reject_non_image(tmp_path):
    p=tmp_path/'volume.pck';p.write_bytes(pickle.dumps({'unexpected':'data'}))
    with pytest.raises(ValueError):load_kneemri(p,config()['preprocessing'])
