"""Restricted importer for the original local KneeMRI numeric-array pickles."""
import pickle, codecs
import numpy as np
from src.data.preprocessing import validate_volume

class NumericUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('_codecs','encode'):codecs.encode}
        if (module,name) not in allowed:raise ValueError('Unsupported pickle object')
        return allowed[module,name]

def load_kneemri(path,cfg):
    if path.stat().st_size>cfg['max_upload_mb']*1024**2:raise ValueError('Volume exceeds size limit')
    with path.open('rb') as stream:volume=NumericUnpickler(stream,encoding='latin1').load()
    return validate_volume(volume,cfg)

