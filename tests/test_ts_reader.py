import sys
from pathlib import Path
import types
import numpy as np

# Stub minimal torch module to satisfy imports in ts_reader
torch_stub = types.SimpleNamespace()
utils_stub = types.SimpleNamespace()
data_stub = types.SimpleNamespace(Dataset=object)
utils_stub.data = data_stub
torch_stub.utils = utils_stub
sys.modules['torch'] = torch_stub
sys.modules['torch.utils'] = utils_stub
sys.modules['torch.utils.data'] = data_stub

# Stub convert module to avoid pandas dependency during import
convert_stub = types.ModuleType('src.dataloader.convert')
def stub_read_mm(*args, **kwargs):
    raise RuntimeError('stub')
convert_stub.read_mm = stub_read_mm
sys.modules['src.dataloader.convert'] = convert_stub

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.dataloader.ts_reader as ts_reader


def fake_read_mm(data_dir, name):
    if name == 'flat':
        data = np.zeros((4, 2), dtype=np.float32)
        info = {'train_len': 2, 'val_len': 1, 'test_len': 1, 'total': 4, 'shape': (4, 2), 'columns': ['f0', 'f1']}
    elif name == 'ts':
        data = np.zeros((4, 1, 1), dtype=np.float32)
        info = {'train_len': 2, 'val_len': 1, 'test_len': 1, 'total': 4, 'shape': (4, 1, 1), 'columns': ['t0']}
    elif name == 'diagnoses':
        data = np.arange(8, dtype=np.float32).reshape(4, 2)
        info = {'train_len': 1, 'val_len': 1, 'test_len': 2, 'total': 4, 'shape': (4, 2), 'columns': ['d0', 'd1']}
    elif name == 'labels':
        data = np.zeros((4, 5), dtype=np.float32)
        info = {
            'train_len': 2,
            'val_len': 1,
            'test_len': 1,
            'total': 4,
            'shape': (4, 5),
            'columns': ['c0', 'ihm', 'c2', 'los', 'c4'],
        }
    else:
        raise ValueError(name)
    return data, info


def test_diagnoses_slice(monkeypatch):
    monkeypatch.setattr(ts_reader, 'read_mm', fake_read_mm)
    expected = {'train': 1, 'val': 1, 'test': 2}
    for split, exp_len in expected.items():
        seq, flat, labels, info, N, train_n, val_n = ts_reader.collect_ts_flat_labels(
            'dummy', True, 'ihm', True, split=split, split_flat_and_diag=True
        )
        diag = flat[1]
        assert diag.shape[0] == exp_len
