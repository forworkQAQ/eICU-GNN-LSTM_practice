import json
import sys
import types
import numpy as np
import pytest
from pathlib import Path

# Provide a minimal torch stub so ts_reader can be imported without installing
# the real torch package.
torch_stub = types.ModuleType("torch")
torch_stub.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
torch_stub.Tensor = np.ndarray
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.utils", torch_stub.utils)
sys.modules.setdefault("torch.utils.data", torch_stub.utils.data)

# Ensure project root is on the path so ``src`` can be imported
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataloader.ts_reader import collect_ts_flat_labels, slice_data


@pytest.fixture
def dummy_mmap(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    N = 6
    # ts
    ts_shape = (N, 2, 3)
    ts_data = np.arange(np.prod(ts_shape), dtype=np.float32).reshape(ts_shape)
    ts_mm = np.memmap(data_dir / "ts.dat", dtype=np.float32, mode="w+", shape=ts_shape)
    ts_mm[:] = ts_data
    ts_mm.flush()
    ts_info = {
        "name": "ts",
        "shape": list(ts_shape),
        "total": N,
        "train_len": 2,
        "val_len": 2,
        "test_len": 2,
        "columns": ["c0", "c1", "c2"],
    }
    (data_dir / "ts_info.json").write_text(json.dumps(ts_info))

    # flat
    flat_shape = (N, 2)
    flat_data = np.arange(np.prod(flat_shape), dtype=np.float32).reshape(flat_shape)
    flat_mm = np.memmap(data_dir / "flat.dat", dtype=np.float32, mode="w+", shape=flat_shape)
    flat_mm[:] = flat_data
    flat_mm.flush()
    flat_info = {
        "name": "flat",
        "shape": list(flat_shape),
        "total": N,
        "train_len": 2,
        "val_len": 2,
        "test_len": 2,
        "columns": ["f0", "f1"],
    }
    (data_dir / "flat_info.json").write_text(json.dumps(flat_info))

    # diagnoses
    diag_shape = (N, 3)
    diag_data = np.arange(np.prod(diag_shape), dtype=np.float32).reshape(diag_shape)
    diag_mm = np.memmap(data_dir / "diagnoses.dat", dtype=np.float32, mode="w+", shape=diag_shape)
    diag_mm[:] = diag_data
    diag_mm.flush()
    diag_info = {
        "name": "diagnoses",
        "shape": list(diag_shape),
        "total": N,
        "train_len": 2,
        "val_len": 2,
        "test_len": 2,
        "columns": ["d0", "d1", "d2"],
    }
    (data_dir / "diagnoses_info.json").write_text(json.dumps(diag_info))

    # labels
    label_shape = (N, 5)
    label_data = np.arange(np.prod(label_shape), dtype=np.float32).reshape(label_shape)
    label_mm = np.memmap(data_dir / "labels.dat", dtype=np.float32, mode="w+", shape=label_shape)
    label_mm[:] = label_data
    label_mm.flush()
    label_info = {
        "name": "labels",
        "shape": list(label_shape),
        "total": N,
        "train_len": 2,
        "val_len": 2,
        "test_len": 2,
        "columns": ["l0", "ihm", "l2", "los", "l4"],
    }
    (data_dir / "labels_info.json").write_text(json.dumps(label_info))

    return data_dir, diag_info


def test_slice_data(dummy_mmap):
    data_dir, diag_info = dummy_mmap
    diag_mm = np.memmap(data_dir / "diagnoses.dat", dtype=np.float32, shape=tuple(diag_info["shape"]))

    train = slice_data(diag_mm, diag_info, "train")
    val = slice_data(diag_mm, diag_info, "val")
    test = slice_data(diag_mm, diag_info, "test")

    assert train.shape == (diag_info["train_len"], diag_info["shape"][1])
    assert val.shape == (diag_info["val_len"], diag_info["shape"][1])
    assert test.shape == (diag_info["test_len"], diag_info["shape"][1])
    np.testing.assert_array_equal(train, diag_mm[:2])
    np.testing.assert_array_equal(val, diag_mm[2:4])
    np.testing.assert_array_equal(test, diag_mm[4:6])


def test_collect_ts_flat_labels(dummy_mmap):
    data_dir, diag_info = dummy_mmap
    seq, (flat, diag), labels, info, N, train_n, val_n = collect_ts_flat_labels(
        data_dir,
        ts_mask=True,
        task="ihm",
        add_diag=True,
        split="val",
        debug=0,
        split_flat_and_diag=True,
    )

    assert seq.shape == (diag_info["val_len"], 2, 3)
    assert flat.shape == (diag_info["val_len"], 2)
    assert diag.shape == (diag_info["val_len"], 3)
    assert labels.shape == (diag_info["val_len"],)

    diag_mm = np.memmap(data_dir / "diagnoses.dat", dtype=np.float32, shape=tuple(diag_info["shape"]))
    expected_diag = diag_mm[diag_info["train_len"] : diag_info["train_len"] + diag_info["val_len"]]
    np.testing.assert_array_equal(diag, expected_diag)