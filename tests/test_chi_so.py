import numpy as np
import pandas as pd
import pytest

from floodlab import chi_so


def _mang(tp, fp, fn, tn):
    nhan = np.array([1] * tp + [0] * fp + [1] * fn + [0] * tn)
    xs = np.array([0.9] * tp + [0.9] * fp + [0.1] * fn + [0.1] * tn)
    return xs, nhan


def test_csi_tay():
    xs, nhan = _mang(6, 2, 2, 10)
    k = chi_so.tinh(xs, nhan, 0.5)
    assert abs(k["csi"] - 0.6) < 1e-9 and abs(k["f1"] - 0.75) < 1e-9 and abs(k["iou"] - 0.6) < 1e-9


def test_bo_qua_255():
    xs, nhan = _mang(6, 2, 2, 10)
    nhan = nhan.copy()
    nhan[0] = 255
    assert chi_so.bang_nham_lan(xs, nhan, 0.5)["tp"] == 5


def test_chi_so_rong_tra_nan():
    k = chi_so.tinh(np.array([0.1, 0.2]), np.array([0, 0]), 0.5)
    assert np.isnan(k["csi"]) and np.isnan(k["f1"]) and np.isnan(k["pr_auc"])


def test_ece_hoan_hao():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    assert chi_so.tinh(p, y, 0.5)["ece_15"] < 0.01


def test_holm_tay():
    assert np.allclose(chi_so.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_bootstrap_on_dinh():
    ds = pd.DataFrame({"tile_id": [f"o{i % 7}" for i in range(50)], "tran": ["a"] * 25 + ["b"] * 25,
                       "tp": np.arange(50) % 5, "fp": 1, "fn": 1, "tn": 10})
    f = lambda d: d.tp.sum() / (d.tp.sum() + d.fp.sum() + d.fn.sum())
    a = chi_so.bootstrap_cum(ds, f, B=300, seed=3)
    b = chi_so.bootstrap_cum(ds, f, B=300, seed=3)
    assert a == b and a[1] <= a[0] <= a[2]


def test_bootstrap_ghep_cap_hieu_0_khi_giong_nhau():
    ds = pd.DataFrame({"tile_id": [f"o{i}" for i in range(30)], "tran": "a", "tp": 3, "fp": 1, "fn": 1, "tn": 5})
    f = lambda d: d.tp.sum() / (d.tp.sum() + d.fp.sum() + d.fn.sum())
    hieu, lo, hi, p = chi_so.bootstrap_ghep_cap(ds, ds, f, B=200, seed=0)
    assert hieu == 0 and lo == 0 and hi == 0 and p == 1.0


def test_bootstrap_ty_so_khop_ban_cham():
    rng = np.random.default_rng(1)
    ds = pd.DataFrame({"tran": np.repeat(["a", "b", "c"], 40), "tile_id": [f"o{i % 25}" for i in range(120)],
                       "tp": rng.integers(0, 50, 120), "fp": rng.integers(0, 30, 120), "fn": rng.integers(0, 30, 120)})
    f = lambda d: d.tp.sum() / (d.tp.sum() + d.fp.sum() + d.fn.sum())
    cham = chi_so.bootstrap_cum(ds, f, B=1500, seed=0)
    nhanh = chi_so.bootstrap_ty_so(ds, ["tp"], ["tp", "fp", "fn"], B=1500, seed=0)
    assert nhanh[0] == pytest.approx(cham[0]) and abs(nhanh[1] - cham[1]) < 0.015 and abs(nhanh[2] - cham[2]) < 0.015
    assert nhanh == chi_so.bootstrap_ty_so(ds, ["tp"], ["tp", "fp", "fn"], B=1500, seed=0)
