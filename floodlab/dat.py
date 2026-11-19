"""Nhóm thủy văn đất (HSG) từ kết cấu SoilGrids và chỉ số CN theo lớp phủ WorldCover.

Nhóm theo Ross et al. 2018 (HYSOGs250m, Sci Data 5:180091, Bảng 1); lớp kết cấu theo tam giác kết cấu USDA;
bảng CN ở cau_hinh/cn_bang.csv (Jaafar et al. 2019, GCN250, Bảng 1; TR-55 cho đô thị).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from floodlab.config import CAU_HINH_DIR

_NHOM_THEO_KET_CAU = {
    "sand": "A",
    "sandy_loam": "B", "loamy_sand": "B",
    "clay_loam": "C", "silty_clay_loam": "C", "sandy_clay_loam": "C", "loam": "C", "silt_loam": "C", "silt": "C",
    "clay": "D", "silty_clay": "D", "sandy_clay": "D",
}
_THU_TU = {"A": 0, "B": 1, "C": 2, "D": 3}


def ket_cau_usda(set_pct: np.ndarray, cat_pct: np.ndarray) -> np.ndarray:
    c, s = np.asarray(set_pct, float), np.asarray(cat_pct, float)
    si = 100.0 - c - s
    lop = np.full(c.shape, "loam", dtype=object)
    dieu_kien = [  # thứ tự quan trọng: lớp sau ghi đè lớp trước
        ("sandy_loam", ((c >= 7) & (c < 20) & (s > 52) & (si + 2 * c >= 30)) | ((c < 7) & (si < 50) & (si + 2 * c >= 30))),
        ("silt_loam", ((si >= 50) & (c >= 12) & (c < 27)) | ((si >= 50) & (si < 80) & (c < 12))),
        ("silt", (si >= 80) & (c < 12)),
        ("sandy_clay_loam", (c >= 20) & (c < 35) & (si < 28) & (s > 45)),
        ("clay_loam", (c >= 27) & (c < 40) & (s > 20) & (s <= 45)),
        ("silty_clay_loam", (c >= 27) & (c < 40) & (s <= 20)),
        ("sandy_clay", (c >= 35) & (s > 45)),
        ("silty_clay", (c >= 40) & (si >= 40)),
        ("clay", (c >= 40) & (s <= 45) & (si < 40)),
        ("loamy_sand", (si + 1.5 * c >= 15) & (si + 2 * c < 30)),
        ("sand", si + 1.5 * c < 15),
    ]
    for ten, m in dieu_kien:
        lop[m] = ten
    return lop


def nhom_thuy_van(set_pct: np.ndarray, cat_pct: np.ndarray) -> np.ndarray:
    lop = ket_cau_usda(set_pct, cat_pct)
    return np.vectorize(_NHOM_THEO_KET_CAU.get, otypes=[object])(lop)


def nhom_pedon(cac_tang: list[np.ndarray]) -> np.ndarray:
    """Nhóm của pedon là nhóm của tầng kém thoát nước nhất tới 1 m (Ross et al. 2018)."""
    so = np.max(np.stack([np.vectorize(_THU_TU.get)(t) for t in cac_tang]), axis=0)
    return np.array(list("ABCD"), dtype=object)[so]


@lru_cache
def _bang():
    return pd.read_csv(CAU_HINH_DIR / "cn_bang.csv").set_index("worldcover")


def cn(nhom: np.ndarray, worldcover: np.ndarray) -> np.ndarray:
    b = _bang()
    nhom, wc = np.asarray(nhom), np.asarray(worldcover)
    out = np.full(wc.shape, np.nan, dtype=float)
    for k in "ABCD":
        for lop, gia_tri in b[k].items():
            out[(nhom == k) & (wc == lop)] = gia_tri
    return out


def lap_thieu(a: np.ndarray) -> tuple[np.ndarray, int]:
    """SoilGrids để trống ở đô thị, mặt nước (giá trị âm sau khi đổi đơn vị): lấy giá trị ô hợp lệ gần nhất.
    Trả mảng đã lấp và số ô phải lấp; nếu cả ô ảnh không có giá trị hợp lệ thì trả nguyên NaN."""
    from scipy.ndimage import distance_transform_edt

    out = np.where(a < 0, np.nan, a).astype(np.float32)
    thieu = np.isnan(out)
    if thieu.any() and not thieu.all():
        _, (ri, ci) = distance_transform_edt(thieu, return_indices=True)
        out = out[ri, ci]
    return out, int(thieu.sum())
