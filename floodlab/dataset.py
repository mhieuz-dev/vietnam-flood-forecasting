"""Dataset PyTorch trên shard zarr (Task 12) và bộ lấy mẫu cân bằng cho train.

Chuẩn hóa: S1 (VV, VH, dB) và DEM theo thống kê tiền huấn luyện TerraMind v1 (terratorch
terramind_register.v1_pretraining_mean/std, khóa untok_sen1grd@224 và untok_dem@224; backbone ViT không tự chuẩn
hóa, chỉ bản sinh ảnh có cờ standardize). 11 lớp tĩnh còn lại và 10 cột quan trắc theo trung bình, độ lệch tính
trên train (cau_hinh/chuan_hoa.json, tinh_chuan_hoa); mưa dự báo dùng thống kê hai cột mưa quan trắc (mua_o,
mua_thuong_nguon) vì nhánh khí tượng ghép chúng vào cùng ô của chuỗi. NaN sau chuẩn hóa thành 0. Ngày dự báo
ngoài hạn L đặt 0 sau chuẩn hóa. khoang_cach_s1 chia 30 (ngày).
co_s2 (chỉ B4): ảnh S2 mùa khô năm của ngày t, chuẩn hóa theo HLS của Prithvi-EO-2.0 (terratorch prithvi_vit
PRITHVI_V2_MEAN/STD, đơn vị phản xạ x 10000); chỉ đọc file đã tải bằng scripts/tai_s2.py.
Chỉ đọc train và val: tập test chỉ chấm qua danh_gia (cửa chia_tap.doc_test).
"""
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from floodlab import khi_tuong, mau, s2
from floodlab.config import CAU_HINH_DIR, LUU_VUC

F_CHUAN_HOA = CAU_HINH_DIR / "chuan_hoa.json"
S1_TB, S1_LS = np.array([-12.599, -20.293], np.float32), np.array([5.195, 5.890], np.float32)
S2_TB = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], np.float32)[:, None, None]
S2_LS = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], np.float32)[:, None, None]
DEM_TB, DEM_LS = 670.665, 951.272
KENH_TINH = mau.KENH_TINH[1:]
KENH_DAT = ("set_0_30", "cat_0_30", "dung_trong_0_30", "set_30_100", "cat_30_100", "cn")
KENH_DIA_HINH = ("do_doc", "hand", "log_dien_tich_thuong_nguon")
THANH_PHAN = {"dat", "cn", "dat_tho", "khi_tuong", "s1", "dia_hinh"}  # "cn": chỉ bỏ CN; "dat_tho": bỏ đất thô, giữ CN


def _chi_muc(vai_tro: str) -> pd.DataFrame:
    return pd.concat([c[c.vai_tro == vai_tro] for c in map(mau.doc_chi_muc, LUU_VUC)], ignore_index=True)


class MauNgap(Dataset):
    def __init__(self, vai_tro: str, nguon_du_bao: str = "gfs", bo: set[str] = frozenset(), tang_cuong: bool = False,
                 nguon_quan_trac: str = "imerg_final", seed: int | None = None, bo_luu_vuc: str | None = None,
                 co_s2: bool = False):
        if vai_tro not in ("train", "val"):
            raise ValueError(f"MauNgap chỉ đọc train hoặc val; {vai_tro} chấm qua danh_gia")
        if set(bo) - THANH_PHAN:
            raise ValueError(f"bo chỉ nhận {THANH_PHAN}, gặp {set(bo) - THANH_PHAN}")
        self.vai_tro, self.nguon_du_bao, self.nguon_quan_trac = vai_tro, nguon_du_bao, nguon_quan_trac
        self.bo, self.tang_cuong, self.seed, self.co_s2 = set(bo), tang_cuong, seed, co_s2
        self.d = _chi_muc(vai_tro)
        if bo_luu_vuc:  # LOBO: bỏ hẳn một lưu vực khỏi tập này
            self.d = self.d[self.d.luu_vuc != bo_luu_vuc].reset_index(drop=True)
        self.tinh_tb, self.tinh_ls, self.qt_tb, self.qt_ls = _thong_ke_chuan_hoa()
        self._g, self._rng = {}, None

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i: int) -> dict:
        r = self.d.iloc[i]
        if r.luu_vuc not in self._g:
            self._g = {r.luu_vuc: mau.mo_zarr(r.luu_vuc)}
        m = mau.doc_mau(self._g[r.luu_vuc], r, nguon_du_bao=self.nguon_du_bao, nguon_quan_trac=self.nguon_quan_trac)
        x = _chuan_hoa(m, self.bo, (self.tinh_tb, self.tinh_ls, self.qt_tb, self.qt_ls),
                       r.tile_id if self.co_s2 else None, r.ngay_t)
        s1, dem, tinh, qt, db, mn, nhan, anh_s2 = (x[k] for k in ("s1", "dem", "tinh", "qt", "db", "mn", "nhan", "s2"))
        if self.tang_cuong:
            if self._rng is None:
                self._rng = np.random.default_rng(self.seed if self.seed is not None else torch.initial_seed() % 2**32)
            k, lat = int(self._rng.integers(4)), bool(self._rng.random() < 0.5)

            def bien(a):
                a = np.rot90(a, k, axes=(-2, -1))
                return np.flip(a, -1) if lat else a

            s1, dem, tinh, nhan = bien(s1), bien(dem), bien(tinh), bien(nhan)
            if anh_s2 is not None:
                anh_s2 = bien(anh_s2)
        t = lambda a: torch.from_numpy(np.nan_to_num(np.ascontiguousarray(a, dtype=np.float32)))
        return {"s1": t(s1), "dem": t(dem), "tinh": t(tinh), "quan_trac": t(qt), "du_bao": t(db),
                "mat_na_du_bao": torch.from_numpy(mn.copy()), "L": int(r.L),
                "khoang_cach_s1": float(r.khoang_cach_s1_ngay) / 30, "nhan": torch.from_numpy(nhan.astype(np.int64)),
                "mau_id": r.mau_id, "bo": ",".join(sorted(self.bo)), **({"s2": t(anh_s2)} if self.co_s2 else {})}


def _chuan_hoa(m: dict, bo: set[str], thong_ke: tuple, tile_s2: str | None, ngay_t) -> dict:
    """Chuẩn hóa và bỏ thành phần cho một mẫu thô của mau.doc_mau; dùng chung cho MauNgap (train/val) và
    mau_cho_mo_hinh (chấm test qua danh_gia) để hai đường không lệch nhau."""
    tinh_tb, tinh_ls, qt_tb, qt_ls = thong_ke
    s1 = (m["s1"] - S1_TB[:, None, None]) / S1_LS[:, None, None]
    dem = (m["tinh"][:1] - DEM_TB) / DEM_LS
    tinh = (m["tinh"][1:] - tinh_tb) / tinh_ls
    qt = (m["quan_trac"] - qt_tb) / qt_ls
    mn = m["mat_na"].astype(bool)
    db = (m["du_bao"] - qt_tb[:2]) / qt_ls[:2] * mn[:, None]
    anh_s2 = None
    if tile_s2 is not None:
        anh_s2 = (s2.anh_mua_kho(tile_s2, None, pd.Timestamp(ngay_t).year) * 10000 - S2_TB) / S2_LS
    if "dat" in bo:
        tinh[[KENH_TINH.index(k) for k in KENH_DAT]] = 0
    if "cn" in bo:
        tinh[KENH_TINH.index("cn")] = 0
    if "dat_tho" in bo:
        tinh[[KENH_TINH.index(k) for k in KENH_DAT if k != "cn"]] = 0
    if "dia_hinh" in bo:
        tinh[[KENH_TINH.index(k) for k in KENH_DIA_HINH]] = 0
        dem = np.zeros_like(dem)
    if "s1" in bo:
        s1 = np.zeros_like(s1)
    if "khi_tuong" in bo:
        qt, db = np.zeros_like(qt), np.zeros_like(db)
    return {"s1": s1, "dem": dem, "tinh": tinh, "qt": qt, "db": db, "mn": mn, "nhan": m["nhan"], "s2": anh_s2}


def _thong_ke_chuan_hoa() -> tuple:
    ch = json.loads(F_CHUAN_HOA.read_text())
    tinh_tb, tinh_ls = (np.array(ch[k], np.float32)[:, None, None] for k in ("tinh_tb", "tinh_ls"))
    return tinh_tb, tinh_ls, np.array(ch["qt_tb"], np.float32), np.array(ch["qt_ls"], np.float32)


def mau_cho_mo_hinh(m: dict, r, bo: set[str] = frozenset(), co_s2: bool = False) -> dict:
    """Mẫu thô (mau.doc_mau) + dòng chỉ mục -> lô 1 mẫu cho mô hình, không tăng cường, chuẩn hóa như MauNgap."""
    x = _chuan_hoa(m, set(bo), _thong_ke_chuan_hoa(), r.tile_id if co_s2 else None, r.ngay_t)
    t = lambda a: torch.from_numpy(np.nan_to_num(np.ascontiguousarray(a, dtype=np.float32)))[None]
    return {"s1": t(x["s1"]), "dem": t(x["dem"]), "tinh": t(x["tinh"]), "quan_trac": t(x["qt"]), "du_bao": t(x["db"]),
            "mat_na_du_bao": torch.from_numpy(x["mn"].copy())[None], "L": torch.tensor([int(r.L)]),
            "khoang_cach_s1": torch.tensor([float(r.khoang_cach_s1_ngay) / 30]),
            **({"s2": t(x["s2"])} if co_s2 else {})}


def lay_mau_can_bang(ds: MauNgap, ty_le_o_co_ngap: float = 0.5, seed: int = 0, so_mau: int | None = None,
                     can_luu_vuc: bool = False) -> WeightedRandomSampler:
    """Rút có hoàn lại sao cho tỉ lệ mẫu có ngập (ty_le_ngap > 0) là ty_le_o_co_ngap. can_luu_vuc: trong mỗi nhóm
    (có, không ngập) mỗi lưu vực có tổng trọng số bằng nhau (mặc định tắt, giống phân bố B2 đã học)."""
    if ds.vai_tro != "train":
        raise ValueError("lay_mau_can_bang chỉ dùng cho train")
    co = ds.d.ty_le_ngap.to_numpy() > 0
    w = np.zeros(len(ds))
    for nhom, p in ((co, ty_le_o_co_ngap), (~co, 1 - ty_le_o_co_ngap)):
        if can_luu_vuc:
            lv = ds.d.luu_vuc.to_numpy()[nhom]
            u, dem = np.unique(lv, return_counts=True)
            w[nhom] = p / len(u) / dem[np.searchsorted(u, lv)]
        else:
            w[nhom] = p / max(nhom.sum(), 1)
    return WeightedRandomSampler(torch.from_numpy(w), so_mau or len(ds), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def tinh_chuan_hoa(so_mau: int = 2000, seed: int = 0) -> dict:
    """Trung bình, độ lệch của 11 lớp tĩnh (lấy 1/16 pixel mỗi mẫu) và 10 cột quan trắc trên so_mau mẫu train ngẫu
    nhiên, bỏ NaN; cột toàn NaN nhận trung bình 0, độ lệch 1 và ghi vào cot_rong. Ghi F_CHUAN_HOA."""
    d = _chi_muc("train")
    d = d.iloc[np.sort(np.random.default_rng(seed).choice(len(d), min(so_mau, len(d)), replace=False))]
    tinh, qt, g = [], [], {}
    for r in d.itertuples():
        if r.luu_vuc not in g:
            g = {r.luu_vuc: mau.mo_zarr(r.luu_vuc)}
        m = mau.doc_mau(g[r.luu_vuc], r)
        tinh.append(m["tinh"][1:, ::4, ::4].reshape(len(KENH_TINH), -1))
        qt.append(m["quan_trac"])
    tinh, qt = np.concatenate(tinh, 1), np.concatenate(qt, 0)
    rong = np.isnan(qt).all(0)  # glofas_q: tùy chọn trong plan, chưa tải nên toàn NaN
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        qt_tb, qt_ls = np.where(rong, 0.0, np.nanmean(qt, 0)), np.where(rong, 1.0, np.nanstd(qt, 0))
    ra = {"tinh_tb": np.nanmean(tinh, 1).tolist(), "tinh_ls": np.maximum(np.nanstd(tinh, 1), 1e-6).tolist(),
          "kenh_tinh": KENH_TINH, "qt_tb": qt_tb.tolist(), "qt_ls": np.maximum(qt_ls, 1e-6).tolist(),
          "cot_rong": [c for c, x in zip(khi_tuong.COT, rong) if x], "so_mau": len(d), "seed": seed,
          "nguon": "train", "luc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    F_CHUAN_HOA.write_text(json.dumps(ra, indent=1, ensure_ascii=False))
    return ra
