"""Lắp mẫu (tile_id, ngày mục tiêu t, L) và ghi shard zarr theo lưu vực.

Một mẫu: ảnh S1 trạng thái mới nhất <= t0 = t - L (30 ngày), lớp tĩnh, chuỗi quan trắc 31 ngày tới t0, mưa dự báo
3 ngày sau t0, nhãn GFM ngày t. Nhãn trạng thái (cho B6 và tập con ngập mới) là bản đồ GFM mới nhất <= t0 trong 30
ngày, không buộc trùng ngày ảnh S1 vì sản phẩm GFM nhiều ngày không phủ ô dù S1 có chụp; không có thì toàn 255.
Mặt nạ loại trừ hợp (data/nhan/_loai_tru) chỉ áp cho nhãn từ sản phẩm GFM cũ thiếu lớp loại trừ (_can_mat_na_gop).
Mẫu thiếu dữ liệu bị loại và ghi lý do vào bi_loai.csv.

data/mau/{lv}.zarr:
  tinh (n_o, 12, 224, 224) f16 theo KENH_TINH; s1 (n_s1, 2, 224, 224) f16; nhan_trang_thai (n_s1, 224, 224) u8
  (i_s1 đánh theo cặp ngày ảnh S1, ngày nhãn trạng thái);
  nhan (n_nhan, 224, 224) u8; quan_trac (n_mau, 2, 31, 10) f32 theo NGUON_QT; du_bao (n_mau, 2, 3, 2) f32 theo
  NGUON_DB; mat_na (n_mau, 3) bool. chi_muc.parquet nối mẫu với các chỉ số i_o, i_s1, i_nhan.
"""
import csv
import json
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import zarr

from floodlab import du_bao, khi_tuong, s1
from floodlab.config import CUA_SO, DATA_DIR, LUU_VUC_KIEM_DINH, NHAN_BO_QUA, TILE_PX
from floodlab.luoi import luoi_o

DIR_MAU = DATA_DIR / "mau"
KENH_TINH = ("dem", "do_doc", "hand", "log_dien_tich_thuong_nguon", "set_0_30", "cat_0_30", "dung_trong_0_30",
             "set_30_100", "cat_30_100", "cn", "worldcover", "nuoc_thuong_xuyen")
NGUON_QT = ("imerg_final", "gsmap_nrt")
NGUON_DB = ("gfs", "imerg_final")
LOAI_TEST = ("test_a", "test_b")
KHOI_MANG_NHO = 4096  # số mẫu mỗi khối cho mảng nhỏ (quan_trac, du_bao, mat_na)


def _trong(n: date, ten: str) -> bool:
    a, b = (date.fromisoformat(x) for x in CUA_SO[ten])
    return a <= n <= b


def vai_tro(luu_vuc: str, t: date, loai: str) -> str:
    """test_a/test_b giữ nguyên; lu/khong_lu thành train, riêng lưu vực kiểm định thành val. Ngày lệch cửa sổ là lỗi."""
    if loai in LOAI_TEST:
        if not _trong(t, loai):
            raise ValueError(f"{t} gắn {loai} nhưng nằm ngoài cửa sổ {loai}")
        return loai
    if not _trong(t, "train"):
        raise ValueError(f"{t} ({loai}) nằm ngoài cửa sổ train 2016-2023")
    return "val" if luu_vuc == LUU_VUC_KIEM_DINH else "train"


def _doc_nhan(tile_id: str, n: date) -> np.ndarray | None:
    p = DATA_DIR / "nhan" / tile_id / f"{n:%Y%m%d}.npy"
    return np.load(p) if p.exists() else None


def _loai_tru(tile_id: str) -> np.ndarray:
    p = DATA_DIR / "nhan" / "_loai_tru" / f"{tile_id}.npy"
    return np.load(p) if p.exists() else np.zeros((TILE_PX, TILE_PX), bool)


@lru_cache(maxsize=4096)
def _ngay_co_nhan(tile_id: str) -> list[date]:
    d = DATA_DIR / "nhan" / tile_id
    return sorted(date(int(f.stem[:4]), int(f.stem[4:6]), int(f.stem[6:8])) for f in d.glob("*.npy")) if d.exists() else []


@lru_cache(maxsize=8)
def _bang_ty_le(luu_vuc: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "nhan" / "_ty_le" / f"{luu_vuc}.csv", parse_dates=["ngay"])


def _can_mat_na_gop(tile_id: str, n: date) -> bool:
    """Mặt nạ loại trừ hợp chỉ cần cho nhãn từ sản phẩm GFM thiếu lớp loại trừ riêng (co_loai_tru = 0, phần lớn
    2016-2017); nhãn có lớp riêng đã mang 255 đúng chỗ. Áp cho mọi nhãn thì mặt nạ hợp (gom từ hàng trăm ảnh, mọi
    quỹ đạo) phủ 97% ô Gianh, Vu Gia và cắt khoảng 70% pixel hợp lệ còn lại. csv chỉ ghi cờ sản phẩm cuối của ngày
    nên 2016-2017 luôn áp; ngày không có trong csv cũng áp."""
    if n.year <= 2017:
        return True
    b = _bang_ty_le(tile_id.rsplit("_", 2)[0])
    r = b[b.ngay == pd.Timestamp(n)]
    return r.empty or bool((r.co_loai_tru == 0).any())


def mau_mot(tile_id: str, t: date, loai: str, L: int):
    """Trả ({chi_muc, s1, nhan, nhan_trang_thai, quan_trac, du_bao, mat_na}, None) hoặc (None, lý do loại)."""
    lv = tile_id.rsplit("_", 2)[0]
    t0 = t - timedelta(days=L)
    nhan = _doc_nhan(tile_id, t)
    if nhan is None:
        return None, "khong_co_nhan_ngay_t"
    lt = _loai_tru(tile_id)
    if _can_mat_na_gop(tile_id, t):
        nhan = np.where(lt, NHAN_BO_QUA, nhan).astype(np.uint8)
    if (nhan == NHAN_BO_QUA).all():
        return None, "nhan_toan_255"
    tt = s1.anh_trang_thai(tile_id, t0)
    if tt is None:
        return None, "khong_co_s1_30_ngay"
    ngay_s1, anh = tt
    qt = khi_tuong.chuoi_quan_trac(tile_id, t0, "imerg_final")
    if qt is None:
        return None, "thieu_quan_trac"
    gfs = du_bao.khung_du_bao(tile_id, t0, L, "gfs")
    if gfs is None:
        return None, "thieu_du_bao_gfs"
    bt = du_bao.khung_du_bao(tile_id, t0, L, "imerg_final")
    if bt is None:
        return None, "thieu_mua_biet_truoc"
    gs = khi_tuong.chuoi_quan_trac(tile_id, t0, "gsmap_nrt") if loai in LOAI_TEST else None
    ngay_tt = max((n for n in _ngay_co_nhan(tile_id) if t0 - timedelta(days=30) <= n <= t0), default=None)
    nhan_tt = (np.full_like(nhan, NHAN_BO_QUA) if ngay_tt is None
               else np.where(lt & _can_mat_na_gop(tile_id, ngay_tt), NHAN_BO_QUA,
                             _doc_nhan(tile_id, ngay_tt)).astype(np.uint8))
    hop_le = nhan != NHAN_BO_QUA
    chi_muc = {
        "tile_id": tile_id, "luu_vuc": lv, "ngay_t": t, "L": L, "t0": t0, "ngay_s1_trang_thai": ngay_s1, "ngay_nhan_trang_thai": ngay_tt,
        "khoang_cach_s1_ngay": (t - ngay_s1).days, "vai_tro": vai_tro(lv, t, loai), "loai_ngay": loai,
        "ty_le_ngap": float((nhan == 1).sum() / hop_le.sum()),
        "co_ngap_moi": bool(((nhan == 1) & (nhan_tt == 0)).any()),  # chỉ pixel trạng thái đã biết là khô
    }
    return {
        "chi_muc": chi_muc, "s1": anh, "nhan": nhan, "nhan_trang_thai": nhan_tt,
        "quan_trac": np.stack([qt, gs if gs is not None else np.full_like(qt, np.nan)]),
        "du_bao": np.stack([gfs[0], bt[0]]), "mat_na": gfs[1],
    }, None


class _Ghi:
    """Ghi nối dần từng mảng zarr (zstd mức 3) để lưu vực lớn không phải giữ hết trong RAM. Mảng ảnh mỗi mẫu một
    khối; mảng nhỏ (dưới 16 KB một mẫu) gộp KHOI_MANG_NHO mẫu một khối. Khóa khối phẳng (c.i.0.0) vì khóa lồng thư
    mục làm mỗi mẫu tốn thêm vài thư mục 4 KB: ca.zarr từng chiếm 477 MB cho du_bao chỉ có 1,5 MB dữ liệu."""

    def __init__(self, g, ten, dang, kieu):
        from zarr.codecs import BloscCodec

        nen = BloscCodec(cname="zstd", clevel=3)
        khoi = KHOI_MANG_NHO if np.prod(dang) * np.dtype(kieu).itemsize < 16384 else 1
        self.a = g.create_array(ten, shape=(0, *dang), chunks=(khoi, *dang), dtype=kieu, compressors=nen,
                                chunk_key_encoding={"name": "default", "separator": "."})
        self.dem = 0
        self.dem_buf, self.buf = 0, []

    def them(self, x) -> int:
        self.buf.append(x)
        i = self.dem + len(self.buf) - 1
        if len(self.buf) >= 64:
            self.xa()
        return i

    def xa(self):
        if self.buf:
            n = self.dem + len(self.buf)
            self.a.resize((n, *self.a.shape[1:]))
            self.a[self.dem:n] = np.stack(self.buf)
            self.dem, self.buf = n, []


def lap(luu_vuc: str) -> dict[str, int]:
    """Lắp mọi mẫu của lưu vực từ data/mau/ngay_muc_tieu/{lv}.json; ghi zarr, chi_muc, bi_loai; trả số đếm."""
    muc_tieu = {date.fromisoformat(k): v for k, v in
                json.loads((DIR_MAU / "ngay_muc_tieu" / f"{luu_vuc}.json").read_text())["ngay"].items()}
    g = zarr.open_group(DIR_MAU / f"{luu_vuc}.zarr", mode="w")
    o = luoi_o(luu_vuc)
    P = TILE_PX
    ghi = {"tinh": _Ghi(g, "tinh", (len(KENH_TINH), P, P), "float16"), "s1": _Ghi(g, "s1", (2, P, P), "float16"),
           "nhan_trang_thai": _Ghi(g, "nhan_trang_thai", (P, P), "uint8"), "nhan": _Ghi(g, "nhan", (P, P), "uint8"),
           "quan_trac": _Ghi(g, "quan_trac", (len(NGUON_QT), 31, 10), "float32"),
           "du_bao": _Ghi(g, "du_bao", (len(NGUON_DB), 3, 2), "float32"), "mat_na": _Ghi(g, "mat_na", (3,), "bool")}
    dong, loai_bo = [], []
    for tid in o.tile_id:
        z = np.load(DATA_DIR / "tinh" / f"{tid}.npz")
        i_o = None
        i_s1, i_nhan = {}, {}
        for t, loai in sorted(muc_tieu.items()):
            for L in (1, 2, 3):
                ket, ly_do = mau_mot(tid, t, loai, L)
                if ket is None:
                    if ly_do != "khong_co_nhan_ngay_t":  # ô ngoài vùng chụp ngày t: không phải mẫu, không ghi
                        loai_bo.append({"tile_id": tid, "ngay_t": t, "L": L, "ly_do": ly_do})
                    continue
                if i_o is None:
                    i_o = ghi["tinh"].them(np.stack([z[k].astype(np.float16) for k in KENH_TINH]))
                cm = ket["chi_muc"]
                k_s1 = (cm["ngay_s1_trang_thai"], cm["ngay_nhan_trang_thai"])
                if k_s1 not in i_s1:
                    i_s1[k_s1] = ghi["s1"].them(ket["s1"])
                    ghi["nhan_trang_thai"].them(ket["nhan_trang_thai"])
                if t not in i_nhan:
                    i_nhan[t] = ghi["nhan"].them(ket["nhan"])
                for k in ("quan_trac", "du_bao", "mat_na"):
                    ghi[k].them(ket[k])
                dong.append({**cm, "i_o": i_o, "i_s1": i_s1[k_s1], "i_nhan": i_nhan[t]})
    for w in ghi.values():
        w.xa()
    cm = pd.DataFrame(dong)
    cm.insert(0, "mau_id", [f"{r.tile_id}_{r.ngay_t:%Y%m%d}_L{r.L}" for r in cm.itertuples()])
    cap = cm.groupby(["tile_id", "ngay_t"]).vai_tro.nunique()
    if (cap > 1).any():
        raise ValueError("có (tile_id, ngay_t) mang hai vai trò")
    cm.to_parquet(DIR_MAU / f"chi_muc_{luu_vuc}.parquet", index=False)
    with open(DIR_MAU / f"bi_loai_{luu_vuc}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_id", "ngay_t", "L", "ly_do"])
        w.writeheader()
        w.writerows(loai_bo)
    return {"giu": len(cm), **pd.Series([x["ly_do"] for x in loai_bo], dtype=str).value_counts().to_dict()}


def doc_chi_muc(luu_vuc: str) -> pd.DataFrame:
    """chi_muc của lưu vực kèm i_mau: vị trí dòng, cũng là chỉ số của mẫu trong quan_trac, du_bao, mat_na (ghi
    cùng thứ tự; đã kiểm trên cả 6 lưu vực: mat_na khớp L và cột GSMaP chỉ có giá trị ở mẫu test, 100%)."""
    return pd.read_parquet(DIR_MAU / f"chi_muc_{luu_vuc}.parquet").assign(i_mau=lambda d: np.arange(len(d)))


def mo_zarr(luu_vuc: str):
    return zarr.open_group(DIR_MAU / f"{luu_vuc}.zarr", mode="r")


def doc_mau(g, r, nguon_du_bao: str = "gfs", nguon_quan_trac: str = "imerg_final") -> dict:
    """Một mẫu từ shard: ảnh float32, nhãn uint8, chuỗi quan trắc (31, 10) và khung dự báo (3, 2) theo nguồn chọn.
    nguon_du_bao='ecmwf' (chỉ chấm test, không có trong zarr): khung tính lúc đọc từ cache, t0 = t - L."""
    if nguon_du_bao == "ecmwf":
        t0 = pd.Timestamp(r.ngay_t).date() - timedelta(days=int(r.L))
        k = du_bao.khung_du_bao(r.tile_id, t0, int(r.L), "ecmwf")
        if k is None:
            raise RuntimeError(f"thiếu ECMWF chu kỳ {t0 + timedelta(days=1)} cho {r.tile_id}: chạy scripts/tai_ecmwf.py")
        return doc_mau(g, r, "gfs", nguon_quan_trac) | {"du_bao": k[0]}
    return {"tinh": g["tinh"][r.i_o].astype(np.float32), "s1": g["s1"][r.i_s1].astype(np.float32),
            "nhan": g["nhan"][r.i_nhan], "nhan_trang_thai": g["nhan_trang_thai"][r.i_s1],
            "quan_trac": g["quan_trac"][r.i_mau, NGUON_QT.index(nguon_quan_trac)],
            "du_bao": g["du_bao"][r.i_mau, NGUON_DB.index(nguon_du_bao)], "mat_na": g["mat_na"][r.i_mau]}
