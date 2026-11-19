"""Khí tượng quan trắc: độ ẩm tương đối (Magnus), chỉ số mưa tích lũy trước (API), khung 31 ngày kết thúc ở t0.

Lưới khí tượng chung: EPSG:4326, 0,1 độ, góc trên-trái (97,0 E; 26,0 N), 130 cột x 180 hàng (KHUNG_MUA_THUONG_NGUON).
"""
import glob
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

from floodlab.config import API_K, DATA_DIR, KHUNG_MUA_THUONG_NGUON, SO_NGAY_QUAN_TRAC

# Thứ tự 10 cột của chuỗi quan trắc (plan Task 8)
COT = ("mua_o", "mua_thuong_nguon", "t2m", "rh", "swvl1", "swvl2", "swvl3", "swvl4", "api", "glofas_q")
MAGNUS_A, MAGNUS_B = 17.625, 243.04  # Alduchov & Eskridge (1996)


def do_am_tuong_doi(t2m_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    e = lambda t: np.exp(MAGNUS_A * t / (MAGNUS_B + t))
    return np.clip(100.0 * e(np.asarray(td_c, float)) / e(np.asarray(t2m_c, float)), 0, 100)


def api(mua: np.ndarray, k: float = API_K) -> np.ndarray:
    """API_t = k * API_{t-1} + P_t, khởi tạo 0 ở phần tử đầu (gọi với chuỗi bắt đầu từ t0 - 120)."""
    out = np.zeros(len(mua), float)
    acc = 0.0
    for i, p in enumerate(np.nan_to_num(np.asarray(mua, float))):
        acc = k * acc + p
        out[i] = acc
    return out


def cua_so_quan_trac(t0: date) -> list[date]:
    """31 ngày UTC từ t0 - 30 đến t0 (tính cả t0, vì thời điểm phát dự báo là cuối ngày t0)."""
    return [t0 - timedelta(days=SO_NGAY_QUAN_TRAC - 1 - i) for i in range(SO_NGAY_QUAN_TRAC)]


# ---------------- trích chuỗi theo ô từ lưới chung 0,1 độ ----------------
LON0, _, _, LAT1 = KHUNG_MUA_THUONG_NGUON
DIR_QT = DATA_DIR / "khi_tuong" / "quan_trac"
DIR_CHUOI = DATA_DIR / "khi_tuong" / "chuoi"
BIEN_LUOI = ("mua_imerg", "t2m", "td2m", "swvl1", "swvl2", "swvl3", "swvl4", "imerg_permanent")


def o_luoi(lon: float, lat: float, do: float) -> tuple[int, int]:
    """(hàng, cột) của ô lưới chứa điểm; lưới bắt đầu ở (97 E, 26 N), hàng tăng về phía nam.
    Lưới 0,1 độ của GEE là ô vuông (góc ô ở bội số 0,1); lưới 0,25 độ GFS là điểm (lấy điểm gần nhất)."""
    if abs(do - 0.25) < 1e-9:
        return int(round((LAT1 - lat) / do)), int(round((lon - LON0) / do))
    return int((LAT1 - lat) / do), int((lon - LON0) / do)


def mat_na_luoi(hinh_4326, do: float, cao: int, rong: int) -> np.ndarray:
    """Ô lưới có tâm nằm trong hình (dùng cho mưa trung bình thượng nguồn)."""
    from shapely import contains_xy

    hang, cot = np.mgrid[0:cao, 0:rong]
    if abs(do - 0.25) < 1e-9:
        lat, lon = LAT1 - hang * do, LON0 + cot * do
    else:
        lat, lon = LAT1 - (hang + 0.5) * do, LON0 + (cot + 0.5) * do
    return contains_xy(hinh_4326, lon, lat)


def _ngay_tu_ten(f: str) -> date:
    s = f.rsplit("_", 1)[1][:8]
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def trich_chuoi_luu_vuc(luu_vuc: str) -> None:
    """Ghi data/khi_tuong/chuoi/{luu_vuc}.npz: ngay (ordinal), tile_id, gia_tri (n_o, n_ngay, 10) theo COT,
    imerg_permanent (n_ngay). glofas_q để NaN (chưa dùng)."""
    from floodlab.luoi import luoi_o, luu_vuc_thuong_nguon
    from pyproj import Transformer

    o = luoi_o(luu_vuc)
    ve_ll = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True).transform
    vt = [o_luoi(*ve_ll(g.centroid.x, g.centroid.y), 0.1) for g in o.geometry]
    hang, cot = np.array([v[0] for v in vt]), np.array([v[1] for v in vt])
    mn_tn = mat_na_luoi(luu_vuc_thuong_nguon(luu_vuc), 0.1, 180, 130)
    files = {b: sorted(glob.glob(str(DIR_QT / f"{b}_*.npy"))) for b in BIEN_LUOI}
    ngay = []
    for f in files["mua_imerg"]:
        d0 = _ngay_tu_ten(f)
        ngay += [d0 + timedelta(days=i) for i in range(np.load(f, mmap_mode="r").shape[0])]
    n_o, n_ng = len(o), len(ngay)
    gt = np.full((n_o, n_ng, len(COT)), np.nan, np.float32)

    def doc(bien):
        a = np.concatenate([np.load(f) for f in files[bien]])
        return a

    mua = doc("mua_imerg")
    gt[:, :, 0] = mua[:, hang, cot].T
    gt[:, :, 1] = np.nanmean(mua[:, mn_tn], axis=1)[None, :]
    perm = np.nanmean(doc("imerg_permanent")[:, mn_tn], axis=1)
    del mua
    t2m = doc("t2m")[:, hang, cot].T
    td = doc("td2m")[:, hang, cot].T
    gt[:, :, 2] = t2m
    gt[:, :, 3] = do_am_tuong_doi(t2m, td)
    for k in range(1, 5):
        gt[:, :, 3 + k] = doc(f"swvl{k}")[:, hang, cot].T
    # API theo từng đoạn ngày liên tục (đoạn đầu có >= 120 ngày đệm trước ngày cần dùng)
    nd = np.array([d.toordinal() for d in ngay])
    cat = np.where(np.diff(nd) != 1)[0] + 1
    for a, b in zip(np.r_[0, cat], np.r_[cat, n_ng]):
        for i in range(n_o):
            gt[i, a:b, 8] = api(gt[i, a:b, 0])
    DIR_CHUOI.mkdir(parents=True, exist_ok=True)
    # GSMaP NRT (chỉ có ở hai đoạn test) cho thí nghiệm mưa vận hành: mua_o, mua_thuong_nguon, api; nan ngoài đoạn
    gs = np.full((n_o, n_ng, 3), np.nan, np.float32)
    f_gs = sorted(glob.glob(str(DIR_QT / "mua_gsmap_*.npy")))
    if f_gs:
        vi_tri = {d.toordinal(): i for i, d in enumerate(ngay)}
        for f in f_gs:
            a, d0 = np.load(f), _ngay_tu_ten(f)
            for k in range(a.shape[0]):
                j = vi_tri.get((d0 + timedelta(days=k)).toordinal())
                if j is not None:
                    gs[:, j, 0] = a[k][hang, cot]
                    gs[:, j, 1] = np.nanmean(a[k][mn_tn])
        co = ~np.isnan(gs[0, :, 0])
        cat_gs = np.where(np.diff(co.astype(int)) != 0)[0] + 1
        for a, b in zip(np.r_[0, cat_gs], np.r_[cat_gs, n_ng]):
            if co[a]:
                for i in range(n_o):
                    gs[i, a:b, 2] = api(gs[i, a:b, 0])
    np.savez_compressed(DIR_CHUOI / f"{luu_vuc}.npz", ngay=nd, tile_id=np.array(o.tile_id, dtype=str), gia_tri=gt,
                        imerg_permanent=perm.astype(np.float32), gsmap=gs)


@lru_cache(maxsize=8)
def _chuoi(luu_vuc: str):
    d = np.load(DIR_CHUOI / f"{luu_vuc}.npz")
    return ({int(n): i for i, n in enumerate(d["ngay"])}, {t: i for i, t in enumerate(d["tile_id"])}, d["gia_tri"],
            d["gsmap"] if "gsmap" in d else None)


def chuoi_quan_trac(tile_id: str, t0: date, nguon_mua: str = "imerg_final") -> np.ndarray | None:
    """(31, 10) theo COT cho các ngày t0-30..t0; None nếu thiếu ngày nào (mẫu sẽ bị loại có ghi lý do).
    nguon_mua="gsmap_nrt" thay mua_o, mua_thuong_nguon, api bằng GSMaP NRT (thay IMERG Early, xem ledger U3)."""
    if nguon_mua not in ("imerg_final", "gsmap_nrt"):
        raise ValueError(nguon_mua)
    luu_vuc = tile_id.rsplit("_", 2)[0]
    chi_so_ngay, chi_so_o, gt, gs = _chuoi(luu_vuc)
    idx = [chi_so_ngay.get(d.toordinal()) for d in cua_so_quan_trac(t0)]
    if any(i is None for i in idx):
        return None
    out = gt[chi_so_o[tile_id], idx]
    if nguon_mua == "gsmap_nrt":
        if gs is None:
            return None
        out = out.copy()
        out[:, [0, 1, 8]] = gs[chi_so_o[tile_id], idx]
    return None if np.isnan(out[:, :9]).any() else out
