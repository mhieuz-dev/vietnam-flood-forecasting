"""Ảnh Sentinel-1 GRD (VV, VH, dB) theo ô và ngày chụp UTC; ảnh trạng thái là ảnh chụp gần nhất <= t0."""
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np

from floodlab.config import DATA_DIR, TILE_PX
from floodlab.gee_tai import tai_luoi

THU_MUC = DATA_DIR / "s1"


def _bo_suu_tap(hinh_ee, tu: date, den: date):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(hinh_ee)
            .filterDate(tu.isoformat(), (den + timedelta(days=1)).isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def ngay_chup(hinh_4326, tu: date, den: date) -> list[date]:
    """Các ngày UTC có ảnh S1 IW (VV+VH) cắt vùng, cả hai chiều quỹ đạo."""
    g = ee.Geometry(hinh_4326.__geo_interface__)
    ms = _bo_suu_tap(g, tu, den).aggregate_array("system:time_start").getInfo()
    return sorted({_utc(m) for m in ms})


def _utc(ms: int) -> date:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


def tai(tile_id: str, o_utm, ngay: date) -> Path | None:
    """Ghép mọi ảnh S1 chụp trong ngày UTC đó phủ ô, lấy trung bình VV, VH ở 20 m. Trả None nếu không có ảnh."""
    p = THU_MUC / tile_id / f"{ngay:%Y%m%d}.npy"
    if p.exists():
        return p
    anh = _bo_suu_tap(ee.Geometry.Rectangle(list(o_utm.bounds), "EPSG:32648", False), ngay, ngay).select(["VV", "VH"]).mean()
    tho = THU_MUC / "_tho" / tile_id / f"{ngay:%Y%m%d}.npy"
    try:
        raw = np.load(tai_luoi(anh, (o_utm.bounds[0], o_utm.bounds[3]), TILE_PX, TILE_PX, tho))
    except ee.ee_exception.EEException as e:
        if "no bands" in str(e).lower() or "0 bands" in str(e):
            return None
        raise
    mang = np.stack([raw["VV"], raw["VH"]]).astype(np.float16)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, mang)
    tho.unlink()
    return p


def anh_trang_thai(tile_id: str, t0: date, toi_da_ngay: int = 30) -> tuple[date, np.ndarray] | None:
    """Ảnh có ngày chụp lớn nhất mà <= t0 và >= t0 - toi_da_ngay; không bao giờ là ảnh sau t0."""
    d = THU_MUC / tile_id
    if not d.exists():
        return None
    co = [date(int(f.stem[:4]), int(f.stem[4:6]), int(f.stem[6:8])) for f in d.glob("*.npy")]
    hop = [n for n in co if t0 - timedelta(days=toi_da_ngay) <= n <= t0]
    if not hop:
        return None
    n = max(hop)
    return n, np.load(d / f"{n:%Y%m%d}.npy")


def ngay_trang_thai_can(ngay_co_anh: list[date], ngay_muc_tieu: list[date], toi_da_ngay: int = 30) -> set[date]:
    """Ngày ảnh trạng thái cần tải: với mỗi t và L = 1..3, ảnh mới nhất trong [t0 - toi_da_ngay, t0], t0 = t - L."""
    co = sorted(ngay_co_anh)
    can = set()
    for t in ngay_muc_tieu:
        for L in (1, 2, 3):
            t0 = t - timedelta(days=L)
            hop = [n for n in co if t0 - timedelta(days=toi_da_ngay) <= n <= t0]
            if hop:
                can.add(hop[-1])
    return can
