"""Ảnh Sentinel-2 L2A tổng hợp mùa khô (trung vị tháng 1-4, lọc mây bằng SCL) cho đối chứng B4 (Prithvi).
Chọn mùa khô vì mùa bão mây che gần hết; ảnh này là bối cảnh mặt đất trước mùa lũ, không phải trạng thái ngày t0."""
from datetime import date
from pathlib import Path

import ee
import numpy as np

from floodlab.config import DATA_DIR, TILE_PX
from floodlab.gee_tai import tai_luoi

BANG_HLS = ("B2", "B3", "B4", "B8A", "B11", "B12")  # Blue, Green, Red, Narrow NIR, SWIR1, SWIR2 như Prithvi-EO-2.0
THU_MUC = DATA_DIR / "s2"
NGUONG_XS_MAY = 40  # % xác suất mây s2cloudless; SCL L2A đầu 2018 dán nhãn sạch cho mây (Trà Khúc, 03/11)
SO_ANH_TOI_THIEU = 3  # số lần chụp sạch tối thiểu để tin trung vị của năm đó
LOP_SCL_SACH = (4, 5, 6)  # SCL: thực vật, đất trống, nước; bỏ 7 (chưa phân loại) và 11 (tuyết) vì giữ mây sót


def khoang_mua_kho(nam: int) -> tuple[date, date]:
    return date(nam, 1, 1), date(nam, 4, 30)


def _tong_hop(vung, nam: int):
    """(trung vị, số lần chụp sạch) mùa khô của năm nam."""
    tu, den = khoang_mua_kho(nam)

    def loc_may(im):
        scl = im.select("SCL")
        xs_may = ee.Image(im.get("xs")).select("probability")
        return im.updateMask(scl.remap(list(LOP_SCL_SACH), [1] * len(LOP_SCL_SACH), 0).And(xs_may.lt(NGUONG_XS_MAY)))

    # Ảnh rỗng bị che hết: năm không có ảnh nào (mùa khô 2015, trước khi S2 phóng 06/2015) vẫn ra đủ 6 băng
    rong = ee.Image.constant([0] * len(BANG_HLS)).rename(list(BANG_HLS)).toFloat().updateMask(ee.Image(0))
    sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(vung).filterDate(tu.isoformat(), den.isoformat())
    xs = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(vung).filterDate(tu.isoformat(), den.isoformat())
    # Ghép xác suất mây s2cloudless theo system:index; ảnh không có bản xác suất bị bỏ (Join mặc định không outer)
    ghep = ee.ImageCollection(ee.Join.saveFirst("xs").apply(
        sr, xs, ee.Filter.equals(leftField="system:index", rightField="system:index")))
    bo = (ghep.map(loc_may).select(list(BANG_HLS))
          .map(lambda im: im.toFloat()).merge(ee.ImageCollection([rong])))
    return bo.median(), bo.select(0).count().unmask(0)


def _anh(o_utm, nam: int):
    """Trung vị mùa khô năm nam ở pixel có >= SO_ANH_TOI_THIEU lần chụp sạch; còn lại lấy năm nam-1 (luôn trước
    ngày t, không rò rỉ; không dùng nam+1 vì sau trận lũ); thiếu cả hai thì dùng chỗ có ảnh ít hơn ngưỡng, nam trước.
    Trung vị của 1-2 ảnh không loại được mây sót (Trà Khúc 2018: trống 80%, phần còn lại phần lớn là mây, 03/11).
    Băng "bu" = 1 ở pixel lấy từ nam-1."""
    vung = ee.Geometry.Rectangle(list(o_utm.bounds), "EPSG:32648", False)
    (nay, dem_nay), (truoc, dem_truoc) = _tong_hop(vung, nam), _tong_hop(vung, nam - 1)
    a, b = dem_nay.gte(SO_ANH_TOI_THIEU), dem_truoc.gte(SO_ANH_TOI_THIEU)
    c, d = dem_nay.gt(0), dem_truoc.gt(0)
    bu = a.Not().And(b.Or(c.Not().And(d))).rename("bu")
    ket = nay.updateMask(a).unmask(truoc.updateMask(b)).unmask(nay).unmask(truoc)
    return ket.toFloat().addBands(bu.toFloat())


def _duong(tile_id: str, nam: int) -> Path:
    return THU_MUC / tile_id / f"{nam}.npz"


def ty_le_bu(tile_id: str, nam: int) -> float:
    with np.load(_duong(tile_id, nam)) as z:
        return float(z["ty_le_bu"])


def anh_mua_kho(tile_id: str, o_utm, nam: int) -> np.ndarray:
    """(6, 224, 224) float32 phản xạ (đã chia 10000); pixel không có ảnh sạch ở cả năm nam lẫn nam-1 là NaN
    (computePixels trả 0 cho băng nguyên, -inf sau toFloat, cho pixel bị che). Lưu đĩa float16 trong .npz (sai số khoảng 5e-4 trên phản
    xạ 0-1; 14.808 cặp ô-năm ở float32 là 17,8 GB), kèm ty_le_bu."""
    p = _duong(tile_id, nam)
    if p.exists():
        with np.load(p) as z:
            a = z["anh"].astype(np.float32)
        a[~np.isfinite(a)] = np.nan  # file tải trước 05/11 còn -inf ở pixel bị che
        return a
    tho = THU_MUC / "_tho" / tile_id / f"{nam}.npy"
    raw = np.load(tai_luoi(_anh(o_utm, nam), (o_utm.bounds[0], o_utm.bounds[3]), TILE_PX, TILE_PX, tho))
    mang = np.stack([raw[b] for b in BANG_HLS]).astype(np.float32) / 10000.0
    mang[:, (mang == 0).all(axis=0)] = np.nan
    mang[~np.isfinite(mang)] = np.nan  # sau toFloat, pixel bị che ở cả hai năm ra -inf (05/11: 11.481/14.808 file)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, anh=mang.astype(np.float16), ty_le_bu=np.float32((raw["bu"] > 0).mean()))
    Path(tho).unlink()
    return mang
