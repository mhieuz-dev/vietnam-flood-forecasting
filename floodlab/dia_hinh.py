"""Địa hình từ DEM: độ dốc, HAND (độ cao so với kênh gần nhất theo dòng chảy) và diện tích thượng nguồn."""
import numpy as np
from affine import Affine
from pysheds.sgrid import sGrid
from pysheds.sview import Raster, ViewFinder

if not hasattr(np, "in1d"):  # pysheds 0.5 còn gọi np.in1d, numpy 2.4 đã bỏ; isin cùng chức năng
    np.in1d = lambda a, b, **kw: np.isin(np.ravel(a), b, **kw)


def _raster(a: np.ndarray, pixel_m: float):
    vf = ViewFinder(affine=Affine(pixel_m, 0, 0, 0, -pixel_m, 0), shape=a.shape, nodata=np.float64(-9999))
    return Raster(a.astype(np.float64), viewfinder=vf)


def dong_chay(dem: np.ndarray, pixel_m: float):
    """Lấp trũng, xử lý vùng phẳng, hướng D8 và tích lũy dòng (số ô)."""
    r = _raster(dem, pixel_m)
    g = sGrid.from_raster(r)
    dem_sach = g.resolve_flats(g.fill_depressions(g.fill_pits(r)))
    fdir = g.flowdir(dem_sach)
    acc = g.accumulation(fdir)
    return g, dem_sach, fdir, acc


def hand(dem: np.ndarray, pixel_m: float, nguong_dong_km2: float = 1.0) -> np.ndarray:
    g, dem_sach, fdir, acc = dong_chay(dem, pixel_m)
    kenh = np.asarray(acc) * pixel_m * pixel_m / 1e6 >= nguong_dong_km2
    h = np.asarray(g.compute_hand(fdir, dem_sach, Raster(kenh, viewfinder=acc.viewfinder)), dtype=np.float32)
    return lap_hand(h, np.asarray(dem_sach), kenh)


def do_doc(dem: np.ndarray, pixel_m: float) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(dem, float), pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def lap_hand(hand: np.ndarray, dem: np.ndarray, kenh: np.ndarray) -> np.ndarray:
    """Ô HAND rỗng (dòng chảy thoát ra mép khối trước khi gặp kênh) lấy độ cao so với ô kênh gần nhất theo khoảng cách."""
    from scipy.ndimage import distance_transform_edt

    out = np.array(hand, dtype=np.float32)
    rong = np.isnan(out)
    if rong.any() and kenh.any():
        _, (ri, ci) = distance_transform_edt(~kenh, return_indices=True)
        out[rong] = np.clip(dem[rong] - dem[ri[rong], ci[rong]], 0, None)
    return out
