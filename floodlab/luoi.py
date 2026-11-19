"""Lưu vực, vùng đồng bằng ngập (AOI hạ du) và lưới ô ảnh 4.480 m trên EPSG:32648."""
import math
from functools import lru_cache

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

from floodlab.config import CAU_HINH_DIR, CRS, TILE_M

DIR_LUU_VUC = CAU_HINH_DIR / "luu_vuc"
_SANG_UTM = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform


def sang_utm(hinh):
    return transform(_SANG_UTM, hinh)


def luoi_tu_aoi(aoi_4326, luu_vuc: str) -> gpd.GeoDataFrame:
    """Mọi ô 4.480 m (gốc chia hết cho 4.480) giao với AOI."""
    aoi = sang_utm(aoi_4326)
    x0, y0, x1, y1 = aoi.bounds
    o = []
    for i in range(math.floor(x0 / TILE_M), math.ceil(x1 / TILE_M)):
        for j in range(math.floor(y0 / TILE_M), math.ceil(y1 / TILE_M)):
            b = box(i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M)
            if b.intersects(aoi):
                o.append({"tile_id": f"{luu_vuc}_{i}_{j}", "luu_vuc": luu_vuc, "geometry": b})
    return gpd.GeoDataFrame(o, geometry="geometry", crs=CRS)


@lru_cache
def _doc(ten: str):
    g = gpd.read_file(DIR_LUU_VUC / f"{ten}.geojson")
    return unary_union([shape(h) for h in g.geometry])


def luoi_o(luu_vuc: str) -> gpd.GeoDataFrame:
    return luoi_tu_aoi(_doc(f"{luu_vuc}_aoi"), luu_vuc)


def luu_vuc_thuong_nguon(luu_vuc: str):
    """Polygon EPSG:4326: hợp các HydroBASINS cấp 10 đổ về điểm cửa ra, kể cả phần ngoài Việt Nam."""
    return _doc(f"{luu_vuc}_thuong_nguon")
