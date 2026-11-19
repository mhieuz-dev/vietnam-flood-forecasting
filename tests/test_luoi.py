import pytest
from shapely.geometry import box

from floodlab import luoi
from floodlab.config import LUU_VUC, TILE_M

AOI_THU = box(107.50, 16.40, 107.66, 16.54)  # khoảng 17 x 15 km quanh Huế, EPSG:4326


def test_goc_o_chia_het():
    g = luoi.luoi_tu_aoi(AOI_THU, "huong")
    b = g.geometry.bounds
    assert ((b.minx % TILE_M) == 0).all() and ((b.miny % TILE_M) == 0).all()
    assert ((b.maxx - b.minx) == TILE_M).all() and ((b.maxy - b.miny) == TILE_M).all()


def test_tile_id_duy_nhat_va_dung_mau():
    g = luoi.luoi_tu_aoi(AOI_THU, "huong")
    assert g.tile_id.is_unique
    assert g.tile_id.str.match(r"^huong_\d+_\d+$").all()
    assert g.crs.to_string() == "EPSG:32648"


def test_o_giao_aoi():
    g = luoi.luoi_tu_aoi(AOI_THU, "huong")
    aoi_utm = luoi.sang_utm(AOI_THU)
    assert len(g) > 0 and g.geometry.intersects(aoi_utm).all()


@pytest.mark.skipif(not luoi.DIR_LUU_VUC.joinpath("hong_thai_binh_thuong_nguon.geojson").exists(),
                    reason="chưa chạy scripts/lap_luu_vuc.py")
def test_thuong_nguon_hong_vuot_bien_gioi():
    assert luoi.luu_vuc_thuong_nguon("hong_thai_binh").bounds[3] > 23.4


@pytest.mark.skipif(not all(luoi.DIR_LUU_VUC.joinpath(f"{lv}_aoi.geojson").exists() for lv in LUU_VUC),
                    reason="chưa chạy scripts/lap_luu_vuc.py")
def test_moi_luu_vuc_co_o():
    for lv in LUU_VUC:
        assert len(luoi.luoi_o(lv)) > 0
