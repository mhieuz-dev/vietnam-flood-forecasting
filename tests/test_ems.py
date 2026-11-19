from datetime import datetime

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from floodlab import ems
from floodlab.config import CRS, PIXEL_M, TILE_M


def test_doc_anh_sau_su_kien_tu_le_ban_do():
    t = ("Pre-event image: SPOT7 © Airbus DS (2020), (acquired on 22/06/2020 at 03:16 UTC, GSD 1.5 m) "
         "Post-event image: COS MO-S k yMed © ASI (2020), distributed by e-GEOS (acquired on 19/10/2020 at 10:34 UTC)")
    assert ems.anh_sau_su_kien(t) == [("COSMO-SkyMed", datetime(2020, 10, 19, 10, 34))]


def test_doc_anh_sentinel_1_nhieu_kieu_ghi():
    t = ("Post-event image: Post-event image: Sentinel-1A (2020) (acquired on 13/10/2020 at 10:56 UTC and 10:57 UTC, "
         "GSD 10.0 m) provided under COPERNICUS")
    assert ems.anh_sau_su_kien(t) == [("Sentinel-1A", datetime(2020, 10, 13, 10, 56)),
                                      ("Sentinel-1A", datetime(2020, 10, 13, 10, 57))]
    t = "Post-event image: Sentinel-1A (2020) (acquired on 10/10/2020 at 22:44 UTC and on 18/10/2020 at 11:05 UTC, GSD"
    assert [n.date().day for _, n in ems.anh_sau_su_kien(t)] == [10, 18]
    t = "derived from post-event images. The estimated accuracy Post-event image: TerraSar-X © Infoterra (acquired on 28/10/2020 at 22:59 UTC)"
    assert ems.anh_sau_su_kien(t) == [("TerraSar-X", datetime(2020, 10, 28, 22, 59))]


def test_ve_o_to_dung_pixel():
    o = box(0, 0, TILE_M, TILE_M)
    g = gpd.GeoDataFrame(geometry=[box(0, TILE_M - 10 * PIXEL_M, 5 * PIXEL_M, TILE_M)], crs=CRS)
    m = ems.ve_o(g, o)
    assert m.shape == (224, 224) and m.sum() == 50 and m[:10, :5].all()


def test_dem_khop_bo_255_va_ngoai_vung():
    gfm = np.array([[1, 1, 0, 255], [0, 1, 0, 0]], np.uint8)
    ems_ngap = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], bool)
    vung = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], bool)
    assert ems.dem_khop(gfm, ems_ngap, vung) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_chi_so_tu_dem():
    c = ems.chi_so({"tp": 2, "fp": 1, "fn": 1, "tn": 2})
    assert c["iou"] == 0.5 and c["bo_sot"] == 1 / 3 and c["bao_nham"] == 1 / 3
