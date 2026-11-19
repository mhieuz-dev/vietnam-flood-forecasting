"""Dựng lưu vực thượng nguồn (HydroBASINS cấp 10) và AOI hạ du (GLO-30 <= 50 m, trong lãnh thổ Việt Nam)
rồi ghi cau_hinh/luu_vuc/*.geojson và results/luoi.json.

Chạy: .venv/bin/python scripts/lap_luu_vuc.py
"""
import json

import ee
import geopandas as gpd
import yaml
from shapely.geometry import shape

from floodlab.config import CAU_HINH_DIR, RESULTS_DIR
from floodlab.luoi import DIR_LUU_VUC, luoi_o, sang_utm

KHUNG = [97.0, 8.0, 110.0, 26.0]


def _do_thi_hybas():
    fc = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_10").filterBounds(ee.Geometry.Rectangle(KHUNG))
    cot = fc.reduceColumns(ee.Reducer.toList(2), ["HYBAS_ID", "NEXT_DOWN"]).get("list").getInfo()
    nguoc = {}
    for hid, xuong in cot:
        nguoc.setdefault(int(xuong), []).append(int(hid))
    return fc, nguoc


def _thuong_nguon(fc, nguoc, diem):
    ids = {int(fc.filterBounds(ee.Geometry.Point(lon, lat)).first().get("HYBAS_ID").getInfo()) for lon, lat in diem}
    ngan_xep = list(ids)
    while ngan_xep:
        for u in nguoc.get(ngan_xep.pop(), []):
            if u not in ids:
                ids.add(u)
                ngan_xep.append(u)
    hinh = fc.filter(ee.Filter.inList("HYBAS_ID", list(ids))).union(1000).geometry().simplify(500)
    return hinh, len(ids)


def _aoi(vung, cao_toi_da):
    vn = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq("ADM0_NAME", "Viet Nam")).geometry()
    vung = vung.intersection(vn, 100)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic()
    thap = dem.lte(cao_toi_da).selfMask().clip(vung)
    vec = thap.reduceToVectors(geometry=vung, scale=300, geometryType="polygon", maxPixels=1e10, bestEffort=True)
    return vec.geometry().simplify(300)


def _ghi(ten, hinh_ee):
    g = shape(hinh_ee.getInfo())
    DIR_LUU_VUC.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[g], crs="EPSG:4326").to_file(DIR_LUU_VUC / f"{ten}.geojson", driver="GeoJSON")
    return g


def main():
    ee.Initialize(project="vn-floodcast")
    cfg = yaml.safe_load((CAU_HINH_DIR / "luu_vuc.yaml").read_text(encoding="utf-8"))
    fc, nguoc = _do_thi_hybas()
    ket_qua = {}
    for lv, c in cfg["luu_vuc"].items():
        tn, so = _thuong_nguon(fc, nguoc, c["cua_ra"])
        g_tn = _ghi(f"{lv}_thuong_nguon", tn)
        vung = ee.Geometry.Rectangle(c["khung_aoi"]) if "khung_aoi" in c else tn
        g_aoi = _ghi(f"{lv}_aoi", _aoi(vung, cfg["cao_do_toi_da_aoi_m"]))
        ket_qua[lv] = {"so_hybas10": so, "km2_thuong_nguon": round(sang_utm(g_tn).area / 1e6),
                       "lat_max_thuong_nguon": round(g_tn.bounds[3], 2),
                       "km2_aoi": round(sang_utm(g_aoi).area / 1e6), "so_o": len(luoi_o(lv))}
        print(lv, ket_qua[lv], flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "luoi.json").write_text(json.dumps(ket_qua, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
