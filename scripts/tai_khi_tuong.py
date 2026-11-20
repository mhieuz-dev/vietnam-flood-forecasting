"""Tải khí tượng quan trắc lên lưới chung 0,1 độ (97-110 E, 8-26 N; 180 x 130) qua GEE, mỗi lượt 30 ngày.

Ghi data/khi_tuong/quan_trac/{bien}_{yyyymmdd}.npy (30, 180, 130) float32 cho bien trong:
mua_imerg (mm/ngày, tổng 48 ảnh 30 phút x 0,5 giờ), imerg_permanent (tỉ lệ ảnh có status permanent),
t2m (độ C), td2m (độ C), swvl1..4 (m3/m3, ERA5-Land).
Khoảng ngày: 2015-09-01..2023-12-31 (train, có 120 ngày đệm cho API), 2024-05-01..2024-09-30, 2025-06-01..2025-11-30.
--gsmap: chỉ tải mua_gsmap (GSMaP v8 NRT, băng hourlyPrecipRate chỉ vệ tinh, tổng 24 giờ UTC) cho hai đoạn test,
dùng thay IMERG Early trong thí nghiệm mưa vận hành (NASA Earthdata không truy cập được, ledger U3).

Chạy: .venv/bin/python scripts/tai_khi_tuong.py [--gsmap]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import ee
import numpy as np

from floodlab.config import DATA_DIR, EE_PROJECT, KHUNG_MUA_THUONG_NGUON
from floodlab.gee_tai import tai_luoi

DO = 0.1
LON0, LAT0, LON1, LAT1 = KHUNG_MUA_THUONG_NGUON
RONG, CAO = int(round((LON1 - LON0) / DO)), int(round((LAT1 - LAT0) / DO))
KHOI = 30
DOAN = [(date(2015, 9, 1), date(2023, 12, 31)), (date(2024, 5, 1), date(2024, 9, 30)), (date(2025, 6, 1), date(2025, 11, 30))]
THU_MUC = DATA_DIR / "khi_tuong" / "quan_trac"


def _imerg(d0, n):
    c = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
    mua, perm = [], []
    for i in range(n):
        d = ee.Date((d0 + timedelta(days=i)).isoformat())
        ngay = c.filterDate(d, d.advance(1, "day"))
        mua.append(ngay.select("precipitation").sum().multiply(0.5).rename(f"b{i}"))
        perm.append(ngay.map(lambda im: im.select("precipitation").multiply(0).add(
            ee.Number(ee.Algorithms.If(ee.String(im.get("status")).equals("permanent"), 1, 0)))).mean().rename(f"b{i}"))
    return {"mua_imerg": ee.Image.cat(mua), "imerg_permanent": ee.Image.cat(perm)}


def _gsmap(d0, n):
    c = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational").select("hourlyPrecipRate")
    mua = []
    for i in range(n):
        d = ee.Date((d0 + timedelta(days=i)).isoformat())
        mua.append(c.filterDate(d, d.advance(1, "day")).sum().rename(f"b{i}"))
    return {"mua_gsmap": ee.Image.cat(mua)}


def _era5(d0, n):
    c = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    bien = {"t2m": ("temperature_2m", -273.15), "td2m": ("dewpoint_temperature_2m", -273.15),
            **{f"swvl{k}": (f"volumetric_soil_water_layer_{k}", 0) for k in range(1, 5)}}
    out = {}
    for ten, (band, cong) in bien.items():
        anh = []
        for i in range(n):
            d = ee.Date((d0 + timedelta(days=i)).isoformat())
            anh.append(c.filterDate(d, d.advance(1, "day")).select(band).first().add(cong).rename(f"b{i}"))
        out[ten] = ee.Image.cat(anh)
    return out


def _mot_khoi(d0, n, cac_nguon=(_imerg, _era5)):
    for nguon in cac_nguon:
        for ten, anh in nguon(d0, n).items():
            p = THU_MUC / f"{ten}_{d0:%Y%m%d}.npy"
            if p.exists():
                continue
            tho = tai_luoi(anh, (LON0, LAT1), RONG, CAO, THU_MUC / "_tho" / p.name, pixel_m=DO, crs="EPSG:4326")
            raw = np.load(tho)
            np.save(p, np.stack([raw[f"b{i}"] for i in range(n)]).astype(np.float32))
            tho.unlink()
    return d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsmap", action="store_true")
    gsmap = ap.parse_args().gsmap
    ee.Initialize(project=EE_PROJECT)
    viec = []
    for a, b in (DOAN[1:] if gsmap else DOAN):
        d = a
        while d <= b:
            n = min(KHOI, (b - d).days + 1)
            viec.append((d, n))
            d += timedelta(days=n)
    print("so khoi:", len(viec), flush=True)
    with ThreadPoolExecutor(6) as ex:
        for i, d0 in enumerate(ex.map(lambda v: _mot_khoi(*v, cac_nguon=(_gsmap,) if gsmap else (_imerg, _era5)), viec)):
            if i % 10 == 0:
                print(f"  {i + 1}/{len(viec)} {d0}", flush=True)


if __name__ == "__main__":
    main()
