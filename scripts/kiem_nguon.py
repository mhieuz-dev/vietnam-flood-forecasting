"""Dò từng nguồn dữ liệu bằng một yêu cầu nhỏ; in OK/LOI kèm thời gian. Kết quả chép vào results/kiem_nguon.md.

Chạy: .venv/bin/python scripts/kiem_nguon.py
"""
import time

import ee
import requests
from dotenv import dotenv_values

from floodlab.config import ROOT

GFM = "https://api.gfm.eodc.eu/v2"
AOI_THU_HUE = "ee29bbccf527fe56e835af415988359ac16cf93a3d29c0e168a2c6a1125ed66e"


def _thu(ten, f):
    t = time.time()
    try:
        print(f"OK  {ten}: {f()} ({time.time() - t:.1f} s)")
    except Exception as e:  # dò nguồn: in lỗi rồi đi tiếp
        print(f"LOI {ten}: {str(e)[:150]}")


def gee():
    ee.Initialize(project="vn-floodcast")
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(ee.Geometry.Point(107.58, 16.46))
          .filterDate("2020-10-13", "2020-10-14").first().select(["VV", "VH"]))
    _thu("GEE computePixels S1 224x224", lambda: ee.data.computePixels({
        "expression": s1, "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": 224, "height": 224}, "crsCode": "EPSG:32648",
                 "affineTransform": {"scaleX": 20, "shearX": 0, "translateX": 769440,
                                     "shearY": 0, "scaleY": -20, "translateY": 1824000}}}).shape)
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
    for d in ("2020-10-13", "2024-09-08", "2025-10-20"):
        _thu(f"IMERG V07 status {d}", lambda d=d: imerg.filterDate(d, ee.Date(d).advance(1, "day")).first().get("status").getInfo())
    _thu("HydroBASINS lv12", lambda: ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12").size().getInfo())


def gfm():
    e = dotenv_values(ROOT / ".env")
    j = requests.post(f"{GFM}/auth/login", json={"email": e["GFM_USER"], "password": e["GFM_PASS"]}, timeout=60).json()
    h = {"Authorization": f"Bearer {j['access_token']}"}
    _thu("GFM san pham Hue 10/2020", lambda: len(requests.get(
        f"{GFM}/aoi/{AOI_THU_HUE}/products", headers=h, timeout=180,
        params={"time": "range", "from": "2020-10-01T00:00:00", "to": "2020-11-07T00:00:00"}).json()["products"]))


def gfs():
    for d, duong in (("20161015", "gfs.{d}/00/gfs.t00z.pgrb2.0p25.f024.idx"),
                     ("20210115", "gfs.{d}/00/gfs.t00z.pgrb2.0p25.f024.idx"),
                     ("20220915", "gfs.{d}/00/atmos/gfs.t00z.pgrb2.0p25.f024.idx")):
        _thu(f"GFS AWS idx {d}", lambda d=d, duong=duong: requests.head(
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/" + duong.format(d=d), timeout=60).status_code)
    _thu("GFS NCAR 2016 (Accept-Ranges)", lambda: requests.head(
        "https://data.gdex.ucar.edu/d084001/2016/20161015/gfs.0p25.2016101500.f024.grib2", timeout=60).headers.get("Accept-Ranges"))


if __name__ == "__main__":
    gee()
    gfm()
    gfs()
