"""Khung mưa dự báo t0+1 đến t0+3: chu kỳ 00Z ngày t0+1, mưa ngày lấy từ tích lũy 0-24, 0-48, 0-72 giờ."""
import os
import re
import tempfile
import time
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import requests

from floodlab.config import DATA_DIR, KHUNG_MUA_THUONG_NGUON


def ngay_chu_ky(t0: date) -> date:
    return t0 + timedelta(days=1)


def mat_na(L: int) -> np.ndarray:
    """True cho các ngày t0+1..t0+L (trong thời hạn), còn lại False và giá trị điền 0."""
    return np.arange(1, 4) <= L


def mua_ngay_tu_tich_luy(tich_luy: np.ndarray) -> np.ndarray:
    """tich_luy (3, ...) là mưa tích lũy từ đầu chu kỳ tới 24, 48, 72 giờ; trả mưa từng ngày, không âm."""
    t = np.asarray(tich_luy, float)
    return np.clip(np.diff(t, axis=0, prepend=np.zeros_like(t[:1])), 0, None)


# ---------------- tải GFS 0,25 độ, khung 97-110 E, 8-26 N (lưới 73 hàng x 53 cột, hàng 0 là 26 N) ----------------
THU_MUC_GFS = DATA_DIR / "du_bao" / "gfs"
_AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d:%Y%m%d}/00/{atmos}gfs.t00z.pgrb2.0p25.f{h:03d}"
_NCSS = ("https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001/{d:%Y}/{d:%Y%m%d}/gfs.0p25.{d:%Y%m%d}00.f{h:03d}.grib2"
         "?var=Total_precipitation_surface_Mixed_intervals_Accumulation&north={n}&south={s}&west={w}&east={e}&accept=netcdf3")


def _get(url, **kw):
    for lan in range(6):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** lan * 3)
    raise RuntimeError(f"khong tai duoc {url}")


def _cat_khung(lat, lon, v):
    w, s, e, n = KHUNG_MUA_THUONG_NGUON
    hang = (lat <= n + 1e-6) & (lat >= s - 1e-6)
    cot = (lon >= w - 1e-6) & (lon <= e + 1e-6)
    out = v[np.ix_(hang, cot)]
    return out if lat[hang][0] > lat[hang][-1] else out[::-1]


def _aws_tich_luy(d: date) -> np.ndarray | None:
    import xarray as xr

    out = []
    for k in (1, 2, 3):
        for atmos in ("atmos/", ""):
            url = _AWS.format(d=d, h=24 * k, atmos=atmos)
            idx = _get(url + ".idx")
            if idx is not None:
                break
        if idx is None:
            return None
        dong = idx.text.strip().splitlines()
        i = next(j for j, l in enumerate(dong) if f":APCP:surface:0-{k} day acc" in l)
        a, b = int(dong[i].split(":")[1]), int(dong[i + 1].split(":")[1]) - 1
        g = _get(url, headers={"Range": f"bytes={a}-{b}"})
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(g.content)
        try:
            ds = xr.open_dataset(f.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
            v = ds[list(ds.data_vars)[0]]
            out.append(_cat_khung(ds.latitude.values, ds.longitude.values, v.values))
        finally:
            os.unlink(f.name)
    return np.stack(out)


def _ncar_khoang(d: date, h: int) -> dict | None:
    """Đọc mọi bản ghi mưa tích lũy trong file f{h}; trả {(giờ đầu, giờ cuối): mảng (73, 53)}.
    Tên biến đổi theo năm (6_Hour_Accumulation hay Mixed_intervals_Accumulation) nên dò từ dataset.xml."""
    from scipy.io import netcdf_file

    w, s, e, n = KHUNG_MUA_THUONG_NGUON
    goc = _NCSS.split("?")[0].format(d=d, h=h)
    xml = _get(goc + "/dataset.xml")
    if xml is None:
        return None
    ten = re.findall(r'name="(Total_precipitation_surface_[^"]*Accumulation)"', xml.text)
    if not ten:
        return None
    r = _get(f"{goc}?var={ten[0]}&north={n}&south={s}&west={w}&east={e}&accept=netcdf3&temporal=all")
    if r is None or not r.content.startswith(b"CDF"):
        return None
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        f.write(r.content)
    try:
        nc = netcdf_file(f.name, "r", mmap=False)
        v = nc.variables[ten[0]][:].astype(float)
        tb = nc.variables[next(b for b in nc.variables if b.endswith("_bounds"))][:]
        lat = nc.variables["latitude"][:].copy()
        nc.close()
    finally:
        os.unlink(f.name)
    return {(int(a), int(b)): (v[i] if lat[0] > lat[-1] else v[i][::-1]) for i, (a, b) in enumerate(tb)}


def _ncar_tich_luy(d: date) -> np.ndarray | None:
    """Tích lũy 0-24, 0-48, 0-72 giờ. Thử f024/f048/f072 có sẵn bản 0-24k; nếu không (vd. 2016 chỉ có 6 giờ)
    thì cộng 12 bản 6 giờ từ f006..f072."""
    out = []
    for k in (1, 2, 3):
        kh = _ncar_khoang(d, 24 * k)
        if kh is None:
            return None
        if (0, 24 * k) not in kh:
            break
        out.append(kh[(0, 24 * k)])
    else:
        return np.stack(out)
    sau = []
    for h in range(6, 73, 6):
        kh = _ncar_khoang(d, h)
        if kh is None or (h - 6, h) not in kh:
            return None
        sau.append(kh[(h - 6, h)])
    return np.cumsum(np.stack(sau), axis=0)[[3, 7, 11]]


def tai_gfs(d_chu_ky: date, nguon: str) -> np.ndarray | None:
    """Mưa ngày (mm) dự báo cho 3 ngày sau chu kỳ 00Z d_chu_ky, dạng (3, 73, 53). Lưu cache data/du_bao/gfs/."""
    p = THU_MUC_GFS / f"{d_chu_ky:%Y%m%d}.npy"
    if p.exists():
        return np.load(p)
    tl = _aws_tich_luy(d_chu_ky) if nguon == "aws_bdp" else _ncar_tich_luy(d_chu_ky)
    if tl is None:
        return None
    m = mua_ngay_tu_tich_luy(tl).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, m)
    return m


@lru_cache(maxsize=4096)
def _vi_tri(tile_id: str):
    """(hàng, cột) điểm GFS 0,25 độ gần tâm ô nhất và mặt nạ điểm trong lưu vực thượng nguồn."""
    from pyproj import Transformer

    from floodlab.khi_tuong import mat_na_luoi, o_luoi
    from floodlab.luoi import luoi_o, luu_vuc_thuong_nguon

    luu_vuc = tile_id.rsplit("_", 2)[0]
    g = luoi_o(luu_vuc).set_index("tile_id").geometry[tile_id].centroid
    lon, lat = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True).transform(g.x, g.y)
    return o_luoi(lon, lat, 0.25), mat_na_luoi(luu_vuc_thuong_nguon(luu_vuc), 0.25, 73, 53)


# ---------------- ECMWF IFS open data 0,25 độ (chỉ để chấm test, Task 16; mở từ 18/01/2023) ----------------
# Bản S3 của ECMWF trả SlowDown cho truy cập ẩn danh (06/11/2026); dùng bản sao Azure của Microsoft Planetary Computer
# (cùng tên file), token SAS miễn phí không cần tài khoản. Chỉ tải đoạn byte của trường tp theo file .index.
THU_MUC_ECMWF = DATA_DIR / "du_bao" / "ecmwf"
_ECMWF = "https://ai4edataeuwest.blob.core.windows.net/ecmwf/{d:%Y%m%d}/00z/ifs/0p25/oper/{d:%Y%m%d}000000-{h}h-oper-fc"
_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/ai4edataeuwest/ecmwf"


@lru_cache(maxsize=1)
def _token_sas() -> str:
    return _get(_SAS).json()["token"]


def _ecmwf_tich_luy(d: date) -> np.ndarray | None:
    """tp tích lũy (m) từ đầu chu kỳ 00Z d tới 24, 48, 72 giờ, cắt khung, dạng (3, 73, 53)."""
    import json

    import xarray as xr

    out = []
    for h in (24, 48, 72):
        url = _ECMWF.format(d=d, h=h)
        idx = _get(f"{url}.index?{_token_sas()}")
        if idx is None:
            return None
        tp = next(j for j in map(json.loads, idx.text.strip().splitlines()) if j.get("param") == "tp")
        a, b = tp["_offset"], tp["_offset"] + tp["_length"] - 1
        g = _get(f"{url}.grib2?{_token_sas()}", headers={"Range": f"bytes={a}-{b}"})
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(g.content)
        try:
            ds = xr.open_dataset(f.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
            v = ds[list(ds.data_vars)[0]]
            out.append(_cat_khung(ds.latitude.values, ds.longitude.values, v.values))
        finally:
            os.unlink(f.name)
    return np.stack(out)


def tai_ecmwf(d_chu_ky: date) -> np.ndarray | None:
    """Mưa ngày (mm) dự báo ECMWF cho 3 ngày sau chu kỳ 00Z d_chu_ky, (3, 73, 53); cache data/du_bao/ecmwf/."""
    p = THU_MUC_ECMWF / f"{d_chu_ky:%Y%m%d}.npy"
    if p.exists():
        return np.load(p)
    tl = _ecmwf_tich_luy(d_chu_ky)
    if tl is None:
        return None
    m = (mua_ngay_tu_tich_luy(tl) * 1000).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, m)
    return m


def doc_ecmwf(d_chu_ky: date) -> np.ndarray | None:
    p = THU_MUC_ECMWF / f"{d_chu_ky:%Y%m%d}.npy"
    return np.load(p) if p.exists() else None


def doc_gfs(d_chu_ky: date) -> np.ndarray | None:
    """Chỉ đọc cache (3, 73, 53) của chu kỳ; thiếu thì None. Lắp mẫu dùng hàm này, không tải qua mạng: tải là việc
    của scripts/tai_gfs.py. Trước đây khung_du_bao gọi tai_gfs nên mỗi (ô, ngày, L) thiếu chu kỳ lại hỏi NCAR."""
    p = THU_MUC_GFS / f"{d_chu_ky:%Y%m%d}.npy"
    return np.load(p) if p.exists() else None


def khung_du_bao(tile_id: str, t0: date, L: int, nguon: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(3, 2) [mua_o, mua_thuong_nguon] cho t0+1..t0+3 và mặt nạ (3,); ngày ngoài thời hạn L điền 0.
    nguon: 'gfs' hoặc 'ecmwf' (chu kỳ 00Z ngày t0+1, chỉ đọc cache) hoặc 'imerg_final' (mưa quan trắc, biết trước)."""
    mn = mat_na(L)
    if nguon in ("gfs", "ecmwf"):
        luoi = (doc_gfs if nguon == "gfs" else doc_ecmwf)(ngay_chu_ky(t0))
        if luoi is None:
            return None
        (r, c), tn = _vi_tri(tile_id)
        gt = np.stack([luoi[:, r, c], luoi[:, tn].mean(axis=1)], axis=1)
    elif nguon == "imerg_final":
        from floodlab.khi_tuong import _chuoi

        chi_so_ngay, chi_so_o, g, _ = _chuoi(tile_id.rsplit("_", 2)[0])
        idx = [chi_so_ngay.get((t0 + timedelta(days=k)).toordinal()) for k in (1, 2, 3)]
        if any(i is None for i in idx):
            return None
        gt = g[chi_so_o[tile_id], idx][:, :2]
    else:
        raise NotImplementedError(nguon)
    gt = np.where(mn[:, None], gt, 0).astype(np.float32)
    return gt, mn
