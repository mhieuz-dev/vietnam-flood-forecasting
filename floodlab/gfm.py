"""Nhãn ngập từ Copernicus Global Flood Monitoring (GFM): đăng nhập API, AOI theo lưu vực, liệt kê và tải sản phẩm,
mã hóa nhãn 0/1/255 và chiếu về lưới ô EPSG:32648 20 m.

Chú giải GFM (README trong gói tải): ENSEMBLE_FLOOD 1 ngập, 0 không, 255 ngoài vùng chụp; REFERENCE_WATER 1 nước
thường xuyên, 2 nước theo mùa của tháng; ENSEMBLE_EXCLAYER 1 vùng không phân loại (đô thị, rừng rậm, bóng radar...).
"""
import io
import json
import time
import zipfile
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
import requests
from dotenv import dotenv_values
from rasterio.warp import Resampling, reproject

from floodlab.config import CAU_HINH_DIR, CRS, NHAN_BO_QUA, PIXEL_M, ROOT, TILE_PX

API = "https://api.gfm.eodc.eu/v2"
FILE_AOI = CAU_HINH_DIR / "gfm_aoi.json"
LOP = {"ngap": "ENSEMBLE_FLOOD", "loai_tru": "ENSEMBLE_EXCLAYER", "tham_chieu": "REFERENCE_WATER"}


def ma_hoa_nhan(flood: np.ndarray, loai_tru: np.ndarray, tham_chieu: np.ndarray, nuoc_jrc: np.ndarray) -> np.ndarray:
    out = np.where(flood == 1, 1, 0).astype(np.uint8)
    bo = (flood == 255) | (loai_tru == 1) | np.isin(tham_chieu, (1, 2)) | nuoc_jrc
    out[bo] = NHAN_BO_QUA
    return out


def chieu_ve_o(mang: np.ndarray, transform, crs: str, o, nodata: int = NHAN_BO_QUA) -> np.ndarray:
    """Chiếu một lớp phân loại về ô 224 x 224 px theo láng giềng gần nhất. o là hình ô EPSG:32648."""
    dich = np.full((TILE_PX, TILE_PX), nodata, np.uint8)
    x0, _, _, y1 = o.bounds
    reproject(mang, dich, src_transform=transform, src_crs=crs, src_nodata=nodata,
              dst_transform=rasterio.Affine(PIXEL_M, 0, x0, 0, -PIXEL_M, y1), dst_crs=CRS, dst_nodata=nodata,
              resampling=Resampling.nearest)
    return dich


def chieu_ve_vung(mang: np.ndarray, transform, crs: str, bounds, nodata: int = NHAN_BO_QUA) -> np.ndarray:
    """Chiếu một lần về cả vùng (bounds EPSG:32648, chia hết cho 20 m) rồi cắt ô bằng cat_o; nhanh hơn chiếu từng ô."""
    x0, y0, x1, y1 = bounds
    dich = np.full((int(round((y1 - y0) / PIXEL_M)), int(round((x1 - x0) / PIXEL_M))), nodata, np.uint8)
    reproject(mang, dich, src_transform=transform, src_crs=crs, src_nodata=nodata,
              dst_transform=rasterio.Affine(PIXEL_M, 0, x0, 0, -PIXEL_M, y1), dst_crs=CRS, dst_nodata=nodata,
              resampling=Resampling.nearest)
    return dich


def cat_o(vung: np.ndarray, bounds_vung, o) -> np.ndarray:
    r0 = int(round((bounds_vung[3] - o.bounds[3]) / PIXEL_M))
    c0 = int(round((o.bounds[0] - bounds_vung[0]) / PIXEL_M))
    return vung[r0:r0 + TILE_PX, c0:c0 + TILE_PX]


def gop_cung_ngay(cac_nhan: list[np.ndarray]) -> np.ndarray:
    """Hai lát cắt S1 liền kề cùng ngày: pixel có quan sát (0/1) thắng pixel 255; nếu cả hai quan sát, 1 thắng 0."""
    out = np.full(cac_nhan[0].shape, NHAN_BO_QUA, np.uint8)
    for n in cac_nhan:
        co = n != NHAN_BO_QUA
        out[co & (out == NHAN_BO_QUA)] = n[co & (out == NHAN_BO_QUA)]
        out[co & (n == 1)] = 1
    return out


def _get_thu_lai(url: str, lan_toi_da: int = 6, **kw):
    """GET thử lại khi máy chủ lỗi 5xx/429 hoặc mất kết nối, treo (GFM thỉnh thoảng không trả lời quá 300 giây)."""
    for lan in range(lan_toi_da):
        try:
            r = requests.get(url, **kw)
            if r.status_code < 500 and r.status_code != 429:
                return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if lan == lan_toi_da - 1:
                raise
        time.sleep(2 ** lan * 5)
    return r


class Phien:
    """Phiên API GFM, tự đăng nhập lại khi token sắp hết hạn."""

    def __init__(self):
        self._het_han = 0.0

    def _dang_nhap(self):
        e = dotenv_values(ROOT / ".env")
        j = requests.post(f"{API}/auth/login", json={"email": e["GFM_USER"], "password": e["GFM_PASS"]}, timeout=60)
        j.raise_for_status()
        j = j.json()
        self.h = {"Authorization": f"Bearer {j['access_token']}"}
        self.uid = j["client_id"]
        self._het_han = time.time() + int(j["expires_in"]) - 300

    def dam_bao(self):
        if time.time() > self._het_han:
            self._dang_nhap()
        return self

    def get(self, duong: str, **kw):
        self.dam_bao()
        r = _get_thu_lai(f"{API}{duong}", headers=self.h, timeout=300, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, duong: str, body: dict):
        self.dam_bao()
        r = requests.post(f"{API}{duong}", headers=self.h, json=body, timeout=120)
        r.raise_for_status()
        return r.json()


@lru_cache
def phien() -> Phien:
    return Phien()


def aoi_id(luu_vuc: str, hinh_4326) -> str:
    """AOI GFM cho lưu vực (tạo một lần, lưu id ở cau_hinh/gfm_aoi.json)."""
    d = json.loads(FILE_AOI.read_text()) if FILE_AOI.exists() else {}
    if luu_vuc not in d:
        p = phien().dam_bao()
        geo = json.loads(json.dumps(hinh_4326.envelope.__geo_interface__))
        d[luu_vuc] = p.post("/aoi/create", {"aoi_name": f"vnfc_{luu_vuc}", "description": "VN-FloodCast",
                                             "user_id": p.uid, "geoJSON": geo})["aoi_id"]
        FILE_AOI.write_text(json.dumps(d, indent=1))
    return d[luu_vuc]


def san_pham(aoi: str, tu: date, den: date) -> list[dict]:
    """Danh sách {product_id, product_time} trong khoảng ngày (tính cả hai đầu)."""
    j = phien().get(f"/aoi/{aoi}/products", params={"time": "range", "from": f"{tu}T00:00:00", "to": f"{den}T23:59:59"})
    return j.get("products", [])


def ngay_co_anh(aoi: str, tu: date, den: date) -> dict[date, list[str]]:
    out: dict[date, list[str]] = {}
    for p in san_pham(aoi, tu, den):
        out.setdefault(date.fromisoformat(p["product_time"][:10]), []).append(p["product_id"])
    return out


def tai_lop(product_id: str) -> dict[str, tuple[np.ndarray, object, str]]:
    """Tải gói zip của sản phẩm vào bộ nhớ, trả {ngap, loai_tru, tham_chieu}: (mảng, transform, crs).
    Sản phẩm cũ (vd. 2016) chỉ có ENSEMBLE_FLOOD: lớp thiếu không có trong dict."""
    p = phien().dam_bao()
    link = p.get(f"/download/product/{product_id}/{p.uid}")["download_link"]
    r = _get_thu_lai(link, timeout=600)
    r.raise_for_status()
    out = {}
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for ten, khoa in LOP.items():
            f = next((n for n in z.namelist() if khoa in n and n.endswith(".tif")), None)
            if f is None:
                continue
            with rasterio.open(io.BytesIO(z.read(f))) as d:
                out[ten] = (d.read(1), d.transform, d.crs.to_string())
    return out


def tai_lop_hoac_bo(product_id: str, lan_zip: int = 3) -> dict | None:
    """Như tai_lop nhưng trả None khi máy chủ GFM vẫn lỗi 5xx sau mọi lần thử lại (sản phẩm hỏng phía máy chủ,
    vd. 00027566... và 7b66295c... lỗi 500 liên tục nhiều giờ 28/10/2026), hoặc gói tải về không phải zip sau
    lan_zip lần (owncloud trả 200 kèm nội dung lỗi, gặp ở gianh và tra_khuc 29/10). Lỗi 4xx vẫn ném."""
    for lan in range(lan_zip):
        try:
            return tai_lop(product_id)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500:
                return None
            raise
        except zipfile.BadZipFile:
            time.sleep(60 * (lan + 1))
    return None
