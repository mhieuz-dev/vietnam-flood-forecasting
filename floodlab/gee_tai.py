"""Tải lưới pixel từ Earth Engine bằng computePixels, ghi .npy kiểu nguyên tử, bỏ qua file đã có, thử lại khi bị giới hạn."""
import os
import time
from pathlib import Path

import ee
import numpy as np

from floodlab.config import CRS

SO_LAN_THU = 6


def _compute_pixels(req):
    return ee.data.computePixels(req)


def tai_luoi(anh, goc_xy: tuple[float, float], rong: int, cao: int, duong_dan: Path,
             pixel_m: float = 20.0, crs: str = CRS) -> Path:
    """goc_xy là góc trên-trái (x, y) theo crs. Trả đường dẫn file .npy (mảng có cấu trúc, mỗi băng một trường)."""
    duong_dan = Path(duong_dan)
    if duong_dan.exists():
        return duong_dan
    req = {"expression": anh, "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": int(rong), "height": int(cao)}, "crsCode": crs,
                    "affineTransform": {"scaleX": float(pixel_m), "shearX": 0, "translateX": float(goc_xy[0]),
                                        "shearY": 0, "scaleY": -float(pixel_m), "translateY": float(goc_xy[1])}}}
    for lan in range(SO_LAN_THU):
        try:
            mang = _compute_pixels(req)
            break
        except ee.ee_exception.EEException as e:
            tam_thoi = any(k in str(e) for k in ("429", "Too Many", "503", "500", "Deadline", "timed out"))
            if not tam_thoi or lan == SO_LAN_THU - 1:
                raise
            time.sleep(2 ** lan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    tam = duong_dan.with_suffix(f".tmp{os.getpid()}.npy")
    try:
        np.save(tam, mang)
        os.replace(tam, duong_dan)
    finally:
        if tam.exists():
            tam.unlink()
    return duong_dan
