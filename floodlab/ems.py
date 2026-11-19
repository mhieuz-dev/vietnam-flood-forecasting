"""Đối chiếu nhãn GFM với bản đồ lũ Copernicus EMS (Task 6 bước 5, spec §10).

Gói vector EMS (data/ems/{EMSR}/*.zip) có observedEventA (vùng ngập), imageFootprintA, areaOfInterestA; ảnh dùng
để vẽ chỉ ghi ở lề bản đồ PDF ("Post-event image: <cảm biến> ... (acquired on dd/mm/yyyy at hh:mm UTC").
Chỉ so trên bản đồ vẽ từ Sentinel-1, cùng ngày UTC với nhãn GFM, trong AOI giao footprint và pixel GFM != 255.
"""
import re
from datetime import datetime

import numpy as np
import rasterio
from rasterio.features import rasterize

from floodlab.config import CRS, NHAN_BO_QUA, PIXEL_M, TILE_PX


def anh_sau_su_kien(chu: str) -> list[tuple[str, datetime]]:
    """[(cảm biến, thời điểm chụp UTC)] của mọi ảnh sau sự kiện ghi ở lề bản đồ. Chữ trích từ PDF có thể bị tách
    ("COS MO-S k yMed") nên bỏ hết khoảng trắng trước khi đọc; một dòng có thể ghi nhiều ngày ("... and on 18/10/2020
    at 11:05 UTC") hoặc nhiều giờ cùng ngày ("at 10:56 UTC and 10:57 UTC")."""
    c = re.sub(r"\s+", "", chu)
    out = {}
    for m in re.finditer(r"Post-eventimages?:", c):
        doan = c[m.end():m.end() + 300]
        if doan.startswith("Post-event"):
            continue
        cam_bien = re.match(r"[^©(]+", doan)[0]
        doan = re.split(r"GSD|provided", doan)[0]
        ngay = None
        for d in re.finditer(r"(?:(\d{2}/\d{2}/\d{4})at)?(\d{2}:\d{2})UTC", doan):
            ngay = d[1] or ngay
            if ngay:
                out[(cam_bien, datetime.strptime(f"{ngay} {d[2]}", "%d/%m/%Y %H:%M"))] = None
    return list(out)


def ve_o(hinh, o) -> np.ndarray:
    """Vẽ các đa giác (GeoDataFrame bất kỳ CRS) lên ô 224 x 224 px EPSG:32648; pixel có tâm trong đa giác là True."""
    g = hinh.to_crs(CRS)
    x0, _, _, y1 = o.bounds
    if g.empty:
        return np.zeros((TILE_PX, TILE_PX), bool)
    return rasterize(((h, 1) for h in g.geometry if h is not None and not h.is_empty), out_shape=(TILE_PX, TILE_PX),
                     transform=rasterio.Affine(PIXEL_M, 0, x0, 0, -PIXEL_M, y1), fill=0, dtype="uint8").astype(bool)


def dem_khop(gfm: np.ndarray, ems_ngap: np.ndarray, vung: np.ndarray) -> dict[str, int]:
    """Bảng nhầm lẫn GFM so với EMS trên pixel trong vùng so và GFM có quan sát."""
    hl = vung & (gfm != NHAN_BO_QUA)
    g, e = (gfm == 1) & hl, ems_ngap & hl
    return {"tp": int((g & e).sum()), "fp": int((g & ~e).sum()), "fn": int((~g & e & hl).sum()),
            "tn": int((~g & ~e & hl).sum())}


def chi_so(d: dict[str, int]) -> dict[str, float]:
    """IoU, tỉ lệ bỏ sót (EMS ngập mà GFM không) và tỉ lệ báo nhầm (GFM ngập mà EMS không)."""
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    return {"iou": tp / max(tp + fp + fn, 1), "bo_sot": fn / max(tp + fn, 1), "bao_nham": fp / max(tp + fp, 1)}
