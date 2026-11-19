"""Kiểm nhãn GFM trên vùng huấn luyện bằng bản đồ Copernicus EMS 2020 (Task 6 bước 5), ghi results/kiem_nhan_train.json.

Cần sẵn gói vector ở data/ems/{EMSR}/*_vector.zip (tải từ trang mapping.emergency.copernicus.eu/activations/{EMSR});
bản đồ PDF tự tải về data/ems/{EMSR}/pdf/ để đọc ảnh nguồn. Chỉ giữ bản đồ mà mọi ảnh sau sự kiện là Sentinel-1
chụp cùng một ngày UTC; bản đồ khác ghi lý do bỏ. Nhãn GFM áp mặt nạ loại trừ hợp như khi lắp mẫu.
Chạy: .venv/bin/python scripts/kiem_nhan_ems.py
"""
import json
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pymupdf
import requests

from floodlab import ems
from floodlab.config import CRS, DATA_DIR, LUU_VUC, NHAN_BO_QUA, RESULTS_DIR
from floodlab.luoi import luoi_o
from floodlab.mau import _loai_tru

KICH_HOAT = ("EMSR450", "EMSR464", "EMSR469", "EMSR472", "EMSR475")
S3 = "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations"


def _lop(f, ten):
    n = next((x for x in zipfile.ZipFile(f).namelist() if f"_{ten}_" in x and x.endswith(".shp")), None)
    return None if n is None else gpd.read_file(f"zip://{f}!{n}")


def _chu_pdf(ma, ten):
    p = DATA_DIR / "ems" / ma / "pdf" / f"{ten}.pdf"
    if not p.exists():
        r = requests.get(f"{S3}/{ma}/{ten}.pdf", timeout=600)
        if r.status_code != 200:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
    with pymupdf.open(p) as d:
        return " ".join(trang.get_text() for trang in d)


def main():
    o = pd.concat([luoi_o(lv) for lv in LUU_VUC], ignore_index=True)
    ban_do, bo = [], []
    for ma in KICH_HOAT:
        for f in sorted((DATA_DIR / "ems" / ma).glob("*_vector.zip")):
            ten = f.name.removesuffix("_vector.zip")
            chu = _chu_pdf(ma, ten)
            anh = ems.anh_sau_su_kien(chu) if chu else []
            ngay = {t.date() for _, t in anh}
            if not anh or any(not s.startswith("Sentinel-1") for s, _ in anh) or len(ngay) != 1:
                bo.append({"ban_do": ten, "ly_do": "khong_doc_duoc_pdf" if not chu else "anh_khong_phai_mot_ngay_s1",
                           "anh_sau_su_kien": [(s, t.isoformat()) for s, t in anh]})
                continue
            n = ngay.pop()
            aoi, vet, ngap = (_lop(f, k) for k in ("areaOfInterestA", "imageFootprintA", "observedEventA"))
            vung = aoi.to_crs(CRS).union_all()
            if vet is not None and not vet.empty:
                vung = vung.intersection(vet.to_crs(CRS).union_all())
            ngap = ngap[ngap.event_type.astype(str).str.contains("Flood", case=False)] if ngap is not None else None
            dem = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            so_o = 0
            for t in o[o.intersects(vung)].itertuples():
                p = DATA_DIR / "nhan" / t.tile_id / f"{n:%Y%m%d}.npy"
                if not p.exists():
                    continue
                g = np.where(_loai_tru(t.tile_id), NHAN_BO_QUA, np.load(p)).astype(np.uint8)
                m_vung = ems.ve_o(gpd.GeoDataFrame(geometry=[vung], crs=CRS), t.geometry)
                m_ngap = ems.ve_o(ngap, t.geometry) if ngap is not None else np.zeros_like(m_vung)
                for k, v in ems.dem_khop(g, m_ngap, m_vung).items():
                    dem[k] += v
                so_o += 1
            if so_o == 0:
                bo.append({"ban_do": ten, "ly_do": "khong_co_nhan_gfm_cung_ngay_trong_vung", "ngay": n.isoformat()})
                continue
            ban_do.append({"kich_hoat": ma, "ban_do": ten, "ngay": n.isoformat(), "anh": anh[0][0], "so_o": so_o,
                           **dem, **ems.chi_so(dem)})
            print(f"{ten} {n} {so_o} ô IoU={ban_do[-1]['iou']:.3f} bỏ sót={ban_do[-1]['bo_sot']:.3f}", flush=True)
    tong = {k: sum(b[k] for b in ban_do) for k in ("tp", "fp", "fn", "tn")}
    theo_kh = {}
    for b in ban_do:
        d = theo_kh.setdefault(b["kich_hoat"], {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        for k in d:
            d[k] += b[k]
    ra = {"tong": {**tong, **ems.chi_so(tong), "so_ban_do": len(ban_do)},
          "theo_kich_hoat": {k: {**v, **ems.chi_so(v)} for k, v in theo_kh.items()},
          "ban_do": ban_do, "bo": bo}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "kiem_nhan_train.json").write_text(json.dumps(ra, indent=1, ensure_ascii=False, default=str))
    print(f"{len(ban_do)} bản đồ so được, {len(bo)} bỏ; tổng IoU={ra['tong']['iou']:.3f} "
          f"bỏ sót={ra['tong']['bo_sot']:.3f} báo nhầm={ra['tong']['bao_nham']:.3f}")


if __name__ == "__main__":
    main()
