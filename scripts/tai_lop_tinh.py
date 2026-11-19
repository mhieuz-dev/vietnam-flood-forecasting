"""Tải và tính lớp tĩnh cho mọi ô của một lưu vực, ghi data/tinh/{tile_id}.npz.

Khóa trong npz (12 lớp, 224 x 224): dem, do_doc, hand, log_dien_tich_thuong_nguon, set_0_30, cat_0_30,
dung_trong_0_30, set_30_100, cat_30_100, cn, worldcover, nuoc_thuong_xuyen.
HAND tính theo khối 2048 px chồng mép 256 px (giới hạn RAM), ô rỗng lấp theo kênh gần nhất. Diện tích thượng nguồn
lấy từ MERIT Hydro `upa` (đã hiệu chỉnh theo mạng sông; D8 tự tính trên GLO-30 90 m bị lạc ở đồng bằng phẳng quanh Huế).
MERIT là CC-BY-NC/ODbL: chỉ dùng làm đầu vào, KHÔNG đưa lớp này vào bản dữ liệu công bố (công bố script tạo lại).
SoilGrids để trống ở đô thị, mặt nước: lấp theo ô hợp lệ gần nhất; ô ảnh trống hoàn toàn lấp bằng trung vị lưu vực.

Chạy: .venv/bin/python scripts/tai_lop_tinh.py --luu_vuc huong
"""
import argparse
import json
import math
import time

import ee
import numpy as np

from floodlab import dat, dia_hinh
from floodlab.config import DATA_DIR, PIXEL_M, RESULTS_DIR, TILE_M, TILE_PX
from floodlab.gee_tai import tai_luoi
from floodlab.luoi import luoi_o

KHOI, CHONG = 2048, 256
TANG = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
DAY = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15, "30-60cm": 30, "60-100cm": 40}
THU_MUC = DATA_DIR / "tinh"


def _dem():
    return ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().rename("dem")


def _ghep_dem(x0, y1, rong, cao, thu_muc_tam, pixel):
    """Ghép DEM thành một mảng (cao, rong), tải từng mảnh 1024 px (dưới giới hạn 48 MB của computePixels)."""
    out = np.empty((cao, rong), np.float32)
    for r in range(0, cao, 1024):
        for c in range(0, rong, 1024):
            h, w = min(1024, cao - r), min(1024, rong - c)
            p = tai_luoi(_dem(), (x0 + c * pixel, y1 - r * pixel), w, h,
                         thu_muc_tam / f"dem_{pixel}_{int(x0 + c * pixel)}_{int(y1 - r * pixel)}_{w}_{h}.npy", pixel_m=pixel)
            out[r:r + h, c:c + w] = np.load(p)["dem"]
    return out


def _hand_theo_khoi(o, thu_muc_tam):
    """HAND và độ dốc trên vùng bao các ô, tính theo khối chồng mép; trả dict tile_id -> (dem, hand, doc)."""
    b = o.total_bounds
    x0, y0 = math.floor(b[0] / TILE_M) * TILE_M - CHONG * PIXEL_M, math.floor(b[1] / TILE_M) * TILE_M - CHONG * PIXEL_M
    x1, y1 = math.ceil(b[2] / TILE_M) * TILE_M + CHONG * PIXEL_M, math.ceil(b[3] / TILE_M) * TILE_M + CHONG * PIXEL_M
    buoc = (KHOI - 2 * CHONG) * PIXEL_M
    kq = {}
    for kx in np.arange(x0 + CHONG * PIXEL_M, x1 - CHONG * PIXEL_M, buoc):
        for ky in np.arange(y0 + CHONG * PIXEL_M, y1 - CHONG * PIXEL_M, buoc):
            trong = o[o.geometry.bounds.minx.between(kx, kx + buoc - 1) & o.geometry.bounds.miny.between(ky, ky + buoc - 1)]
            if trong.empty:
                continue
            gx0, gy1 = kx - CHONG * PIXEL_M, ky + buoc + CHONG * PIXEL_M
            dem = _ghep_dem(gx0, gy1, KHOI, KHOI, thu_muc_tam, PIXEL_M)
            hand = dia_hinh.hand(dem, PIXEL_M, nguong_dong_km2=1.0)
            doc = dia_hinh.do_doc(dem, PIXEL_M)
            for _, t in trong.iterrows():
                c0 = int((t.geometry.bounds[0] - gx0) / PIXEL_M)
                r0 = int((gy1 - t.geometry.bounds[3]) / PIXEL_M)
                s = (slice(r0, r0 + TILE_PX), slice(c0, c0 + TILE_PX))
                kq[t.tile_id] = (dem[s], hand[s], doc[s])
    return kq


def _anh_dat_lop_phu():
    sg = "projects/soilgrids-isric/"
    bang = [ee.Image(sg + f"{bien}_mean").select(f"{bien}_{t}_mean").rename(f"{bien}_{t}")
            for bien in ("clay", "sand", "bdod") for t in TANG]
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().rename("worldcover")
    mua = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality").unmask(0).rename("seasonality")
    upa = ee.Image("MERIT/Hydro/v1_0_1").select("upa").rename("upa")
    return ee.Image.cat(bang + [wc, mua, upa])


def _tb_theo_do_day(a, cac_tang):
    w = np.array([DAY[t] for t in cac_tang], float)
    return (np.stack([a[t] for t in cac_tang]) * w[:, None, None]).sum(0) / w.sum()


def _trung_vi_luu_vuc(thu_muc_dat):
    """Trung vị SoilGrids hợp lệ (>0) trên mọi ô đã tải của lưu vực, theo từng lớp và tầng (đơn vị thô)."""
    gom = {}
    for f in sorted(thu_muc_dat.glob("*.npy")):
        r = np.load(f)
        for b in ("clay", "sand", "bdod"):
            for tg in TANG:
                v = r[f"{b}_{tg}"]
                gom.setdefault(f"{b}_{tg}", []).append(v[v > 0][::50])
    return {k: float(np.median(np.concatenate(v))) for k, v in gom.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luu_vuc", required=True)
    lv = ap.parse_args().luu_vuc
    ee.Initialize(project="vn-floodcast")
    o = luoi_o(lv)
    THU_MUC.mkdir(parents=True, exist_ok=True)
    tam = DATA_DIR / "tam_tinh" / lv
    can = o[[not (THU_MUC / f"{t}.npz").exists() for t in o.tile_id]]
    print(lv, "so o:", len(o), "can lam:", len(can), flush=True)
    if can.empty:
        return
    t0 = time.time()
    dia = _hand_theo_khoi(can, tam)
    t1 = time.time()
    so_lap, o_trong, tv = 0, [], None
    anh = _anh_dat_lop_phu()
    for i, (_, t) in enumerate(can.iterrows()):
        g = t.geometry.bounds
        raw = np.load(tai_luoi(anh, (g[0], g[3]), TILE_PX, TILE_PX, tam / "dat" / f"{t.tile_id}.npy"))
        a = {}
        for b in ("clay", "sand", "bdod"):
            for tg in TANG:
                a[f"{b}_{tg}"], n = dat.lap_thieu(raw[f"{b}_{tg}"].astype(np.float32) / (100.0 if b == "bdod" else 10.0))
                so_lap += n
        if np.isnan(a["clay_0-5cm"]).all():
            o_trong.append(t.tile_id)
            tv = tv or _trung_vi_luu_vuc(tam / "dat")
            for k in a:
                a[k] = np.full((TILE_PX, TILE_PX), tv[k] / (100.0 if k.startswith("bdod") else 10.0), np.float32)
        nhom = dat.nhom_pedon([dat.nhom_thuy_van(a[f"clay_{tg}"], a[f"sand_{tg}"]) for tg in TANG])
        wc = raw["worldcover"].astype(np.int16)
        dem, hand, doc = dia[t.tile_id]
        np.savez_compressed(
            THU_MUC / f"{t.tile_id}.npz",
            dem=dem, do_doc=doc, hand=hand, log_dien_tich_thuong_nguon=np.log10(np.clip(raw["upa"].astype(np.float32), 0.01, None)),
            set_0_30=_tb_theo_do_day({k[5:]: v for k, v in a.items() if k.startswith("clay_")}, TANG[:3]),
            cat_0_30=_tb_theo_do_day({k[5:]: v for k, v in a.items() if k.startswith("sand_")}, TANG[:3]),
            dung_trong_0_30=_tb_theo_do_day({k[5:]: v for k, v in a.items() if k.startswith("bdod_")}, TANG[:3]),
            set_30_100=_tb_theo_do_day({k[5:]: v for k, v in a.items() if k.startswith("clay_")}, TANG[3:]),
            cat_30_100=_tb_theo_do_day({k[5:]: v for k, v in a.items() if k.startswith("sand_")}, TANG[3:]),
            cn=dat.cn(nhom, wc).astype(np.float32), worldcover=wc,
            nuoc_thuong_xuyen=(raw["seasonality"] == 12))
        if i % 25 == 0:
            print(f"  {i + 1}/{len(can)}", flush=True)
    if o_trong:
        print(f"{len(o_trong)} ô không có SoilGrids, đã lấp bằng trung vị lưu vực: {o_trong}", flush=True)
    t2 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    nk = RESULTS_DIR / "tai_lop_tinh.json"
    d = json.loads(nk.read_text()) if nk.exists() else {}
    d[lv] = {"so_o": len(can), "giay_dia_hinh": round(t1 - t0), "giay_dat_lop_phu": round(t2 - t1),
             "pixel_dat_phai_lap": so_lap, "o_lap_trung_vi_luu_vuc": o_trong}
    nk.write_text(json.dumps(d, indent=1))
    print(lv, d[lv], flush=True)


if __name__ == "__main__":
    main()
