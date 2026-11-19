"""Điền lại kênh log_dien_tich_thuong_nguon (kênh 4 của mảng tinh) vào shard zarr từ MERIT Hydro `upa`.

Bản dữ liệu trên Zenodo để NaN ở kênh này vì giấy phép MERIT Hydro (CC BY-NC 4.0 hoặc ODbL) không cho phát hành lại
theo CC BY 4.0. Script tải `upa` qua Earth Engine trên đúng lưới 20 m của từng ô và tính như scripts/tai_lop_tinh.py:
log10(max(upa, 0,01)) rồi ép float16. --kiem chỉ so với giá trị đang có trong shard, không ghi (dùng để kiểm bản gốc).

Chạy: .venv/bin/python scripts/them_dien_tich_thuong_nguon.py --du_an <earth-engine-project> [--luu_vuc huong] [--kiem]
"""
import argparse

import ee
import numpy as np
import zarr

from floodlab.config import DATA_DIR, TILE_PX
from floodlab.gee_tai import tai_luoi
from floodlab.luoi import luoi_o
from floodlab.mau import DIR_MAU, KENH_TINH, doc_chi_muc

KENH = KENH_TINH.index("log_dien_tich_thuong_nguon")
LUU_VUC = ("hong_thai_binh", "ca", "gianh", "huong", "vu_gia_thu_bon", "tra_khuc")


def log_upa(tile, tam) -> np.ndarray:
    g = tile.geometry.bounds
    upa = ee.Image("MERIT/Hydro/v1_0_1").select("upa")
    raw = np.load(tai_luoi(upa, (g[0], g[3]), TILE_PX, TILE_PX, tam / f"{tile.tile_id}.npy"))
    return np.log10(np.clip(raw["upa"].astype(np.float32), 0.01, None)).astype(np.float16)


def mot_luu_vuc(lv: str, kiem: bool) -> dict:
    g = zarr.open_group(DIR_MAU / f"{lv}.zarr", mode="r" if kiem else "r+")
    i_o = doc_chi_muc(lv).groupby("tile_id").i_o.first()
    o = luoi_o(lv).set_index("tile_id").loc[i_o.index].reset_index()
    tam = DATA_DIR / "tam_upa" / lv
    khac = 0
    for _, t in o.iterrows():
        moi = log_upa(t, tam)
        i = int(i_o[t.tile_id])
        if kiem:
            khac += int(not np.array_equal(g["tinh"][i, KENH].view(np.uint16), moi.view(np.uint16)))
        else:
            g["tinh"][i, KENH] = moi
    return {"so_o": len(o), "o_khac_ban_goc": khac} if kiem else {"so_o_da_dien": len(o)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--du_an", required=True, help="Google Cloud project đã bật Earth Engine")
    ap.add_argument("--luu_vuc", nargs="*", default=list(LUU_VUC))
    ap.add_argument("--kiem", action="store_true")
    a = ap.parse_args()
    ee.Initialize(project=a.du_an)
    for lv in a.luu_vuc:
        print(lv, mot_luu_vuc(lv, a.kiem), flush=True)


if __name__ == "__main__":
    main()
