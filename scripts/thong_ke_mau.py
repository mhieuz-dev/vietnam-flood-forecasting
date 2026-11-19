"""Thống kê mẫu đã lắp (Task 9 bước 4), ghi results/thong_ke_mau.json; về sau thành Bảng dữ liệu trong bài.
Số mẫu theo vai trò, lưu vực, L; tỉ lệ pixel ngập; phân bố khoang_cach_s1_ngay; tỉ lệ mẫu có nhãn trạng thái và có
ngập mới; số mẫu bị loại theo lý do.
Chạy: .venv/bin/python scripts/thong_ke_mau.py
"""
import json

import numpy as np
import pandas as pd
import zarr

from floodlab.config import LUU_VUC, NHAN_BO_QUA, RESULTS_DIR
from floodlab.mau import DIR_MAU


def _phan_vi(s: pd.Series) -> dict:
    return {k: round(float(v), 4) for k, v in s.quantile([0, 0.25, 0.5, 0.75, 0.9, 1]).items()}


def _hop_le(lv: str) -> dict:
    """Phân bố tỉ lệ pixel nhãn khác 255 trên mọi nhãn ngày t của lưu vực (GFM loại đồi núi HAND cao, bóng radar,
    đô thị; mặt nước thường xuyên cũng là 255), đọc từng khối 512 nhãn."""
    a = zarr.open_group(DIR_MAU / f"{lv}.zarr", mode="r")["nhan"]
    hl = np.concatenate([(a[i:i + 512] != NHAN_BO_QUA).mean(axis=(1, 2)) for i in range(0, a.shape[0], 512)])
    return _phan_vi(pd.Series(hl))


def main():
    cm = pd.concat([pd.read_parquet(DIR_MAU / f"chi_muc_{lv}.parquet") for lv in LUU_VUC], ignore_index=True)
    bl = pd.concat([pd.read_csv(DIR_MAU / f"bi_loai_{lv}.csv").assign(luu_vuc=lv) for lv in LUU_VUC], ignore_index=True)
    ra = {
        "tong_mau": len(cm),
        "theo_vai_tro_luu_vuc": cm.groupby(["vai_tro", "luu_vuc"]).size().unstack(fill_value=0).to_dict("index"),
        "theo_vai_tro_L": cm.groupby(["vai_tro", "L"]).size().unstack(fill_value=0).to_dict("index"),
        "theo_loai_ngay": cm.groupby(["luu_vuc", "loai_ngay"]).size().unstack(fill_value=0).to_dict("index"),
        "so_o_so_ngay": cm.groupby("luu_vuc").agg(o=("tile_id", "nunique"), ngay=("ngay_t", "nunique")).to_dict("index"),
        "ty_le_ngap_theo_vai_tro": {k: _phan_vi(g.ty_le_ngap) for k, g in cm.groupby("vai_tro")},
        "ty_le_mau_co_ngap": cm.assign(c=cm.ty_le_ngap > 0).groupby("vai_tro").c.mean().round(4).to_dict(),
        "khoang_cach_s1_ngay": {k: _phan_vi(g.khoang_cach_s1_ngay) for k, g in cm.groupby("L")},
        "ty_le_co_nhan_trang_thai": cm.assign(c=cm.ngay_nhan_trang_thai.notna()).groupby("vai_tro").c.mean().round(4).to_dict(),
        "ty_le_co_ngap_moi": cm.groupby("vai_tro").co_ngap_moi.mean().round(4).to_dict(),
        "ty_le_pixel_hop_le_nhan": {lv: _hop_le(lv) for lv in LUU_VUC},
        "bi_loai": bl.groupby(["luu_vuc", "ly_do"]).size().unstack(fill_value=0).to_dict("index"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "thong_ke_mau.json").write_text(json.dumps(ra, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(ra, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
