"""Chấm các lượt đã train qua danh_gia (Task 16): val trước (chọn ngưỡng), rồi test_a, test_b.

Mặc định nguồn mưa dự báo là nguồn đã train (cfg nguon_du_bao_train). --nguon khác (ecmwf) hoặc --nguon_qt gsmap_nrt
dùng ngưỡng từ val của nguồn train; --kiem_dat nhieu|hoan_vi ghi tên <ten>__<phép> với ngưỡng từ val của <ten>.
Bỏ qua kết quả đã có. Xác suất từng pixel chỉ lưu khi --luu_xs (mỗi vai trò ~0,5 GB).
Chạy: .venv/bin/python scripts/cham_mo_hinh.py --ten hop_nhat_gfs_s0 [--vai_tro val test_a test_b] [--nguon ecmwf]
      [--nguon_qt gsmap_nrt] [--kiem_dat nhieu] [--luu_xs]
"""
import argparse
import time

import pandas as pd
import torch

from floodlab import danh_gia, du_doan, kiem_dat, mau
from floodlab.config import LUU_VUC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", nargs="+", required=True)
    ap.add_argument("--vai_tro", nargs="+", default=["val", "test_a", "test_b"])
    ap.add_argument("--nguon", default=None)
    ap.add_argument("--nguon_qt", default="imerg_final")
    ap.add_argument("--kiem_dat", choices=["nhieu", "hoan_vi"], default=None)
    ap.add_argument("--luu_xs", action="store_true")
    a = ap.parse_args()
    for ten in a.ten:
        ck = torch.load(du_doan.DIR_CKPT / ten / "best.pt", map_location="cpu", weights_only=False)
        nguon_train = ck["cfg"]["nguon_du_bao_train"]
        nguon = a.nguon or nguon_train
        khac_nguon = nguon != nguon_train or a.nguon_qt != "imerg_final"
        ten_ra = f"{ten}__{a.kiem_dat}" if a.kiem_dat else ten
        ham = None
        for vt in a.vai_tro:
            if vt == "val" and (khac_nguon or a.kiem_dat):
                continue  # ngưỡng của biến thể lấy từ val của mô hình gốc với nguồn train
            nh = nguon if a.nguon_qt == "imerg_final" else f"{nguon}_{a.nguon_qt}"
            f = danh_gia.DIR_DANH_GIA / f"{ten_ra}_{vt}_{nh}.json"
            if f.exists():
                print(f"bỏ qua {f.name} (đã có)", flush=True)
                continue
            bien_doi = None
            if a.kiem_dat == "nhieu":
                bien_doi = kiem_dat.nhieu(seed=0)
            elif a.kiem_dat == "hoan_vi":
                d = pd.concat([c[c.vai_tro == vt] for c in map(mau.doc_chi_muc, LUU_VUC)], ignore_index=True)
                bien_doi = kiem_dat.hoan_vi(d, seed=0)
            if ham is None or bien_doi is not None:
                ham = du_doan.ham_du_doan(ten, bien_doi=bien_doi)
            t = time.time()
            p = danh_gia.danh_gia(ham, vt, nguon, ten_ra, nguon_quan_trac=a.nguon_qt, luu_xac_suat=a.luu_xs,
                                  ten_nguong=ten, nguon_nguong=nguon_train if (khac_nguon or a.kiem_dat) else None)
            print(f"{p.name} xong {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
