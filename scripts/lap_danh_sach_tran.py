"""Lập danh sách trận huấn luyện 2016-2023 từ tỉ lệ ngập GFM theo lưu vực (data/nhan/_ty_le/{lv}.csv).

Ghi ngưỡng từng lưu vực (khóa nguong_tran) và các trận (nguon: GFM) vào cau_hinh/su_kien.yaml, thay các trận
GFM cũ; số trận theo lưu vực và năm ghi vào results/danh_sach_tran.json.
Chạy: .venv/bin/python scripts/lap_danh_sach_tran.py [--chi_in]
"""
import argparse
import json
from collections import Counter

import pandas as pd
import yaml

from floodlab import su_kien
from floodlab.config import CAU_HINH_DIR, DATA_DIR, LUU_VUC, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi_in", action="store_true", help="chỉ in, không ghi yaml/json")
    chi_in = ap.parse_args().chi_in
    nguong, tran_moi, dem = {}, [], {}
    for lv in LUU_VUC:
        f = DATA_DIR / "nhan" / "_ty_le" / f"{lv}.csv"
        if not f.exists():
            print(lv, "chưa có nhãn, bỏ qua")
            continue
        d = pd.read_csv(f, parse_dates=["ngay"])
        d = d[[su_kien.vai_tro_cua_ngay(lv, n.date()) == "train" for n in d.ngay]]
        d = d[d.loai.isin(["mua_mua", "mua_kho"])]  # ngày đệm chỉ là nhãn trạng thái
        d = su_kien.bo_thang_cay_lua(d, lv)
        if (d.loai == "mua_kho").sum() < 10:
            print(lv, f"mới có {(d.loai == 'mua_kho').sum()} ngày mùa khô (nhãn chưa tải xong), bỏ qua")
            continue
        nguong[lv] = round(su_kien.nguong_tran(d), 6)
        ds = su_kien.tach_tran(d, nguong[lv])
        for t in ds:
            tran_moi.append({
                "ma": f"{lv}_{t['bat_dau']:%Y%m%d}", "luu_vuc": lv, "bat_dau": t["bat_dau"],
                "ket_thuc": t["ket_thuc"], "vai_tro": "train", "ems": [],
                "nguon": "GFM", "ghi_chu": f"{t['so_ngay_anh']} ngày ảnh, tỉ lệ ngập lớn nhất {t['ty_le_max']:.4f}"})
        dem[lv] = dict(sorted(Counter(t["bat_dau"].year for t in ds).items()))
        print(f"{lv}: ngưỡng {nguong[lv]:.4f}, {len(ds)} trận, theo năm {dem[lv]}")
    if chi_in:
        return
    p = CAU_HINH_DIR / "su_kien.yaml"
    dau = "".join(dong for dong in p.read_text(encoding="utf-8").splitlines(True) if dong.startswith("#"))
    cu = yaml.safe_load(p.read_text(encoding="utf-8"))
    giu = [s for s in cu["su_kien"] if s.get("nguon") != "GFM"]
    noi_dung = {"nguong_tran": nguong, "su_kien": giu + tran_moi}
    p.write_text(dau + yaml.safe_dump(noi_dung, allow_unicode=True, sort_keys=False), encoding="utf-8")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "danh_sach_tran.json").write_text(json.dumps(
        {"nguong": nguong, "so_tran_theo_nam": dem, "tong": len(tran_moi),
         "quy_tac": {"san": su_kien.SAN_NGUONG_TRAN, "phu_toi_thieu": su_kien.PHU_TOI_THIEU,
                     "khe_ngay_toi_da": su_kien.KHE_NGAY_TOI_DA}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
