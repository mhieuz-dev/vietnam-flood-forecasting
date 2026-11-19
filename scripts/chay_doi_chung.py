"""Chạy đối chứng B0, B1, B2, B6 (Task 11 bước 3). Trên val: chọn ngưỡng B1 và huấn luyện B2 rồi chấm; ngưỡng ghi
results/doi_chung.json, mô hình B2 ở results/raw/b2_xgb.json. Trên test (Task 16): đọc lại ngưỡng và mô hình đã có,
không chọn lại gì.
B2 học trên 8.000 mẫu train (4.000 có ngập, 4.000 không, seed 0), cùng tỉ lệ 50% của bộ lấy mẫu Task 12; dừng sớm
trên 1.000 mẫu val (lưu vực gianh) cùng cách chọn.
Chạy: .venv/bin/python scripts/chay_doi_chung.py [--vai_tro val] [--nguon gfs] [--bo b0,b1,b2,b6]
LOBO B2 (spec mục LOBO): --lobo huong học lại B2 trên train bỏ lưu vực đó (cùng cách lấy 8.000 mẫu, cùng dừng sớm trên
val gianh, XGBoost trên CPU để không tranh GPU), lưu results/raw/b2_xgb_lobo_<lv>.json, chấm tên b2_lobo_<lv> trên
val (ngưỡng) rồi trên vai trò lobo_<lv>.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from floodlab import danh_gia, doi_chung, mau
from floodlab.config import LUU_VUC, NHAN_BO_QUA, RESULTS_DIR

F_NGUONG = RESULTS_DIR / "doi_chung.json"
F_B2 = RESULTS_DIR / "raw" / "b2_xgb.json"
KENH_HAND = mau.KENH_TINH.index("hand")
COT_API = 8  # khi_tuong.COT


def _chi_muc(vai_tro):
    return pd.concat([c[c.vai_tro == vai_tro] for c in map(mau.doc_chi_muc, LUU_VUC)], ignore_index=True)


def _chon_can_bang(d, n, seed=0):
    rng = np.random.default_rng(seed)
    co = d[d.ty_le_ngap > 0]
    khong = d[d.ty_le_ngap == 0]
    return pd.concat([co.iloc[rng.choice(len(co), min(n // 2, len(co)), replace=False)],
                      khong.iloc[rng.choice(len(khong), min(n // 2, len(khong)), replace=False)]])


def _doc(d, vai_tro, nguon):
    g = {}
    for r in d.sort_values(["luu_vuc", "i_s1"]).itertuples():
        if r.luu_vuc not in g:
            g = {r.luu_vuc: mau.mo_zarr(r.luu_vuc)}
        yield mau.doc_mau(g[r.luu_vuc], r, nguon_du_bao=nguon) | {"vai_tro": vai_tro}


def _nguong_b1(nguon):
    mau_val = []
    for m in _doc(_chi_muc("val"), "val", nguon):
        hl = m["nhan"] != NHAN_BO_QUA
        mau_val.append({"vai_tro": "val", "hand": m["tinh"][KENH_HAND][hl], "api": m["quan_trac"][-1, COT_API],
                        "nhan": m["nhan"][hl]})
    return doi_chung.chon_nguong_b1(mau_val), len(mau_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vai_tro", default="val", choices=["val", "test_a", "test_b"])
    ap.add_argument("--nguon", default="gfs", choices=["gfs", "imerg_final"])
    ap.add_argument("--bo", default="b0,b1,b2,b6")
    ap.add_argument("--dung_b2_da_luu", action="store_true", help="val: nạp results/raw/b2_xgb.json, không học lại")
    ap.add_argument("--lobo", default=None, choices=list(LUU_VUC))
    a = ap.parse_args()
    if a.lobo:
        return lobo_b2(a.lobo, a.nguon)
    bo = a.bo.split(",")
    luu = json.loads(F_NGUONG.read_text()) if F_NGUONG.exists() else {}
    if "b1" in bo and a.vai_tro == "val":
        t = time.time()
        (nh, na), n = _nguong_b1(a.nguon)
        luu["b1"] = {"nguong_hand_m": nh, "nguong_api_mm": na, "chon_tren": f"val ({n} mẫu gianh)",
                     "luoi_hand": doi_chung.LUOI_HAND, "phan_vi_api": doi_chung.PHAN_VI_API,
                     "thoi_gian_giay": round(time.time() - t, 1)}
        print("B1:", luu["b1"], flush=True)
    if "b2" in bo:
        b2 = doi_chung.B2XGB()
        if a.vai_tro == "val" and not a.dung_b2_da_luu:
            t = time.time()
            b2.fit(_doc(_chon_can_bang(_chi_muc("train"), 8000), "train", a.nguon),
                   _doc(_chon_can_bang(_chi_muc("val"), 1000), "val", a.nguon))
            F_B2.parent.mkdir(parents=True, exist_ok=True)
            b2.mo_hinh.save_model(str(F_B2))
            gain = b2.mo_hinh.get_score(importance_type="gain")
            quan_trong = {ten: round(gain.get(f"f{i}", 0.0), 2) for i, ten in enumerate(doi_chung.TEN_DAC_TRUNG_B2)}
            luu["b2"] = {"so_dong_train": b2.so_dong_train, "ty_le_ngap_train": round(b2.ty_le_ngap_train, 4),
                         "so_cay_tot_nhat": int(b2.mo_hinh.best_iteration) + 1, "tham_so": b2.tham_so,
                         "do_quan_trong_gain": dict(sorted(quan_trong.items(), key=lambda x: -x[1])),
                         "thoi_gian_huan_luyen_giay": round(time.time() - t, 1), "file": F_B2.name}
            print("B2:", {k: v for k, v in luu["b2"].items() if k != "do_quan_trong_gain"}, flush=True)
        else:
            import xgboost as xgb

            b2.mo_hinh = xgb.Booster(model_file=str(F_B2))
    F_NGUONG.write_text(json.dumps(luu, indent=1, ensure_ascii=False))
    du_doan = {
        "b0": lambda m: doi_chung.b0_tan_suat(m["chi_muc"]["tile_id"]),
        "b1": lambda m: doi_chung.b1_hand_mua(m["tinh"][KENH_HAND], m["quan_trac"][-1, COT_API],
                                              luu["b1"]["nguong_hand_m"], luu["b1"]["nguong_api_mm"]),
        "b2": lambda m: b2.predict(m),
        "b6": lambda m: doi_chung.b6_giu_trang_thai(m["nhan_trang_thai"]),
    }
    for ten in bo:
        f = danh_gia.danh_gia(du_doan[ten], a.vai_tro, a.nguon, ten)
        j = json.loads(f.read_text())
        print(f"{ten} {a.vai_tro}: CSI@0,5 {j['tong']['csi']:.3f}, CSI@tối ưu {j['tong']['csi_nguong_toi_uu']:.3f} "
              f"(ngưỡng {j['nguong_toi_uu']['gia_tri']}), ngập mới {j['ngap_moi']['csi']:.3f}, PR-AUC "
              f"{j['tong']['pr_auc']:.3f}, {j['thoi_gian_giay']} s", flush=True)


def lobo_b2(lv: str, nguon: str):
    f_mo_hinh = RESULTS_DIR / "raw" / f"b2_xgb_lobo_{lv}.json"
    b2 = doi_chung.B2XGB(thiet_bi="cpu")
    if f_mo_hinh.exists():
        import xgboost as xgb

        b2.mo_hinh = xgb.Booster(model_file=str(f_mo_hinh))
    else:
        t = time.time()
        train = _chi_muc("train")
        b2.fit(_doc(_chon_can_bang(train[train.luu_vuc != lv], 8000), "train", nguon),
               _doc(_chon_can_bang(_chi_muc("val"), 1000), "val", nguon))
        b2.mo_hinh.save_model(str(f_mo_hinh))
        print(f"B2 LOBO {lv}: {b2.so_dong_train} dòng, {int(b2.mo_hinh.best_iteration) + 1} cây, "
              f"{time.time() - t:.0f} s", flush=True)
    for vt in ("val", f"lobo_{lv}"):
        f = danh_gia.DIR_DANH_GIA / f"b2_lobo_{lv}_{vt}_{nguon}.json"
        if not f.exists():
            f = danh_gia.danh_gia(lambda m: b2.predict(m), vt, nguon, f"b2_lobo_{lv}", luu_xac_suat=False)
        j = json.loads(f.read_text())
        print(f"b2_lobo_{lv} {vt}: CSI@0,5 {j['tong']['csi']:.3f}, ngập mới {j['ngap_moi']['csi']:.3f}, "
              f"{j['so_mau']} mẫu", flush=True)


if __name__ == "__main__":
    main()
