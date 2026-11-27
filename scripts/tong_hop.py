"""Sinh bảng (bai_bao/bang/*.tex) và hình thống kê (bai_bao/hinh/*.pdf) của bài từ results (Task 17).

Mọi số trong bảng đi qua floodlab.tong_hop.SoLieu: ô kèm "% nguon: <khóa>", khóa ghi trong results/tong_hop.json cùng
cách tính lại từ json gốc; tests/test_tong_hop.py::test_moi_so_trong_bang_co_nguon kiểm ngược từng số.
CSI chính là CSI ở ngưỡng chọn trên val (csi_nguong_toi_uu); mô hình 3 seed ghi trung bình ± độ lệch chuẩn.
So sánh ghép cặp: hiệu CSI gộp 3 seed (cộng tp/fp/fn theo ô, mỗi seed ở ngưỡng val của nó), bootstrap cụm theo ô,
p hiệu chỉnh Holm trong từng tập test. Hình bản đồ và độ tin cậy (cần đọc test, GPU) ở scripts/hinh_ban_do.py.
Chạy: .venv/bin/python scripts/tong_hop.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font hình khớp chữ Times của bài (Liberation Serif cùng kích thước Times), nhúng TrueType thay vì Type 3 (IEEE PDF eXpress)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd

from floodlab.config import RESULTS_DIR, ROOT
from floodlab.tong_hop import SoLieu, bootstrap_cap, bootstrap_cap_phan_cap, holm

DG = "results/danh_gia"
BANG, HINH = ROOT / "bai_bao" / "bang", ROOT / "bai_bao" / "hinh"
SEED = range(3)
VT = {"test_a": "Test A", "test_b": "Test B"}
LV = {"vu_gia_thu_bon": "Vu Gia-Thu Bon", "huong": "Huong", "tra_khuc": "Tra Khuc", "gianh": "Gianh", "ca": "Ca",
      "hong_thai_binh": "Red-Thai Binh"}
# nhãn trong bài: (tên lượt, nguồn mưa trong tên json)
MO_HINH = {
    "B0 Flood frequency": (["b0"], "gfs"),
    "B1 HAND + rainfall": (["b1"], "gfs"),
    "B2 XGBoost": (["b2"], "gfs"),
    "B6 Persistence": (["b6"], "gfs"),
    "B3 U-Net": ([f"unet_b3_gfs_s{s}" for s in SEED], "gfs"),
    "B4 Prithvi-EO-2.0": ([f"prithvi_b4_gfs_s{s}" for s in SEED], "gfs"),
    "B5 U-TAE": ([f"utae_b5_gfs_s{s}" for s in SEED], "gfs"),
    "TerraMind (GFS)": ([f"hop_nhat_gfs_s{s}" for s in SEED], "gfs"),
}
CHINH = "TerraMind (GFS)"
sl = SoLieu()


def tep(ten: list[str], vt: str, nguon: str, hau_to: str = "") -> list[str]:
    return [f"{DG}/{t}{hau_to}_{vt}_{nguon}.json" for t in ten]


def o(khoa: str, ds: list[str], k: str) -> str:
    """Một ô: một tệp -> giá trị; nhiều tệp -> trung bình ± độ lệch chuẩn."""
    if len(ds) == 1:
        return sl.o(khoa, "gt", ds, k)
    return sl.o(f"{khoa}/tb", "tb", ds, k) + "\n${}\\pm{}$" + sl.o(f"{khoa}/dlc", "dlc", ds, k)


def dong(nhan: str, o_list: list[str]) -> str:
    return nhan + "".join(f"\n& {x}" for x in o_list) + "\n\\\\"  # \\\\ dòng riêng: ô cuối kết thúc bằng chú thích %


def ghi_bang(ten: str, cot: str, dau: str, than: list[str]):
    BANG.mkdir(parents=True, exist_ok=True)
    (BANG / f"{ten}.tex").write_text(
        f"% Sinh tự động bởi scripts/tong_hop.py, không sửa tay.\n\\begin{{tabular}}{{{cot}}}\n\\toprule\n{dau}\n"
        "\\midrule\n" + "\n".join(than) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def bang_du_lieu():
    f = "results/thong_ke_mau.json"
    d = json.loads((ROOT / f).read_text())
    than = []
    for lv, nhan in LV.items():
        c = [sl.o(f"du_lieu/{lv}/{k}", "gt", [f], kj, "\\num{{{:,.0f}}}") for k, kj in
             (("o", f"so_o_so_ngay/{lv}/o"), ("ngay", f"so_o_so_ngay/{lv}/ngay"))]
        c += ["-" if d["theo_vai_tro_luu_vuc"][vt][lv] == 0 else
              sl.o(f"du_lieu/{lv}/{vt}", "gt", [f], f"theo_vai_tro_luu_vuc/{vt}/{lv}", "\\num{{{:,.0f}}}")
              for vt in ("train", "val", "test_a", "test_b")]
        than.append(dong(nhan, c))
    than.append("\\midrule")
    than.append(dong("Samples with flooding (fraction)", ["", ""] + [
        sl.o(f"du_lieu/ty_le_co_ngap/{vt}", "gt", [f], f"ty_le_mau_co_ngap/{vt}") for vt in ("train", "val", "test_a", "test_b")]))
    ghi_bang("bang1_du_lieu", "lrrrrrr", "Basin & Tiles & Dates & Train & Val & Test A & Test B \\\\", than)


def bang_ket_qua():
    than = []
    for vt, nvt in VT.items():
        than.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{nvt}}}}} \\\\")
        for nhan, (ten, ng) in MO_HINH.items():
            ds = tep(ten, vt, ng)
            k = f"ket_qua/{vt}/{ten[0]}"
            than.append(dong(nhan, [o(f"{k}/csi", ds, "tong/csi_nguong_toi_uu"),
                                    o(f"{k}/csi_moi", ds, "ngap_moi/csi_nguong_toi_uu"),
                                    o(f"{k}/pr_auc", ds, "tong/pr_auc"), o(f"{k}/brier", ds, "tong/brier"),
                                    o(f"{k}/ece", ds, "tong/ece_15")]))
        if vt == "test_a":
            than.append("\\addlinespace")
    ghi_bang("bang2_ket_qua", "lccccc", "Model & CSI & CSI (new flooding) & PR-AUC & Brier & ECE \\\\", than)


def bang_bo_thanh_phan():
    bien = {"Full model": "", "No soil (texture + CN)": "_bo_dat", "No curve number": "_bo_cn",
            "No soil texture": "_bo_dat_tho", "No weather": "_bo_khi_tuong", "No Sentinel-1": "_bo_s1",
            "No terrain": "_bo_dia_hinh"}
    than = []
    for nhan, hau in bien.items():
        ten = [f"hop_nhat_gfs{hau}_s{s}" for s in SEED]
        than.append(dong(nhan, [o(f"bo/{hau or 'day_du'}/{vt}", tep(ten, vt, "gfs"), "tong/csi_nguong_toi_uu")
                                for vt in ("val", "test_a", "test_b")]))
    than.append("\\midrule")
    for nhan, phep in (("Soil replaced by Gaussian noise", "nhieu"), ("Soil maps permuted across tiles", "hoan_vi")):
        ten = [f"hop_nhat_gfs_s{s}" for s in SEED]
        than.append(dong(nhan, ["-"] + [o(f"bo/{phep}/{vt}", tep(ten, vt, "gfs", f"__{phep}"), "tong/csi_nguong_toi_uu")
                                         for vt in ("test_a", "test_b")]))
    ghi_bang("bang3_bo_thanh_phan", "lccc", "Variant & Val & Test A & Test B \\\\", than)


def bang_mua():
    ten = [f"hop_nhat_gfs_s{s}" for s in SEED]
    bien = {"GFS forecast (operational, as trained)": (ten, "gfs"),
            "ECMWF IFS forecast (swapped at inference)": (ten, "ecmwf"),
            "GFS + GSMaP-NRT antecedent rainfall": (ten, "gfs_gsmap_nrt"),
            "IMERG Final, oracle (trained and tested)": ([f"hop_nhat_imerg_s{s}" for s in SEED], "imerg_final")}
    than = [dong(nhan, [o(f"mua/{ng}/{vt}/{k.split('/')[1]}", tep(t, vt, ng), k) for vt in VT
                        for k in ("tong/csi_nguong_toi_uu", "tong/pr_auc")]) for nhan, (t, ng) in bien.items()]
    ghi_bang("bang4_mua", "lcccc", "Rainfall input & \\multicolumn{2}{c}{Test A} & \\multicolumn{2}{c}{Test B} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & CSI & PR-AUC & CSI & PR-AUC \\\\", than)


def bang_lobo():
    mh = {"TerraMind": "hop_nhat_gfs_lobo_{}_s0", "U-Net": "unet_b3_gfs_lobo_{}_s0", "XGBoost": "b2_lobo_{}"}
    lv5 = [lv for lv in LV if lv != "gianh"]
    f = {m: {lv: f"{DG}/{t.format(lv)}_lobo_{lv}_gfs.json" for lv in lv5} for m, t in mh.items()}
    than = [dong(LV[lv], [sl.o(f"lobo/{m}/{lv}/{k.split('/')[1]}", "gt", [f[m][lv]], k) for m in mh
                          for k in ("tong/csi_nguong_toi_uu", "tong/pr_auc")]) for lv in lv5]
    than.append("\\midrule")
    than.append(dong("Mean", [sl.o(f"lobo/{m}/tb/{k.split('/')[1]}", "tb", list(f[m].values()), k) for m in mh
                              for k in ("tong/csi_nguong_toi_uu", "tong/pr_auc")]))
    ghi_bang("bang5_lobo", "lcccccc", "Held-out basin & \\multicolumn{2}{c}{TerraMind} & \\multicolumn{2}{c}{U-Net} & "
             "\\multicolumn{2}{c}{XGBoost} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
             " & CSI & PR-AUC & CSI & PR-AUC & CSI & PR-AUC \\\\", than)


def bang_kiem_nhan():
    f = ["results/kiem_nhan_test.json"]
    d = json.loads((ROOT / f[0]).read_text())
    than = []
    for phan, nhan_tc in (("test_a_unosat", "UNOSAT"), ("test_b_ems", "Copernicus EMS")):
        g = d[phan]["gfm_so_tham_chieu"]
        for bd in g["theo_ban_do"]:
            kj = f"{phan}/gfm_so_tham_chieu/theo_ban_do/{bd}"
            aoi = " " + " ".join(bd.split("_")[:2]) if phan == "test_b_ems" else ""  # vd EMSR848 AOI01: 3 bản đồ cùng ngày
            nhan = f"{nhan_tc}{aoi}, {g['theo_ban_do'][bd]['ngay']}"
            than.append(dong(nhan, [sl.o(f"kiem/{bd}/so_o", "gt", f, f"{kj}/so_o", "\\num{{{:,.0f}}}")] +
                             [sl.o(f"kiem/{bd}/{k}", "gt", f, f"{kj}/{k}") for k in ("iou", "bo_sot", "bao_nham")]))
        kj = f"{phan}/gfm_so_tham_chieu/tong"
        than.append(dong(f"\\textit{{{nhan_tc}, all}}", [sl.o(f"kiem/{phan}/so_ban_do", "gt", f, f"{kj}/so_ban_do",
                                                               "\\num{{{:,.0f}}} maps")] +
                         [sl.o(f"kiem/{phan}/{k}", "gt", f, f"{kj}/{k}") for k in ("iou", "bo_sot", "bao_nham")]))
        than.append("\\addlinespace")
    # kiểm tra độc lập cảm biến: bản đồ EMS quang học (EMSR754), chụp sau ảnh S1 1-2 ngày; GFM và UNOSAT trên cùng pixel
    than.append("\\multicolumn{5}{l}{\\textit{Optical EMS maps of Yagi, 1 to 2 days after the Sentinel-1 image}} \\\\")
    for nhan, phan in (("GFM vs.\\ optical EMS", "gfm_so_tham_chieu"), ("UNOSAT vs.\\ optical EMS", "phu_so_tham_chieu")):
        kj = f"test_a_quang_hoc/{phan}/tong"
        dau = (sl.o(f"kiem/quang_hoc/so_ban_do", "gt", f, "test_a_quang_hoc/gfm_so_tham_chieu/tong/so_ban_do",
                    "\\num{{{:,.0f}}} maps") if phan == "gfm_so_tham_chieu" else "")
        than.append(dong(nhan, [dau] + [sl.o(f"kiem/quang_hoc/{phan}/{k}", "gt", f, f"{kj}/{k}")
                                        for k in ("iou", "bo_sot", "bao_nham")]))
    # IoU nhỏ nhất, lớn nhất giữa các bản đồ quang học có nước ở ít nhất một sản phẩm (AOI05: không pixel nước nào)
    g = d["test_a_quang_hoc"]["gfm_so_tham_chieu"]["theo_ban_do"]
    co_nuoc = sorted((v["iou"], k) for k, v in g.items() if v["tp"] + v["fp"] + v["fn"] > 0)
    for ten, (_, k) in (("iou_min", co_nuoc[0]), ("iou_max", co_nuoc[-1])):
        sl.o(f"kiem/quang_hoc/{ten}", "gt", f, f"test_a_quang_hoc/gfm_so_tham_chieu/theo_ban_do/{k}/iou")
    sl.ghi("kiem/quang_hoc/so_ban_do_co_nuoc", len(co_nuoc), "tinh_san", dinh_dang="{:d}",
           mo_ta="số bản đồ quang học EMSR754 có tp+fp+fn > 0 trong results/kiem_nhan_test.json")
    ghi_bang("bang6a_kiem_nhan", "lcccc", "Reference map & Tiles (maps) & IoU & Omission & Commission \\\\", than)
    than = []
    for nhan, h in (("B3 U-Net", "unet_b3_gfs"), ("B4 Prithvi-EO-2.0", "prithvi_b4_gfs"), ("B5 U-TAE", "utae_b5_gfs"),
                    (CHINH, "hop_nhat_gfs")):
        c = []
        for phan in ("test_a_unosat", "test_a_quang_hoc", "test_b_ems"):
            for k in ("theo_tham_chieu", "theo_gfm"):
                kj = f"{phan}/mo_hinh_tb_3_seed/{h}/{k}"
                c.append(sl.o(f"kiem/{phan}/{h}/{k}/tb", "gt", f, f"{kj}/tb") + "\n${}\\pm{}$" +
                         sl.o(f"kiem/{phan}/{h}/{k}/dlc", "gt", f, f"{kj}/dlc"))
        than.append(dong(nhan, c))
    ghi_bang("bang6b_diem_theo_tham_chieu", "lcccccc",
             "Model & \\multicolumn{2}{c}{Test A (UNOSAT)} & \\multicolumn{2}{c}{Test A (optical EMS)} & "
             "\\multicolumn{2}{c}{Test B (EMS)} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & vs.\\ ref. & vs.\\ GFM & vs.\\ ref. & vs.\\ GFM"
             " & vs.\\ ref. & vs.\\ GFM \\\\", than)


def _dem_gop(ten: list[str], vt: str, nguon: str) -> pd.DataFrame:
    """Cộng tp/fp/fn theo ô qua các seed, mỗi seed ở ngưỡng chọn trên val của nó."""
    ra = []
    for t in ten:
        j = json.loads((ROOT / DG / f"{t}_{vt}_{nguon}.json").read_text())
        k = int(np.argmin(np.abs(np.array(j["luoi_nguong"]) - j["nguong_toi_uu"]["gia_tri"])))
        p = pd.read_parquet(RESULTS_DIR / "raw" / "du_doan" / f"{t}_{vt}_{nguon}_dem.parquet")
        ra.append(p[["tile_id"]].assign(tp=p[f"tp_{k}"], fp=p[f"fp_{k}"], fn=p[f"fn_{k}"]))
    return pd.concat(ra)


def _dem_seed(ten: list[str], vt: str, nguon: str) -> list[pd.DataFrame]:
    """Mỗi seed một DataFrame tp/fp/fn theo ô (gộp các hạn L), ở ngưỡng chọn trên val của seed đó."""
    return [_dem_gop([t], vt, nguon).groupby("tile_id", as_index=False)[["tp", "fp", "fn"]].sum() for t in ten]


MO_TA_2T = "bootstrap hai tầng (seed rồi khối {k}x{k} ô) B=2000 seed=0, floodlab.tong_hop.bootstrap_cap_phan_cap"


def bang_ghep_cap_phan_cap():
    """Bảng 7b: như bảng 7 nhưng bootstrap hai tầng; p Holm ở khối 1, 2, 5 ô (độ nhạy theo tương quan không gian)."""
    doi = {k: v for k, v in MO_HINH.items() if k != CHINH}
    doi["TerraMind, IMERG oracle"] = ([f"hop_nhat_imerg_s{s}" for s in SEED], "imerg_final")
    than = []
    for vt, nvt in VT.items():
        chinh = _dem_seed(MO_HINH[CHINH][0], vt, "gfs")
        kq = {kk: {n: bootstrap_cap_phan_cap(chinh, _dem_seed(t, vt, ng), B=2000, seed=0, k=kk)
                   for n, (t, ng) in doi.items()} for kk in (1, 2, 5)}
        ph = {kk: dict(zip(kq[kk], holm([r["p"] for r in kq[kk].values()]))) for kk in kq}
        than.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{nvt}}}}} \\\\")
        for n in doi:
            k0, r = f"ghep_cap2/{vt}/{doi[n][0][0]}", kq[1][n]
            c = [sl.o_san(f"{k0}/hieu", r["hieu"], MO_TA_2T.format(k=1), "{:+.3f}"),
                 "[" + sl.o_san(f"{k0}/thap", r["thap"], MO_TA_2T.format(k=1), "{:+.3f}") + "\n, " +
                 sl.o_san(f"{k0}/cao", r["cao"], MO_TA_2T.format(k=1), "{:+.3f}") + "\n]"]
            for kk in (1, 2, 5):
                pv = ph[kk][n]
                c.append(sl.o_san(f"{k0}/p_holm_k{kk}", pv, MO_TA_2T.format(k=kk) + ", Holm",
                                  "$<$0.001" if pv < 0.001 else "{:.3f}"))
            than.append(dong(n, c))
        if vt == "test_a":
            than.append("\\addlinespace")
    ghi_bang("bang7b_ghep_cap_phan_cap", "lccccc",
             "Compared with & $\\Delta$CSI & 95\\% CI (two-level) & \\multicolumn{3}{c}{$p$ (Holm), block size} \\\\\n"
             "\\cmidrule(lr){4-6}\n & (TerraMind $-$ model) & & 1 tile & $2\\times2$ & $5\\times5$ \\\\", than)


def bang_bo_thanh_phan_kiem_dinh():
    """Bảng 3b: hiệu CSI mô hình đầy đủ - biến thể, bootstrap hai tầng ở mức ô; Holm trong từng tập. Biến thể huấn
    luyện lại: seed độc lập; phép thử đất trên chính mô hình đã train: ghép seed."""
    day_du = [f"hop_nhat_gfs_s{s}" for s in SEED]
    bien = {"No soil (texture + CN)": "_bo_dat", "No curve number": "_bo_cn", "No soil texture": "_bo_dat_tho",
            "No weather": "_bo_khi_tuong", "No Sentinel-1": "_bo_s1", "No terrain": "_bo_dia_hinh"}
    phep = {"Soil replaced by Gaussian noise": "nhieu", "Soil maps permuted across tiles": "hoan_vi"}
    kq = {}
    for vt in ("val", "test_a", "test_b"):
        dd = _dem_seed(day_du, vt, "gfs")
        kq[vt] = {n: bootstrap_cap_phan_cap(dd, _dem_seed([f"hop_nhat_gfs{h}_s{s}" for s in SEED], vt, "gfs"), B=2000,
                                            seed=0) for n, h in bien.items()}
        if vt != "val":
            for n, ph in phep.items():
                kq[vt][n] = bootstrap_cap_phan_cap(dd, _dem_seed([f"hop_nhat_gfs_s{s}__{ph}" for s in SEED], vt, "gfs"),
                                                   B=2000, seed=0, ghep_seed=True)
    ph_holm = {vt: dict(zip(kq[vt], holm([r["p"] for r in kq[vt].values()]))) for vt in kq}
    ma = {**{n: h for n, h in bien.items()}, **{n: "__" + p for n, p in phep.items()}}
    p_khoi = {}  # (kk, vt, n) -> p Holm theo khối, in cạnh p mức ô để người đọc kiểm tiêu chí "ô và 2x2"
    for kk in (2, 5):  # độ nhạy theo khối không gian, như bảng 7b
        for vt in ("val", "test_a", "test_b"):
            dd = _dem_seed(day_du, vt, "gfs")
            r = {n: bootstrap_cap_phan_cap(dd, _dem_seed([f"hop_nhat_gfs{h}_s{s}" for s in SEED], vt, "gfs"), B=2000,
                                           seed=0, k=kk) for n, h in bien.items()}
            if vt != "val":
                for n, ph in phep.items():
                    r[n] = bootstrap_cap_phan_cap(dd, _dem_seed([f"hop_nhat_gfs_s{s}__{ph}" for s in SEED], vt, "gfs"),
                                                  B=2000, seed=0, k=kk, ghep_seed=True)
            for n, pv in zip(r, holm([x["p"] for x in r.values()])):
                p_khoi[kk, vt, n] = pv
    than = []
    for n in list(bien) + list(phep):
        if n in phep and n == list(phep)[0]:
            than.append("\\midrule")
        c = []
        for vt in ("val", "test_a", "test_b"):
            if n not in kq[vt]:
                c += ["-", "-"]
                continue
            k0, r = f"bo2/{ma[n]}/{vt}", kq[vt][n]
            pv = ph_holm[vt][n]
            c += [sl.o_san(f"{k0}/hieu", r["hieu"], MO_TA_2T.format(k=1), "{:+.3f}"),
                  "\n\\,/\\,".join(sl.o_san(f"{k0}/p_holm" + ("" if kk == 1 else f"_k{kk}"), p, MO_TA_2T.format(k=kk) + ", Holm",
                                        "$<$0.001" if p < 0.001 else "{:.3f}")
                               for kk, p in ((1, pv), (2, p_khoi[2, vt, n]), (5, p_khoi[5, vt, n])))]
        than.append(dong(n, c))
    ghi_bang("bang3b_bo_thanh_phan_kiem_dinh", "lcccccc",
             "Variant & \\multicolumn{2}{c}{Val} & \\multicolumn{2}{c}{Test A} & \\multicolumn{2}{c}{Test B} \\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
             " & $\\Delta$ & $p$ & $\\Delta$ & $p$ & $\\Delta$ & $p$ \\\\", than)


def kiem_dinh_mua():
    """Đổi nguồn mưa lúc suy luận trên cùng mô hình GFS (ghép seed): GFS - ECMWF, IMERG Final - GSMaP-NRT (phần quan
    trắc); Holm trên 2 phép so mỗi tập test. Chỉ ghi khóa cho lời văn."""
    ten = [f"hop_nhat_gfs_s{s}" for s in SEED]
    for vt in VT:
        goc = _dem_seed(ten, vt, "gfs")
        r = {ng: bootstrap_cap_phan_cap(goc, _dem_seed(ten, vt, ng), B=2000, seed=0, ghep_seed=True)
             for ng in ("ecmwf", "gfs_gsmap_nrt")}
        for ng, pv in zip(r, holm([x["p"] for x in r.values()])):
            mo = MO_TA_2T.format(k=1) + ", ghép seed, Holm trên 2 phép đổi nguồn"
            sl.ghi(f"mua2/{ng}/{vt}/hieu", r[ng]["hieu"], "tinh_san", dinh_dang="{:+.3f}", mo_ta=mo)
            sl.ghi(f"mua2/{ng}/{vt}/p_holm", pv, "tinh_san", dinh_dang="$<$0.001" if pv < 0.001 else "{:.3f}", mo_ta=mo)
        for kk in (2, 5):
            r2 = {ng: bootstrap_cap_phan_cap(goc, _dem_seed(ten, vt, ng), B=2000, seed=0, k=kk, ghep_seed=True)
                  for ng in ("ecmwf", "gfs_gsmap_nrt")}
            for ng, pv in zip(r2, holm([x["p"] for x in r2.values()])):
                sl.ghi(f"mua2/{ng}/{vt}/p_holm_k{kk}", pv, "tinh_san", dinh_dang="$<$0.001" if pv < 0.001 else "{:.3f}",
                       mo_ta=MO_TA_2T.format(k=kk) + ", ghép seed, Holm trên 2 phép đổi nguồn")


def bang_ghep_cap():
    doi = {k: v for k, v in MO_HINH.items() if k != CHINH}
    doi["TerraMind, IMERG oracle"] = ([f"hop_nhat_imerg_s{s}" for s in SEED], "imerg_final")
    than = []
    for vt, nvt in VT.items():
        chinh = _dem_gop(MO_HINH[CHINH][0], vt, "gfs")
        kq = {n: bootstrap_cap(chinh, _dem_gop(t, vt, ng), B=2000, seed=0) for n, (t, ng) in doi.items()}
        ph = dict(zip(kq, holm([r["p"] for r in kq.values()])))
        than.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{nvt}}}}} \\\\")
        mo = f"bootstrap cụm theo ô B=2000 seed=0, {vt}, gộp 3 seed (scripts/tong_hop.py::bang_ghep_cap)"
        for n, r in kq.items():
            k = f"ghep_cap/{vt}/{doi[n][0][0]}"
            than.append(dong(n, [sl.o_san(f"{k}/hieu", r["hieu"], mo, "{:+.3f}"),
                                 "[" + sl.o_san(f"{k}/thap", r["thap"], mo, "{:+.3f}") + "\n, " +
                                 sl.o_san(f"{k}/cao", r["cao"], mo, "{:+.3f}") + "\n]",
                                 sl.o_san(f"{k}/p_holm", ph[n], mo + ", Holm",
                                          "$<$0.001" if ph[n] < 0.001 else "{:.3f}")]))
        if vt == "test_a":
            than.append("\\addlinespace")
    ghi_bang("bang7_ghep_cap", "lccc", "Compared with & $\\Delta$CSI (TerraMind $-$ model) & 95\\% CI & "
             "$p$ (Holm) \\\\", than)


def hinh_theo_L():
    HINH.mkdir(parents=True, exist_ok=True)
    nhom = {"B2 XGBoost": MO_HINH["B2 XGBoost"], "B3 U-Net": MO_HINH["B3 U-Net"],
            "B4 Prithvi-EO-2.0": MO_HINH["B4 Prithvi-EO-2.0"], "B5 U-TAE": MO_HINH["B5 U-TAE"], CHINH: MO_HINH[CHINH]}
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.9), sharey=True)  # in ở toàn khổ chữ (5,4 in), chữ không bị thu nhỏ
    for a, (vt, nvt) in zip(ax, VT.items()):
        for nhan, (ten, ng) in nhom.items():
            y = np.array([[json.loads((ROOT / f).read_text())["theo_L"][str(L)]["csi_nguong_toi_uu"] for L in (1, 2, 3)]
                          for f in tep(ten, vt, ng)])
            a.errorbar([1, 2, 3], y.mean(0), yerr=y.std(0, ddof=1) if len(y) > 1 else None, marker="o", capsize=3,
                       lw=2 if nhan == CHINH else 1.2, label=nhan)
        a.set_title(nvt)
        a.set_xticks([1, 2, 3])
        a.set_xlabel("Lead time L (days)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("CSI (validation threshold)")
    tay, nhan = ax[0].get_legend_handles_labels()  # chú giải dưới hình: đặt trong ô Test B thì đè lên đường
    fig.legend(tay, nhan, fontsize=7.5, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HINH / "hinh_csi_theo_L.pdf")
    plt.close(fig)


def hinh_khoang_cach_s1():
    from floodlab import mau
    from floodlab.config import LUU_VUC
    c = pd.concat([mau.doc_chi_muc(lv)[["vai_tro", "L", "khoang_cach_s1_ngay"]] for lv in LUU_VUC])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for vt, nhan in (("train", "Train"), ("val", "Val"), ("test_a", "Test A"), ("test_b", "Test B")):
        x = c[c.vai_tro == vt].khoang_cach_s1_ngay
        ax.hist(x, bins=np.arange(0.5, 25.5), histtype="step", density=True, lw=1.3, label=nhan)
    ax.set_xlabel("Days since last Sentinel-1 image")
    ax.set_ylabel("Fraction of samples")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(HINH / "hinh_khoang_cach_s1.pdf")
    plt.close(fig)


def so_lieu_loi_van():
    """bai_bao/so_lieu.tex: macro \\sTen cho số dùng trong lời văn; lấy khóa đã có trong bảng hoặc ghi thêm (val, dữ liệu)."""
    for ten, (t, ng) in MO_HINH.items():
        o(f"val/{t[0]}/csi", tep(t, "val", ng), "tong/csi_nguong_toi_uu")
    o("val/hop_nhat_imerg_s0/csi", tep([f"hop_nhat_imerg_s{s}" for s in SEED], "val", "imerg_final"),
      "tong/csi_nguong_toi_uu")
    sl.o("du_lieu/tong_mau", "gt", ["results/thong_ke_mau.json"], "tong_mau", "\\num{{{:,.0f}}}")
    for vt in VT:
        sl.o(f"du_lieu/so_mau_{vt}", "gt", ["results/thong_ke_mau.json"], f"theo_vai_tro_L/{vt}/1", "\\num{{{:,.0f}}}")
    tm, ma = "hop_nhat_gfs_s0", {}
    for tat, t in (("Tm", tm), ("Unet", "unet_b3_gfs_s0"), ("Utae", "utae_b5_gfs_s0"), ("Pri", "prithvi_b4_gfs_s0"),
                   ("Xgb", "b2"), ("Bzero", "b0"), ("Bsix", "b6"), ("Bone", "b1")):
        for vt, c in (("test_a", "A"), ("test_b", "B")):
            k = f"ket_qua/{vt}/{t}/csi"
            if f"{k}/dlc" in sl.muc:
                ma[f"{tat}{c}"] = f"{k}/tb"
                ma[f"{tat}{c}sd"] = f"{k}/dlc"
            else:
                ma[f"{tat}{c}"] = k
        if f"val/{t}/csi/tb" in sl.muc:
            ma[f"{tat}Val"] = f"val/{t}/csi/tb"
        elif f"val/{t}/csi" in sl.muc:
            ma[f"{tat}Val"] = f"val/{t}/csi"
    ma["ImergVal"] = "val/hop_nhat_imerg_s0/csi/tb"
    for c, vt in (("A", "test_a"), ("B", "test_b")):
        ma[f"Noise{c}"], ma[f"Perm{c}"] = f"bo/nhieu/{vt}/tb", f"bo/hoan_vi/{vt}/tb"
        for tat, h in (("NoSoil", "_bo_dat"), ("NoCn", "_bo_cn"), ("NoTex", "_bo_dat_tho"), ("NoWx", "_bo_khi_tuong"),
                       ("NoSone", "_bo_s1"), ("NoTerr", "_bo_dia_hinh")):
            ma[f"{tat}{c}"] = f"bo/{h}/{vt}/tb"
        for tat, ng in (("Ecm", "ecmwf"), ("Gsm", "gfs_gsmap_nrt"), ("Imerg", "imerg_final")):
            ma[f"{tat}{c}"] = f"mua/{ng}/{vt}/csi_nguong_toi_uu/tb"
        for tat, t in (("Utae", "utae_b5_gfs_s0"), ("Unet", "unet_b3_gfs_s0"), ("Xgb", "b2"), ("Imerg", "hop_nhat_imerg_s0"),
                       ("Pri", "prithvi_b4_gfs_s0"), ("Bsix", "b6")):
            for hau, k in (("", "hieu"), ("Lo", "thap"), ("Hi", "cao"), ("P", "p_holm_k1"), ("PBlockTwo", "p_holm_k2"),
                           ("PBlockFive", "p_holm_k5")):
                ma[f"d{tat}{c}{hau}"] = f"ghep_cap2/{vt}/{t}/{k}"  # bootstrap hai tầng (bảng 7b)
    for tat, m in (("Tm", "TerraMind"), ("Unet", "U-Net"), ("Xgb", "XGBoost")):
        ma[f"Lobo{tat}"] = f"lobo/{m}/tb/csi_nguong_toi_uu"
        for lv, c in (("hong_thai_binh", "Htb"), ("tra_khuc", "Tk"), ("huong", "Huong"), ("ca", "Ca"), ("vu_gia_thu_bon", "Vgtb")):
            ma[f"Lobo{tat}{c}"] = f"lobo/{m}/{lv}/csi_nguong_toi_uu"
    for c, ph in (("Unosat", "test_a_unosat"), ("Ems", "test_b_ems"), ("Opt", "test_a_quang_hoc")):
        if c != "Opt":  # chỉ số nhãn quang học có khóa riêng (kiem/quang_hoc/..., macro OptGfm*, OptUnosat*)
            for k, tat in (("iou", "IoU"), ("bo_sot", "Om"), ("bao_nham", "Com")):
                ma[f"{c}{tat}"] = f"kiem/{ph}/{k}"
        for tat, h in (("Tm", "hop_nhat_gfs"), ("Unet", "unet_b3_gfs"), ("Utae", "utae_b5_gfs"), ("Pri", "prithvi_b4_gfs")):
            ma[f"{tat}{c}"] = f"kiem/{ph}/{h}/theo_tham_chieu/tb"
            ma[f"{tat}{c}Gfm"] = f"kiem/{ph}/{h}/theo_gfm/tb"
    for vt, c in (("test_a", "A"), ("test_b", "B")):
        ma[f"TmMoi{c}"] = f"ket_qua/{vt}/hop_nhat_gfs_s0/csi_moi/tb"
        ma[f"BsixMoi{c}"] = f"ket_qua/{vt}/b6/csi_moi"
        ma[f"TmEce{c}"] = f"ket_qua/{vt}/hop_nhat_gfs_s0/ece/tb"
        ma[f"TmPr{c}"] = f"ket_qua/{vt}/hop_nhat_gfs_s0/pr_auc/tb"
        ma[f"UnetPr{c}"] = f"ket_qua/{vt}/unet_b3_gfs_s0/pr_auc/tb"
        ma[f"UtaePr{c}"] = f"ket_qua/{vt}/utae_b5_gfs_s0/pr_auc/tb"
    for ma_bo, tat in (("_bo_dat", "Soil"), ("_bo_cn", "Cn"), ("_bo_dat_tho", "Tex"), ("_bo_khi_tuong", "Wx"),
                       ("_bo_s1", "Sone"), ("_bo_dia_hinh", "Terr"), ("__nhieu", "Noise"), ("__hoan_vi", "Perm")):
        for vt, c in (("val", "Val"), ("test_a", "A"), ("test_b", "B")):
            if f"bo2/{ma_bo}/{vt}/hieu" in sl.muc:
                ma[f"Dbo{tat}{c}"] = f"bo2/{ma_bo}/{vt}/hieu"
                ma[f"Dbo{tat}{c}P"] = f"bo2/{ma_bo}/{vt}/p_holm"
    for t, tat in (("hop_nhat_gfs_s0", "Tm"), ("unet_b3_gfs_s0", "Unet"), ("utae_b5_gfs_s0", "Utae"),
                   ("prithvi_b4_gfs_s0", "Pri")):
        ten = [t.replace("_s0", f"_s{s_}") for s_ in SEED]
        for vt, c in (("test_a", "A"), ("test_b", "B")):
            o(f"csi05/{vt}/{t}", tep(ten, vt, "gfs"), "tong/csi")
            ma[f"{tat}HalfThr{c}"] = f"csi05/{vt}/{t}/tb"
    for k, tat in (("iou", "IoU"), ("bo_sot", "Om"), ("bao_nham", "Com")):
        ma[f"OptGfm{tat}"] = f"kiem/quang_hoc/gfm_so_tham_chieu/{k}"
        ma[f"OptUnosat{tat}"] = f"kiem/quang_hoc/phu_so_tham_chieu/{k}"
    for vt, c in (("test_a", "A"), ("test_b", "B")):
        for ng, tat in (("ecmwf", "Ecm"), ("gfs_gsmap_nrt", "Gsm")):
            ma[f"dSwap{tat}{c}"], ma[f"dSwap{tat}{c}P"] = f"mua2/{ng}/{vt}/hieu", f"mua2/{ng}/{vt}/p_holm"
            ma[f"dSwap{tat}{c}PBlockTwo"] = f"mua2/{ng}/{vt}/p_holm_k2"
            ma[f"dSwap{tat}{c}PBlockFive"] = f"mua2/{ng}/{vt}/p_holm_k5"
        for ma_bo, tat in (("_bo_dat", "Soil"), ("_bo_khi_tuong", "Wx"), ("_bo_s1", "Sone"), ("__hoan_vi", "Perm")):
            for kk, chu in ((2, "Two"), (5, "Five")):
                ma[f"Dbo{tat}{c}PBlock{chu}"] = f"bo2/{ma_bo}/{vt}/p_holm_k{kk}"
    from floodlab import mau  # tuổi ảnh S1: tỉ lệ mẫu Test A dùng ảnh 9 ngày, mẫu train dùng ảnh 3 ngày (hình 2)
    from floodlab.config import LUU_VUC
    c = pd.concat([mau.doc_chi_muc(lv)[["vai_tro", "khoang_cach_s1_ngay"]] for lv in LUU_VUC])
    for vt, ngay, tat in (("test_a", 9, "AgeNineA"), ("train", 3, "AgeThreeTrain")):
        x = c[c.vai_tro == vt].khoang_cach_s1_ngay
        sl.ghi(f"du_lieu/tuoi_anh/{vt}_{ngay}", float((x == ngay).mean() * 100), "tinh_san", dinh_dang="{:.0f}\\%",
               mo_ta=f"% mẫu {vt} có khoang_cach_s1_ngay == {ngay}, data/mau/chi_muc_*.parquet")
        ma[tat] = f"du_lieu/tuoi_anh/{vt}_{ngay}"
    for tat, t in (("Tm", "hop_nhat_gfs_s0"), ("Utae", "utae_b5_gfs_s0")):  # CSI ngập mới, mục cảnh báo sớm
        ma[f"{tat}NewA"] = f"ket_qua/test_a/{t}/csi_moi/tb"
    ma["BsixNewA"] = "ket_qua/test_a/b6/csi_moi"
    ma["OptIoUMin"], ma["OptIoUMax"] = "kiem/quang_hoc/iou_min", "kiem/quang_hoc/iou_max"
    ma["OptNWet"] = "kiem/quang_hoc/so_ban_do_co_nuoc"
    for c, vt in (("A", "test_a"), ("B", "test_b")):
        ma[f"DboNoise{c}PBlockTwo"], ma[f"DboNoise{c}PBlockFive"] = (f"bo2/__nhieu/{vt}/p_holm_k2",
                                                                    f"bo2/__nhieu/{vt}/p_holm_k5")
    ma["NSamples"] = "du_lieu/tong_mau"
    ma["NTestA"], ma["NTestB"] = "du_lieu/so_mau_test_a", "du_lieu/so_mau_test_b"
    dong_ra = ["% Sinh tự động bởi scripts/tong_hop.py, không sửa tay. Mỗi macro kèm khóa trong results/tong_hop.json."]
    for ten, k in ma.items():
        if k not in sl.muc:
            raise KeyError(f"macro {ten}: thiếu khóa {k}")
        v = sl.muc[k]
        dong_ra.append(f"\\newcommand{{\\s{ten}}}{{{v['dinh_dang'].format(v['gia_tri'])}}}% nguon: {k}")
    (ROOT / "bai_bao" / "so_lieu.tex").write_text("\n".join(dong_ra) + "\n", encoding="utf-8")
    return len(ma)


def main():
    bang_du_lieu()
    bang_ket_qua()
    bang_bo_thanh_phan()
    bang_mua()
    bang_lobo()
    bang_kiem_nhan()
    bang_ghep_cap()
    bang_ghep_cap_phan_cap()
    bang_bo_thanh_phan_kiem_dinh()
    kiem_dinh_mua()
    n = so_lieu_loi_van()
    sl.luu(RESULTS_DIR / "tong_hop.json")
    print(f"{n} macro lời văn")
    hinh_theo_L()
    hinh_khoang_cach_s1()
    print(f"{len(sl.muc)} số, {len(list(BANG.glob('*.tex')))} bảng, {len(list(HINH.glob('*.pdf')))} hình")


if __name__ == "__main__":
    main()
