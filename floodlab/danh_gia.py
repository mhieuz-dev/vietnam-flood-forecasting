"""Chấm một bộ dự đoán trên một vai trò (val, test_a, test_b, lobo_<lưu vực>); dùng chung cho đối chứng và mô hình học sâu.

du_doan(mau) -> xác suất (224, 224), mau là dict của mau.doc_mau kèm khóa chi_muc (dòng chỉ mục). Mẫu test chỉ đọc qua chia_tap.doc_test (có ghi
nhật ký) và phải nằm trong khóa. Ngưỡng tối ưu CSI chỉ chọn trên val; test dùng lại ngưỡng đó từ json của val.
Chỉ số gộp pixel trên toàn vai trò; PR-AUC tính từ biểu đồ tần suất 1000 khoảng (sai khác với AP chính xác chỉ ở
các xác suất cùng khoảng 0,001). Khoảng tin cậy 95% bootstrap theo cụm ô trong từng trận (chi_so.bootstrap_ty_so).
Ghi results/danh_gia/{ten}_{vai_tro}_{nguon}.json, bảng đếm từng mẫu ở results/raw/du_doan/..._dem.parquet (cho
so sánh ghép cặp) và xác suất từng pixel uint8 (p * 254) ở results/raw/du_doan/....zarr.
"""
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import zarr

from floodlab import chi_so, chia_tap, mau
from floodlab.config import LUU_VUC, NHAN_BO_QUA, RESULTS_DIR, TILE_PX
from floodlab.su_kien import doc_su_kien

DIR_DANH_GIA = RESULTS_DIR / "danh_gia"
DIR_DU_DOAN = RESULTS_DIR / "raw" / "du_doan"
LUOI_NGUONG = np.round(np.arange(0.05, 0.951, 0.05), 2)
K05 = int(np.flatnonzero(LUOI_NGUONG == 0.5)[0])
SO_KHOANG = 1000
KENH_WC = mau.KENH_TINH.index("worldcover")
DEM = ("tp", "fp", "fn", "tn")


def _ngay(x) -> str:
    return pd.Timestamp(x).date().isoformat()


def _tran(r, su_kien) -> str:
    t = pd.Timestamp(r.ngay_t).date()
    for sk in su_kien:
        if r.luu_vuc in sk.luu_vuc and sk.bat_dau <= t <= sk.ket_thuc:
            return sk.ma
    return f"{r.luu_vuc}_ngoai_tran_{t.year}"


def _mau_vai_tro(vai_tro: str, ten: str, nguon: str) -> pd.DataFrame:
    if vai_tro.startswith("lobo_"):  # LOBO: mẫu train của lưu vực đã bị bỏ khỏi lúc train mô hình đó
        lv = vai_tro.removeprefix("lobo_")
        if lv not in LUU_VUC:
            raise ValueError(f"lưu vực LOBO lạ: {lv}")
        c = mau.doc_chi_muc(lv)
        return c[c.vai_tro == "train"].reset_index(drop=True)
    d = pd.concat([c[c.vai_tro == vai_tro] for c in (mau.doc_chi_muc(lv) for lv in LUU_VUC)], ignore_index=True)
    if vai_tro in ("test_a", "test_b"):
        khoa = chia_tap.doc_test(vai_tro, ly_do=f"danh_gia {ten} {nguon}")
        cho_phep = set(zip(khoa.tile_id, khoa.ngay_t.map(_ngay)))
        ngoai = [c for c in zip(d.tile_id, d.ngay_t.map(_ngay)) if c not in cho_phep]
        if ngoai:
            raise RuntimeError(f"{len(ngoai)} mẫu {vai_tro} không có trong tập test đã khóa, vd {ngoai[:3]}")
    return d


def _dem(p, y_ngap, chon) -> np.ndarray:
    """(4, len(LUOI_NGUONG)) tp, fp, fn, tn trên các pixel chon."""
    pp, yy = p[chon], y_ngap[chon]
    du = pp[:, None] >= LUOI_NGUONG[None, :]
    tp = (du & yy[:, None]).sum(0)
    fp = du.sum(0) - tp
    return np.stack([tp, fp, yy.sum() - tp, (~yy).sum() - fp])


def _chi_so(d: np.ndarray, k: int) -> dict:
    return chi_so.tu_bang(*(int(v) for v in d[:, k])) | {"so_pixel": int(d[:, k].sum()), "so_ngap": int(d[0, k] + d[2, k])}


def _ci_csi(dem: pd.DataFrame, tien_to: str, k: int, B: int):
    return chi_so.bootstrap_ty_so(dem, [f"{tien_to}tp_{k}"], [f"{tien_to}{c}_{k}" for c in ("tp", "fp", "fn")], B=B)


def danh_gia(du_doan, vai_tro: str, nguon_du_bao: str, ten: str, nguon_quan_trac: str = "imerg_final",
             B: int = 1000, luu_xac_suat: bool = True, ten_nguong: str | None = None, nguon_nguong: str | None = None):
    """ten_nguong, nguon_nguong: lấy ngưỡng từ val của lượt/nguồn khác (mặc định chính nó): phép thử đất đặt tên
    riêng, còn mưa ECMWF/GSMaP không có val cùng nguồn nên dùng val GFS của chính mô hình."""
    t_bd = time.time()
    nguon = nguon_du_bao if nguon_quan_trac == "imerg_final" else f"{nguon_du_bao}_{nguon_quan_trac}"
    if vai_tro == "val":
        f_val = None
    else:
        f_val = DIR_DANH_GIA / f"{ten_nguong or ten}_val_{nguon_nguong or nguon}.json"
        if not f_val.exists():
            raise FileNotFoundError(f"chưa có kết quả val {f_val}: ngưỡng cho {vai_tro} phải chọn trên val trước")
    d = _mau_vai_tro(vai_tro, ten, nguon)
    su_kien = doc_su_kien()
    DIR_DU_DOAN.mkdir(parents=True, exist_ok=True)
    ghi_p = (mau._Ghi(zarr.open_group(DIR_DU_DOAN / f"{ten}_{vai_tro}_{nguon}.zarr", mode="w"), "xac_suat",
                      (TILE_PX, TILE_PX), "uint8") if luu_xac_suat else None)
    dong, lop_phu = [], {}
    h_ngap, h_kho = np.zeros(SO_KHOANG), np.zeros(SO_KHOANG)
    brier, ece_p, ece_y, ece_n = 0.0, np.zeros(15), np.zeros(15), np.zeros(15)
    g_mo = {}
    for r in d.itertuples():
        if r.luu_vuc not in g_mo:
            g_mo = {r.luu_vuc: mau.mo_zarr(r.luu_vuc)}
        m = mau.doc_mau(g_mo[r.luu_vuc], r, nguon_du_bao=nguon_du_bao, nguon_quan_trac=nguon_quan_trac)
        m["chi_muc"] = r._asdict()
        p = np.clip(np.asarray(du_doan(m), np.float32), 0, 1)
        if p.shape != (TILE_PX, TILE_PX):
            raise ValueError(f"du_doan trả {p.shape}, cần ({TILE_PX}, {TILE_PX})")
        if ghi_p is not None:
            ghi_p.them(np.round(p * 254).astype(np.uint8))
        y = m["nhan"]
        hl, y_ngap = y != NHAN_BO_QUA, y == 1
        tong, moi = _dem(p, y_ngap, hl), _dem(p, y_ngap, hl & (m["nhan_trang_thai"] == 0))
        wc = m["tinh"][KENH_WC]
        for c in np.unique(wc[hl]):
            lop_phu[int(c)] = lop_phu.get(int(c), 0) + _dem(p, y_ngap, hl & (wc == c))
        pp, yy = p[hl], y_ngap[hl]
        h_ngap += np.histogram(pp[yy], SO_KHOANG, (0, 1))[0]
        h_kho += np.histogram(pp[~yy], SO_KHOANG, (0, 1))[0]
        brier += float(((pp - yy) ** 2).sum())
        kh = np.minimum((pp * 15).astype(int), 14)
        ece_p += np.bincount(kh, pp, 15)
        ece_y += np.bincount(kh, yy, 15)
        ece_n += np.bincount(kh, minlength=15)
        hang = {"tile_id": r.tile_id, "luu_vuc": r.luu_vuc, "ngay_t": _ngay(r.ngay_t), "L": int(r.L),
                "tran": _tran(r, su_kien)}
        for tien_to, a in (("", tong), ("moi_", moi)):
            for i, c in enumerate(DEM):
                hang |= {f"{tien_to}{c}_{k}": int(a[i, k]) for k in range(len(LUOI_NGUONG))}
        dong.append(hang)
    if ghi_p is not None:
        ghi_p.xa()
    dem = pd.DataFrame(dong)
    dem.to_parquet(DIR_DU_DOAN / f"{ten}_{vai_tro}_{nguon}_dem.parquet", index=False)

    def gop(ds, tien_to=""):
        return np.stack([[ds[f"{tien_to}{c}_{k}"].sum() for k in range(len(LUOI_NGUONG))] for c in DEM])

    tong = gop(dem)
    if f_val is None:
        csi = [_chi_so(tong, k)["csi"] for k in range(len(LUOI_NGUONG))]
        k_opt = int(np.nanargmax(csi))
        nguong_toi_uu = {"gia_tri": float(LUOI_NGUONG[k_opt]), "nguon": "val"}
    else:
        nguong_toi_uu = {"gia_tri": json.loads(f_val.read_text())["nguong_toi_uu"]["gia_tri"], "nguon": "val",
                         "file": f_val.name}
        k_opt = int(np.flatnonzero(LUOI_NGUONG == nguong_toi_uu["gia_tri"])[0])

    def hai_nguong(a):
        return _chi_so(a, K05) | {"csi_nguong_toi_uu": _chi_so(a, k_opt)["csi"]}

    # PR-AUC (average precision) từ biểu đồ tần suất, duyệt ngưỡng từ cao xuống thấp
    tp_c, fp_c = np.cumsum(h_ngap[::-1]), np.cumsum(h_kho[::-1])
    pr_auc = float((tp_c / np.maximum(tp_c + fp_c, 1) * h_ngap[::-1]).sum() / max(h_ngap.sum(), 1))
    n_px = h_ngap.sum() + h_kho.sum()
    ra = {
        "ten": ten, "vai_tro": vai_tro, "nguon_du_bao": nguon_du_bao, "nguon_quan_trac": nguon_quan_trac,
        "so_mau": len(dem), "nguong_toi_uu": nguong_toi_uu, "luoi_nguong": LUOI_NGUONG.tolist(),
        "tong": hai_nguong(tong) | {"pr_auc": pr_auc, "brier": brier / max(n_px, 1),
                                    "ece_15": float(np.abs(ece_p - ece_y).sum() / max(n_px, 1))},
        "ngap_moi": hai_nguong(gop(dem, "moi_")),
        "theo_L": {str(L): hai_nguong(gop(g)) for L, g in dem.groupby("L")},
        "theo_tran": {t: hai_nguong(gop(g)) | {"so_mau": len(g)} for t, g in dem.groupby("tran")},
        "theo_lop_phu": {str(c): hai_nguong(a) for c, a in sorted(lop_phu.items())},
        "khoang_tin_cay_95": {"B": B, "cum": "tile_id trong từng trận",
                              "csi_0.5": _ci_csi(dem, "", K05, B), "csi_nguong_toi_uu": _ci_csi(dem, "", k_opt, B),
                              "ngap_moi_csi_0.5": _ci_csi(dem, "moi_", K05, B)},
        "git": chia_tap._git_hash(), "lenh": " ".join(sys.argv), "thoi_gian_giay": round(time.time() - t_bd, 1),
        "luc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DIR_DANH_GIA.mkdir(parents=True, exist_ok=True)
    f = DIR_DANH_GIA / f"{ten}_{vai_tro}_{nguon}.json"
    f.write_text(json.dumps(ra, indent=1, ensure_ascii=False, default=float))
    return f
