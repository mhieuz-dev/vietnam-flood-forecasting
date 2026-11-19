"""Số liệu cho bảng của bài (Task 17): mỗi số trong bai_bao/bang/*.tex kèm chú thích "% nguon: <khóa>" trỏ vào
results/tong_hop.json; mỗi khóa ở đó ghi cách tính lại từ json gốc (ham, nguon, khoa_json) để kiểm ngược.

ham: gt (một tệp), tb / dlc (trung bình / độ lệch chuẩn mẫu qua các tệp, vd 3 seed), tinh_san (số tính sẵn như
bootstrap ghép cặp, kèm mo_ta; kiểm bằng test riêng của hàm tính). Khóa json dùng "/" (vd tong/csi_nguong_toi_uu,
khoang_tin_cay_95/csi_nguong_toi_uu/1); phần tử danh sách viết bằng chỉ số.
"""
import json
import re
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

from floodlab.config import ROOT

_SO = re.compile(r"(?<![\w.-])-?\d+\.\d+|\\num\{[^}]*\}")  # bỏ số dính chữ như Prithvi-EO-2.0


def doc_khoa(d, khoa: str):
    for k in khoa.split("/"):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def _doc(f: str):
    p = Path(f)
    return json.loads((p if p.is_absolute() else ROOT / p).read_text(encoding="utf-8"))


def tinh(ham: str, nguon: list[str], khoa: str) -> float:
    x = [float(doc_khoa(_doc(f), khoa)) for f in nguon]
    if ham == "gt":
        assert len(x) == 1
        return x[0]
    if ham == "tb":
        return st.mean(x)
    if ham == "dlc":
        return st.stdev(x)
    raise ValueError(ham)


class SoLieu:
    def __init__(self):
        self.muc = {}

    def ghi(self, khoa: str, gia_tri: float, ham: str, nguon=(), khoa_json: str = "", dinh_dang: str = "{:.3f}",
            mo_ta: str = "") -> float:
        if khoa in self.muc:
            raise KeyError(f"trùng khóa {khoa}")
        self.muc[khoa] = {"gia_tri": gia_tri, "ham": ham, "nguon": [str(Path(f).relative_to(ROOT)) if Path(f).is_absolute()
                                                                    else str(f) for f in nguon],
                          "khoa_json": khoa_json, "dinh_dang": dinh_dang, "mo_ta": mo_ta}
        return gia_tri

    def o(self, khoa: str, ham: str, nguon, khoa_json: str, dinh_dang: str = "{:.3f}") -> str:
        """Tính từ json gốc, ghi lại, trả chuỗi ô bảng kèm chú thích nguồn (kết thúc dòng)."""
        v = self.ghi(khoa, tinh(ham, list(nguon), khoa_json), ham, nguon, khoa_json, dinh_dang)
        return f"{dinh_dang.format(v)}% nguon: {khoa}"

    def o_san(self, khoa: str, gia_tri: float, mo_ta: str, dinh_dang: str = "{:.3f}") -> str:
        v = self.ghi(khoa, gia_tri, "tinh_san", (), "", dinh_dang, mo_ta)
        return f"{dinh_dang.format(v)}% nguon: {khoa}"

    def luu(self, f: Path):
        Path(f).write_text(json.dumps(self.muc, indent=1, ensure_ascii=False), encoding="utf-8")


def kiem_bang(bang: Path, f_so_lieu: Path) -> list[str]:
    """Lỗi trong một bảng (hoặc tệp macro \\newcommand): số có nguồn mà giá trị lệch tong_hop.json, hoặc số (thập phân, \\num) không có nguồn."""
    muc = json.loads(Path(f_so_lieu).read_text(encoding="utf-8"))
    loi, trong_than = [], False
    for i, dong in enumerate(Path(bang).read_text(encoding="utf-8").splitlines(), 1):
        if dong.strip().startswith("\\midrule"):
            trong_than = True
            continue
        if dong.strip().startswith("\\bottomrule"):
            trong_than = False
        if not trong_than and not dong.lstrip().startswith("\\newcommand"):  # bảng hoặc macro số liệu cho lời văn
            continue
        pc = re.search(r"(?<!\\)%", dong)
        ma, cm = (dong[:pc.start()], dong[pc.end():]) if pc else (dong, "")
        ma = re.sub(r"\\(cmidrule|addlinespace)\S*", "", ma)
        so = _SO.findall(ma)
        m = re.search(r"nguon:\s*(\S+)", cm)
        if not m:
            if so:
                loi.append(f"{bang.name}:{i}: số {so} không có nguồn")
            continue
        k = m.group(1)
        if k not in muc:
            loi.append(f"{bang.name}:{i}: khóa {k} không có trong tong_hop.json")
            continue
        chuoi = muc[k]["dinh_dang"].format(muc[k]["gia_tri"])
        if so != _SO.findall(chuoi):
            loi.append(f"{bang.name}:{i}: {so} khác {chuoi} ({k})")
    return loi


def kiem_nguon(f_so_lieu: Path) -> list[str]:
    """Tính lại mọi khóa gt/tb/dlc từ json gốc, so với giá trị đã ghi."""
    loi = []
    for k, v in json.loads(Path(f_so_lieu).read_text(encoding="utf-8")).items():
        if v["ham"] == "tinh_san":
            if not v["mo_ta"]:
                loi.append(f"{k}: số tính sẵn thiếu mô tả")
            continue
        x = tinh(v["ham"], v["nguon"], v["khoa_json"])
        if abs(x - v["gia_tri"]) > 1e-12:
            loi.append(f"{k}: {v['gia_tri']} khác tính lại {x}")
    return loi


def bootstrap_cap(a: pd.DataFrame, b: pd.DataFrame, B: int = 2000, seed: int = 0) -> dict:
    """Hiệu CSI gộp (a - b) với bootstrap cụm theo ô (cùng mẫu ô cho hai mô hình), khoảng 95% và p hai phía."""
    g = [x.groupby("tile_id")[["tp", "fp", "fn"]].sum() for x in (a, b)]
    o = g[0].index.union(g[1].index)
    xa, xb = (x.reindex(o, fill_value=0).to_numpy(float) for x in g)

    def csi(x):
        return x[..., 0] / np.maximum(x[..., 0] + x[..., 1] + x[..., 2], 1)

    hieu = float(csi(xa.sum(0)) - csi(xb.sum(0)))
    chon = np.random.default_rng(seed).integers(0, len(o), (B, len(o)))
    d = csi(xa[chon].sum(1)) - csi(xb[chon].sum(1))
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"hieu": hieu, "thap": float(np.quantile(d, 0.025)), "cao": float(np.quantile(d, 0.975)), "p": p, "B": B,
            "so_o": len(o)}


def holm(p: list[float]) -> list[float]:
    thu_tu = np.argsort(p)
    ra, cao = np.empty(len(p)), 0.0
    for r, i in enumerate(thu_tu):
        cao = max(cao, min(1.0, (len(p) - r) * p[i]))
        ra[i] = cao
    return ra.tolist()


def khoi_o(tile_id: str, k: int) -> str:
    """Khối không gian k x k ô (ô 4,48 km) chứa tile_id dạng <lưu vực>_<i>_<j>; k = 1 trả nguyên ô."""
    if k == 1:
        return tile_id
    lv, i, j = tile_id.rsplit("_", 2)
    return f"{lv}_{int(i) // k}_{int(j) // k}"


def _mang_seed(ds: list[pd.DataFrame], don_vi: pd.Index, k: int) -> np.ndarray:
    """(số seed, số đơn vị lấy mẫu, 3) đếm tp/fp/fn theo khối k x k."""
    ra = []
    for d in ds:
        g = d.assign(khoi=d.tile_id.map(lambda t: khoi_o(t, k))).groupby("khoi")[["tp", "fp", "fn"]].sum()
        ra.append(g.reindex(don_vi, fill_value=0).to_numpy(float))
    return np.stack(ra)


def bootstrap_cap_phan_cap(a: list[pd.DataFrame], b: list[pd.DataFrame], B: int = 2000, seed: int = 0, k: int = 1,
                           ghep_seed: bool = False) -> dict:
    """Bootstrap hai tầng cho hiệu CSI gộp (a - b): mỗi vòng rút lại có hoàn lại các seed của từng mô hình (mô hình
    tất định truyền 1 phần tử thì giữ nguyên), rồi rút lại các khối không gian k x k ô, dùng chung chỉ số khối cho hai
    mô hình. ghep_seed=True (cùng mô hình, vd phép thử đất trên chính lượt đó): dùng chung chỉ số seed cho a và b.
    Ước lượng điểm là hiệu CSI gộp mọi seed, như bootstrap_cap."""
    don_vi = pd.Index(sorted({khoi_o(t, k) for d in a + b for t in d.tile_id.unique()}))
    xa, xb = _mang_seed(a, don_vi, k), _mang_seed(b, don_vi, k)
    if ghep_seed and len(a) != len(b):
        raise ValueError("ghep_seed cần cùng số seed")

    def csi(x):
        return x[..., 0] / np.maximum(x[..., 0] + x[..., 1] + x[..., 2], 1)

    hieu = float(csi(xa.sum((0, 1))) - csi(xb.sum((0, 1))))
    rng = np.random.default_rng(seed)
    n = len(don_vi)
    w_u = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)  # (B, n)
    w_a = np.stack([np.bincount(rng.integers(0, len(a), len(a)), minlength=len(a)) for _ in range(B)]).astype(float)
    w_b = w_a if ghep_seed else np.stack([np.bincount(rng.integers(0, len(b), len(b)), minlength=len(b))
                                          for _ in range(B)]).astype(float)
    ta = sum(w_a[:, s, None] * (w_u @ xa[s]) for s in range(len(a)))  # (B, 3)
    tb = sum(w_b[:, s, None] * (w_u @ xb[s]) for s in range(len(b)))
    d = csi(ta) - csi(tb)
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"hieu": hieu, "thap": float(np.quantile(d, 0.025)), "cao": float(np.quantile(d, 0.975)), "p": p, "B": B,
            "k": k, "so_don_vi": n, "so_seed": [len(a), len(b)]}
