"""Chỉ số trên lớp ngập (bỏ pixel nhãn 255) và thống kê bootstrap theo cụm ô trong từng trận."""
import numpy as np
import pandas as pd

from floodlab.config import NHAN_BO_QUA


def _hop_le(xac_suat, nhan, mat_na=None):
    xs, y = np.ravel(xac_suat), np.ravel(nhan)
    m = y != NHAN_BO_QUA
    if mat_na is not None:
        m &= np.ravel(mat_na)
    return xs[m].astype(float), y[m].astype(int)


def bang_nham_lan(xac_suat, nhan, nguong=0.5, mat_na=None) -> dict[str, int]:
    xs, y = _hop_le(xac_suat, nhan, mat_na)
    du = xs >= nguong
    return {"tp": int((du & (y == 1)).sum()), "fp": int((du & (y == 0)).sum()),
            "fn": int((~du & (y == 1)).sum()), "tn": int((~du & (y == 0)).sum())}


def _chia(a, b):
    return a / b if b else float("nan")


def tu_bang(tp, fp, fn, tn) -> dict[str, float]:
    return {"csi": _chia(tp, tp + fp + fn), "iou": _chia(tp, tp + fp + fn), "f1": _chia(2 * tp, 2 * tp + fp + fn),
            "precision": _chia(tp, tp + fp), "recall": _chia(tp, tp + fn)}


def _pr_auc(xs, y):
    if y.sum() == 0:
        return float("nan")
    thu_tu = np.argsort(-xs, kind="mergesort")
    y = y[thu_tu]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())  # average precision


def _ece(xs, y, so_khoang=15):
    if len(y) == 0:
        return float("nan")
    k = np.minimum((xs * so_khoang).astype(int), so_khoang - 1)
    tong = 0.0
    for b in range(so_khoang):
        m = k == b
        if m.any():
            tong += m.sum() * abs(xs[m].mean() - y[m].mean())
    return float(tong / len(y))


def tinh(xac_suat, nhan, nguong=0.5, mat_na=None) -> dict[str, float]:
    xs, y = _hop_le(xac_suat, nhan, mat_na)
    b = bang_nham_lan(xs, y, nguong)
    out = tu_bang(**b)
    out.update(pr_auc=_pr_auc(xs, y), brier=float(np.mean((xs - y) ** 2)) if len(y) else float("nan"),
               ece_15=_ece(xs, y), so_pixel=int(len(y)), so_ngap=int(y.sum()))
    return out


def _lay_mau_cum(ds: pd.DataFrame, rng) -> pd.DataFrame:
    """Lấy mẫu lại có hoàn lại các tile_id trong từng trận (cụm = ô ảnh)."""
    phan = []
    for _, g in ds.groupby("tran", sort=True):
        o = g.tile_id.unique()
        chon = rng.choice(o, size=len(o), replace=True)
        phan += [g[g.tile_id == t] for t in chon]
    return pd.concat(phan)


def bootstrap_cum(ds: pd.DataFrame, ham, B=10_000, seed=0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = [ham(_lay_mau_cum(ds, rng)) for _ in range(B)]
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(ham(ds)), float(lo), float(hi)


def bootstrap_ty_so(ds: pd.DataFrame, cot_tu: list[str], cot_mau: list[str], B=10_000, seed=0
                    ) -> tuple[float, float, float]:
    """Như bootstrap_cum cho chỉ số sum(cot_tu) / sum(cot_mau) (CSI, recall...), nhưng vector hóa: gộp theo
    (tran, tile_id), mỗi vòng rút lại các ô trong từng trận thành ma trận trọng số. Trên val 9.807 mẫu, bản lọc
    DataFrame từng vòng mất 27 phút cho 3 khoảng tin cậy với B = 1000."""
    g = ds.groupby(["tran", "tile_id"])[list(dict.fromkeys(cot_tu + cot_mau))].sum().reset_index()
    tu, mau = g[cot_tu].sum(axis=1).to_numpy(float), g[cot_mau].sum(axis=1).to_numpy(float)
    rng = np.random.default_rng(seed)
    tong_tu, tong_mau = np.zeros(B), np.zeros(B)
    for _, chi_so_nhom in g.groupby("tran", sort=True).indices.items():
        n = len(chi_so_nhom)
        w = np.zeros((B, n))
        np.add.at(w, (np.arange(B)[:, None], rng.integers(0, n, (B, n))), 1)
        tong_tu += w @ tu[chi_so_nhom]
        tong_mau += w @ mau[chi_so_nhom]
    v = np.divide(tong_tu, tong_mau, out=np.full(B, np.nan), where=tong_mau > 0)
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(tu.sum() / mau.sum()) if mau.sum() else float("nan"), float(lo), float(hi)


def bootstrap_ghep_cap(ds_a, ds_b, ham, B=10_000, seed=0) -> tuple[float, float, float, float]:
    """Hiệu ham(a) - ham(b) trên cùng các ô (lấy mẫu lại chung chỉ số ô); trả hiệu, 2 cận 95%, p hai phía."""
    rng = np.random.default_rng(seed)
    khoa = ["tran", "tile_id"]
    a, b = ds_a.set_index(khoa).sort_index(), ds_b.set_index(khoa).sort_index()
    nhom = a.reset_index()[khoa].drop_duplicates()
    hieu0 = float(ham(ds_a) - ham(ds_b))
    v = []
    for _ in range(B):
        chon = []
        for tran, g in nhom.groupby("tran", sort=True):
            chon += [(tran, t) for t in rng.choice(g.tile_id.values, size=len(g), replace=True)]
        idx = pd.MultiIndex.from_tuples(chon, names=khoa)
        v.append(ham(a.loc[idx].reset_index()) - ham(b.loc[idx].reset_index()))
    v = np.asarray(v, float)
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))) if hieu0 != 0 or np.any(v != 0) else 1.0
    return hieu0, float(lo), float(hi), float(p)


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    thu_tu = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(thu_tu):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()
