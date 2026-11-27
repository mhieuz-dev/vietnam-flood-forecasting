"""Hình bản đồ và độ tin cậy của bài (Task 17), ghi bai_bao/hinh/*.pdf.

- hinh_vung_nghien_cuu: lưu vực thượng nguồn, vùng ô mẫu, vai trò (train/val/test) trên nền ranh giới Natural Earth 1:50m
  (data/ne/ne_50m_admin_0_countries.zip, miền công cộng).
- hinh_yagi: ghép ô Test A ngày 12/09/2024, L = 1: nhãn GFM, vùng nước UNOSAT (ảnh S1 cùng ngày), xác suất TerraMind
  seed 0 (đã lưu lúc chấm, results/raw/du_doan/*.zarr) và U-TAE seed 0 (tính lại). Đọc test qua danh_gia._mau_vai_tro.
- hinh_do_tin_cay: biểu đồ độ tin cậy trên Test A và B cho các lượt có lưu xác suất (TerraMind s0, B2, B0).
Chạy: .venv/bin/python scripts/hinh_ban_do.py
"""
import json

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Font hình khớp chữ Times của bài (Liberation Serif cùng kích thước Times), nhúng TrueType thay vì Type 3 (IEEE PDF eXpress)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from floodlab import danh_gia, du_doan, ems, mau
from floodlab.config import CRS, DATA_DIR, LUU_VUC, NHAN_BO_QUA, RESULTS_DIR, ROOT, TILE_PX
from floodlab.luoi import _doc, luoi_o

HINH = ROOT / "bai_bao" / "hinh"
DU_DOAN = RESULTS_DIR / "raw" / "du_doan"
TEN = {"vu_gia_thu_bon": "Vu Gia-Thu Bon", "huong": "Huong", "tra_khuc": "Tra Khuc", "gianh": "Gianh", "ca": "Ca",
       "hong_thai_binh": "Red-Thai Binh"}
VAI_TRO = {"hong_thai_binh": "train + Test A", "ca": "train + Test B", "gianh": "val + Test B",
           "huong": "train + Test B", "tra_khuc": "train + Test B", "vu_gia_thu_bon": "train + Test B"}


# dời nhãn theo vĩ độ để ba lưu vực phía nam (cách nhau < 1°) không đè chữ lên nhau
LECH_NHAN = {"hong_thai_binh": 0.35, "huong": 0.45, "vu_gia_thu_bon": -0.05, "tra_khuc": -0.55}


def hinh_vung_nghien_cuu():
    nen = gpd.read_file(f"zip://{DATA_DIR / 'ne' / 'ne_50m_admin_0_countries.zip'}").cx[98:114, 8:27]
    # khung 3,5 in + cột nhãn bên phải, in ở 0,85 khổ chữ nên chữ 7 pt in ra khoảng 7 pt; khung dừng ở 110,8°E, không với tới quần đảo Hoàng Sa
    fig, ax = plt.subplots(figsize=(3.5, 4.6))
    nen.plot(ax=ax, color="#f2f2f2", edgecolor="#9a9a9a", lw=0.5)
    nen[nen.ADMIN == "Vietnam"].plot(ax=ax, color="#e3e3e3", edgecolor="#666666", lw=0.7)
    mau_lv = plt.get_cmap("tab10")
    for i, lv in enumerate(LUU_VUC):
        gpd.GeoSeries([_doc(f"{lv}_thuong_nguon")], crs=4326).plot(ax=ax, color=mau_lv(i), alpha=0.25, lw=0)
        o = luoi_o(lv).to_crs(4326)
        gpd.GeoSeries([o.union_all()], crs=4326).plot(ax=ax, color=mau_lv(i), alpha=0.9, lw=0)
        c = o.union_all().centroid
        # nhãn đặt ngoài khung bên phải (trục x theo phần trăm khung, trục y theo vĩ độ): trong khung thì bị
        # đường viền khung và bờ biển Hải Nam cắt ngang chữ
        ax.annotate(f"{TEN[lv]}\n({VAI_TRO[lv]})", (c.x, c.y), xytext=(1.04, c.y + LECH_NHAN.get(lv, 0)),
                    textcoords=("axes fraction", "data"), fontsize=7, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#444444"})
    ax.set_xlim(100.5, 110.8)
    ax.set_ylim(13.5, 25.6)
    x0, y0, km = 108.6, 24.9, 100  # thước tỉ lệ 100 km, góc trên phải cho khỏi đè nhãn
    dx = km / (111.32 * np.cos(np.radians(y0)))  # độ kinh tuyến ứng với 100 km tại vĩ độ đặt thước
    ax.plot([x0, x0 + dx], [y0, y0], color="#222222", lw=1.5, solid_capstyle="butt")
    ax.text(x0 + dx / 2, y0 + 0.15, f"{km} km", ha="center", va="bottom", fontsize=7)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.legend(handles=[Patch(color="#888888", alpha=0.25, label="Upstream catchment"),
                       Patch(color="#888888", alpha=0.9, label="Sample tiles (4.48 km)")], fontsize=7, frameon=False,
              loc="lower left")
    fig.savefig(HINH / "hinh_vung_nghien_cuu.pdf", bbox_inches="tight")  # tight để giữ nhãn ngoài khung
    plt.close(fig)


def _manh(ten: str) -> gpd.GeoDataFrame:
    """Một lớp UNOSAT TC20240905VNM (ảnh S1 12/09/2024 22:58 UTC), chiếu sang CRS ô, tách mảnh để tra sindex."""
    f = DATA_DIR / "unosat" / "3981_TC20240910VNM_SHP.zip"
    g = gpd.read_file(f"zip://{f}!TC20240910VNM_SHP/{ten}.shp").to_crs(CRS)
    return g.explode(index_parts=False).reset_index(drop=True)[["geometry"]]


def _xac_suat_da_luu(ten: str, vt: str, d: pd.DataFrame, chon: np.ndarray) -> np.ndarray:
    """Xác suất đã lưu (uint8, p * 254) theo đúng thứ tự dòng của danh_gia._mau_vai_tro."""
    z = zarr.open_group(DU_DOAN / f"{ten}_{vt}_gfs.zarr", mode="r")["xac_suat"]
    assert z.shape[0] == len(d), (z.shape, len(d))
    return np.stack([np.asarray(z[i], np.float32) / 254 for i in chon])


def hinh_yagi(ngay="2024-09-12", L=1, thu=2, cua=10):
    d = danh_gia._mau_vai_tro("test_a", "hinh_yagi", "gfs")  # đọc qua doc_test
    chon = np.flatnonzero((pd.to_datetime(d.ngay_t).dt.strftime("%Y-%m-%d") == ngay).to_numpy() & (d.L == L).to_numpy())
    rows = d.iloc[chon]
    o = luoi_o("hong_thai_binh").set_index("tile_id")
    p_tm = _xac_suat_da_luu("hop_nhat_gfs_s0", "test_a", d, chon)
    f_b5 = du_doan.ham_du_doan("utae_b5_gfs_s0")
    z = mau.mo_zarr("hong_thai_binh")
    vung, ngap = (_manh(f"ST1_20240912_{k}_RedRiverDelta") for k in ("AnalysisExtent", "FloodExtent"))
    x0 = min(o.geometry[t].bounds[0] for t in rows.tile_id)
    y1 = max(o.geometry[t].bounds[3] for t in rows.tile_id)
    x1 = max(o.geometry[t].bounds[2] for t in rows.tile_id)
    y0 = min(o.geometry[t].bounds[1] for t in rows.tile_id)
    k = TILE_PX // thu
    W, H = int(round((x1 - x0) / 4480)) * k, int(round((y1 - y0) / 4480)) * k
    lop = {t: np.full((H, W), np.nan, np.float32) for t in ("gfm", "unosat", "tm", "b5")}
    for i, r in enumerate(rows.itertuples()):
        m = mau.doc_mau(z, r)
        m["chi_muc"] = {kk: v for kk, v in r._asdict().items() if kk != "Index"}
        g = o.geometry[r.tile_id]
        cot, hang = int(round((g.bounds[0] - x0) / 4480)) * k, int(round((y1 - g.bounds[3]) / 4480)) * k
        y = m["nhan"].astype(np.float32)
        y[m["nhan"] == NHAN_BO_QUA] = np.nan
        m_vung = ems.ve_o(vung.iloc[vung.sindex.query(g, predicate="intersects")], g)
        un = np.where(m_vung, ems.ve_o(ngap.iloc[ngap.sindex.query(g, predicate="intersects")], g), np.nan)
        un = un.astype(np.float32)
        for t, a in (("gfm", y), ("unosat", un), ("tm", p_tm[i]), ("b5", f_b5(m))):
            with np.errstate(all="ignore"):  # khối toàn NaN (ngoài vùng so) giữ NaN
                lop[t][hang:hang + k, cot:cot + k] = np.nanmean(a.reshape(k, thu, k, thu), (1, 3))
    # phóng vào khung cua x cua ô có nhiều nước UNOSAT nhất (nhìn được chi tiết thay vì cả đồng bằng)
    cua *= k
    un = np.nan_to_num(lop["unosat"])
    tot = max(((un[i:i + cua, j:j + cua].sum(), i, j) for i in range(0, max(H - cua, 0) + 1, k)
               for j in range(0, max(W - cua, 0) + 1, k)))
    lop = {t: a[tot[1]:tot[1] + cua, tot[2]:tot[2] + cua] for t, a in lop.items()}
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.6))
    tieu_de = {"gfm": "(a) GFM label", "unosat": "(b) UNOSAT (S1)", "tm": "(c) TerraMind",
               "b5": "(d) U-TAE"}
    for a, (t, tt) in zip(ax, tieu_de.items()):
        im = a.imshow(lop[t], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        a.set_facecolor("#d9d9d9")
        a.set_title(tt, fontsize=8)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Flooded fraction / probability")
    fig.savefig(HINH / "hinh_yagi_12_09_2024.pdf", bbox_inches="tight")
    plt.close(fig)
    return len(rows)


LUOT_TIN_CAY = {"TerraMind (GFS), seed 0": ("hop_nhat_gfs_s0", "#1f77b4"), "B2 XGBoost": ("b2", "#ff7f0e"),
                "B0 Flood frequency": ("b0", "#2ca02c")}


def tinh_do_tin_cay(so_khoang=10):
    """Cộng theo khoảng xác suất (tổng p, tổng nhãn ngập, số pixel) cho các lượt có lưu xác suất, ghi
    results/do_tin_cay.json để vẽ lại không phải đọc lại nhãn test."""
    bien = np.linspace(0, 1, so_khoang + 1)
    ra = {"bien": bien.tolist()}
    for vt in ("test_a", "test_b"):
        d = danh_gia._mau_vai_tro(vt, "hinh_do_tin_cay", "gfs")  # đọc qua doc_test
        cong = {n: np.zeros((3, so_khoang)) for n in LUOT_TIN_CAY}
        z_p = {n: zarr.open_group(DU_DOAN / f"{t}_{vt}_gfs.zarr", mode="r")["xac_suat"]
               for n, (t, _) in LUOT_TIN_CAY.items()}
        g_mo = {}
        for i, r in enumerate(d.itertuples()):
            if r.luu_vuc not in g_mo:
                g_mo = {r.luu_vuc: mau.mo_zarr(r.luu_vuc)}
            y = np.asarray(g_mo[r.luu_vuc]["nhan"][r.i_nhan])
            hl = y != NHAN_BO_QUA
            for n in LUOT_TIN_CAY:
                p = np.asarray(z_p[n][i], np.float32)[hl] / 254
                b = np.clip(np.digitize(p, bien) - 1, 0, so_khoang - 1)
                cong[n] += [np.bincount(b, p, so_khoang), np.bincount(b, (y[hl] == 1).astype(float), so_khoang),
                            np.bincount(b, minlength=so_khoang)]
        ra[vt] = {n: {"tong_p": c[0].tolist(), "tong_ngap": c[1].tolist(), "so_pixel": c[2].tolist()}
                  for n, c in cong.items()}
    (RESULTS_DIR / "do_tin_cay.json").write_text(json.dumps(ra, indent=1), encoding="utf-8")


def hinh_do_tin_cay(toi_thieu=1000):
    """Vẽ từ results/do_tin_cay.json; bỏ khoảng có ít hơn toi_thieu pixel."""
    ra = json.loads((RESULTS_DIR / "do_tin_cay.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.9), sharey=True)  # in ở toàn khổ chữ (5,4 in)
    for a, vt in zip(ax, ("test_a", "test_b")):
        a.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls="--", label="_nolegend_")
        for n, (_, mau_ve) in LUOT_TIN_CAY.items():
            s, sy, c = (np.array(ra[vt][n][k]) for k in ("tong_p", "tong_ngap", "so_pixel"))
            ok = c >= toi_thieu
            a.plot(s[ok] / c[ok], sy[ok] / c[ok], marker="o", ms=3, color=mau_ve, label=n,
                   lw=1.6 if "TerraMind" in n else 1.0)
        a.set_title({"test_a": "Test A", "test_b": "Test B"}[vt])
        a.set_xlabel("Forecast probability")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Observed flood frequency (GFM)")
    # chú thích đặt dưới hai khung: trong khung nó đè lên đường B0 ở Test A
    tay, nhan = ax[0].get_legend_handles_labels()
    fig.legend(tay, nhan, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(HINH / "hinh_do_tin_cay.pdf")
    plt.close(fig)


def main():
    HINH.mkdir(parents=True, exist_ok=True)
    hinh_vung_nghien_cuu()
    print("vùng nghiên cứu xong", flush=True)
    print(f"Yagi: {hinh_yagi()} ô", flush=True)
    tinh_do_tin_cay()
    hinh_do_tin_cay()
    print("độ tin cậy xong", flush=True)


if __name__ == "__main__":
    main()
