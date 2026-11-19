"""Tải nhãn GFM cho mọi ô của một lưu vực, ghi data/nhan/{tile_id}/{yyyymmdd}.npy (uint8 0/1/255)
và tỉ lệ ngập toàn lưu vực theo ngày vào data/nhan/_ty_le/{luu_vuc}.csv.

Sản phẩm cũ không có lớp loại trừ / nước tham chiếu: nhãn khi đó chỉ bỏ qua nước JRC; mỗi ô gom HỢP mọi lớp loại trừ
đã thấy vào data/nhan/_loai_tru/{tile_id}.npy, Task 9 áp mặt nạ hợp này lên MỌI nhãn (thận trọng).

Cửa sổ: mùa mưa (miền Trung 9-12, Bắc Bộ 6-10) và mùa khô 2-4 các năm 2016-2023 (mùa khô dùng làm mốc phát hiện trận);
2024-2025 chỉ lấy đúng cửa sổ Test A / Test B. Thêm 33 ngày đệm trước mỗi cửa sổ (loai "dem"), chỉ làm nhãn
trạng thái, không dùng tìm trận hay làm mẫu. Gói zip xử lý trong bộ nhớ, không ghi ra ổ.

Chạy: .venv/bin/python scripts/tai_nhan.py --luu_vuc huong
"""
import argparse
import csv
import time
from datetime import date, timedelta

import numpy as np

from floodlab import gfm
from floodlab.config import CUA_SO, DATA_DIR, NHAN_BO_QUA
from floodlab.luoi import _doc, luoi_o

MUA = {"hong_thai_binh": (6, 10)}
MUA_MAC_DINH = (9, 12)
MUA_KHO = (2, 4)
DEM_NGAY = 33


def cua_so(luu_vuc: str) -> list[tuple[date, date, str]]:
    m0, m1 = MUA.get(luu_vuc, MUA_MAC_DINH)
    out = []
    for y in range(2016, 2024):
        out.append((date(y, m0, 1), date(y + (m1 == 12), 1 if m1 == 12 else m1 + 1, 1), "mua_mua"))
        out.append((date(y, MUA_KHO[0], 1), date(y, MUA_KHO[1] + 1, 1), "mua_kho"))
    ten_test = "test_a" if luu_vuc == "hong_thai_binh" else "test_b"
    a, b = CUA_SO[ten_test]
    out.append((date.fromisoformat(a), date.fromisoformat(b), ten_test))
    # Đệm 33 ngày trước mỗi cửa sổ: nhãn trạng thái cho t0 = t - L (L <= 3) ở đầu cửa sổ, lùi tối đa 30 ngày.
    # Đặt sau cùng để ngày trùng cửa sổ chính (vd. cuối tháng 12) đã có trong csv và được bỏ qua.
    return out + [(tu - timedelta(days=DEM_NGAY), tu, "dem") for tu, _, _ in out]


def _ghi_bo(lv, ngay, pid):
    """Sản phẩm máy chủ GFM lỗi 5xx liên tục hoặc zip hỏng: bỏ, ghi lại để biết nhãn ngày đó thiếu phần nào."""
    f = DATA_DIR / "nhan" / "_san_pham_bo" / f"{lv}.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    moi = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if moi:
            w.writerow(["ngay", "product_id", "ly_do"])
        w.writerow([ngay.isoformat(), pid, "gfm_5xx_hoac_zip_hong"])
    print(f"{lv} {ngay} BỎ sản phẩm {pid[:12]} (máy chủ GFM lỗi 5xx hoặc zip hỏng)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luu_vuc", required=True)
    lv = ap.parse_args().luu_vuc
    o = luoi_o(lv)
    bvung = tuple(float(v) for v in o.total_bounds)
    jrc = {t: np.load(DATA_DIR / "tinh" / f"{t}.npz")["nuoc_thuong_xuyen"] for t in o.tile_id}
    aoi = gfm.aoi_id(lv, _doc(f"{lv}_aoi"))
    dir_tl = DATA_DIR / "nhan" / "_ty_le"
    dir_tl.mkdir(parents=True, exist_ok=True)
    f_tl = dir_tl / f"{lv}.csv"
    da_co = {r["ngay"] for r in csv.DictReader(open(f_tl))} if f_tl.exists() else set()
    t0, n_sp = time.time(), 0
    for tu, den, loai in cua_so(lv):
        for ngay, pids in sorted(gfm.ngay_co_anh(aoi, tu, den).items()):
            if ngay.isoformat() in da_co or ngay >= den:
                continue
            theo_o = {t: [] for t in o.tile_id}
            dung, du_lop = 0, False
            for pid in pids:
                lop = gfm.tai_lop_hoac_bo(pid)
                n_sp += 1
                if lop is None:
                    _ghi_bo(lv, ngay, pid)
                    continue
                dung += 1
                du_lop = "loai_tru" in lop
                vung = {k: gfm.chieu_ve_vung(*lop[k], bvung) for k in lop}
                for _, t in o.iterrows():
                    ch = {k: gfm.cat_o(vung[k], bvung, t.geometry) for k in lop}
                    if (ch["ngap"] == NHAN_BO_QUA).all():
                        continue
                    zero = np.zeros_like(ch["ngap"])
                    theo_o[t.tile_id].append(gfm.ma_hoa_nhan(ch["ngap"], ch.get("loai_tru", zero),
                                                             ch.get("tham_chieu", zero), jrc[t.tile_id]))
                    if du_lop:
                        f_lt = DATA_DIR / "nhan" / "_loai_tru" / f"{t.tile_id}.npy"
                        f_lt.parent.mkdir(parents=True, exist_ok=True)
                        cu = np.load(f_lt) if f_lt.exists() else np.zeros_like(ch["ngap"], dtype=bool)
                        np.save(f_lt, cu | (ch["loai_tru"] == 1))
            ngap = hop_le = so_o = 0
            for tid, ds in theo_o.items():
                if not ds:
                    continue
                nhan = gfm.gop_cung_ngay(ds)
                hl = int((nhan != NHAN_BO_QUA).sum())
                if hl == 0:
                    continue
                p = DATA_DIR / "nhan" / tid / f"{ngay:%Y%m%d}.npy"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, nhan)
                ngap += int((nhan == 1).sum())
                hop_le += hl
                so_o += 1
            moi = not f_tl.exists()
            with open(f_tl, "a", newline="") as f:
                w = csv.writer(f)
                if moi:
                    w.writerow(["ngay", "loai", "so_san_pham", "co_loai_tru", "so_o", "pixel_hop_le", "pixel_ngap", "ty_le_ngap"])
                w.writerow([ngay.isoformat(), loai, dung, int(du_lop), so_o, hop_le, ngap,
                            round(ngap / hop_le, 6) if hop_le else ""])
            print(f"{lv} {ngay} {loai} sp={dung}/{len(pids)} o={so_o} ty_le={ngap / max(hop_le, 1):.4f} "
                  f"({n_sp} sp, {time.time() - t0:.0f} s)", flush=True)
            da_co.add(ngay.isoformat())  # cửa sổ đệm chồng cửa sổ chính không tải lại


if __name__ == "__main__":
    main()
