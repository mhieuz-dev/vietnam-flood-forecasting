"""Kiểm nhãn GFM trên tập test bằng bản đồ độc lập (Task 16 bước 5), ghi results/kiem_nhan_test.json.

Test B: bản đồ khoanh vùng Copernicus EMS (DEL) mà mọi ảnh sau sự kiện là Sentinel-1 chụp cùng một ngày UTC.
Test A (Yagi): cả 11 bản đồ EMSR754 dùng ảnh quang học thương mại nên bỏ (ghi lý do); thay bằng vùng nước lũ UNOSAT
TC20240905VNM từ Sentinel-1 (CC BY-SA, data.humdata.org), ảnh 10/9 10:58 UTC, 12/9 22:58 UTC, 24/9 22:58 UTC (tệp
UNOSAT ghi ngày giờ địa phương 25/9). Chỉ so trên mẫu test có nhãn GFM đúng ngày UTC đó (cùng ảnh S1).
Mẫu test đọc qua danh_gia._mau_vai_tro (gọi chia_tap.doc_test, kiểm băm, ghi nhật ký).
Ngoài IoU GFM so với EMS, chấm thêm các mô hình (ngưỡng chọn trên val) theo EMS và theo GFM trên cùng các pixel:
trong vùng so của bản đồ và GFM có quan sát.
Gói vector EMS tự tải từ rapidmapping.emergency.copernicus.eu về data/ems_test/; UNOSAT tải tay về data/unosat/.
Chạy: .venv/bin/python scripts/kiem_nhan_ems_test.py
"""
import json
import statistics as st
import zipfile
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely

from floodlab import danh_gia, du_doan, ems, mau
from floodlab.config import CRS, DATA_DIR, LUU_VUC, NHAN_BO_QUA, RESULTS_DIR
from floodlab.luoi import luoi_o

KICH_HOAT = ("EMSR754", "EMSR844", "EMSR848")
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code={}"
THU_MUC = DATA_DIR / "ems_test"
MO_HINH = [f"{h}_s{s}" for h in ("hop_nhat_gfs", "unet_b3_gfs", "prithvi_b4_gfs", "utae_b5_gfs") for s in range(3)]
KHONG = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
LO = 500
UNOSAT = DATA_DIR / "unosat" / "3981_TC20240910VNM_SHP.zip"
UNOSAT_BAN_DO = (("ST1_20240910", "BacGiang", date(2024, 9, 10)), ("ST1_20240912", "RedRiverDelta", date(2024, 9, 12)),
                 ("ST1_20240925", "RedRiverDelta", date(2024, 9, 24)))


def _kich_hoat(ma: str) -> dict:
    p = THU_MUC / f"{ma}.json"
    if not p.exists():
        THU_MUC.mkdir(parents=True, exist_ok=True)
        p.write_text(requests.get(API.format(ma), timeout=120).text, encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))["results"][0]


def ban_do_s1() -> tuple[list[dict], list[dict]]:
    """(bản đồ dùng được, bản đồ bỏ kèm lý do) trong các kích hoạt test."""
    dung, bo = [], []
    for ma in KICH_HOAT:
        for a in _kich_hoat(ma)["aois"]:
            for p in a["products"]:
                if p["type"] != "DEL":
                    continue
                anh = [(i["sensorName"], i["acquisitionTime"]) for i in p.get("images", [])]
                ten = f"{ma}_AOI{a['number']:02d}_mon{p['monitoringNumber']}"
                ngay = {t[:10] for _, t in anh}
                if not anh or any("sentinel-1" not in s.lower() for s, _ in anh) or len(ngay) != 1:
                    bo.append({"ban_do": ten, "ly_do": "anh_khong_phai_mot_ngay_s1", "anh": anh})
                elif not p.get("downloadPath"):
                    bo.append({"ban_do": ten, "ly_do": "khong_co_goi_tai", "anh": anh})
                else:
                    dung.append({"kich_hoat": ma, "ban_do": ten, "ngay": date.fromisoformat(ngay.pop()), "anh": anh,
                                 "url": p["downloadPath"]})
    return dung, bo


def _lop(f, ten):
    n = next((x for x in zipfile.ZipFile(f).namelist() if f"_{ten}_" in x and x.endswith(".shp")), None)
    return None if n is None else gpd.read_file(f"zip://{f}!{n}")


def _goi(b: dict):
    f = THU_MUC / b["url"].rsplit("/", 1)[1]
    if not f.exists():
        r = requests.get(b["url"], timeout=900)
        r.raise_for_status()
        f.write_bytes(r.content)
    aoi, vet, ngap = (_lop(f, k) for k in ("areaOfInterestA", "imageFootprintA", "observedEventA"))
    vung = aoi.to_crs(CRS).union_all()
    if vet is not None and not vet.empty:
        vung = vung.intersection(vet.to_crs(CRS).union_all())
    if ngap is not None:
        ngap = ngap[ngap.event_type.astype(str).str.contains("Flood", case=False)]
        if "notation" in ngap:  # bỏ "Flood trace" (vết lũ đã rút): GFM chỉ ghi nước đang có
            ngap = ngap[ngap.notation.astype(str).str.lower() != "flood trace"]
    return vung, ngap


def _cong(a: dict, b: dict):
    for k, v in b.items():
        a[k] += v


def _unosat(tien_to: str, vung_ten: str):
    doc = lambda k: gpd.read_file(f"zip://{UNOSAT}!TC20240910VNM_SHP/{tien_to}_{k}_{vung_ten}.shp").to_crs(CRS)
    return doc("AnalysisExtent").union_all(), doc("FloodExtent")


def kiem(vai_tro: str, ban_do: list[dict], bo: list[dict], fs: dict) -> dict:
    """ban_do: {ban_do, ngay, nap() -> (vùng so, GeoDataFrame ngập)}."""
    d = danh_gia._mau_vai_tro(vai_tro, "kiem_nhan_doc_lap", "gfs")  # đọc qua doc_test
    d["ngay"] = pd.to_datetime(d.ngay_t).dt.date
    o = pd.concat([luoi_o(lv) for lv in LUU_VUC], ignore_index=True).set_index("tile_id")
    viec, hinh = [], {}  # viec: (bản đồ, dòng chỉ mục); hinh: tên bản đồ -> (vùng, ngập) đã chiếu, tách mảnh, có sindex
    for b in ban_do:
        dd = d[d.ngay == b["ngay"]]
        if dd.empty:
            bo.append({"ban_do": b["ban_do"], "ly_do": "khong_co_mau_test_cung_ngay", "ngay": b["ngay"].isoformat()})
            continue
        vung, ngap = b["nap"]()
        phu = b["phu"]() if "phu" in b else None  # (vùng, ngập) của sản phẩm phụ so cùng tham chiếu, vd UNOSAT
        if phu is not None:
            vung = vung.intersection(phu[0])  # mọi so sánh trên cùng pixel: trong cả hai vùng phân tích
        shapely.prepare(vung)
        trong = {t for t in dd.tile_id.unique() if vung.intersects(o.geometry[t])}
        dd = dd[dd.tile_id.isin(trong)]
        if dd.empty:
            bo.append({"ban_do": b["ban_do"], "ly_do": "khong_co_o_test_trong_vung", "ngay": b["ngay"].isoformat()})
            continue
        manh = lambda h: h.to_crs(CRS).explode(index_parts=False).reset_index(drop=True)[["geometry"]]
        hinh[b["ban_do"]] = (manh(gpd.GeoDataFrame(geometry=[vung], crs=CRS)),
                             manh(ngap) if ngap is not None and not ngap.empty else None,
                             manh(phu[1]) if phu is not None else None)
        viec += [(b, r) for r in dd.sort_values(["tile_id", "L"]).itertuples()]

    def mat_na(ten_bd: str, tile_id: str):
        g = o.geometry[tile_id]
        v, n, ph = hinh[ten_bd]
        m_vung = ems.ve_o(v.iloc[v.sindex.query(g, predicate="intersects")], g)
        m_ngap = ems.ve_o(n.iloc[n.sindex.query(g, predicate="intersects")], g) if n is not None else np.zeros_like(m_vung)
        m_phu = ems.ve_o(ph.iloc[ph.sindex.query(g, predicate="intersects")], g) if ph is not None else None
        return m_vung, m_ngap, m_phu
    print(f"{vai_tro}: {len(viec)} mẫu test trong {len({v[0]['ban_do'] for v in viec})} bản đồ", flush=True)

    # chấm theo lô (Test A ~6.000 mẫu không vừa RAM nếu nạp hết); fs: 12 hàm dự đoán nạp sẵn một lần trong main
    nguong = {t: json.loads((danh_gia.DIR_DANH_GIA / f"{t}_val_gfs.json").read_text())["nguong_toi_uu"]["gia_tri"]
              for t in fs}
    theo_bd, da, phu_bd = {}, set(), {}
    dem_mh = {t: {"theo_tham_chieu": dict(KHONG), "theo_gfm": dict(KHONG)} for t in fs}
    g_mo = {}
    for a in range(0, len(viec), LO):
        lo = viec[a:a + LO]
        mau_tho, mn = [], {}
        for b, r in lo:
            if (b["ban_do"], r.tile_id) not in mn:
                mn[(b["ban_do"], r.tile_id)] = mat_na(b["ban_do"], r.tile_id)
            if r.luu_vuc not in g_mo:
                g_mo = {r.luu_vuc: mau.mo_zarr(r.luu_vuc)}
            m = mau.doc_mau(g_mo[r.luu_vuc], r, nguon_du_bao="gfs", nguon_quan_trac="imerg_final")
            m["chi_muc"] = {k: v for k, v in r._asdict().items() if k not in ("Index", "ngay")}
            mau_tho.append(m)
        # GFM so với bản đồ tham chiếu: mỗi cặp (bản đồ, ô) một lần (các hạn L cùng một nhãn)
        for (b, r), m in zip(lo, mau_tho):
            m_vung, m_ngap, m_phu = mn[(b["ban_do"], r.tile_id)]
            if (b["ban_do"], r.tile_id) in da:
                continue
            da.add((b["ban_do"], r.tile_id))
            x = theo_bd.setdefault(b["ban_do"], {"ngay": b["ngay"].isoformat(), "so_o": 0, **KHONG})
            _cong(x, ems.dem_khop(m["nhan"], m_ngap, m_vung) | {"so_o": 1})
            if m_phu is not None:  # sản phẩm phụ trên đúng các pixel GFM có quan sát
                g_phu = np.where(m["nhan"] == NHAN_BO_QUA, NHAN_BO_QUA, m_phu.astype(np.uint8))
                xp = phu_bd.setdefault(b["ban_do"], dict(KHONG))
                _cong(xp, ems.dem_khop(g_phu, m_ngap, m_vung))
        # mô hình theo tham chiếu và theo GFM, cùng pixel (vùng so, GFM có quan sát), mọi hạn L
        for ten, f in fs.items():
            for (b, r), m in zip(lo, mau_tho):
                m_vung, m_ngap, _ = mn[(b["ban_do"], r.tile_id)]
                y = m["nhan"]
                du = np.where(y == NHAN_BO_QUA, NHAN_BO_QUA, (f(m) >= nguong[ten]).astype(np.uint8))
                _cong(dem_mh[ten]["theo_tham_chieu"], ems.dem_khop(du, m_ngap, m_vung))
                _cong(dem_mh[ten]["theo_gfm"], ems.dem_khop(du, y == 1, m_vung))
        print(f"{vai_tro}: xong {a + len(lo)}/{len(viec)} mẫu", flush=True)
    tong = dict(KHONG)
    for x in theo_bd.values():
        _cong(tong, {k: x[k] for k in KHONG})
        x.update(ems.chi_so(x))
    c = ems.chi_so(tong)
    print(f"{vai_tro} GFM so với tham chiếu: IoU={c['iou']:.3f} bỏ sót={c['bo_sot']:.3f} báo nhầm={c['bao_nham']:.3f}",
          flush=True)
    diem = {}
    for ten, v in dem_mh.items():
        diem[ten] = {"nguong_val": nguong[ten], **{k: {**x, **ems.chi_so(x)} for k, x in v.items()}}
        print(f"{vai_tro} {ten}: CSI theo tham chiếu {diem[ten]['theo_tham_chieu']['iou']:.3f}, "
              f"theo GFM {diem[ten]['theo_gfm']['iou']:.3f}", flush=True)
    nhom = {}
    for ten, v in diem.items():
        h = nhom.setdefault(ten.rsplit("_s", 1)[0], {"theo_tham_chieu": [], "theo_gfm": []})
        for k in h:
            h[k].append(v[k]["iou"])
    tom = {h: {k: {"tb": st.mean(x), "dlc": st.stdev(x)} for k, x in v.items()} for h, v in nhom.items()}
    for h, v in tom.items():
        print(f"{vai_tro} {h}: theo tham chiếu {v['theo_tham_chieu']['tb']:.3f}±{v['theo_tham_chieu']['dlc']:.3f}, "
              f"theo GFM {v['theo_gfm']['tb']:.3f}±{v['theo_gfm']['dlc']:.3f}", flush=True)
    phu = None
    if phu_bd:
        tp = dict(KHONG)
        for x in phu_bd.values():
            _cong(tp, x)
        phu = {"tong": {**tp, **ems.chi_so(tp)}, "theo_ban_do": {k: {**x, **ems.chi_so(x)} for k, x in phu_bd.items()}}
        print(f"{vai_tro} sản phẩm phụ so với tham chiếu: IoU={phu['tong']['iou']:.3f} bỏ sót={phu['tong']['bo_sot']:.3f}"
              f" báo nhầm={phu['tong']['bao_nham']:.3f}", flush=True)
    return {"gfm_so_tham_chieu": {"tong": {**tong, **c, "so_ban_do": len(theo_bd)}, "theo_ban_do": theo_bd},
            "phu_so_tham_chieu": phu,
            "mo_hinh": diem, "mo_hinh_tb_3_seed": tom, "so_mau_mo_hinh": len(viec),
            "ban_do": [{"ban_do": b["ban_do"], "ngay": b["ngay"].isoformat(), **{k: b[k] for k in ("anh", "url") if k in b}}
                       for b in ban_do]}


def ban_do_quang_hoc() -> list[dict]:
    """EMSR754 (Yagi): bản đồ khoanh vùng từ ảnh quang học thương mại, lệch 1-2 ngày so với ngày test gần nhất
    (13, 14/9 -> 12/9; 20/9 -> 22/9). Kiểm tra độc lập cảm biến; so cùng pixel với UNOSAT 12/9 khi có."""
    ra = []
    for a in _kich_hoat("EMSR754")["aois"]:
        for p in a["products"]:
            if p["type"] != "DEL" or not p.get("downloadPath"):
                continue
            anh = [(i["sensorName"], i["acquisitionTime"]) for i in p["images"]]
            n_ems = date.fromisoformat(anh[0][1][:10])
            n_test = date(2024, 9, 12) if n_ems.day in (13, 14) else date(2024, 9, 22)
            b = {"kich_hoat": "EMSR754", "ban_do": f"EMSR754_AOI{a['number']:02d}_{anh[0][0]}_{n_ems:%d%m}",
                 "ngay": n_test, "anh": anh, "url": p["downloadPath"]}
            b["nap"] = (lambda b=b: _goi(b))
            if n_test == date(2024, 9, 12):
                b["phu"] = lambda: _unosat("ST1_20240912", "RedRiverDelta")
            ra.append(b)
    return ra


def main():
    import sys
    if "quang_hoc" in sys.argv[1:]:  # chỉ chạy kiểm tra quang học, ghi thêm khóa vào kiem_nhan_test.json đã có
        f = RESULTS_DIR / "kiem_nhan_test.json"
        ra = json.loads(f.read_text())
        bo = []
        nhom = next((x.split("=", 1)[1].split(",") for x in sys.argv[1:] if x.startswith("nhom=")), None)
        if "khong_mo_hinh" in sys.argv[1:]:
            fs = {}
        else:  # nhom=<tiền tố,...>: chỉ nạp một nhóm mô hình (VRAM không đủ cho cả 12 khi GPU đang dùng chung)
            fs = {t: du_doan.ham_du_doan(t) for t in MO_HINH if nhom is None or any(t.startswith(n) for n in nhom)}
        kq = kiem("test_a", ban_do_quang_hoc(), bo, fs)
        cu = ra.get("test_a_quang_hoc")
        if cu is not None and fs:  # gộp điểm mô hình vào kết quả đã có (giữ phần so nhãn), tính lại trung bình 3 seed
            cu["mo_hinh"] = {**cu.get("mo_hinh", {}), **kq["mo_hinh"]}
            nh = {}
            for ten, v in cu["mo_hinh"].items():
                h = nh.setdefault(ten.rsplit("_s", 1)[0], {"theo_tham_chieu": [], "theo_gfm": []})
                for k in h:
                    h[k].append(v[k]["iou"])
            cu["mo_hinh_tb_3_seed"] = {h: {k: {"tb": st.mean(x), "dlc": st.stdev(x)} for k, x in v.items()}
                                       for h, v in nh.items() if len(v["theo_gfm"]) == 3}
        else:
            ra["test_a_quang_hoc"] = kq
            ra["test_a_quang_hoc"]["bo"] = bo
        f.write_text(json.dumps(ra, indent=1, ensure_ascii=False, default=str))
        return
    # 12 mô hình nạp một lần (~5 GB VRAM) dùng cho cả hai tập: nạp/xóa lặp lại làm rò bộ nhớ vì B4 bị một frame giữ
    fs = {t: du_doan.ham_du_doan(t) for t in MO_HINH}
    ems_dung, bo = ban_do_s1()
    for b in ems_dung:
        b["nap"] = (lambda b=b: _goi(b))
    ra = {"test_b_ems": kiem("test_b", [b for b in ems_dung if b["ngay"].year == 2025], bo, fs)}
    un = [{"ban_do": f"UNOSAT_{t}_{v}", "ngay": n, "nap": (lambda t=t, v=v: _unosat(t, v))} for t, v, n in UNOSAT_BAN_DO]
    ra["test_a_unosat"] = kiem("test_a", un, bo, fs)
    ra["bo"] = bo
    (RESULTS_DIR / "kiem_nhan_test.json").write_text(json.dumps(ra, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
