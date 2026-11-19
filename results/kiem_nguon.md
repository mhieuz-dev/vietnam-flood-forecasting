# Dò nguồn dữ liệu (Task 3), phần không cần tài khoản

Ngày chạy: 28/10/2026, bằng curl từ máy laptop.

| Nguồn | Kết quả | Hệ quả |
|---|---|---|
| GFS 0,25 độ trên AWS `noaa-gfs-bdp-pds` | Có file `.idx`. Đường dẫn `gfs.YYYYMMDD/00/atmos/...` dùng được cho 2022-09-15 và 2024-09-08. Đường dẫn không có `atmos/` dùng được cho 2021-01-15, nhưng 2021-06-01 trả 404 ở dạng này. Năm 2016, 2019, 2020 trả 404 | Trường APCP "0-1 day acc fcst" nặng khoảng 370 KB mỗi lượt đọc theo byte. Từ 2021 trở đi đọc theo byte được |
| GFS trên NCAR GDEX `d084001` | Có file 2016-10-15 f024 (238.404.643 byte), hỗ trợ đọc theo byte, nhưng KHÔNG có `.idx` (404) | 2016 đến 2020 phải tự quét đầu các bản tin GRIB, hoặc dùng nguồn khác. Chưa chốt |
| ECMWF open data trên AWS | 503 cả hai lần thử (endpoint chung và endpoint eu-central-1) | Thử lại bằng thư viện `ecmwf-opendata` ở bước có tài khoản |

Còn lại (cần tài khoản của user, mục U1 và U2 trong plan): GEE `computePixels`, GFM API, IMERG Early trên GEE, HydroBASINS.

## Phần cần tài khoản Earth Engine (28/10/2026, project vn-floodcast, billingEnabled: false)

| Nguồn | Kết quả | Hệ quả |
|---|---|---|
| GEE `computePixels`, S1 GRD | 1 ảnh IW tại Huế ngày 2020-10-13. Ô 224 x 224 px, 20 m, EPSG:32648, hai kênh VV và VH, tải mất 1,62 giây. VV có min -31,8, trung vị -6,8, max 26,3 dB | Tải thẳng về máy, không cần Drive. Ước lượng hạn mức GEE làm ở Task 4, khi đã biết số ô |
| IMERG V07 trên GEE (`NASA/GPM_L3/IMERG_V07`) | Có tới 2026-10-26. Thuộc tính `status`: 2020-10-13 và 2024-09-08 là `permanent`; 2025-10-20 và 2026-10-25 là `provisional` | Test A (Yagi 09/2024) dùng bản Final; Test B (10-11/2025) chỉ có bản provisional. Khớp với việc NASA chuyển sang V08, không còn bản V07 permanent sau 30/09/2025. Phải ghi rõ trong bài; trước khi dùng, kiểm cột `status` của từng ngày test |
| IMERG Early, Late riêng trên GEE | `IMERG_V07_EARLY` và `IMERG_V07_LATE` không tồn tại | Thí nghiệm IMERG Early (RQ3) phải tải từ GES DISC, tức cần tài khoản NASA Earthdata (U3) |
| HydroBASINS cấp 12 (`WWF/HydroSHEDS/v1/Basins/hybas_12`) | Có, kèm `NEXT_DOWN`, `UP_AREA` | Dựng lưu vực thượng nguồn được (Task 4). Giấy phép HydroSHEDS cần đọc trước khi công bố dữ liệu dẫn xuất |
| SoilGrids (`projects/soilgrids-isric/clay_mean`) | Có đủ 6 tầng sâu từ 0-5 đến 100-200 cm | Dùng cho Task 5 |
| ERA5-Land daily | Có tới ít nhất 2025-11-01 | Dùng cho Task 8 |
| GLO-30 | `COPERNICUS/DEM/GLO30` báo đã cũ, thay bằng `GLO30_2024_1`, đúng như config | Dùng `GLO30_2024_1` |

## GFM và GFS 2016-2020 (28/10/2026)

| Nguồn | Kết quả | Hệ quả |
|---|---|---|
| GFM API v2 (`api.gfm.eodc.eu/v2`, có swagger) | Đăng nhập trả Bearer token sống 18.000 giây. Tạo AOI thử ở Huế (aoi_id `ee29bbcc...`). AOI Huế có 11 sản phẩm từ 01/10 đến 07/11/2020, trùng lịch chụp S1 đã đếm; năm 2016 có 7 sản phẩm từ tháng 10 đến 12 | Nhãn có từ 2016, đủ cho cửa sổ huấn luyện |
| Tải sản phẩm GFM | Trả link owncloud tới file zip 20.833.131 byte, chứa NGUYÊN ô Equi7 (10.623 x 12.774 px, EPSG:4326, khoảng 20 m), không cắt theo AOI. Có các lớp FLOOD, OBSWATER, EXCLAYER (0/1/255), UNCERTAINTY, REFERENCE_WATER, ADVFLAG, kèm ảnh dân số và lớp phủ | Tải, cắt theo ô, rồi xóa ngay để không đầy ổ |
| GFS 2016 qua NCAR THREDDS NetcdfSubset | `accept=netcdf3` trả 18.760 byte cho `Total_precipitation_surface_6_Hour_Accumulation` trong khung 97 đến 110 E, 8 đến 26 N (lưới 73 x 53, kg m-2, max 64,6). Lần đầu mất 16 giây. `netcdf4` không được hỗ trợ | 2016 đến 2020 dùng NCSS: tổng mưa ngày là 4 file 6 giờ. Đọc bằng engine scipy, vì xarray với netCDF4 và rasterio trong venv gây lỗi double free khi thoát |
