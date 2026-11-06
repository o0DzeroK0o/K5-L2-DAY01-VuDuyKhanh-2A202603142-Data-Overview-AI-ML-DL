# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

****Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Record hạng 1:** `class_id = 468`; `class_name = cab`; `rank = 1`; `score = 0.510915`; `taxonomy_name = ImageNet-1K`. Checkpoint là `yolo11n-cls.pt`.
2. Record mô tả dự đoán cho **cả bức ảnh**, không phải một chiếc xe riêng. Đây là tác vụ `image_classification`; record không có bounding box hay polygon để định vị một đối tượng. Ảnh minh họa có nhiều xe nhưng biểu đồ vẫn thể hiện top-5 lớp cho ảnh.
3. Danh sách lớp đi kèm taxonomy/checkpoint đã huấn luyện, không do model tự đặt tên trong lúc suy luận. Ở đây là ImageNet-1K.
4. `class_id` xác định lớp bằng mã; `class_name` giúp con người đọc; `taxonomy_name` xác định hệ lớp dùng để diễn giải mã và tên. Cùng một mã số có thể mang nghĩa khác trong taxonomy khác, vì vậy không nên chỉ lưu tên lớp hoặc mã số đơn lẻ.
5. Guideline phải quy định chọn lớp theo chủ thể chính hay chủ đề toàn ảnh, tiêu chí ưu tiên khi có nhiều loại vật và cách xử lý khi không có lớp phù hợp.
6. `score` là mức tin cậy của model đối với dự đoán, không phải xác nhận nhãn đúng hoặc điểm chất lượng ground truth. Ảnh có nhiều xe buýt và ô tô, trong khi top-1 là `cab` với `0.510915`, top-2 là `minibus` với `0.164284`; sự chênh lệch đó không chứng minh toàn cảnh phải mang nhãn taxi. Ground truth phải do người gán và kiểm tra theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Record được chọn:** `class_name = person`; `class_id = 0`; `score = 0.912625`; `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`; `bbox_width = 113.58`; `bbox_height = 279.68`; `taxonomy_name = COCO-80`.
2. Gốc tọa độ ở góc trên-trái ảnh; x tăng sang phải, y tăng xuống dưới. Box bắt đầu tại `(385.33, 69.24)` và kết thúc tại `(498.92, 348.92)`, bao người đứng ở phía phải ảnh. Chiều rộng và cao được JSON ghi lần lượt là `113.58` và `279.68` pixel. Trừ trực tiếp hai tọa độ x đã làm tròn cho `113.59`; chênh `0.01` pixel là do các trường được làm tròn độc lập, nên báo cáo giữ nguyên record, không sửa số liệu.
3. **So sánh ngưỡng:** JSON đã xuất tại `0.35` có **11 prediction** cho `kitchen`: 2 `person`, 5 `bowl`, 2 `oven`, 2 `cup`. Khi lọc chính các record đã xuất với điều kiện `score >= 0.60`, còn **6 record**. Đây là phép lọc hậu kỳ trên JSON `0.35`, **không phải** kết quả đã quan sát từ một lần inference riêng tại `0.60`. Chưa có output ô so sánh `0.20 / 0.35 / 0.60` của notebook; không thể khôi phục số prediction tại `0.20` vì các record dưới `0.35` không nằm trong JSON. Cần đối chiếu output ô đó để hoàn tất bằng chứng so sánh ngưỡng theo yêu cầu bài lab.
4. Khi hạ ngưỡng, reviewer phải kiểm tra thêm các hộp độ tin cậy thấp: đối chiếu có vật thật hay không, đúng lớp hay không, có hộp trùng hoặc box lệch không. Giữ nhiều box hơn có thể tăng khả năng tìm lại vật bị bỏ sót, đồng thời tăng false positive và công việc loại/sửa nhãn; không được tự động chấp nhận tất cả box.
5. **Quy tắc box đề xuất:** Với chính sách chỉ gán phần nhìn thấy, vẽ hình chữ nhật trục song song nhỏ nhất bao toàn bộ phần nhìn thấy của từng đối tượng hợp lệ; không suy đoán phần bị che hoặc kéo box vượt biên ảnh. Mỗi đối tượng có box và lớp riêng theo taxonomy. Chính sách này cần được thống nhất trong guideline trước khi áp dụng cho cả tập dữ liệu.
6. Ở mép trái gần góc dưới ảnh có một phần người/cánh tay bị cắt bởi biên ảnh, được model dự đoán `person`; không nhìn thấy đầy đủ cơ thể. Nếu guideline đã quy định ngưỡng phần nhìn thấy tối thiểu và cách gán đối tượng cắt mép thì áp dụng nhất quán. Nếu chưa có quy định, **escalation lên Lab Coach** để chốt có giữ đối tượng này hay không; không tự đoán hình dạng cơ thể ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Record được chọn:** `instance_id = kitchen-001`; `class_name = person`; `class_id = 0`; `score = 0.899318`; `polygon_point_count = 348`. Năm điểm đầu của `polygon_xy` theo thứ tự JSON, đơn vị pixel:

   ```text
   [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
   ```

   Record dùng `taxonomy_name = COCO-80`, checkpoint `yolo11n-seg.pt`, ngưỡng `0.35`.
2. Box chỉ cho hình chữ nhật bao vật, thường chứa cả vùng nền. Polygon mô tả đường biên hình dạng đối tượng chi tiết hơn, cho phép xác định vùng pixel thuộc vật và phân biệt với nền hoặc đối tượng lân cận. Mask người bên phải trong ảnh đi theo đầu, thân, tay và chân thay vì tô kín cả box.
3. `instance_id` phân biệt từng đối tượng trong kết quả, ví dụ `kitchen-001`. Hai người có thể cùng `class_id = 0` nhưng phải có instance khác nhau. Nó không phải class ID vì không biểu diễn loại vật; không phải tracking ID vì không bảo đảm nhận diện cùng một người xuyên qua nhiều khung hình hoặc các lần chạy model.
4. **Quy tắc biên mask đề xuất:** Bám theo biên phần nhìn thấy của từng đối tượng; không tô nền, không nối qua vùng bị vật khác che và không gộp hai đối tượng cùng lớp thành một instance. Với vùng che khuất, áp dụng thống nhất chính sách visible-only và quy định polygon nhiều phần nếu định dạng/guideline hỗ trợ; không tự vẽ phần khuất.
5. Vùng bàn phía dưới có bát, dụng cụ và vật khác tiếp xúc hoặc che mặt bàn; đây là nơi khó quyết định pixel nào thuộc bàn, pixel nào thuộc vật đặt trên bàn. Nếu guideline đã quy định visible-only, loại vùng vật che khỏi mask bàn theo định dạng được phép. Nếu chưa quy định cách biểu diễn lỗ/vùng bị che hoặc biên không đủ rõ, **escalation lên Lab Coach**, không lấy mask màu của model làm chuẩn đúng mặc định.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):


| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn ảnh theo taxonomy và chính sách một lớp của dự án. | `traffic` có nhiều loại xe; top-1 `cab` = `0.510915` nhưng không chứng minh toàn cảnh là taxi. | Chọn lớp theo quy tắc chủ thể/chủ đề và thứ tự ưu tiên; hỏi khi không có lớp phù hợp. | Kiểm tra lớp, taxonomy, tính nhất quán với các ảnh tương tự; không dùng score làm điểm đúng của nhãn người. |
| Phát hiện vật thể | Một box và một class cho từng đối tượng hợp lệ, tọa độ/đơn vị thống nhất. | `kitchen` có phần người bị cắt ở mép trái; JSON tại `0.35` có 11 box, các bát nhỏ và vật che khuất cần kiểm tra. | Gán đúng lớp và box sát phần nhìn thấy theo guideline; không suy diễn phần ngoài ảnh. | Đối chiếu vật thật, box sai/trùng/bỏ sót, lớp và quy tắc cắt mép; yêu cầu sửa hoặc escalation nếu chưa có chính sách. |
| Instance segmentation | Một mask/polygon cùng class và định danh instance cho từng đối tượng. | Bàn bị nhiều vật che; JSON có `kitchen-001` gồm 348 điểm, nhưng nhiều điểm không tự chứng minh biên đúng. | Vẽ biên phần nhìn thấy, tách từng instance và xử lý vùng che theo guideline. | Soi biên, vùng nền bị tô nhầm, phần vật bị thiếu, instance bị gộp/tách sai và tính hợp lệ của polygon. |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai cố định của bài lab, giữ thông tin nguồn/giấy phép, không thêm dữ liệu cá nhân hoặc ảnh ngoài phạm vi vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, dừng xử lý/chia sẻ dữ liệu đó và báo Lab Coach hoặc giảng viên phụ trách để được hướng dẫn trước khi tiếp tục.

Theo `IMAGE_ATTRIBUTION.md`, ảnh `traffic` (COCO 210273) là “Wuhan” của Tauno Tõhk (toehk); `kitchen` (COCO 397133) là “Kitchen” của Maggie Stephens (Pot Noodle); `dining` (COCO 166918) là “Big Wine Thing” của WordRidden. Các ảnh nguồn có giấy phép CC BY 2.0; ảnh minh họa đã được thêm prediction/chart và cần giữ nguyên file attribution khi nộp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS