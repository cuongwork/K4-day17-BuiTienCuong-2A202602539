# Bước 8: Phân tích kết quả benchmark

Chạy `python src/benchmark.py` ở chế độ offline. Kết quả chi tiết và phương pháp đo nằm trong [Analysis.md](Analysis.md).

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ message trong cùng thread. Câu hỏi recall được gửi ở thread mới nên Baseline không còn fact đã cung cấp. Advanced ghi các fact ổn định vào `User.md`, rồi đọc lại hồ sơ khi mở thread mới. Trên cả Standard và Long-Context Stress, recall của Baseline là 0%, Advanced là 100% theo phép so khớp chuỗi của benchmark.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Advanced đưa hồ sơ người dùng vào ngữ cảnh mỗi lượt, kể cả khi hội thoại chưa đủ dài để compact. Ở Standard, không agent nào compact; Advanced xử lý 19.862 prompt token so với 13.911 của Baseline, nhiều hơn 5.951 token (khoảng 42,8%). Đó là chi phí của persistent memory trong bộ dữ liệu này.

## 3. Vì sao compact giúp ở hội thoại dài?

Baseline kéo theo toàn bộ lịch sử của thread. Advanced tóm tắt phần cũ và giữ các message gần nhất khi vượt ngưỡng. Ở stress test 16 lượt, Advanced compact 2 lần và xử lý 12.514 prompt token so với 22.089 của Baseline, ít hơn 9.575 token (khoảng 43,3%). Lợi ích nằm ở prompt token đã xử lý; token phản hồi của Advanced vẫn cao hơn Baseline (291 so với 260).

## 4. Memory tăng trưởng ra sao và có rủi ro gì?

`User.md` tăng 225 byte ở Standard và 168 byte ở Stress; Baseline không tạo hồ sơ. File lớn dần sẽ tăng chi phí prompt. Fact trích sai hoặc fact cũ không được sửa có thể gây nhớ sai qua nhiều phiên. Advanced dùng confidence threshold 0,8 để hạn chế lưu câu giả định; đây vẫn là heuristic và có thể bỏ sót fact thật khi cách diễn đạt mơ hồ. Chỉ số token là ước lượng offline, chưa phải usage thực từ API.
