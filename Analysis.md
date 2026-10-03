# Phân tích kết quả Day 17: Memory Systems for AI Agent

## Cách tái hiện

Chạy từ root repo với Python đã cài các dependency trong `README.md`:

```bash
python -m pytest -q
python src/benchmark.py
python src/eval_bonus.py
```

Benchmark chạy offline tất định, dùng cùng dữ liệu và cấu hình compact cho Baseline và Advanced. Mỗi lần chạy tạo state tạm riêng; câu hỏi recall được hỏi ở thread mới. `Agent tokens only` cộng token phản hồi, còn `Prompt tokens processed` cộng lượng ngữ cảnh ước lượng ở từng lượt, kể cả lượt recall. Đây là estimator, không phải số token tính phí do API trả về.

`Cross-session recall` dựa trên các chuỗi `expected_contains`: 0 nếu không khớp, 0,5 nếu khớp một phần, 1 nếu khớp hết. `Response quality` chấm độc lập với đáp án recall: 50% trả lời trực tiếp, 25% ngắn gọn, 25% có cấu trúc. Một câu trả lời sai fact vẫn có thể đạt quality cao; vì vậy cần đọc chỉ số này cùng recall.

## Kết quả benchmark

| Bộ dữ liệu | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | Baseline | 1.386 | 13.911 | 0% | 0,25 | 0 | 0 |
| Standard | Advanced | 1.429 | 19.862 | 100% | 1,00 | 225 | 0 |
| Long-Context Stress | Baseline | 260 | 22.089 | 0% | 0,25 | 0 | 0 |
| Long-Context Stress | Advanced | 291 | 12.514 | 100% | 1,00 | 168 | 2 |

**Recall qua phiên.** Baseline chỉ giữ lịch sử trong cùng thread, nên câu hỏi ở thread mới không có fact cũ. Advanced ghi fact ổn định vào `User.md` và đọc lại ở thread mới. Correction trực tiếp thay fact cũ; câu giả định hoặc lời kể về người khác được lọc trước khi ghi.

**Chi phí ở hội thoại thường.** Standard có 10 hội thoại tương đối ngắn và không kích hoạt compact. Advanced xử lý nhiều hơn Baseline 5.951 prompt token, khoảng 42,8%, vì mang hồ sơ người dùng vào ngữ cảnh. Token phản hồi nhiều hơn 43.

**Chi phí ở hội thoại dài.** Stress có 16 lượt dài trong một thread. Advanced compact 2 lần và xử lý ít hơn Baseline 9.575 prompt token, khoảng 43,3%. Token phản hồi của Advanced nhiều hơn 31. Lợi ích của compact nằm ở lượng prompt phải xử lý, không nhất thiết ở độ dài phản hồi.

**Tăng trưởng memory.** `User.md` tăng 225 byte ở Standard và 168 byte ở Stress. Cột này đo file hồ sơ, không tính state tạm của compact manager. `upsert_fact()` giữ lại ghi chú Markdown viết tay và chỉ thay dòng fact tương ứng; các thread được tách theo cả `user_id` và `thread_id` để tránh lộ ngữ cảnh giữa hai người dùng dùng trùng tên thread.

## Bonus bước 9: thí nghiệm confidence threshold

Extractor gắn confidence heuristic 0,95 cho fact trực tiếp và 0,4 cho câu giả định, lịch sử hoặc lời dẫn của người khác. Advanced mặc định chỉ ghi candidate đạt ít nhất 0,8; có thể đổi bằng `profile_confidence_threshold`. Hai điểm số này là quy tắc, chưa phải xác suất được hiệu chuẩn.

`data/bonus_ablation.json` gồm hai ca đối kháng có fact đã xác nhận, giả định, lời dẫn và correction. `src/eval_bonus.py` chạy cùng ca với ngưỡng mặc định và ngưỡng 0,0:

| Chế độ | Ngưỡng | Fact precision | Fact recall | False facts | Memory bytes | Prompt tokens processed |
|---|---:|---:|---:|---:|---:|---:|
| protected | 0,8 | 100% | 100% | 0 | 141 | 533 |
| unfiltered | 0,0 | 40% | 40% | 3 | 146 | 535 |

Trong ca kiểm soát này, ngưỡng 0,8 ngăn fact giả định ghi đè fact thật. Chênh lệch 2 prompt token và 5 byte quá nhỏ để kết luận hiệu quả chi phí tổng quát. Tập ca này nhỏ và được thiết kế để thử lỗi cụ thể, không đại diện cho mọi cách nói tiếng Việt.

## Giới hạn và việc nên làm tiếp

Benchmark và ablation đều offline với đáp án và heuristic cố định. Summary compact ưu tiên message gần đây và các mốc quan trọng theo từ khóa, nên vẫn có thể bỏ sót một fact cũ không chứa từ khóa. Extractor vẫn có thể hiểu sai câu nhiều mệnh đề hoặc cách nói mơ hồ. Chưa đo độ trễ, token thực do API báo, chất lượng nhánh live hoặc đánh giá của người đọc. Kết quả kiểm thử hiện tại: 31 test pass.
