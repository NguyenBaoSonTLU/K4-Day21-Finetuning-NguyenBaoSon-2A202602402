# Reflection — Lab 21

**Nguyễn Bảo Sơn · 2A202602402 · 07/10/2026**

Bản phản tư được AI hỗ trợ soạn từ nhật ký thực thi. Nhận xét kỹ thuật dưới đây
có bằng chứng trong repository; không gán cho học viên cảm xúc, dự đoán hoặc niềm
tin trước lab chưa được cung cấp. Học viên cần đọc và xác nhận cách diễn đạt cá
nhân trước khi nộp.

**1. Điều gì làm bạn ngạc nhiên nhất?**

Quan sát đáng chú ý nhất là chat template render được câu trả lời bình thường
nhưng API assistant mask trả về zero token vì thiếu generation markers. Mask của
lab supervise 37/91 token trong mẫu đầu và loại câu hỏi. Kiểm tra chuỗi render đẹp
chưa đủ: phải theo dõi nhãn qua collator đến trainer. `mask_agreement.json` và
`trainer_mask_*.json` kiểm chứng đúng đoạn nối đó, không chỉ dùng tokenizer giả.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Thời gian tập trung vào tải PyTorch CUDA/model và huấn luyện các đối chứng. GPU
chỉ có 4 GB, ổ workspace thiếu chỗ nên cache model chuyển sang thư mục tạm trên C:.
Một lần train ban đầu bị loại vì phát hiện seed của Trainer đến sau khởi tạo LoRA;
các run được chạy lại với seed 42 đặt trước khi tạo adapter. Không có nhật ký dự
đoán thời gian của học viên trước lab, nên không khẳng định đây là bất ngờ cá nhân.
Bài học thực hành là kiểm tra thứ tự khởi tạo và tài nguyên trước khi train dài.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Không có dữ liệu về niềm tin trước đây của học viên. Kết luận kỹ thuật cần xem lại
là “loss giảm nghĩa là fine-tuning đáng triển khai”. Cùng base, prompt ngắn đạt
target 0.0000 còn prompt tối ưu đạt 0.3950 và format 1.0000 trước train: chỉ so với
prompt ngắn đặt mốc quá dễ. Ngoài ra, text-linear rank 16 có 8,798,208 tham số,
còn q/v phải lên rank 130 mới có 8,785,920; cùng rank không có nghĩa cùng ngân sách.
Do đó phải đọc target, regression và các ca thua, thay vì chỉ đọc loss hoặc rank.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

AI đọc yêu cầu, chọn model nhỏ, cài môi trường, thực thi notebook, lưu predictions
và loss curves, kiểm tra nhãn thật, đối chiếu checksum, soạn report và đóng gói
bằng chứng. Kế hoạch ban đầu chưa bắt được việc Trainer đặt seed sau LoRA
initialization; phải dừng và chạy lại trước evaluation. Lệnh freeze package đầu
tiên cũng thất bại do cache mặc định ngoài workspace; đã chạy lại với cache trong
workspace. Các lỗi được sửa theo log, không che bằng số liệu mẫu hay sửa điểm chấm.
AI không thể xác nhận trải nghiệm chủ quan của học viên; phần này công khai giới hạn đó.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đầu tiên xác định lỗi kinh doanh cần giảm và xây tập đánh giá đại diện, tách khỏi
dữ liệu train; thống nhất tiêu chí độ đúng, format, regression và latency. Đo base
với prompt tốt rồi đóng băng mốc. Chỉ train khi đã kiểm tra nhãn, quyền sử dụng
dữ liệu, prompt alignment và loss mask. Với bài này, ưu tiên đọc ca thua trong
`REPORT.md`, kiểm tra trên ticket thực tế độc lập và lặp thêm seed, thay vì tăng
rank hoặc nới gate chỉ để có chữ PASSED.
