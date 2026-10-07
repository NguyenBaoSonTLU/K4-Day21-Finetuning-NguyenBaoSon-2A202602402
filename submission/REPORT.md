# Lab 21 — Báo cáo thí nghiệm LoRA trên GPU 4 GB

**Học viên:** Nguyễn Bảo Sơn · **MSSV:** 2A202602402 · **Ngày:** 07/10/2026.
Tên và MSSV lấy từ tên thư mục bài làm. Báo cáo được AI hỗ trợ soạn từ các lần chạy
thật trên máy này; phần phản tư không giả định trải nghiệm hay niềm tin riêng của học viên.

## 1. Kết quả và phạm vi

Cổng đánh giá cho kết quả **FAILED**:
target Δ = **+0.5300**, regression Δ = **-0.3333**
so với base dùng prompt tối ưu. Đây là kết quả của **Qwen/Qwen2.5-0.5B-Instruct**,
không phải kết quả của Qwen3.5-4B trong cấu hình mẫu.

Chọn model instruction-tuned nhỏ để cả LoRA 16-bit và đối chứng QLoRA chạy được
trên NVIDIA GeForce RTX 3050 Laptop GPU 4 GB. Giữ nguyên một base cho baseline và
bốn adapter. Model có 24 lớp full attention; các nhận xét riêng về hybrid attention,
vision tower hay khuyến cáo QLoRA của Qwen3.5 không được suy rộng sang thí nghiệm này.
Nguồn model: [model card chính thức](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

Dataset là corpus đi kèm: 250 ticket CSKH tiếng Việt với JSON gồm `intent`,
`urgency`, `product`, `sentiment`. Lý do chọn là có nhãn kiểm tra khách quan và giữ
nguyên benchmark của lab. Split seed 42 cho 225 train / 25 validation. Validation
không được dùng chọn checkpoint. Eval đủ 50 ticket target và 15 câu regression;
không bật `EVAL_LIMIT`, không dùng secret holdout để học, chọn prompt hay chọn ví dụ.
Giới hạn: corpus nhỏ, mẫu tổng hợp cùng nguồn, một seed; kết quả chưa chứng minh
khả năng khái quát trên ticket thực tế hay ưu thế có ý nghĩa thống kê.

## 2. Pipeline và bằng chứng mask

Tier LAPTOP, batch vật lý 1, tích lũy gradient 8, batch hiệu dụng 8; bf16 theo GPU.
Ngân sách khai báo 2 epoch, ép rõ **58 optimizer steps** cho cả bốn run.
`max_length` huấn luyện **256** được đọc từ NB1: p95=100,
max=105, độ dài đề xuất=256. Hàm của lab có
sàn 256 nên không trả về 128 dù p95=100. NB1 ban đầu kiểm tra ở trần tier 1024;
NB3/NB4 dùng 256 và không cắt mẫu nào vì mọi mẫu đều ngắn hơn ngưỡng này.
Packing và padding-free tắt để giữ căn chỉnh nhãn; loss dùng `chunked_nll`.

`results/mask_proof.json`: answer_is_supervised=true,
question_is_masked=true,
supervised_fraction=0.4066 (37/91 token).
Phần thực sự chịu loss ở ví dụ kiểm chứng:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Template giữ nguyên đoạn `<think>` nhân tạo (`template_check.json`). Điều này chỉ
kiểm tra thao tác render, không chứng minh Qwen2.5 là reasoning model. Corpus không
có reasoning trace. Tokenizer không có generation markers, nên API assistant mask
của tokenizer trả về mask rỗng; script đối chiếu báo FAIL đúng cho đường API đó.
Pipeline khắc phục bằng nhãn tokenize sẵn từ `labkit.data`, thay vì bật cờ
`assistant_only_loss`. `mask_agreement.json` xác nhận cả 250 prompt train/eval khớp,
không có mẫu mất hết supervision và collator giữ mask. Bốn file
`trainer_mask_*.json` xác nhận thêm trên trainer thật trước optimizer step.

## 3. Mốc đóng băng và so sánh ba phương án

NB2 đóng băng tại `2026-10-07T11:24:56.903147+00:00` (UTC), trước NB3.
Prompt tối ưu giữ nguyên, SHA-256 rút gọn `719e74d3b6232053`.
Điểm target của (b)=0.3950 > (a)=0.0000.
Không điều chỉnh prompt theo kết quả fine-tune. Full predictions được lưu trước khi
train trong `baseline_predictions.json`; dữ liệu và raw eval hashes được giữ nguyên.

| run | target | regression | format | latency_ms | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.4667 | 0.0 | 1789.3 | 50 |
| (b) base + optimized prompt | 0.395 | 0.4667 | 1.0 | 678.9 | 50 |
| (c) LoRA fine-tune | 0.925 | 0.1333 | 1.0 | 1176.7 | 50 |

Target là trung bình độ chính xác bốn trường, không phải tỷ lệ ticket đúng toàn bộ.
Regression là keyword recall của 15 câu; đây là phép đo hẹp, không phải năng lực
tổng quát đầy đủ. Format là parse theo scorer của lab và đủ bốn khóa, không đồng
nghĩa mọi giá trị đều hợp lệ. Latency là ms/mẫu trong batch greedy 4, tối đa 160 token
target; regression giới hạn 96 token. Đây là một lượt đo trên GPU laptop dùng chung
với desktop, không phải benchmark latency lặp nhiều lần. Prompt (b) dài hơn, còn
fine-tune dùng prompt ngắn giống train: so sánh là hai phương án triển khai của lab.

## 4. Ba đối chứng và lập luận nhân quả

| run | vị trí | r | tham số | LR | mean train loss | target | giây | peak GB |
|---|---|---|---|---|---|---|---|---|
| correct | text-linear | 16 | 8798208 | 0.0001 | 0.5043 | 0.925 | 273.8 | 1.33 |
| attn_only | attn-only | 130 | 8785920 | 0.0001 | 0.5624 | 0.705 | 200.1 | 1.33 |
| wrong_lr | text-linear | 16 | 8798208 | 1e-05 | 1.995 | 0.115 | 127.3 | 1.33 |
| qlora | text-linear | 16 | 8798208 | 0.0001 | 0.5315 | 0.86 | 290.5 | 0.75 |

Tất cả có seed 42, mask, corpus, max_length, số step, scheduler cosine và 6 warmup
steps giống nhau. `final_loss` trong CSV là **mean training loss** do Trainer trả
về, không phải loss của batch cuối. Đường loss thật nằm trong `loss_curves.json`
và từng `adapters/*/trainer_state.json`.

Seed 42 được đặt ngay trước khi tạo adapter, không chỉ trong cấu hình Trainer.
Lần train ban đầu thiếu thao tác này đã bị loại và chạy lại trước evaluation;
không trộn số liệu lần thử đó vào bảng và không đo lại baseline theo kết quả train.

**Vị trí so với rank.** `attn_only` đổi tập module thành q/v, nâng rank lên
130 để khớp ngân sách: 8,785,920 so với
8,798,208 tham số, sai lệch 0.1397% (<5%). Đây là phép
so sánh vị trí ở ngân sách gần bằng nhau; không phải giữ rank cố định. Chênh lệch
target attention-only trừ correct là -0.2200. Thứ tự theo target:
`correct > qlora > attn_only > wrong_lr` (điểm bằng nhau phải đọc là hòa); theo mean train loss thấp
đến cao: `correct < qlora < attn_only < wrong_lr`. Rank lớn tự nó không chứng minh tốt hơn;
muốn tách riêng tác động rank cần thêm sweep giữ vị trí cố định. Một lần chạy cũng
không đủ khẳng định quy luật cho mọi model hay tác vụ.

**Learning rate.** `wrong_lr` chỉ hạ LR từ 0.0001 xuống
1e-05. Loss ghi ở step 5 và
55 của correct là
3.1297 → 0.0239;
wrong_lr là 3.2863 → 1.2654.
Target tương ứng 0.9250 và 0.1150.
Với cùng step budget, LR thay đổi tốc độ cập nhật; nhìn loss riêng lẻ dễ nhầm
chưa học đủ thành thiếu khả năng biểu diễn của LoRA. Kết luận cuối phải đối chiếu
target thay vì chỉ so độ dốc loss; dữ liệu hiện tại không chứng minh LR tối ưu toàn cục.

**QLoRA.** Đổi cách nạp base sang NF4 4-bit, vẫn cùng vị trí/rank/LR/step budget.
Peak allocated VRAM là 0.75 GB so với 1.33 GB,
chênh lệch tiết kiệm +0.58 GB (+43.6%).
Thời gian 290.5 s so với 273.8 s;
target 0.8600 so với 0.9250.
Các số VRAM là bộ nhớ PyTorch cấp phát, không phải tổng VRAM trong nvidia-smi.
QLoRA được đánh giá trên base 4-bit, đúng với lúc train. Phép đổi này bao gồm cách
chuẩn bị model k-bit của thư viện, nên không thể quy mọi khác biệt chỉ cho sai số
lượng tử. Dù kết quả theo hướng nào, nó không xác nhận hay bác bỏ khuyến cáo cho
Qwen3.5 vì thí nghiệm này dùng Qwen2.5.

## 5. Diễn giải phán quyết

**FAILED**. Lý do máy ghi nhận:
general capability regressed by 0.333 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.

Cổng của lab đòi target tăng nghiêm ngặt và regression không giảm quá 0.02.
Các ngưỡng giữ nguyên từ trước khi train; không sửa scorer để tạo một kết quả đẹp.
Format và latency vẫn được đo, nhưng mã gate hiện tại chỉ dùng target và regression
để quyết định pass/fail. Vì vậy không nên diễn giải PASSED thành đủ mọi điều kiện
vận hành. Ngược lại, FAILED là kết quả có giá trị: nó cho biết phương án này chưa
đáp ứng tiêu chí đã đặt dù loss có thể giảm hoặc JSON trông đúng hơn. Trên tập nhỏ,
một câu regression có thể làm điểm thay đổi đáng kể; cần đọc từng output và chạy
thêm tập độc lập trước khi quyết định triển khai. `valid_trace_rate` đo được là
0.0; vì model/corpus không được thiết kế cho reasoning trace,
chỉ số này không phải bằng chứng reasoning-trace collapse và không dùng nhận bonus B3.

## 6. Ví dụ định tính có đối chiếu đầy đủ

Toàn bộ target: 50 thắng, 0 thua,
0 hòa so với (b). Regression: 0 thắng,
5 thua, 10 hòa theo keyword recall.
Quy tắc chọn cố định: lấy tối đa hai ca thua, hai ca thắng rồi thêm ca chưa chọn
để đủ năm; ưu tiên target, dùng ca regression khi target không đủ ca thua.
Mỗi ví dụ ghi rõ nhóm để không đánh tráo hai metric. Lưu tất cả so sánh trong
`qualitative_comparison.json`, không chỉ các ví dụ được trích.

### Ví dụ 1: regression #0 — loss

**Đầu vào:** Thủ đô của Việt Nam là thành phố nào?

**Nhãn / keywords:** `["Hà Nội"]`

**(b), điểm 1.0000:**

````text
Thủ đô của Việt Nam là Hà Nội.
````

**Fine-tune, điểm 0.0000:**

````text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": "thủ đô", "sentiment": "trung_tinh"}
````

Chênh lệch -1.0000. Đây là mất/được điểm trên câu hỏi tổng quát, đo bằng keywords; cần đọc nội dung vì keyword recall không đánh giá toàn bộ chất lượng câu trả lời.

### Ví dụ 2: regression #2 — loss

**Đầu vào:** 1 km bằng bao nhiêu mét?

**Nhãn / keywords:** `["1000"]`

**(b), điểm 1.0000:**

````text
1 km bằng 1000 mét.
````

**Fine-tune, điểm 0.0000:**

````text
"1 km" được coi là "một chục mét".
````

Chênh lệch -1.0000. Đây là mất/được điểm trên câu hỏi tổng quát, đo bằng keywords; cần đọc nội dung vì keyword recall không đánh giá toàn bộ chất lượng câu trả lời.

### Ví dụ 3: target #8 — win

**Đầu vào:** Xin chào, mình đặt chuột không dây mã đơn DH139158. Bảo hành bao lâu. Không vội. Mình vẫn tin tưởng shop.

**Nhãn / keywords:** `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "chuột không dây", "sentiment": "tich_cuc"}`

**(b), điểm 0.2500:**

````text
{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây", "sentiment": "tieu_cuc"}
````

**Fine-tune, điểm 1.0000:**

````text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": "chuột không dây", "sentiment": "tich_cuc"}
````

Chênh lệch +0.7500. So sánh từng trường JSON với nhãn; đúng format chưa đủ để đúng intent/urgency/sentiment.

### Ví dụ 4: target #9 — win

**Đầu vào:** Cho mình hỏi, mình đặt chuột không dây mã đơn OD538419. Hoàn tiền. Mong shop phản hồi. Mình vẫn tin tưởng shop.

**Nhãn / keywords:** `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "chuột không dây", "sentiment": "tich_cuc"}`

**(b), điểm 0.2500:**

````text
{"intent": "hoan_tien", "urgency": "cao", "product": "chuột không dây OD538419", "sentiment": "tieu_cuc"}
````

**Fine-tune, điểm 1.0000:**

````text
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "chuột không dây", "sentiment": "tich_cuc"}
````

Chênh lệch +0.7500. So sánh từng trường JSON với nhãn; đúng format chưa đủ để đúng intent/urgency/sentiment.

### Ví dụ 5: target #0 — win

**Đầu vào:** Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.

**Nhãn / keywords:** `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`

**(b), điểm 0.2500:**

````text
{"intent": "hoan_tien", "urgency": "thap", "product": "chuột không dây", "sentiment": "tieu_cuc"}
````

**Fine-tune, điểm 0.7500:**

````text
{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "trung_tinh"}
````

Chênh lệch +0.5000. So sánh từng trường JSON với nhãn; đúng format chưa đủ để đúng intent/urgency/sentiment.

## 7. Kết luận và bài học

Thí nghiệm này cho thấy việc đánh giá fine-tuning phải bắt đầu từ một phép so sánh
có thể kiểm tra lại. Nếu chỉ đặt prompt ngắn cho model gốc rồi so với adapter đã
được học schema, phần tăng điểm có thể chỉ phản ánh thông tin mà prompt chưa cung
cấp. Vì vậy mốc có prompt tối ưu được đo và đóng băng trước khi huấn luyện. Mask
là điều kiện để các bước tối ưu có nghĩa: một loss thấp trên prompt hoặc trên nhãn
rỗng không chứng minh model học trả lời. Việc kiểm tra nhãn ngay tại collator giúp
nối bằng chứng NB1 với dữ liệu mà trainer thực sự dùng. Sau đó, đối chứng learning
rate giúp phân biệt tốc độ học với giới hạn biểu diễn; đối chứng vị trí chỉ có ý
nghĩa khi ngân sách tham số gần bằng nhau. Không thể dùng training loss thay cho
target, cũng không thể bỏ qua regression chỉ vì JSON đẹp hơn. Quyết định triển
khai phải dựa vào phán quyết và ví dụ lỗi ở trên, thêm chi phí vận hành, rồi xác
nhận trên dữ liệu thực tế độc lập. Chưa triển khai cho khách hàng thật chỉ dựa
trên corpus tổng hợp này. Nếu có thêm hai giờ, ưu tiên phân tích lỗi theo từng
trường và chạy lại nhiều seed trên tập phát triển riêng; không sửa tập eval đã
đóng băng. Các thí nghiệm tiếp theo cần khai báo trước giả thuyết và ngân sách,
để việc cải thiện không biến thành chọn kết quả thuận lợi sau khi đã nhìn điểm.

Ba bài học cụ thể rút ra từ bằng chứng trong lần chạy:

1. Tokenizer có thể render chat đúng nhưng trả assistant mask rỗng; cần kiểm tra
   nhãn thật sau collator, không chỉ tin tên một cờ cấu hình.
2. Cùng rank không có nghĩa cùng ngân sách: q/v cần rank 130 để gần khớp text-linear
   rank 16 trên model này. Không mang con số đó sang model khác mà không tính lại.
3. Khác biệt checksum có thể do checkout CRLF; phải chứng minh bằng hash chuẩn hóa
   và Git gốc, không viết lại checksum để làm cổng kiểm tra chuyển xanh.

## 8. Tái lập, phạm vi bonus và trung thực báo cáo

Xem `submission/REPRODUCE.md` và `submission/requirements-lock.txt`. Không sử dụng
số liệu mẫu trong tài liệu làm kết quả cá nhân. Không công bố adapter lên Hub.
Không nhận bonus dataset mới, reasoning collapse hoặc rank sweep vì chưa thực
hiện các thí nghiệm tương ứng. Chi tiết merge/hot-swap nếu đã đo nằm ở phụ lục.

### Phụ lục B1 — Merge

Trước merge=0.9250; sau merge=0.9250; Δ=+0.0000, ngưỡng=0.01, n=50.

Đã hot-swap trên một base: correct, attn_only, wrong_lr (`hot_swap.json`). Merge bỏ phép tính adapter riêng lúc suy luận nhưng gắn checkpoint với adapter đã gộp; giữ adapter riêng phù hợp khi cần đổi tác vụ/khách hàng hoặc rollback nhanh.
