"""Build the submission report from measured artifacts, never invented metrics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import evaluate as ev, report


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main():
    proof, stats = read("mask_proof.json"), read("token_stats.json")
    frozen, verdict = read("baselines_frozen.json"), read("verdict.json")
    autopsy = {r["run"]: r for r in read("autopsy.json")}
    runs = {r["run"]: r for r in report.read_rows(results_dir=ROOT / "results")}
    assert set(runs) >= {"correct", "attn_only", "wrong_lr", "qlora"}
    histories = {}
    for key in runs:
        state = json.loads((ROOT / "adapters" / key / "trainer_state.json").read_text(encoding="utf-8"))
        histories[key] = [{"step": row["step"], "loss": row["loss"]}
                          for row in state["log_history"] if "loss" in row]
    report.write_json(histories, "loss_curves.json")
    base, ft = read("baseline_predictions.json"), read("finetune_predictions.json")
    cases = [dict(r, group="target") for r in read("qualitative.json")]
    regression = [json.loads(line) for line in (ROOT / "data" / "eval_regression.jsonl").read_text(encoding="utf-8").splitlines()]
    for i, (row, b, c) in enumerate(zip(regression, base["regression_b"], ft["regression"])):
        bs, cs = ev.keyword_recall(b, row["keywords"]), ev.keyword_recall(c, row["keywords"])
        cases.append({"group": "regression", "i": i, "ticket": row["instruction"],
                      "label": row["keywords"], "baseline_b_pred": b, "ft_pred": c,
                      "baseline_b_score": bs, "ft_score": cs, "delta": cs-bs,
                      "outcome": "win" if cs > bs else "loss" if cs < bs else "tie"})
    # Prefer target losses. If absent, show genuine regression losses explicitly.
    losses = sorted([c for c in cases if c["outcome"] == "loss"],
                    key=lambda c: (c["group"] != "target", c["delta"], c["i"]))
    wins = sorted([c for c in cases if c["outcome"] == "win"],
                  key=lambda c: (c["group"] != "target", -c["delta"], c["i"]))
    chosen = losses[:2] + wins[:2]
    for c in cases:
        if len(chosen) >= 5:
            break
        if c not in chosen:
            chosen.append(c)
    counts = {g: {o: sum(c["group"] == g and c["outcome"] == o for c in cases)
                  for o in ("win", "loss", "tie")} for g in ("target", "regression")}
    report.write_json({"counts": counts, "selected": chosen, "all_cases": cases}, "qualitative_comparison.json")
    correct, attn, low, quant = (runs[k] for k in ("correct", "attn_only", "wrong_lr", "qlora"))
    gap = abs(int(attn["trainable_params"])-int(correct["trainable_params"])) / int(correct["trainable_params"])*100
    delta_attn = autopsy["attn_only"]["target"]-autopsy["correct"]["target"]
    saving = float(correct["peak_vram_gb"])-float(quant["peak_vram_gb"])
    v = verdict["verdict"]
    ranking = sorted(autopsy, key=lambda k: -autopsy[k]["target"])
    loss_ranking = sorted(runs, key=lambda k: float(runs[k]["final_loss"]))
    rows = []
    for key in ("correct", "attn_only", "wrong_lr", "qlora"):
        r = runs[key]
        rows.append({"run": key, "vị trí": r["placement"], "r": r["r"],
                     "tham số": r["trainable_params"], "LR": r["learning_rate"],
                     "mean train loss": r["final_loss"], "target": autopsy[key]["target"],
                     "giây": r["train_seconds"], "peak GB": r["peak_vram_gb"]})
    text = f'''# Lab 21 — Báo cáo thí nghiệm LoRA trên GPU 4 GB

**Học viên:** Nguyễn Bảo Sơn · **MSSV:** 2A202602402 · **Ngày:** 07/10/2026.
Tên và MSSV lấy từ tên thư mục bài làm. Báo cáo được AI hỗ trợ soạn từ các lần chạy
thật trên máy này; phần phản tư không giả định trải nghiệm hay niềm tin riêng của học viên.

## 1. Kết quả và phạm vi

Cổng đánh giá cho kết quả **{"PASSED" if v["passed"] else "FAILED"}**:
target Δ = **{v["target_delta"]:+.4f}**, regression Δ = **{v["regression_delta"]:+.4f}**
so với base dùng prompt tối ưu. Đây là kết quả của **{frozen["model"]}**,
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
Ngân sách khai báo 2 epoch, ép rõ **{correct["max_steps"]} optimizer steps** cho cả bốn run.
`max_length` huấn luyện **{correct["max_length"]}** được đọc từ NB1: p95={stats["p95"]},
max={stats["max"]}, độ dài đề xuất={stats["suggested_max_length"]}. Hàm của lab có
sàn 256 nên không trả về 128 dù p95=100. NB1 ban đầu kiểm tra ở trần tier 1024;
NB3/NB4 dùng 256 và không cắt mẫu nào vì mọi mẫu đều ngắn hơn ngưỡng này.
Packing và padding-free tắt để giữ căn chỉnh nhãn; loss dùng `chunked_nll`.

`results/mask_proof.json`: answer_is_supervised={str(proof["answer_is_supervised"]).lower()},
question_is_masked={str(proof["question_is_masked"]).lower()},
supervised_fraction={proof["supervised_fraction"]} ({proof["n_supervised"]}/{proof["n_total"]} token).
Phần thực sự chịu loss ở ví dụ kiểm chứng:

```text
{proof["supervised_preview"].strip()}
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

NB2 đóng băng tại `{frozen["frozen_at_utc"]}` (UTC), trước NB3.
Prompt tối ưu giữ nguyên, SHA-256 rút gọn `{frozen["optimized_prompt_sha"]}`.
Điểm target của (b)={frozen["baseline_b"]["target"]:.4f} > (a)={frozen["baseline_a"]["target"]:.4f}.
Không điều chỉnh prompt theo kết quả fine-tune. Full predictions được lưu trước khi
train trong `baseline_predictions.json`; dữ liệu và raw eval hashes được giữ nguyên.

{report.markdown_table(verdict["comparison"])}

Target là trung bình độ chính xác bốn trường, không phải tỷ lệ ticket đúng toàn bộ.
Regression là keyword recall của 15 câu; đây là phép đo hẹp, không phải năng lực
tổng quát đầy đủ. Format là parse theo scorer của lab và đủ bốn khóa, không đồng
nghĩa mọi giá trị đều hợp lệ. Latency là ms/mẫu trong batch greedy 4, tối đa 160 token
target; regression giới hạn 96 token. Đây là một lượt đo trên GPU laptop dùng chung
với desktop, không phải benchmark latency lặp nhiều lần. Prompt (b) dài hơn, còn
fine-tune dùng prompt ngắn giống train: so sánh là hai phương án triển khai của lab.

## 4. Ba đối chứng và lập luận nhân quả

{report.markdown_table(rows)}

Tất cả có seed 42, mask, corpus, max_length, số step, scheduler cosine và 6 warmup
steps giống nhau. `final_loss` trong CSV là **mean training loss** do Trainer trả
về, không phải loss của batch cuối. Đường loss thật nằm trong `loss_curves.json`
và từng `adapters/*/trainer_state.json`.

Seed 42 được đặt ngay trước khi tạo adapter, không chỉ trong cấu hình Trainer.
Lần train ban đầu thiếu thao tác này đã bị loại và chạy lại trước evaluation;
không trộn số liệu lần thử đó vào bảng và không đo lại baseline theo kết quả train.

**Vị trí so với rank.** `attn_only` đổi tập module thành q/v, nâng rank lên
{attn["r"]} để khớp ngân sách: {int(attn["trainable_params"]):,} so với
{int(correct["trainable_params"]):,} tham số, sai lệch {gap:.4f}% (<5%). Đây là phép
so sánh vị trí ở ngân sách gần bằng nhau; không phải giữ rank cố định. Chênh lệch
target attention-only trừ correct là {delta_attn:+.4f}. Thứ tự theo target:
`{" > ".join(ranking)}` (điểm bằng nhau phải đọc là hòa); theo mean train loss thấp
đến cao: `{" < ".join(loss_ranking)}`. Rank lớn tự nó không chứng minh tốt hơn;
muốn tách riêng tác động rank cần thêm sweep giữ vị trí cố định. Một lần chạy cũng
không đủ khẳng định quy luật cho mọi model hay tác vụ.

**Learning rate.** `wrong_lr` chỉ hạ LR từ {correct["learning_rate"]} xuống
{low["learning_rate"]}. Loss ghi ở step {histories["correct"][0]["step"]} và
{histories["correct"][-1]["step"]} của correct là
{histories["correct"][0]["loss"]:.4f} → {histories["correct"][-1]["loss"]:.4f};
wrong_lr là {histories["wrong_lr"][0]["loss"]:.4f} → {histories["wrong_lr"][-1]["loss"]:.4f}.
Target tương ứng {autopsy["correct"]["target"]:.4f} và {autopsy["wrong_lr"]["target"]:.4f}.
Với cùng step budget, LR thay đổi tốc độ cập nhật; nhìn loss riêng lẻ dễ nhầm
chưa học đủ thành thiếu khả năng biểu diễn của LoRA. Kết luận cuối phải đối chiếu
target thay vì chỉ so độ dốc loss; dữ liệu hiện tại không chứng minh LR tối ưu toàn cục.

**QLoRA.** Đổi cách nạp base sang NF4 4-bit, vẫn cùng vị trí/rank/LR/step budget.
Peak allocated VRAM là {quant["peak_vram_gb"]} GB so với {correct["peak_vram_gb"]} GB,
chênh lệch tiết kiệm {saving:+.2f} GB ({saving/float(correct["peak_vram_gb"])*100:+.1f}%).
Thời gian {quant["train_seconds"]} s so với {correct["train_seconds"]} s;
target {autopsy["qlora"]["target"]:.4f} so với {autopsy["correct"]["target"]:.4f}.
Các số VRAM là bộ nhớ PyTorch cấp phát, không phải tổng VRAM trong nvidia-smi.
QLoRA được đánh giá trên base 4-bit, đúng với lúc train. Phép đổi này bao gồm cách
chuẩn bị model k-bit của thư viện, nên không thể quy mọi khác biệt chỉ cho sai số
lượng tử. Dù kết quả theo hướng nào, nó không xác nhận hay bác bỏ khuyến cáo cho
Qwen3.5 vì thí nghiệm này dùng Qwen2.5.

## 5. Diễn giải phán quyết

**{"PASSED" if v["passed"] else "FAILED"}**. Lý do máy ghi nhận:
{" ".join(v["reasons"])}

Cổng của lab đòi target tăng nghiêm ngặt và regression không giảm quá 0.02.
Các ngưỡng giữ nguyên từ trước khi train; không sửa scorer để tạo một kết quả đẹp.
Format và latency vẫn được đo, nhưng mã gate hiện tại chỉ dùng target và regression
để quyết định pass/fail. Vì vậy không nên diễn giải PASSED thành đủ mọi điều kiện
vận hành. Ngược lại, FAILED là kết quả có giá trị: nó cho biết phương án này chưa
đáp ứng tiêu chí đã đặt dù loss có thể giảm hoặc JSON trông đúng hơn. Trên tập nhỏ,
một câu regression có thể làm điểm thay đổi đáng kể; cần đọc từng output và chạy
thêm tập độc lập trước khi quyết định triển khai. `valid_trace_rate` đo được là
{verdict["valid_trace_rate"]}; vì model/corpus không được thiết kế cho reasoning trace,
chỉ số này không phải bằng chứng reasoning-trace collapse và không dùng nhận bonus B3.

## 6. Ví dụ định tính có đối chiếu đầy đủ

Toàn bộ target: {counts["target"]["win"]} thắng, {counts["target"]["loss"]} thua,
{counts["target"]["tie"]} hòa so với (b). Regression: {counts["regression"]["win"]} thắng,
{counts["regression"]["loss"]} thua, {counts["regression"]["tie"]} hòa theo keyword recall.
Quy tắc chọn cố định: lấy tối đa hai ca thua, hai ca thắng rồi thêm ca chưa chọn
để đủ năm; ưu tiên target, dùng ca regression khi target không đủ ca thua.
Mỗi ví dụ ghi rõ nhóm để không đánh tráo hai metric. Lưu tất cả so sánh trong
`qualitative_comparison.json`, không chỉ các ví dụ được trích.
'''
    if len(losses) < 2:
        text += f"\nChỉ đo được {len(losses)} ca thua trên hai nhóm; không bịa thêm ca để đủ rubric.\n"
    for j, c in enumerate(chosen, 1):
        text += f'''
### Ví dụ {j}: {c["group"]} #{c["i"]} — {c["outcome"]}

**Đầu vào:** {c["ticket"]}

**Nhãn / keywords:** `{json.dumps(c["label"], ensure_ascii=False)}`

**(b), điểm {c["baseline_b_score"]:.4f}:**

````text
{c["baseline_b_pred"]}
````

**Fine-tune, điểm {c["ft_score"]:.4f}:**

````text
{c["ft_pred"]}
````

Chênh lệch {c["delta"]:+.4f}. {"So sánh từng trường JSON với nhãn; đúng format chưa đủ để đúng intent/urgency/sentiment." if c["group"] == "target" else "Đây là mất/được điểm trên câu hỏi tổng quát, đo bằng keywords; cần đọc nội dung vì keyword recall không đánh giá toàn bộ chất lượng câu trả lời."}
'''
    text += '''
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
'''
    merge_path = ROOT / "results" / "merge_check.json"
    if merge_path.exists():
        merge = read("merge_check.json")
        text += f'\n### Phụ lục B1 — Merge\n\nTrước merge={merge["before_merge"]:.4f}; sau merge={merge["after_merge"]:.4f}; Δ={merge["delta"]:+.4f}, ngưỡng={merge["tolerance"]}, n={merge["n"]}.\n'
        swap_path = ROOT / "results" / "hot_swap.json"
        if swap_path.exists():
            swap = read("hot_swap.json")
            text += f'\nĐã hot-swap trên một base: {", ".join(swap)} (`hot_swap.json`). Merge bỏ phép tính adapter riêng lúc suy luận nhưng gắn checkpoint với adapter đã gộp; giữ adapter riêng phù hợp khi cần đổi tác vụ/khách hàng hoặc rollback nhanh.\n'
    (ROOT / "submission" / "REPORT.md").write_text(text, encoding="utf-8")
    print("Wrote REPORT.md from measured results; qualitative counts:", counts)


if __name__ == "__main__":
    main()
