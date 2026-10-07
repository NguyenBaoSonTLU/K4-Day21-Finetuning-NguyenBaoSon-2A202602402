"""Independently recompute scores from saved predictions and check training artifacts."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import evaluate as ev, report


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def close(actual, expected, tolerance=0.000051):
    assert abs(actual-expected) <= tolerance, (actual, expected)


def main():
    target = [json.loads(l) for l in (ROOT / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines()]
    regression = [json.loads(l) for l in (ROOT / "data/eval_regression.jsonl").read_text(encoding="utf-8").splitlines()]
    frozen, baseline = read("baselines_frozen.json"), read("baseline_predictions.json")
    verdict, ft = read("verdict.json"), read("finetune_predictions.json")
    checks = []
    for name, expected in frozen["eval_sha256"].items():
        assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected
    checks.append("raw eval hashes unchanged since baseline freeze")

    def score(preds, regpreds, expected):
        assert len(preds) == len(target)
        close(sum(ev.triage_field_accuracy(p, r["label"]) for p, r in zip(preds, target))/len(target), expected["target"])
        close(sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in preds)/len(target), expected["format"])
        if regpreds is not None:
            assert len(regpreds) == len(regression)
            close(sum(ev.keyword_recall(p, r["keywords"]) for p, r in zip(regpreds, regression))/len(regression), expected["regression"])

    for key in ("a", "b"):
        score(baseline[f"target_{key}"], baseline[f"regression_{key}"], frozen[f"baseline_{key}"])
    score(ft["target"], ft["regression"], verdict["comparison"][2])
    for row in read("autopsy.json"):
        preds = ft["target"] if row["run"] == "correct" else read(row["run"]+"_predictions.json")["target"]
        score(preds, None, row)
    checks.append("all target, format and regression scores recomputed from full predictions")

    scores_ft = ev.GroupScores(**{k:v for k,v in verdict["comparison"][2].items() if k != "run"})
    scores_b = ev.GroupScores(**frozen["baseline_b"])
    computed = ev.regression_gate(scores_ft, scores_b)
    assert computed.passed == verdict["verdict"]["passed"]
    close(computed.target_delta, verdict["verdict"]["target_delta"])
    close(computed.regression_delta, verdict["verdict"]["regression_delta"])
    checks.append("verdict agrees with unchanged regression gate")

    rows = {r["run"]:r for r in report.read_rows()}
    assert set(rows) == {"correct", "attn_only", "wrong_lr", "qlora"}
    assert len({r["max_steps"] for r in rows.values()}) == 1
    from safetensors import safe_open
    for key, row in rows.items():
        assert row["model"] == frozen["model"]
        assert int(row["initialization_seed"]) == 42
        assert read(f"trainer_mask_{key}.json")["labels_preserved"]
        path = ROOT / "adapters" / key
        if path.exists():
            state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
            assert state["global_step"] == int(row["max_steps"])
            with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
                actual = sum(math.prod(tensors.get_slice(k).get_shape()) for k in tensors.keys())
            assert actual == int(row["trainable_params"]), (key, actual, row["trainable_params"])
    checks.append("shared model, initialization seed, actual steps, trainer masks and saved parameter counts verified")
    selected = read("qualitative_comparison.json")["selected"]
    assert len(selected) >= 5
    assert sum(c["outcome"] == "loss" for c in selected) >= 2, "Rubric requires two genuine losses"
    checks.append("five qualitative examples include at least two measured losses")
    merge_path = ROOT / "results" / "merge_check.json"
    if merge_path.exists():
        merge = read("merge_check.json")
        assert merge["delta"] >= -merge["tolerance"]
        assert merge["n"] == len(target)
        swaps = read("hot_swap.json")
        assert len(swaps) >= 2
        checks.append("full-set merge score and at least two hot-swapped adapters verified")
    report.write_json({"passed": True, "checks": checks}, "validation.json")
    print("Validated:")
    print("\n".join("- "+c for c in checks))


if __name__ == "__main__":
    main()
