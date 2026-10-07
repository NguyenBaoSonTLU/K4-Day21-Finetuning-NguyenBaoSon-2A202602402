"""A Windows checkout must pass, while actual eval edits must remain detectable."""
import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "lab_verify", Path(__file__).resolve().parents[1] / "scripts" / "verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_checksum_accepts_crlf_but_rejects_content_edits(tmp_path):
    original = b'{"label": "correct"}\n'
    expected = hashlib.sha256(original).hexdigest()[:16]
    path = tmp_path / "eval.jsonl"
    path.write_bytes(original)
    assert verify._matches_checksum(path, expected)
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert verify._matches_checksum(path, expected)
    path.write_bytes(original.replace(b"correct", b"changed"))
    assert not verify._matches_checksum(path, expected)
    path.write_bytes(original.replace(b": ", b":"))
    assert not verify._matches_checksum(path, expected)
