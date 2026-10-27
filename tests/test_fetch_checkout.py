"""Offline tests for serial GitHub source snapshot reconstruction."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from runpy import run_path

import pytest

module = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "github_fetch_checkout.py"),
    run_name="__test__",
)


def test_git_blob_sha_matches_native_git_blob_contract():
    data = b"example data\n"
    expected = hashlib.sha1(b"blob 13\0example data\n").hexdigest()
    assert module["_git_blob_sha"](data) == expected


def test_safe_path_rejects_traversal_and_dot_git(tmp_path):
    safe_target = module["_safe_target"]
    assert safe_target(tmp_path, "docs/README.md") == tmp_path / "docs" / "README.md"
    with pytest.raises(ValueError):
        safe_target(tmp_path, "../escape.txt")
    with pytest.raises(ValueError):
        safe_target(tmp_path, ".git/config")


def test_sequential_fetch_verifies_and_reuses_blobs(tmp_path):
    content = b"hello world\n"
    sha = module["_git_blob_sha"](content)
    payloads = {
        "commits/": {
            "sha": "head_sha", "commit": {"tree": {"sha": "tree_sha"}}
        },
        "trees/": {
            "truncated": False,
            "tree": [{"type": "blob", "path": "src/hello.txt",
                      "sha": sha, "mode": "100644"}],
        },
        "blobs/": {
            "encoding": "base64",
            "content": base64.b64encode(content).decode("ascii"),
        },
    }
    def fake_api(url, token, *, attempts=4):
        return next(data for part, data in payloads.items() if part in url)

    # Functions returned by run_path use the module's globals, not this test's globals.
    module["checkout"].__globals__["_api_json"] = fake_api
    checkout = module["checkout"]
    first = checkout("example/repo", "main", tmp_path, token=None)
    second = checkout("example/repo", "main", tmp_path, token=None)
    assert first["downloaded_files"] == 1
    assert second["reused_files"] == 1
    assert (tmp_path / "src/hello.txt").read_bytes() == content
