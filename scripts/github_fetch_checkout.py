"""Reconstruct a GitHub working-tree snapshot by fetching each blob individually.

A snapshot is not a true git clone: it has no .git history, refs or local remote.
Use 'git clone --filter=blob:none' if history is required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from loguru import logger

REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _api_json(url: str, token: str | None, *, attempts: int = 4) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "chunk-experiment-snapshot-fetcher",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == attempts - 1:
                raise RuntimeError(
                    f"GitHub API failed HTTP {exc.code}. "
                    "Check GITHUB_TOKEN permissions and API rate limits."
                ) from exc
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("unreachable API retry state")


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _safe_target(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or ".git" in candidate.parts
        or not candidate.parts
    ):
        raise ValueError(f"unsafe repository path: {relative}")
    output = root / candidate
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"repository path escapes destination: {relative}")
    return output


def checkout(
    repo: str,
    ref: str,
    target: Path,
    *,
    token: str | None,
    excludes: tuple[str, ...] = (),
    limit: int | None = None,
    force: bool = False,
) -> dict:
    if not REPO_PATTERN.fullmatch(repo):
        raise ValueError("repo must be 'owner/name'")
    target.mkdir(parents=True, exist_ok=True)
    base = f"https://api.github.com/repos/{repo}"
    commit = _api_json(f"{base}/commits/{quote(ref, safe='')}", token)
    head_sha = commit["sha"]
    tree_sha = commit["commit"]["tree"]["sha"]
    tree = _api_json(f"{base}/git/trees/{tree_sha}?recursive=1", token)
    if tree.get("truncated"):
        raise RuntimeError("GitHub truncated the tree; snapshot would be incomplete")
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob"]
    entries = [
        entry for entry in entries
        if not any(entry["path"] == prefix.rstrip("/")
                   or entry["path"].startswith(prefix.rstrip("/") + "/")
                   for prefix in excludes)
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        entries = entries[:limit]
    downloaded = 0
    reused = 0
    for ordinal, item in enumerate(entries, start=1):
        path = item["path"]
        output = _safe_target(target, path)
        if item["mode"] not in ("100644", "100755"):
            raise ValueError(f"unsupported blob mode {item['mode']} for {path}")
        sha = item["sha"]
        if output.is_file() and _git_blob_sha(output.read_bytes()) == sha:
            reused += 1
            logger.info("reuse {}/{} {} sha={}", ordinal, len(entries), path, sha)
            continue
        if output.exists() and not force:
            raise ValueError(
                f"existing local file differs: {output}. Use --force to overwrite."
            )
        response = _api_json(f"{base}/git/blobs/{sha}", token)
        if response.get("encoding") != "base64":
            raise RuntimeError(f"unexpected blob encoding for {path}")
        data = base64.b64decode(response["content"], validate=False)
        if _git_blob_sha(data) != sha:
            raise RuntimeError(f"blob SHA validation failed: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".fetch-partial")
        temporary.write_bytes(data)
        temporary.replace(output)
        if item["mode"] == "100755":
            output.chmod(output.stat().st_mode | stat.S_IXUSR)
        downloaded += 1
        logger.info("fetch {}/{} {} bytes={} sha={}", ordinal, len(entries),
                    path, len(data), sha)

    summary = {
        "repo": repo,
        "commit_sha": head_sha,
        "tree_sha": tree_sha,
        "selected_files": len(entries),
        "downloaded_files": downloaded,
        "reused_files": reused,
        "excluded_prefixes": list(excludes),
        "full_snapshot": limit is None and not excludes,
        "note": "Working-tree file snapshot only; no .git history.",
    }
    (target / ".github_snapshot_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial verified GitHub blob fetcher")
    parser.add_argument("--repo", default="1998x-stack/chunk_experiment")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.target.mkdir(parents=True, exist_ok=True)
    sink = logger.add(args.target / ".fetch.jsonl", serialize=True, rotation="10 MB")
    try:
        result = checkout(
            args.repo, args.ref, args.target, token=os.getenv("GITHUB_TOKEN"),
            excludes=tuple(args.exclude_prefix), limit=args.max_files, force=args.force,
        )
    finally:
        logger.remove(sink)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
