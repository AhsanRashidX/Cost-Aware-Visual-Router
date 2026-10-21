"""Assemble the folder to copy to the T4 serving machine: scripts, README, request set and pre-resized page images.

Runs t4/build_requests.py (requests.jsonl, images/, requests_meta.json, policies_summary.json, with all its
verification checks), copies the server-side scripts from t4/ with Unix line endings, checks that config.sh's pixel
limits equal generate.py's, and writes package_info.json (git commit, build time, sizes) and SHA1SUMS (check on the
server with `sha1sum -c SHA1SUMS --quiet`).
Output: t4_package/ (about 0.26 GB; git-ignored).
Usage:  python make_t4_package.py [--check-n -1]     (-1 = token-count check on every test prompt, 0 = skip)
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time

from common import ROOT, _git_commit

sys.path.insert(0, str(ROOT / "t4"))
import build_requests  # noqa: E402  (t4/build_requests.py)
import generate as G  # noqa: E402

OUT = ROOT / "t4_package"
SCRIPTS = ["README_T4.md", "config.sh", "serve.sh", "stop.sh", "client.sh", "download_model.sh", "run_matrix.sh",
           "pilot.py", "loadgen.py", "client_common.py", "vqa_metrics.py", "mock_server.py"]
SKIP_DIRS = {"results", "logs", "run", "__pycache__"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-n", type=int, default=-1)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    cfg = (ROOT / "t4" / "config.sh").read_text(encoding="utf-8")
    px = {k: int(re.search(rf"^{k}=(\d+)", cfg, re.M).group(1)) for k in ("MIN_PIXELS", "MAX_PIXELS")}
    if px != {"MIN_PIXELS": G.MIN_PIXELS, "MAX_PIXELS": G.MAX_PIXELS}:
        raise SystemExit(f"config.sh pixel limits {px} != generate.py ({G.MIN_PIXELS}, {G.MAX_PIXELS})")

    meta = build_requests.main(OUT, args.check_n)

    for f in SCRIPTS:
        data = (ROOT / "t4" / f).read_bytes().replace(b"\r\n", b"\n")   # the checkout may use CRLF
        (OUT / f).write_bytes(data)
        if f.endswith(".sh"):
            (OUT / f).chmod(0o755)

    try:
        dirty = bool(subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent.parent.as_posix()}", "status",
                                              "--porcelain", "--", "."], cwd=ROOT, text=True).strip())
    except Exception:  # noqa: BLE001
        dirty = None
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(OUT).parts[:-1])
                   and p.name not in ("SHA1SUMS", "package_info.json"))
    size = sum(p.stat().st_size for p in files)
    info = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": _git_commit(), "git_dirty_revision_dir": dirty,
            "n_requests": meta["n_requests"], "n_images": meta["n_images"], "n_files": len(files),
            "size_mb": round(size / 2**20, 1), "verification": meta["verification"]}
    (OUT / "package_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8", newline="\n")
    with open(OUT / "SHA1SUMS", "w", encoding="utf-8", newline="\n") as f:
        for p in files + [OUT / "package_info.json"]:
            f.write(f"{hashlib.sha1(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}\n")
    print(f"wrote {OUT}: {len(files) + 2} files, {size / 2**20:.0f} MB "
          f"({meta['n_requests']} requests, {meta['n_images']} images)")


if __name__ == "__main__":
    main()
