import math
import random
import asyncio
import argparse
import os
import shutil
import re
import heapq
import aioboto3
import gzip
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, Any, cast, TypeAlias

LANGUAGES = {"Python"}

CONFIG_MARKERS = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "pipfile", "pipfile.lock", "environment.yml", "tox.ini", "poetry.lock",
    "makefile", "manage.py",
}

JUNK_TOKENS = {
    "sample", "demo", "example", "exercise", "assignment", "homework",
    "tutorial", "playground", "practice", "course", "learn", "study",
    "tmp", "temp", "backup", "archive", "leetc", "katas",
}

# Non-core: excluded from core density calc, but still allowed to download
NONCORE_DIRS = {
    ".github", ".gitlab", ".circleci",
    "docs", "doc",
    "examples", "example",
    "scripts", "script",
    "notebooks", "notebook",
    "bench", "benchmark",
    "assets", "data",
    "tests", "test", "testing",
    "venv", "env", "node_modules",
}

# Generated code patterns
GENERATED_SUBSTR = {
    "/migrations/",
    "/alembic/versions/",
    "/__pycache__/",
    "/build/",
    "/dist/",
    "/generated/",
    "/gen/",
    "/openapi/",
    "/swagger_client/",
}
GENERATED_SUFFIX = (
    "_pb2.py", "_pb2_grpc.py", "_grpc.py",
    "_generated.py", "_gen.py",
    "ui_mainwindow.py", "_ui.py",
    "resources_rc.py",
)
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff",
    ".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav", ".flac",
    ".zip", ".tar", ".tgz", ".gz", ".rar", ".7z", ".bz2", ".xz",
    ".jar", ".war",
    ".pyc", ".pyo", ".o", ".obj", ".dll", ".so", ".dylib", ".exe", ".bin", ".class",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".db", ".sqlite", ".parquet", ".arrow",
    ".ds_store",
    ".ipynb",  # skip notebooks (JSON-based) unless you implement extraction
}

@dataclass
class RepoStats:
    name_is_junk: bool
    owner: str
    has_config: bool
    has_init: bool
    root_folders: set[str]
    total_bytes_est: int
    core_bytes_est: int
    core_py_bytes_est: int
    core_py_files: int
    core_py_ratio: float
    gha_language: Optional[str]
    stars: int
    forks: int
    
def score_repo(st: RepoStats) -> tuple[float, str]:
    score = 0.0
    reasons = []
    # Core python ratio
    score += 110 * st.core_py_ratio
    reasons.append(f"core_ratio={st.core_py_ratio:.2f}")
    # Absolute core pythons files
    if st.core_py_files >= 40:
        score += 12
        reasons.append("core_py>=40")
    elif st.core_py_files >= 25:
        score += 7
        reasons.append("core_py>=25")
    
    if st.has_config:
        score += 18
        reasons.append("config")
    if any(d in {"src", "lib"} for d in st.root_folders):
        score += 8
        reasons.append("src/lib")
    
    if st.has_init:
        score += 3
        reasons.append("init")
    # Github language is soft
    if st.gha_language is not None:
        lang = st.gha_language.lower()
        if lang == "python":
            score += 8
            reasons.append("gha=py")
        else:
            score -= 8
            reasons.append(f"gha={lang}")
    # Popularity (log)
    if st.stars > 0:
        score += 3 * math.log1p(st.stars)
    if st.forks > 0:
        score += 2 * math.log1p(st.forks)
    
    return score, " ;".join(reasons)
def parse_buckets(specs: list[str]) -> list[tuple[int, int, int]]:
    result = []
    for spec in specs:
        rng, quota = spec.strip().split(":")
        lo, hi = rng.split("-")
        result.append((
            int(float(lo) * 1024 ** 2),
            int(float(hi) * 1024 ** 2),
            int(float(quota) * 1024 ** 2)
        ))
    return result
def is_text_candidates(f: dict[str, Any], max_file_kb: int, max_line_len: int, min_alphanum: float) -> bool:
    path: str = f.get("path", "")
    path = path.replace("\\", "/")
    if not path:
        return False
    lower = path.lower()
    if f.get("is_vendor") or f.get("is_generated"):
        return False
    if ".git" in lower.split("/"):
        return False
    # Binary guard
    if any(lower.endswith(ext) for ext in BINARY_EXTS):
        return False
    # Generated code guard
    if any(s in lower for s in GENERATED_SUBSTR):
        return False
    if any(lower.endswith(suf) for suf in (GENERATED_SUFFIX)):
        return False
    size_est = int(f.get("length_bytes", 0))
    if size_est <= 0 or size_est > max_file_kb * 1024:
        return False
    if f.get("max_line_length", max_line_len+1) > max_line_len:
        return False
    if f.get("alphanum_fraction", min_alphanum-1) < min_alphanum:
        return False
    # Pass all criteas
    return True
def tokenize_repo_name(repo_name: str) -> set[str]:
    base = repo_name.split("/")[-1].lower()
    return set(t for t in re.split(r"[^a-z0-9]+", base) if t)

def repo_stats(repo: dict[str, Any], eligible: list[dict[str, Any]]) -> RepoStats:
    repo_name: str = repo.get("repo_name", "")
    owner = repo_name.split("/")[0].lower() if "/" in repo_name else "unknown"
    toks = tokenize_repo_name(repo_name)
    name_is_junk = any(t in toks for t in JUNK_TOKENS)
    has_config = False
    has_init = False
    root_folders = set()
    for file in (repo.get("files", [])):
        file = cast(dict[str, Any], file)
        path: str = file.get("path", "")
        path = path.replace("\\", "/").lower()
        parts = path.split("/")
        if len(parts) > 1:
            root_folders.add(parts[0])
        fn = parts[-1]
        if fn in CONFIG_MARKERS:
            has_config = True
        if fn == "__init__.py":
            has_init = True
    total_bytes = 0
    core_bytes = 0
    core_python_bytes = 0
    core_python_count = 0
    for file in eligible:
        size = int(file.get("length_bytes", 0))
        total_bytes += size
        path = path.replace("\\", "/").lower()
        parts = path.split("/")
        # core if not in NONCORE at top-level
        is_core = True
        if len(parts) > 1 and parts[0] in NONCORE_DIRS:
            is_core = False
        if is_core:
            core_bytes += size
            language: str = file["language"]
            if language in LANGUAGES or path.lower().endswith(".py"):
                core_python_bytes += size
                core_python_count += 1
    ratio = (core_python_bytes / core_bytes) if core_bytes > 0 else 0.0
    # if (core_python_bytes > 0 and core_bytes > 0):
    #     print(core_python_bytes, core_bytes)
    return RepoStats(
        name_is_junk=name_is_junk,
        owner=owner,
        has_config=has_config,
        has_init=has_init,
        root_folders=root_folders,
        total_bytes_est=total_bytes,
        core_bytes_est=core_bytes,
        core_py_bytes_est=core_python_bytes,
        core_py_files=core_python_count,
        core_py_ratio=ratio,
        gha_language=repo.get("gha_language"),
        stars=int(repo.get("star_events_count", 0)),
        forks=int(repo.get("fork_events_count", 0)),
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token", default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--buckets", nargs="+", type=str, default=["0.5-1:10", "1-5:50", "5-15:50"])# default=["8-15:40", "15-25:60"])
    parser.add_argument("--target_total_mb", type=int, default=100)
    parser.add_argument("--max_single_mb", type=int, default=15)
    parser.add_argument("--min_core_py_files", type=int, default=25)
    parser.add_argument("--min_core_py_ratio", type=float, default=0.55)
    parser.add_argument("--score_threshold", type=float, default=60.0)
    parser.add_argument("--max_file_kb", type=int, default=2048)
    parser.add_argument("--max_actual_kb", type=int, default=4096)
    parser.add_argument("--max_line_length", type=int, default=2000)
    parser.add_argument("--min_alphanum", type=float, default=0.25)
    # ap.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_inflight_files", type=int, default=64) # For each workers
    # Scanning controls
    parser.add_argument("--candidates_per_bucket", type=int, default=500)
    parser.add_argument("--scan_max", type=int, default=300_000)
    parser.add_argument("--topk_random_pick", type=int, default=8) # Pick random among top k buckets
    parser.add_argument("--max_repos_per_owner", type=int, default=1)
    # Post check on real files
    parser.add_argument("--post_min_py_mb", type=float, default=0.1)
    parser.add_argument("--post_min_py_ratio", type=float, default=0.55)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--backoff_s", type=float, default=0.25)
    parser.add_argument("--file_timeout", type=float, default=5)
    # Output
    parser.add_argument("--out_dir", type=str, default="stackv2_100mb")
    parser.add_argument("--force_exit", action="store_true")
    
    args = parser.parse_args()
    return args
def bucket_index(bytes_val: int, buckets: list[tuple[int, int, int]]) -> Optional[int]:
    for i, (lo, hi, _) in enumerate(buckets):
        if lo <= bytes_val < hi:
            return i
    return None
def print_debug(args: argparse.Namespace, st: RepoStats, repo: dict[str, Any]):
    reject_reason = ""
    if st.name_is_junk: reject_reason = "Junk name"
    elif st.core_py_files < args.min_core_py_files: reject_reason = f"Few Files ({st.core_py_files})"
    elif st.core_py_ratio < args.min_core_py_ratio: reject_reason = f"Low Ratio ({st.core_py_ratio:.2f})"
    else:
        est = st.total_bytes_est
        bi = bucket_index(est, parse_buckets(args.buckets))
        if bi is None: reject_reason = f"Size Mismatch ({est/1024:.1f}KB)"
        else:
            s_tmp, _ = score_repo(st)
            if s_tmp < args.score_threshold: reject_reason = f"Low Score ({s_tmp:.1f})"
    
    if reject_reason:
        print(f"DEBUG REJECT: {repo.get("repo_name")} | Size: {st.total_bytes_est/1024:.1f}KB | {reject_reason}")
# candidates: store(score, tie, repo_obj, eligible, stats, reason)
HeapItem: TypeAlias = tuple[float, float, dict[str, Any], list[dict[str, Any]], RepoStats, str]
# candidates: store(score, repo_obj, eligible, stats, reason)
Item: TypeAlias = tuple[float, dict[str, Any], list[dict[str, Any]], RepoStats, str]
def scan(args: argparse.Namespace, rng: random.Random) -> list[list[Item]]:
    buckets = parse_buckets(args.buckets)
    max_single_bytes = args.max_single_mb * 1024 ** 2
    
    kwargs = {
        "split": args.split,
        "streaming": True
    }
    if args.token:
        kwargs["token"] = args.token
    elif os.environ.get("HF_TOKEN"):
        kwargs["token"] = os.environ.get("HF_TOKEN")
    print("Loading dataset")
    dataset = load_dataset("bigcode/the-stack-v2-train-full-ids", **kwargs).shuffle(seed=args.seed, buffer_size=50000) #type:ignore
    # per bucket min-heap of candidates: store(score, tie, repo_obj, eligible, stats, reason)
    heaps: list[list[HeapItem]] = [
        [] for _ in buckets
    ]
    print("=== SCAN (shortlist candidates) ===")
    scanned = 0
    for repo in tqdm(dataset, total=args.scan_max):
        if scanned % 10000 == 0:
            sizes = [len(h) for h in heaps]
            print(f"scanned={scanned} | candidates_per_bucket={sizes}")
        repo = cast(dict[str, Any], repo)
        scanned += 1
        if scanned > args.scan_max:
            break
        repo_name = repo.get("repo_name", "")
        files = repo.get("files", [])
        if not repo_name or not files: continue
        eligible = [f for f in files if is_text_candidates(f, args.max_file_kb, args.max_line_length, args.min_alphanum)]
        if not eligible: continue
        stats = repo_stats(repo, eligible)
        # --- [CHÈN ĐOẠN NÀY ĐỂ DEBUG] ---
        # Chỉ in ra lý do loại bỏ của 10 repo đầu tiên
        if scanned <= 10: 
            print_debug(args, stats, repo)
        est: int = stats.total_bytes_est
        bi = bucket_index(est, buckets)
        score, reason = score_repo(stats)
        skip = (
            stats.name_is_junk 
            or stats.core_py_ratio < args.min_core_py_ratio 
            or stats.core_py_files < args.min_core_py_files
            or est <= 0 or est > max_single_bytes
            or bi is None
            or score < args.score_threshold
        )
        # if (stats.core_py_ratio > 0.0):
        #     print(stats.core_py_ratio)
        # if (score >= args.score_threshold and stats.core_py_files >= args.min_core_py_files):
        # #     print(score, stats.name_is_junk, stats.core_py_files, stats.core_py_files, est, bi)
        #     print_debug(args, stats, repo)
        if skip: continue
        bi = cast(int, bi)
        bucket_heap = heaps[bi]
        # Push into heap
        tie = rng.random()
        heapq.heappush(bucket_heap, (score, tie, repo, eligible, stats, reason))
        if len(bucket_heap) > args.candidates_per_bucket:
            heapq.heappop(bucket_heap)
            
        # early stop if all heaps are “full”
        if all(len(h) >= args.candidates_per_bucket for h in heaps):
            break
    # Convert heap to sorted desc lists
    shortlists: list[list[Item]] = []
    for heap in heaps:
        items = [(s, r, el, st, rsn) for (s, _t, r, el, st, rsn) in heap]
        items.sort(key=lambda x: x[0], reverse=True)
        shortlists.append(items)
    return shortlists
def sanitize_dirname(name: str) -> str:
    return (re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") or "repo")[:180]
def safe_relpath(p: str) -> str:
    p = p.replace("\\", "/").lstrip("/")
    return re.sub(r"(^|/)\.\.(?=/|$)", r"\1__UP__", p)
async def download_one(client: Any, file: dict[str, Any], out_repo_dir: Path, max_actual_kb: int, retries: int, back_off: float) -> tuple[int, int]:
    blob_id = file.get("blob_id")
    rel_path = file.get("path")
    if not blob_id or not rel_path:
        return 0, 0
    bucket = "softwareheritage"
    key = f"content/{blob_id}"
    out_path = out_repo_dir / safe_relpath(rel_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        try:
            obj = await client.get_object(Bucket=bucket, Key=key)
            async with obj["Body"] as stream:
                raw = await stream.read()
            raw = gzip.decompress(raw)
            if not raw:
                return 0, 0
            if len(raw) > max_actual_kb * 1024:
                return 0, 0
            if b"\x00" in raw:
                return 0, 0
            enc = file.get("src_encoding", 'utf-8')
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                # skip noisy encodings
                return 0, 0
            out_path.write_text(text, encoding="utf-8", errors="replace")
            return len(raw), 1
        except Exception:
            # import traceback
            # traceback.print_exc()
            await asyncio.sleep(back_off)
    return 0, 0
def make_s3_client(threads: int) -> Any:
    cfg = Config(
        signature_version=UNSIGNED,
        retries={"max_attempts": 10, "mode": "adaptive"},
        max_pool_connections=max(32, threads * 4),
        connect_timeout=10,
        read_timeout=90,
    )

    session = aioboto3.Session()

    # NOTE: do NOT "await" here — this returns an async context manager
    return session.client(
        "s3",
        region_name="us-east-1",
        config=cfg,
    )
async def download_repo_task(eligible: list[dict[str, Any]], repo_dir: Path, max_inflight: int, timeout: float, max_actual_kb: int, retries: int, backoff: float) -> tuple[int, int]:
    limiter = asyncio.Semaphore(max_inflight)
    s3_client_cm = make_s3_client(16)
    async with s3_client_cm as client:
        async def task_job(file: dict[str, Any]):
            await limiter.acquire()
            try:
                return await download_one(
                    client,
                    file,
                    repo_dir,
                    max_actual_kb,
                    retries,
                    backoff
                )
            finally:
                limiter.release()
        got_bytes = 0
        got_count = 0
        tasks = []
        for file in eligible:
            tasks.append(task_job(file))
        result = await tqdm_asyncio.gather(*tasks)
        for byte_, count_ in result:
            got_bytes += byte_
            got_count += count_
    return got_bytes, got_count
def postcheck_repo(repo_dir: Path, min_py_bytes: int, min_py_ratio: float) -> bool:
    """
    Sanity check on actual downloaded bytes.
    """
    total = 0
    py = 0
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        total += sz
        if p.name.endswith(".py") or p.name.endswith(".pyi"):
            py += sz
    if py < min_py_bytes:
        return False
    ratio = (py / total) if total > 0 else 0.0
    return ratio >= min_py_ratio
async def download(args: argparse.Namespace, rng: random.Random, shorlists: list[list[Item]]):
    out_root = Path(args.out_dir)
    total_bytes = 0
    repo_count = 0
    buckets = parse_buckets(args.buckets)
    bucket_bytes = [0] * len(buckets)
    target_bytes = args.target_total_mb * 1024 ** 2
    post_min_py_bytes = args.post_min_py_mb * 1024 ** 2
    max_repos_per_owner = args.max_repos_per_owner
    topk: int = args.topk_random_pick
    owner_counts: dict[str, int] = {}
    print("=== DOWNLOAD (bucketed + diversity + bounded futures + postcheck) ===")
    for bi, (lo, hi, quota) in enumerate(buckets):
        if total_bytes > target_bytes:
            break
        candidates = shorlists[bi]   
        while candidates and bucket_bytes[bi] < quota and total_bytes < target_bytes:
            # Pick random among top k 
            k = min(topk, len(candidates))
            pick_idx = rng.randrange(k)
            score, repo, eligible, stats, reason = candidates.pop(pick_idx)
            repo_name = repo.get("repo_name", "unknown")
            owner = stats.owner
            if owner_counts.get(owner, 0) >= max_repos_per_owner:
                continue
            est = stats.total_bytes_est
            if total_bytes + est > target_bytes + (5 * 1024 ** 2):
                continue
            repo_dir = out_root / sanitize_dirname(repo_name)
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)
            repo_dir.mkdir(parents=True, exist_ok=True)
            print(f"--> PICK b{bi} [{score:.1f}] {repo_name} | est={est/1024**2:.2f}MB | {reason}")
            got_b, got_f = await download_repo_task(
                eligible=eligible,
                repo_dir=repo_dir,
                max_inflight=args.max_inflight_files,
                max_actual_kb=args.max_actual_kb,
                retries=args.retries,
                timeout=args.file_timeout,
                backoff=args.backoff_s
            )
            if got_f == 0:
                shutil.rmtree(repo_dir, ignore_errors=True)
                continue
            # postcheck on actual bytes
            ok = postcheck_repo(repo_dir, min_py_bytes=post_min_py_bytes, min_py_ratio=args.post_min_py_ratio)
            if not ok:
                shutil.rmtree(repo_dir, ignore_errors=True)
                print(f"    DROP (postcheck failed): {repo_name}")
                continue
            # save small meta
            (repo_dir / "_meta.txt").write_text(
                f"repo={repo_name}\nscore={score:.3f}\nreason={reason}\n"
                f"got_mb={got_b/1024**2:.3f}\ncore_py_ratio_est={stats.core_py_ratio:.3f}\n",
                encoding="utf-8",
            )
            repo_count += 1
            total_bytes += got_b
            bucket_bytes[bi] += got_b
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
            print(f"    DONE: got={got_b/1024**2:.2f}MB files={got_f} | total={total_bytes/1024**2:.2f}MB")

async def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    shortlists = scan(args, rng)
    await download(args, rng, shortlists)
    
if __name__ == "__main__":
    asyncio.run(main())