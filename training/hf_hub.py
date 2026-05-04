"""Hugging Face Hub upload helpers.

Lazy-imports `huggingface_hub` so this module is importable in environments
where the package isn't installed; callers either gate on `available()` or
let `_api()` raise a clear ImportError pointing at the install command.
"""
import time
from pathlib import Path


_HF_API = None
_HF_AVAILABLE = None


def available():
    global _HF_AVAILABLE
    if _HF_AVAILABLE is None:
        try:
            import huggingface_hub  # noqa: F401
            _HF_AVAILABLE = True
        except ImportError:
            _HF_AVAILABLE = False
    return _HF_AVAILABLE


def _api():
    global _HF_API
    if _HF_API is None:
        if not available():
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )
        from huggingface_hub import HfApi
        _HF_API = HfApi()
    return _HF_API


def verify_repo(repo_id, private=False, repo_type="model"):
    """Confirm we can write to ``repo_id``; create it if missing.

    Run this once before kicking off a long training campaign so an
    auth/typo failure surfaces in seconds rather than after hours of compute.
    """
    from huggingface_hub import create_repo

    api = _api()
    info = api.whoami()
    user = info.get("name") or info.get("fullname") or str(info)
    print(f"[hf] authed as: {user}")

    create_repo(repo_id, private=private, repo_type=repo_type, exist_ok=True)
    api.repo_info(repo_id, repo_type=repo_type)
    visibility = "private" if private else "public"
    print(f"[hf] repo OK: {repo_id} ({visibility})")
    return f"https://huggingface.co/{repo_id}"


def upload_checkpoint(local_path, repo_id, path_in_repo,
                      repo_type="model", retries=3, verbose=True):
    """Upload ``local_path`` to ``repo_id`` at ``path_in_repo``.

    Retries with exponential backoff on transient failures. HF stores blobs
    by content hash, so re-uploading an identical checkpoint is a no-op.
    """
    api = _api()
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {local_path}")

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            url = api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"upload {path_in_repo}",
            )
            if verbose:
                size_mb = local_path.stat().st_size / 1e6
                dt = time.time() - t0
                print(f"      [hf-upload] {path_in_repo}  "
                      f"({size_mb:.1f} MB, {dt:.1f}s)  -> {url}")
            return url
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"      [hf-upload] attempt {attempt}/{retries} failed: {e}; "
                  f"sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"HF upload failed after {retries} attempts: {last_exc}")
