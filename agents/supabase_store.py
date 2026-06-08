import httpx
from supabase import create_client
from app.config import Config

_write = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)
_read  = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
BUCKET = Config.SUPABASE_BUCKET


def upload_doc(content: str, version: int) -> str:
    """Upload markdown docs to Supabase Storage. Each version in its own folder."""
    path = f"v{version}/DOCUMENTATION.md"
    try:
        _write.storage.from_(BUCKET).upload(
            path=path,
            file=content.encode("utf-8"),
            file_options={"content-type": "text/markdown", "upsert": "true"}
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase Storage while uploading documentation. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc
    return path


def fetch_latest_doc() -> tuple[str, int]:
    """Get the most recent documentation. Returns (content, version_number)."""
    try:
        row = _read.table("doc_versions").select("version, storage_path").order("version", desc=True).limit(1).execute()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while loading documentation versions. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc
    if not row.data:
        return "# No documentation yet.", 0
    latest = row.data[0]
    try:
        content = _write.storage.from_(BUCKET).download(latest["storage_path"]).decode("utf-8")
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase Storage while downloading documentation. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc
    return content, latest["version"]


def save_version(version: int, storage_path: str, commit_sha: str, trigger: str) -> int:
    """Insert version metadata. Returns new row id."""
    try:
        resp = _write.table("doc_versions").insert({
            "version": version, "storage_path": storage_path,
            "commit_sha": commit_sha, "trigger_source": trigger
        }).execute()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while saving documentation metadata. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc
    return resp.data[0]["id"]


def save_agent_log(version_id: int, agent_name: str, iteration: int, message: str):
    """Save one agent thinking step."""
    try:
        _write.table("agent_logs").insert({
            "version_id": version_id, "agent_name": agent_name,
            "iteration": iteration, "message": message
        }).execute()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while saving agent logs. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc


def get_all_versions() -> list[dict]:
    try:
        return _read.table("doc_versions").select("id, version, created_at, commit_sha, trigger_source").order("version", desc=True).execute().data
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while loading documentation versions. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc


def get_version_doc(version: int) -> str:
    try:
        resp = _read.table("doc_versions").select("storage_path").eq("version", version).single().execute()
        return _write.storage.from_(BUCKET).download(resp.data["storage_path"]).decode("utf-8")
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while loading documentation content. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc


def get_agent_logs(version_id: int) -> list[dict]:
    try:
        return _read.table("agent_logs").select("agent_name, iteration, message, created_at").eq("version_id", version_id).order("created_at").execute().data
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while loading agent logs. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc