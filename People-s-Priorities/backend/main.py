from __future__ import annotations

import math
import mimetypes
import json
import os
from threading import Lock
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ai_engine import (
    analyze_submission,
    check_deduplication,
    fetch_demographics_from_bigquery,
    formalize_submission,
)

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"
SUBMISSIONS_FILE = BASE_DIR / "data" / "submissions.json"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────

ADMIN_PASSKEY = "MP-2026"

WORKFLOW_STATUSES = {
    "Noticed by Government",
    "Under Process",
    "Work Started",
    "Work Done",
}

SUBMISSION_DEFAULTS: Dict[str, Any] = {
    "status": "Noticed by Government",
    "mp_explanation": "",
    "citizen_review": None,
    "is_archived": False,
    "linked_reports": [],
    "report_count": 1,
    "triage_category": "quick_fix",
    "photo_url": None,
    "formal_description": "",
    "raw_description": "",
    "detected_language": "en",
    "profanity_detected": False,
}

# Logarithmic cluster boost coefficient.
# Each natural-log unit of report_count growth adds this many urgency points.
# With CLUSTER_BOOST_FACTOR = 1.5:
#   1 report  → +0.00   5 reports → +2.42
#   2 reports → +1.04  10 reports → +3.45
#   3 reports → +1.65  20 reports → +4.35
CLUSTER_BOOST_FACTOR: float = 1.5


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extension_for(content_type: str, filename: Optional[str] = None) -> str:
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix[:16]
    return mimetypes.guess_extension((content_type or "").split(";")[0]) or ".bin"


def _with_submission_defaults(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backfills any missing fields on a record.
    Handles both V2 records (new format) and V1 legacy records
    that predate the clustering and triage features.
    """
    record.setdefault("status", SUBMISSION_DEFAULTS["status"])
    if record.get("status") == "new":
        record["status"] = SUBMISSION_DEFAULTS["status"]
    record.setdefault("mp_explanation", SUBMISSION_DEFAULTS["mp_explanation"])
    record.setdefault("citizen_review", SUBMISSION_DEFAULTS["citizen_review"])
    record.setdefault("is_archived", SUBMISSION_DEFAULTS["is_archived"])
    record.setdefault("linked_reports", list(SUBMISSION_DEFAULTS["linked_reports"]))
    record.setdefault("report_count", SUBMISSION_DEFAULTS["report_count"])
    record.setdefault("triage_category", SUBMISSION_DEFAULTS["triage_category"])
    record.setdefault("photo_url", SUBMISSION_DEFAULTS["photo_url"])
    # Back-fill formal_description from text for V1 legacy records
    if not record.get("formal_description"):
        record["formal_description"] = record.get("text", "")
    record.setdefault("raw_description", record.get("text", ""))
    record.setdefault("detected_language", SUBMISSION_DEFAULTS["detected_language"])
    record.setdefault("profanity_detected", SUBMISSION_DEFAULTS["profanity_detected"])
    return record


def _is_active_submission(record: Dict[str, Any]) -> bool:
    return not bool(record.get("is_archived"))


def _log_cluster_boost(report_count: int) -> float:
    """
    Returns the logarithmic urgency boost for a clustered issue.

    Formula: boost = ln(report_count) × CLUSTER_BOOST_FACTOR

    Properties:
    - Single reports receive zero boost (ln(1) = 0)
    - Boost grows sub-linearly: doubling the report count does NOT double the boost
    - This prevents a single highly-reported trivial issue from dominating the priority list
    - The caller enforces a hard cap of 10.0 on the resulting demand_score

    See CLUSTER_BOOST_FACTOR constant for per-count example values.
    """
    n = max(1, _parse_int(report_count, 1))
    return math.log(n) * CLUSTER_BOOST_FACTOR


# ─── Storage Backend ──────────────────────────────────────────────────────────

class StorageBackend:
    """
    Offline-first JSON file storage for hackathon demos.
    All operations are thread-safe via a threading.Lock.
    No cloud database, no network dependency.
    """

    def __init__(self, path: Path = SUBMISSIONS_FILE) -> None:
        self.path = path
        self._lock = Lock()

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read_rows(self) -> List[Dict[str, Any]]:
        self._ensure_file()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        return payload if isinstance(payload, list) else []

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        self._ensure_file()
        self.path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")

    async def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        def _append() -> Dict[str, Any]:
            with self._lock:
                record["id"] = record.get("id") or str(uuid.uuid4())
                _with_submission_defaults(record)
                rows = self._read_rows()
                rows.append(_json_safe(record))
                self._write_rows(rows)
            return record

        await run_in_threadpool(_append)
        return record

    async def load(self, limit: int = 250) -> List[Dict[str, Any]]:
        def _load() -> List[Dict[str, Any]]:
            with self._lock:
                rows = self._read_rows()
            rows = [_with_submission_defaults(row) for row in rows]
            rows.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
            return rows[:limit]

        return await run_in_threadpool(_load)

    async def update(self, submission_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        def _update() -> Optional[Dict[str, Any]]:
            with self._lock:
                rows = self._read_rows()
                for row in rows:
                    if str(row.get("id")) != submission_id:
                        continue
                    _with_submission_defaults(row)
                    row.update(updates)
                    # Auto-archive when Work Done + citizen review submitted
                    if row.get("status") == "Work Done" and row.get("citizen_review") not in (None, ""):
                        row["is_archived"] = True
                    row["updated_at"] = _now_iso()
                    _with_submission_defaults(row)
                    self._write_rows(rows)
                    return row
            return None

        return await run_in_threadpool(_update)

    async def link_report(
        self,
        master_id: str,
        linked_entry: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Append a linked citizen report entry to an existing master issue.

        - Appends the linked_entry to linked_reports[]
        - Increments report_count = 1 + len(linked_reports)
        - Persists and returns the updated master record
        - Returns None if master_id is not found (caller should create a new issue)
        """
        def _link() -> Optional[Dict[str, Any]]:
            with self._lock:
                rows = self._read_rows()
                for row in rows:
                    if str(row.get("id")) != master_id:
                        continue
                    _with_submission_defaults(row)
                    row.setdefault("linked_reports", [])
                    row["linked_reports"].append(_json_safe(linked_entry))
                    row["report_count"] = 1 + len(row["linked_reports"])
                    row["updated_at"] = _now_iso()
                    self._write_rows(rows)
                    return row
            return None

        return await run_in_threadpool(_link)


storage_backend = StorageBackend()


# ─── Media Handling ───────────────────────────────────────────────────────────

def _local_media_record(media: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe media metadata, stripping raw bytes before saving."""
    return {k: v for k, v in media.items() if k != "bytes"}


async def save_bytes_to_uploads(
    content: bytes,
    content_type: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save raw bytes to backend/static/uploads/ and return metadata.
    The returned `url` field (/static/uploads/<file>) is directly browser-accessible
    via the /static StaticFiles mount.
    """
    if not content:
        raise HTTPException(status_code=400, detail="Cannot upload an empty media file.")

    extension = _extension_for(content_type, filename)
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / unique_filename
    await run_in_threadpool(destination.write_bytes, content)

    relative_path = (Path("static") / "uploads" / unique_filename).as_posix()
    browser_url = f"/static/uploads/{unique_filename}"
    return {
        "bytes": content,
        "mime_type": content_type or "application/octet-stream",
        "filename": unique_filename,
        "original_filename": filename,
        "path": relative_path,          # relative to BASE_DIR, for local disk ops
        "url": browser_url,             # browser-accessible URL via /static mount
    }


async def save_upload_file(upload: UploadFile) -> Optional[Dict[str, Any]]:
    if upload is None or not upload.filename:
        return None
    content = await upload.read()
    content_type = upload.content_type or "application/octet-stream"
    return await save_bytes_to_uploads(content, content_type, upload.filename)


async def download_external_media_to_local(media: Dict[str, Any]) -> Dict[str, Any]:
    url = media.get("url")
    provider = media.get("provider")
    if not url and media.get("id") and provider == "meta":
        url = await resolve_meta_media_url(str(media["id"]))
    if not url:
        raise HTTPException(status_code=400, detail="Webhook media needs a url or resolvable Meta media id.")

    headers: Dict[str, str] = {}
    auth = None
    parsed = urlparse(url)

    if "api.twilio.com" in parsed.netloc:
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        if sid and token:
            auth = (sid, token)

    if provider == "meta":
        token = os.getenv("META_WHATSAPP_ACCESS_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, auth=auth, headers=headers)
        response.raise_for_status()
        content_type = (
            media.get("mime_type")
            or response.headers.get("content-type")
            or "application/octet-stream"
        )
        fname = Path(parsed.path).name or None
        return await save_bytes_to_uploads(response.content, content_type, fname)


async def resolve_meta_media_url(media_id: str) -> str:
    token = os.getenv("META_WHATSAPP_ACCESS_TOKEN")
    if not token:
        raise HTTPException(status_code=400, detail="Set META_WHATSAPP_ACCESS_TOKEN to resolve Meta media ids.")
    graph_version = os.getenv("META_GRAPH_VERSION", "v21.0")
    url = f"https://graph.facebook.com/{graph_version}/{media_id}"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        payload = response.json()
        media_url = payload.get("url")
        if not media_url:
            raise HTTPException(status_code=400, detail="Meta media response did not include a download URL.")
        return media_url


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="People's Priorities API", version="2.0.0")

allowed_origins = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_dashboard_user() -> Dict[str, Any]:
    """
    Offline demo auth shim.
    Cloud auth is intentionally bypassed for the hackathon demo so billing or
    network issues cannot block the MP dashboard.
    """
    return {"uid": "local-dev"}


def public_dashboard_config() -> Dict[str, Any]:
    return {
        "mapsApiKey": os.getenv("GOOGLE_MAPS_API_KEY", ""),
        "requireAuth": False,
        "defaultMapCenter": {
            "lat": _env_float("DEFAULT_MAP_LAT", 28.6139),
            "lng": _env_float("DEFAULT_MAP_LNG", 77.2090),
        },
        "heatmapRadius": _parse_int(os.getenv("HEATMAP_RADIUS"), 38),
    }


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_HTML)


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    return public_dashboard_config()


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "storage_file": str(storage_backend.path),
        "upload_dir": str(UPLOAD_DIR),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "cluster_boost_factor": CLUSTER_BOOST_FACTOR,
        "version": "2.0.0",
    }


# ─── V2.0 Three-Step Intake Pipeline ─────────────────────────────────────────

async def analyze_and_store(
    *,
    channel: str,
    raw_text: str,
    media_refs: List[Dict[str, Any]],
    address: Optional[str] = None,
    sender: Optional[str] = None,
    raw_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    V2.0 three-step intake pipeline for every new citizen submission.

    ┌─ Step 1: Formalize ──────────────────────────────────────────────────────┐
    │  Translate regional languages → English                                  │
    │  Remove profanity, scrub slang → formal professional English             │
    │  Always succeeds (graceful fallback on Gemini error)                     │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Step 2: Deduplicate ────────────────────────────────────────────────────┐
    │  Compare against up to 20 active issues via Gemini                       │
    │  Confidence gate: only merge if confidence >= 0.85                       │
    │  If duplicate → link_report() on master, return merged response          │
    │  Always succeeds (falls back to creating new issue on error)             │
    └──────────────────────────────────────────────────────────────────────────┘
    ┌─ Step 3: Analyze ────────────────────────────────────────────────────────┐
    │  Full AI analysis: category, triage_category, urgency_score, etc.        │
    │  Only runs for unique (non-duplicate) new issues                         │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    citizen_id = str(uuid.uuid4())

    # ── Step 1: Formalize ──────────────────────────────────────────────────────
    formalization = await run_in_threadpool(formalize_submission, raw_text, media_refs)
    formal_description = formalization.formal_description

    # ── Step 2: Deduplicate ────────────────────────────────────────────────────
    all_rows = await storage_backend.load(limit=10000)
    active_rows = [r for r in all_rows if _is_active_submission(r)]

    dup_result = await run_in_threadpool(
        check_deduplication,
        formal_description,
        address or "",
        active_rows,
    )

    if dup_result.is_duplicate and dup_result.master_id:
        linked_entry: Dict[str, Any] = {
            "citizen_id": citizen_id,
            "submitted_at": _now_iso(),
            "address": address or "",
            "raw_description": raw_text,
            "channel": channel,
        }
        master = await storage_backend.link_report(dup_result.master_id, linked_entry)
        if master:
            return {
                "merged": True,
                "id": citizen_id,            # Citizen's personal tracking ID
                "master_id": dup_result.master_id,
                "master": master,
                "message": (
                    "Your report has been linked to an existing active issue. "
                    "Use your unique ID below to track the shared status and MP updates."
                ),
                "deduplication_confidence": dup_result.confidence,
            }
        # Race condition: master was archived between load and link — fall through to new issue

    # ── Step 3: Full Analysis (unique new issue) ───────────────────────────────
    demographics = await run_in_threadpool(
        lambda: fetch_demographics_from_bigquery(constituency_id=os.getenv("CONSTITUENCY_ID"))
    )
    analysis = await run_in_threadpool(
        analyze_submission,
        formal_description,
        media_refs,
        demographics.get("rows", []),
    )

    # Derive browser-accessible photo URL from the first uploaded media ref
    photo_url: Optional[str] = None
    for m in media_refs:
        candidate = m.get("url") or m.get("path", "")
        if candidate:
            photo_url = candidate if candidate.startswith("/") else f"/{candidate}"
            break

    record: Dict[str, Any] = {
        "id": citizen_id,
        "channel": channel,
        "sender": sender,
        # V2 description fields
        "raw_description": raw_text,
        "formal_description": formal_description,
        "detected_language": formalization.detected_language,
        "profanity_detected": formalization.profanity_detected,
        "text": formal_description,       # backwards-compat alias for V1 code paths
        # Location
        "address": address,
        # Media
        "photo_url": photo_url,
        "media": [_local_media_record(m) for m in media_refs],
        # AI analysis
        "analysis": analysis,
        "category": analysis.get("category"),
        "triage_category": analysis.get("triage_category", "quick_fix"),
        "urgency_score": analysis.get("urgency_score"),
        # Clustering
        "linked_reports": [],
        "report_count": 1,
        # Workflow
        "status": SUBMISSION_DEFAULTS["status"],
        "mp_explanation": SUBMISSION_DEFAULTS["mp_explanation"],
        "citizen_review": SUBMISSION_DEFAULTS["citizen_review"],
        "is_archived": SUBMISSION_DEFAULTS["is_archived"],
        # Metadata
        "demographic_source": demographics.get("source"),
        "created_at": _now_iso(),
        "raw_metadata": raw_metadata or {},
    }
    return await storage_backend.append(record)


# ─── Submission Endpoints ─────────────────────────────────────────────────────

@app.post("/api/submissions")
@app.post("/api/submit")
async def create_submission(
    request: Request,
    text: str = Form(default=""),
    address: str = Form(default=""),
    photo: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    legacy_file: Optional[UploadFile] = File(default=None, alias="file"),
) -> Dict[str, Any]:
    media_refs: List[Dict[str, Any]] = []

    image_upload = photo or image or legacy_file
    uploaded_image = await save_upload_file(image_upload) if image_upload else None
    if uploaded_image:
        media_refs.append(uploaded_image)

    if not text.strip() and not media_refs:
        raise HTTPException(status_code=400, detail="Please describe the issue or attach a photo.")

    return await analyze_and_store(
        channel="Date of Submission Form",
        raw_text=text.strip(),
        media_refs=media_refs,
        address=address.strip() or None,
        raw_metadata={"client_host": request.client.host if request.client else None},
    )


@app.get("/api/submissions")
async def list_submissions(
    limit: int = Query(default=250, ge=1, le=1000),
    user: Dict[str, Any] = Depends(require_dashboard_user),
) -> Dict[str, Any]:
    del user
    rows = await storage_backend.load(limit=10000)
    active_rows = [row for row in rows if _is_active_submission(row)]
    return {"items": active_rows[:limit]}


@app.get("/api/submissions/{submission_id}")
async def get_submission(submission_id: str) -> Dict[str, Any]:
    """
    Fetch a single submission by ID — handles both master IDs and linked citizen IDs.

    Resolution order:
    1. Exact master ID match → returns {item, is_linked: false}
    2. Linked citizen ID match (inside any master's linked_reports[]) →
       returns the master issue with {is_linked: true, master_id, linked_entry}
       so the citizen sees live MP status and explanation.
    3. Archived issues are still returned (citizens should be able to track closed reports).
    """
    rows = await storage_backend.load(limit=10000)

    # 1. Check master IDs
    for row in rows:
        if str(row.get("id")) == submission_id:
            return {"item": row, "is_linked": False}

    # 2. Check linked citizen IDs within each master issue
    for row in rows:
        for linked in row.get("linked_reports", []):
            if str(linked.get("citizen_id")) == submission_id:
                return {
                    "item": row,
                    "is_linked": True,
                    "master_id": str(row.get("id")),
                    "linked_entry": linked,
                }

    raise HTTPException(status_code=404, detail="Submission not found.")


async def _read_update_payload(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    return dict(await request.form())


@app.post("/api/submissions/{submission_id}/mp_update")
async def mp_update_submission(
    submission_id: str,
    request: Request,
    x_admin_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """
    MP-only update endpoint. Requires x-admin-key: MP-2026 header.
    Updating a master issue automatically broadcasts to all linked citizens —
    they will see the updated status and mp_explanation when they track their ID.
    """
    if x_admin_key != ADMIN_PASSKEY:
        raise HTTPException(status_code=403, detail="Invalid or missing admin passkey.")

    payload = await _read_update_payload(request)
    updates: Dict[str, Any] = {}

    if "status" in payload and payload.get("status") not in (None, ""):
        status = str(payload["status"]).strip()
        if status not in WORKFLOW_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Use one of: {', '.join(sorted(WORKFLOW_STATUSES))}.",
            )
        updates["status"] = status

    if "mp_explanation" in payload:
        updates["mp_explanation"] = (
            "" if payload.get("mp_explanation") is None
            else str(payload["mp_explanation"]).strip()
        )

    if not updates:
        raise HTTPException(status_code=400, detail="Provide status or mp_explanation to update.")

    updated = await storage_backend.update(submission_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Submission not found.")

    return {"ok": True, "item": updated}


@app.post("/api/submissions/{submission_id}/citizen_review")
async def citizen_review_submission(
    submission_id: str,
    request: Request,
) -> Dict[str, Any]:
    payload = await _read_update_payload(request)
    raw_review = payload.get("citizen_review")
    if raw_review in (None, ""):
        raise HTTPException(status_code=400, detail="Provide citizen_review to submit.")

    updates = {"citizen_review": raw_review}
    updated = await storage_backend.update(submission_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Submission not found.")

    return {"ok": True, "item": updated}


# ─── Priority Ranking ─────────────────────────────────────────────────────────

def _demographic_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("ward_id", "")).lower(): row for row in rows if row.get("ward_id")}


@app.get("/api/priorities")
async def ranked_priorities(
    limit: int = Query(default=500, ge=1, le=1000),
    user: Dict[str, Any] = Depends(require_dashboard_user),
) -> Dict[str, Any]:
    """
    Returns ranked priority buckets (grouped by ward + civic category) with:

    SCORING FORMULA:
      base_urgency     = AI urgency score (1–10)
      cluster_boost    = ln(report_count) × CLUSTER_BOOST_FACTOR
      effective_urgency= min(base_urgency + cluster_boost, 10.0)   ← hard cap
      pop_weight       = min(population / 50,000, 4.0)
      vuln_weight      = min(vulnerability_index / 20, 4.0)
      demand_score     = min(effective_urgency + pop_weight + vuln_weight, 10.0)  ← hard cap

    Cluster boost is logarithmic so that:
    - A single-report issue receives zero boost
    - A 10-report clustered issue gets +3.45 urgency points
    - A 50-report clustered issue gets +5.77 urgency points
    - But no issue can exceed demand_score = 10.0

    Response includes:
    - items: all priority buckets sorted by demand_score desc
    - by_triage: same buckets grouped by triage_category for frontend tab rendering
    """
    del user
    submissions = [
        item
        for item in await storage_backend.load(limit=10000)
        if _is_active_submission(item)
    ][:limit]

    demographics = await run_in_threadpool(
        lambda: fetch_demographics_from_bigquery(constituency_id=os.getenv("CONSTITUENCY_ID"))
    )
    demo_by_ward = _demographic_index(demographics.get("rows", []))

    grouped: Dict[str, Dict[str, Any]] = {}
    triage_vote: Dict[str, Dict[str, int]] = {}  # key → {triage_category: vote_count}

    for item in submissions:
        ward_id = str(item.get("ward_id") or "unknown")
        category = str(
            item.get("category")
            or (item.get("analysis") or {}).get("category")
            or "other"
        )
        triage_category = str(item.get("triage_category") or "quick_fix")
        key = f"{ward_id}:{category}"

        base_urgency = _number(
            item.get("urgency_score") or (item.get("analysis") or {}).get("urgency_score"),
            default=1.0,
        )
        report_count = _parse_int(item.get("report_count"), 1)

        # Logarithmic cluster boost (capped so urgency never exceeds 10)
        cluster_boost = _log_cluster_boost(report_count)
        effective_urgency = min(base_urgency + cluster_boost, 10.0)

        demo = demo_by_ward.get(ward_id.lower(), {})
        population_weight = min(_number(demo.get("population")) / 50_000, 4.0)
        vulnerability_weight = min(_number(demo.get("vulnerability_index")) / 20.0, 4.0)

        # Total demand score hard-capped at 10.0
        demand_score = min(
            effective_urgency + population_weight + vulnerability_weight,
            10.0,
        )

        bucket = grouped.setdefault(
            key,
            {
                "ward_id": ward_id,
                "ward_name": demo.get("ward_name") or ward_id,
                "category": category,
                "triage_category": triage_category,
                "count": 0,             # distinct master issues in this ward:category bucket
                "total_reports": 0,     # sum of all report_counts (includes linked citizens)
                "max_urgency": 0.0,
                "demand_score": 0.0,
                "demographics": demo,
                "latest_summary": "",
            },
        )

        bucket["count"] += 1
        bucket["total_reports"] += report_count
        bucket["max_urgency"] = round(max(bucket["max_urgency"], effective_urgency), 1)
        bucket["demand_score"] = round(max(bucket["demand_score"], demand_score), 1)
        bucket["latest_summary"] = (
            (item.get("analysis") or {}).get("summary")
            or item.get("formal_description")
            or item.get("text")
            or ""
        )

        # Vote for dominant triage_category in this bucket
        votes = triage_vote.setdefault(key, {})
        votes[triage_category] = votes.get(triage_category, 0) + 1

    # Assign dominant triage_category to each bucket by plurality vote
    for key, bucket in grouped.items():
        votes = triage_vote.get(key, {})
        if votes:
            bucket["triage_category"] = max(votes, key=votes.__getitem__)

    priorities = sorted(grouped.values(), key=lambda r: r["demand_score"], reverse=True)

    # Build triage-grouped convenience dict for frontend tab rendering
    by_triage: Dict[str, List[Dict[str, Any]]] = {
        "critical_emergency": [],
        "quick_fix": [],
        "urgent_infrastructure": [],
        "long_term_planning": [],
    }
    for p in priorities:
        tc = p.get("triage_category", "quick_fix")
        by_triage.setdefault(tc, []).append(p)

    return {
        "items": priorities,
        "by_triage": by_triage,
        "demographics_source": demographics.get("source"),
        "demographics_query": demographics.get("query"),
    }


# ─── WhatsApp Webhook ─────────────────────────────────────────────────────────

async def _read_webhook_payload(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    form = await request.form()
    return dict(form)


def _extract_twilio_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    media: List[Dict[str, Any]] = []
    for index in range(_parse_int(payload.get("NumMedia"))):
        url = payload.get(f"MediaUrl{index}")
        mime_type = payload.get(f"MediaContentType{index}") or "application/octet-stream"
        if url and not mime_type.startswith("audio/"):
            media.append({"provider": "twilio", "url": url, "mime_type": mime_type})
    return {
        "sender": payload.get("From") or payload.get("WaId"),
        "text": payload.get("Body", ""),
        "address": payload.get("Address", ""),
        "media": media,
        "raw": payload,
    }


def _media_from_meta(message: Dict[str, Any], msg_type: str) -> Optional[Dict[str, Any]]:
    media_obj = message.get(msg_type) or {}
    if not media_obj:
        return None
    return {
        "provider": "meta",
        "id": media_obj.get("id"),
        "url": media_obj.get("url"),
        "mime_type": media_obj.get("mime_type", "application/octet-stream"),
    }


def _extract_meta_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                msg_type = message.get("type")
                text = ""
                media: List[Dict[str, Any]] = []
                if msg_type == "text":
                    text = (message.get("text") or {}).get("body", "")
                elif msg_type in {"image", "document"}:
                    media_item = _media_from_meta(message, msg_type)
                    if media_item:
                        media.append(media_item)
                location = message.get("location") or {}
                extracted.append({
                    "sender": message.get("from"),
                    "text": text,
                    "address": location.get("address") or location.get("name") or "",
                    "media": media,
                    "raw": message,
                })
    return extracted


def _extract_mock_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = (
        payload.get("messages")
        if isinstance(payload.get("messages"), list)
        else [payload]
    )
    extracted = []
    for message in messages:
        media = message.get("media", [])
        if isinstance(media, dict):
            media = [media]
        extracted.append({
            "sender": message.get("from") or message.get("sender"),
            "text": message.get("text") or message.get("body") or "",
            "address": message.get("address", ""),
            "media": media,
            "raw": message,
        })
    return extracted


def extract_whatsapp_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "Body" in payload or "NumMedia" in payload:
        return [_extract_twilio_message(payload)]
    if "entry" in payload:
        return _extract_meta_messages(payload)
    return _extract_mock_messages(payload)


async def load_existing_local_media(media: Dict[str, Any]) -> Dict[str, Any]:
    raw_path = Path(str(media["path"]))
    file_path = raw_path if raw_path.is_absolute() else BASE_DIR / raw_path
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"Local media path not found: {media['path']}")
    content = await run_in_threadpool(file_path.read_bytes)
    mime_type = (
        media.get("mime_type")
        or mimetypes.guess_type(file_path.name)[0]
        or "application/octet-stream"
    )
    try:
        relative_path = file_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        relative_path = file_path.name
    return {
        "bytes": content,
        "mime_type": mime_type,
        "filename": file_path.name,
        "original_filename": media.get("filename") or file_path.name,
        "path": relative_path,
        "url": f"/{relative_path}",
    }


async def _materialize_webhook_media(media_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for item in media_items:
        if item.get("path"):
            refs.append(await load_existing_local_media(item))
            continue
        refs.append(await download_external_media_to_local(item))
    return refs


@app.post("/api/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> Dict[str, Any]:
    payload = await _read_webhook_payload(request)
    messages = extract_whatsapp_messages(payload)
    saved: List[Dict[str, Any]] = []

    for message in messages:
        media_refs = await _materialize_webhook_media(message.get("media", []))
        if not str(message.get("text") or "").strip() and not media_refs:
            continue
        saved.append(
            await analyze_and_store(
                channel="whatsapp",
                raw_text=str(message.get("text") or "").strip(),
                media_refs=media_refs,
                address=str(message.get("address") or "").strip() or None,
                sender=message.get("sender"),
                raw_metadata=message.get("raw"),
            )
        )

    return {"ok": True, "saved": saved}


# ─── Static File Serving ──────────────────────────────────────────────────────

# /static → backend/static/ (serves uploaded images at /static/uploads/<file>)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# / → frontend/ (must be LAST — after all API routes and the /static mount)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ─── Dev Server Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    import uvicorn

    def _find_available_port(bind_host: str, preferred_port: int, attempts: int = 25) -> int:
        for candidate in range(preferred_port, preferred_port + attempts):
            family = (
                socket.AF_INET6
                if ":" in bind_host and bind_host != "0.0.0.0"
                else socket.AF_INET
            )
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((bind_host, candidate))
                    return candidate
                except OSError:
                    continue
        return preferred_port

    host = os.getenv("HOST", "0.0.0.0")
    requested_port = _parse_int(os.getenv("PORT"), 8000)
    port = _find_available_port(host, requested_port)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if port != requested_port:
        print(f"Port {requested_port} busy — using {port} instead.")
    print(f"People's Priorities V2.0 → http://{display_host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info"))
