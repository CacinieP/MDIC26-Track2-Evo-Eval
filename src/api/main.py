"""
FastAPI application - MinerU DataAgent API Server.
===================================================

Provides REST API for:
- Submitting document processing tasks (with file upload)
- Tracking task lifecycle and progress
- Retrieving structured results and execution logs
- Querying system capabilities

The API wires the LangGraph agent to HTTP endpoints with async background processing.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.api.task_store import TaskStore, TaskStatus, TaskRecord
from src.tools.mineru_parser import MinerUParser
from src.tools.table_parser import TableParser
from src.tools.crosspage_merger import CrossPageMerger
from src.tools.image_enhancer import ImageEnhancer
from src.tools.chart_analyzer import ChartAnalyzer
from src.agents.graph import create_agent_graph, _BuiltinVerifier, _BuiltinExporter
from src.utils.logger import setup_logging
from src.utils.config import load_config, get_default_config

# ---------------------------------------------------------------------------
# App & globals
# ---------------------------------------------------------------------------

VERSION = "1.0.0"

app = FastAPI(
    title="MinerU DataAgent API",
    description=(
        "Data Agent API - Intelligent document processing powered by MinerU\n\n"
        "Competition Track 2 | MinerU"
    ),
    version=VERSION,
)

# --- CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request logging middleware ---
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with method, path, status code, and duration."""
    start = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    logger.info(f"[{request_id}] --> {request.method} {request.url.path}")

    try:
        response: Response = await call_next(request)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(
            f"[{request_id}] <-- {request.method} {request.url.path} "
            f"500 ({elapsed:.3f}s) exception={exc}"
        )
        raise

    elapsed = time.perf_counter() - start
    logger.info(
        f"[{request_id}] <-- {request.method} {request.url.path} "
        f"{response.status_code} ({elapsed:.3f}s)"
    )
    return response


# Global stores and state — initialized in lifespan
task_store = TaskStore()
_agent_graph = None
_config: dict = {}
_temp_dir = Path("./data/temp")
_tools_loaded: list[str] = []

# Supported file types
SUPPORTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
    ".docx", ".pptx", ".html", ".htm",
}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Initialize tools, agent graph, and directories on startup."""
    global _agent_graph, _config, _temp_dir, _tools_loaded

    setup_logging({"level": "INFO"})

    _config = load_config()
    _temp_dir = Path(_config.get("storage", {}).get("temp_dir", "./data/temp"))
    _temp_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output dir exists too
    output_dir = Path(_config.get("storage", {}).get("output_dir", "./data/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize individual tool instances
    mineru_cfg = _config.get("mineru", {})
    tools = {
        "mineru_parser": MinerUParser(mineru_cfg),
        "table_parser": TableParser(),
        "chart_analyzer": ChartAnalyzer(),
        "image_enhancer": ImageEnhancer(),
        "cross_page_merger": CrossPageMerger(),
        "verifier": _BuiltinVerifier(),
        "exporter": _BuiltinExporter(),
    }
    _tools_loaded = sorted(tools.keys())

    try:
        _agent_graph = create_agent_graph(tool_registry=tools, config=_config.get("pipeline"))
        logger.info(f"API startup: agent graph ready with tools: {_tools_loaded}")
    except Exception as e:
        logger.error(f"API startup: failed to initialize agent graph: {e}")
        _agent_graph = None

    logger.info("MinerU DataAgent API server started")


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    tools_loaded: list[str] = Field(default_factory=list, description="Loaded tool names")
    agent_ready: bool = Field(False, description="Whether the agent graph is ready")
    active_tasks: int = Field(0, description="Number of active (non-terminal) tasks")


class TaskSubmitRequest(BaseModel):
    """Task submission request (JSON body, no file)."""
    task_description: str = Field(..., description="Natural language task description")
    file_url: str | None = Field(None, description="URL or path to the document file")
    options: dict[str, Any] = Field(default_factory=dict, description="Processing options")


class TaskResponse(BaseModel):
    """Task submission response."""
    task_id: str
    status: str
    message: str


class SubtaskDetail(BaseModel):
    """Details of a single subtask / execution step."""
    step_id: str = ""
    tool_name: str = ""
    description: str = ""
    status: str = "pending"
    retries: int = 0


class TaskStatusResponse(BaseModel):
    """Full task status with execution details."""
    task_id: str
    status: str
    progress: float
    file_name: str
    current_step: str
    total_steps: int
    completed_steps: int
    execution_plan: list[SubtaskDetail] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    logs: list[str] = []
    errors: list[str] = []
    duration: float | None = None


class TaskResultResponse(BaseModel):
    """Final result download response."""
    task_id: str
    status: str
    result: dict[str, Any] | None = None
    errors: list[str] = []
    duration: float | None = None


class TaskLogsResponse(BaseModel):
    """Execution logs response."""
    task_id: str
    total_logs: int
    logs: list[str]


class BatchResponse(BaseModel):
    """Batch submission response."""
    task_ids: list[str]
    status: str
    message: str


class TaskListResponse(BaseModel):
    """Task list response."""
    tasks: list[dict]
    total: int


class CapabilityDetail(BaseModel):
    """Single processing capability."""
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    scenarios: list[str] = Field(default_factory=list)
    types: list[str] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    """System capabilities response."""
    supported_file_types: list[str] = Field(default_factory=list)
    max_file_size_mb: int = 100
    processing_capabilities: dict[str, CapabilityDetail] = Field(default_factory=dict)
    output_formats: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str


# ---------------------------------------------------------------------------
# Background processor
# ---------------------------------------------------------------------------

async def _process_task(task_id: str, file_path: str, request: str, options: dict):
    """Background task: run the agent graph and update the task store."""
    global _agent_graph

    task_store.update_status(task_id, TaskStatus.PROCESSING)
    task_store.add_log(task_id, f"Processing started: {request[:120]}")

    if _agent_graph is None:
        task_store.set_failed(task_id, "Agent graph not initialized — server startup may have failed")
        return

    try:
        # Build file_info metadata
        file_info: dict[str, Any] = {}
        p = Path(file_path)
        if p.exists():
            file_info = {
                "name": p.name,
                "suffix": p.suffix.lower(),
                "size": p.stat().st_size,
            }
        elif file_path:
            file_info = {"name": file_path, "suffix": "", "size": 0}

        # Invoke the LangGraph agent
        result = await _agent_graph.ainvoke({
            "task_id": task_id,
            "request": request,
            "file_path": file_path,
            "file_info": file_info,
            "options": options,
        })

        # --- Extract data from the graph result ---
        final = result.get("final_output", {})
        exec_summary = final.get("execution_summary", [])
        verification = final.get("verification", {})
        all_logs = result.get("logs", [])
        all_errors = result.get("errors", [])
        assessment = final.get("assessment")
        structured = final.get("structured_content")

        # Update execution plan in task record
        plan_steps = []
        for step in exec_summary:
            plan_steps.append({
                "step_id": step.get("step_id", ""),
                "tool_name": step.get("tool_name", ""),
                "description": step.get("description", ""),
                "status": step.get("status", "unknown"),
                "retries": step.get("retries", 0),
            })
        task_store.set_execution_plan(task_id, plan_steps)

        if assessment:
            task_store.set_assessment(task_id, assessment)

        # Update progress
        completed = sum(1 for s in exec_summary if s.get("status") == "completed")
        progress = completed / max(len(exec_summary), 1)
        task_store.update_progress(
            task_id,
            progress=progress,
            total_steps=len(exec_summary),
            completed_steps=completed,
        )

        # Transfer logs
        for log in all_logs:
            task_store.add_log(task_id, log)
        for err in all_errors:
            task_store.add_error(task_id, err)

        # Store final result
        task_store.set_result(task_id, final)

        # Attach verification if present
        if verification:
            record = task_store.get_task(task_id)
            if record:
                record.verification = verification

        logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        task_store.set_failed(task_id, str(e))


# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------

def _validate_file_extension(filename: str) -> str:
    """Validate and return the file extension. Raises HTTPException on failure."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )
    return suffix


async def _save_upload_file(file: UploadFile, task_id: str, suffix: str) -> Path:
    """
    Read an UploadFile, validate size, save to _temp_dir, return the save path.
    Raises HTTPException on validation failure.
    """
    save_path = _temp_dir / f"{task_id}{suffix}"
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({len(content)} bytes). Max: {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )
        with open(save_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        task_store.set_failed(task_id, f"File save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    task_store.add_log(task_id, f"File uploaded: {file.filename} ({len(content)} bytes)")
    return save_path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    active = len([
        t for t in task_store.list_tasks(1000)
        if t.get("status") not in ("completed", "failed")
    ])
    return HealthResponse(
        status="healthy",
        version=VERSION,
        tools_loaded=_tools_loaded,
        agent_ready=_agent_graph is not None,
        active_tasks=active,
    )


@app.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    """List supported document types and processing capabilities."""
    return CapabilitiesResponse(
        supported_file_types=sorted(SUPPORTED_EXTENSIONS),
        max_file_size_mb=MAX_FILE_SIZE // (1024 * 1024),
        processing_capabilities={
            "document_parsing": CapabilityDetail(
                description="Document parsing: layout analysis + OCR + table recognition",
                tools=["MinerU", "PaddleOCR"],
            ),
            "table_extraction": CapabilityDetail(
                description="Table structuring: merged cell expansion + numeric validation + confidence",
                scenarios=["Financial statements", "Balance sheets", "Income statements"],
            ),
            "chart_analysis": CapabilityDetail(
                description="Chart analysis: classification + data extraction + description",
                types=["Bar", "Line", "Pie", "Scatter"],
            ),
            "cross_page_merge": CapabilityDetail(
                description="Cross-page merge: table continuation + paragraph rejoin + reference resolution",
            ),
            "image_enhancement": CapabilityDetail(
                description="Image enhancement: CLAHE + denoise + deskew + stamp removal",
                scenarios=["Blurry photos", "Scanned documents", "Handwritten stamps"],
            ),
            "reference_resolution": CapabilityDetail(
                description="Reference resolution: pronoun/abbreviation parsing + entity consistency",
            ),
        },
        output_formats=["json", "markdown", "csv"],
    )


@app.post("/tasks", response_model=TaskResponse)
async def submit_task(request: TaskSubmitRequest, background_tasks: BackgroundTasks):
    """
    Submit a document processing task (with file URL or path).
    """
    task_id = task_store.create_task(
        request=request.task_description,
        options=request.options,
    )

    file_path = request.file_url or ""
    if file_path and not Path(file_path).exists():
        task_store.set_failed(task_id, f"File not found: {file_path}")
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    background_tasks.add_task(
        _process_task, task_id, file_path, request.task_description, request.options
    )

    return TaskResponse(
        task_id=task_id,
        status="accepted",
        message=f"Task submitted. Use GET /tasks/{task_id} to check status.",
    )


@app.post("/tasks/upload", response_model=TaskResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    task_description: str = Form("Parse document and extract structured data"),
    options: str = Form("{}"),
):
    """
    Upload a document file for processing (multipart form).

    Supported formats: PDF, PNG/JPG/JPEG, DOCX, PPTX, HTML.
    Max file size: 100MB.
    """
    suffix = _validate_file_extension(file.filename or "unknown")

    # Parse options JSON
    try:
        opts = json.loads(options) if options else {}
    except json.JSONDecodeError:
        opts = {}

    task_id = task_store.create_task(
        request=task_description,
        file_name=file.filename or "unknown",
        options=opts,
    )

    save_path = await _save_upload_file(file, task_id, suffix)

    background_tasks.add_task(
        _process_task, task_id, str(save_path), task_description, opts
    )

    return TaskResponse(
        task_id=task_id,
        status="accepted",
        message=f"File '{file.filename}' uploaded. Use GET /tasks/{task_id} to check status.",
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Query task execution status and details with all subtask info.
    """
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    # Build subtask details from the execution plan
    subtasks = []
    for step in record.execution_plan:
        subtasks.append(SubtaskDetail(
            step_id=step.get("step_id", ""),
            tool_name=step.get("tool_name", ""),
            description=step.get("description", ""),
            status=step.get("status", "pending"),
            retries=step.get("retries", 0),
        ))

    status_value = record.status.value if isinstance(record.status, TaskStatus) else record.status

    return TaskStatusResponse(
        task_id=record.task_id,
        status=status_value,
        progress=record.progress,
        file_name=record.file_name,
        current_step=record.current_step,
        total_steps=record.total_steps,
        completed_steps=record.completed_steps,
        execution_plan=subtasks,
        result=record.result,
        verification=record.verification,
        logs=record.logs[-100:],
        errors=record.errors,
        duration=record.duration,
    )


@app.get("/tasks/{task_id}/result", response_model=TaskResultResponse)
async def get_task_result(task_id: str):
    """
    Get the final structured result of a completed task.
    """
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    if record.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Task '{task_id}' is still "
                f"{record.status.value if isinstance(record.status, TaskStatus) else record.status}. "
                f"Use GET /tasks/{task_id} for status."
            ),
        )

    status_value = record.status.value if isinstance(record.status, TaskStatus) else record.status

    return TaskResultResponse(
        task_id=task_id,
        status=status_value,
        result=record.result,
        errors=record.errors,
        duration=record.duration,
    )


@app.get("/tasks/{task_id}/logs", response_model=TaskLogsResponse)
async def get_task_logs(task_id: str, limit: int = 200):
    """
    Get execution logs for a task.

    Returns recent log entries. For live streaming, use the streaming variant.
    """
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    return TaskLogsResponse(
        task_id=task_id,
        total_logs=len(record.logs),
        logs=record.logs[-limit:],
    )


@app.get("/tasks/{task_id}/logs/stream")
async def stream_task_logs(task_id: str):
    """
    Stream execution logs as NDJSON (Newline-Delimited JSON).

    Each line is a JSON object: {"ts": "...", "message": "..."}
    The stream stays open until the task reaches a terminal state (completed/failed)
    plus a short tail to deliver final logs.
    """
    record = task_store.get_task(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    async def log_stream() -> AsyncIterator[str]:
        sent = 0
        idle_rounds = 0
        max_idle = 30  # ~30 seconds of idle before closing stream

        while True:
            rec = task_store.get_task(task_id)
            if rec is None:
                yield json.dumps({"error": "task not found"}) + "\n"
                return

            new_logs = rec.logs[sent:]
            for log_entry in new_logs:
                yield json.dumps({"ts": time.strftime("%H:%M:%S"), "message": log_entry}) + "\n"
                sent += 1

            is_terminal = rec.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

            if is_terminal and not new_logs:
                # Deliver any errors as final log lines
                for err in rec.errors:
                    yield json.dumps({"ts": time.strftime("%H:%M:%S"), "message": f"ERROR: {err}", "level": "error"}) + "\n"
                yield json.dumps({
                    "ts": time.strftime("%H:%M:%S"),
                    "message": f"Task {task_id} finished with status: {rec.status.value}",
                    "level": "done",
                }) + "\n"
                return

            if new_logs:
                idle_rounds = 0
            else:
                idle_rounds += 1
                if idle_rounds >= max_idle:
                    yield json.dumps({"ts": time.strftime("%H:%M:%S"), "message": "stream timeout", "level": "timeout"}) + "\n"
                    return

            await asyncio.sleep(1)

    return StreamingResponse(
        log_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/tasks")
async def list_tasks(limit: int = 20, offset: int = 0):
    """
    List all tasks, newest first.
    """
    tasks = task_store.list_tasks(limit=limit, offset=offset)
    return TaskListResponse(
        tasks=tasks,
        total=task_store.count(),
    )


@app.post("/tasks/batch", response_model=BatchResponse)
async def batch_submit(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Multiple document files"),
    task_description: str = Form("Batch parse documents and extract structured data"),
    options: str = Form("{}"),
):
    """
    Batch submit multiple files for processing.
    """
    try:
        opts = json.loads(options) if options else {}
    except json.JSONDecodeError:
        opts = {}

    task_ids: list[str] = []
    rejected: list[str] = []

    for file in files:
        filename = file.filename or "unknown"
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            rejected.append(f"{filename}: unsupported type '{suffix}'")
            continue

        task_id = task_store.create_task(
            request=task_description,
            file_name=filename,
            options=opts,
        )

        save_path = _temp_dir / f"{task_id}{suffix}"
        try:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                task_store.set_failed(task_id, f"File too large: {len(content)} bytes")
                rejected.append(f"{filename}: exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit")
                continue
            with open(save_path, "wb") as f:
                f.write(content)
            task_store.add_log(task_id, f"File uploaded: {filename} ({len(content)} bytes)")
        except Exception as e:
            task_store.set_failed(task_id, str(e))
            rejected.append(f"{filename}: save failed ({e})")
            continue

        background_tasks.add_task(
            _process_task, task_id, str(save_path), task_description, opts
        )
        task_ids.append(task_id)

    msg = f"{len(task_ids)} file(s) submitted for processing."
    if rejected:
        msg += f" {len(rejected)} rejected: {'; '.join(rejected[:3])}"

    return BatchResponse(
        task_ids=task_ids,
        status="accepted",
        message=msg,
    )


# ---------------------------------------------------------------------------
# Custom exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return JSON error responses for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
