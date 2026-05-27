"""
In-memory task store for tracking task lifecycle.

Thread-safe dict-based store that tracks:
- Task submission, progress, completion
- Per-task logs and results
- Subtask execution details
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING = "processing"


@dataclass
class TaskRecord:
    """Complete record for a single processing task."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    request: str = ""
    file_name: str = ""
    file_path: str = ""
    options: dict = field(default_factory=dict)

    # Progress
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0

    # Results
    result: dict | None = None
    assessment: dict | None = None
    execution_plan: list[dict] = field(default_factory=list)
    verification: dict | None = None

    # Logs
    logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Timing
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at and self.completed_at:
            return round(self.completed_at - self.started_at, 2)
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        d["duration"] = self.duration
        return d


class TaskStore:
    """
    Thread-safe in-memory task store.

    Usage::

        store = TaskStore()
        task_id = store.create_task("Parse financial report", "report.pdf")
        store.update_status(task_id, TaskStatus.PROCESSING)
        store.add_log(task_id, "Started parsing...")
        record = store.get_task(task_id)
    """

    def __init__(self, max_tasks: int = 1000):
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()
        self._max_tasks = max_tasks

    @staticmethod
    def _touch(task: TaskRecord) -> None:
        """Update the updated_at timestamp on a task record."""
        task.updated_at = time.time()

    def create_task(
        self,
        request: str = "",
        file_name: str = "",
        file_path: str = "",
        options: dict | None = None,
    ) -> str:
        """Create a new task and return its ID."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        record = TaskRecord(
            task_id=task_id,
            request=request,
            file_name=file_name,
            file_path=file_path,
            options=options or {},
        )
        with self._lock:
            # Evict oldest if at capacity
            if len(self._tasks) >= self._max_tasks:
                oldest_id = min(self._tasks, key=lambda k: self._tasks[k].created_at)
                del self._tasks[oldest_id]
            self._tasks[task_id] = record
        return task_id

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Get a task record by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                if status == TaskStatus.PROCESSING and task.started_at is None:
                    task.started_at = time.time()
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    task.completed_at = time.time()
                self._touch(task)

    def update_progress(
        self,
        task_id: str,
        progress: float,
        current_step: str = "",
        completed_steps: int = 0,
        total_steps: int = 0,
    ) -> None:
        """Update task progress."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.progress = progress
                task.current_step = current_step
                task.completed_steps = completed_steps
                task.total_steps = total_steps
                self._touch(task)

    def set_result(self, task_id: str, result: dict) -> None:
        """Set the final result for a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.result = result
                task.completed_at = time.time()
                task.status = TaskStatus.COMPLETED
                self._touch(task)

    def set_assessment(self, task_id: str, assessment: dict) -> None:
        """Set task assessment."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.assessment = assessment
                self._touch(task)

    def set_execution_plan(self, task_id: str, plan: list[dict]) -> None:
        """Set execution plan."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.execution_plan = plan
                task.total_steps = len(plan)
                self._touch(task)

    def add_log(self, task_id: str, message: str) -> None:
        """Append a log entry to a task."""
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.logs.append(entry)
                self._touch(task)

    def add_error(self, task_id: str, error: str) -> None:
        """Append an error to a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.errors.append(error)
                self._touch(task)

    def set_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.errors.append(error)
                self._touch(task)

    def list_tasks(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List all tasks, newest first."""
        with self._lock:
            tasks = sorted(
                self._tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
            return [t.to_dict() for t in tasks[offset : offset + limit]]

    def count(self) -> int:
        """Return the total number of tasks in the store."""
        with self._lock:
            return len(self._tasks)

    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Remove tasks older than max_age_seconds. Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = [
                tid
                for tid, task in self._tasks.items()
                if task.completed_at and (now - task.completed_at) > max_age_seconds
            ]
            for tid in to_remove:
                del self._tasks[tid]
                removed += 1
        return removed
