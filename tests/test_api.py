"""
Tests for the FastAPI application and TaskStore.
Tests: API endpoints, task lifecycle, file validation, TaskStore operations.
"""

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# TaskStore unit tests (no HTTP needed)
# ---------------------------------------------------------------------------

class TestTaskStore:
    """Tests for the in-memory TaskStore."""

    def test_create_task(self):
        from src.api.task_store import TaskStore, TaskStatus

        store = TaskStore()
        task_id = store.create_task(request="Parse doc", file_name="test.pdf")
        assert task_id.startswith("task_")

        record = store.get_task(task_id)
        assert record is not None
        assert record.request == "Parse doc"
        assert record.file_name == "test.pdf"
        assert record.status == TaskStatus.PENDING

    def test_task_lifecycle(self):
        from src.api.task_store import TaskStore, TaskStatus

        store = TaskStore()
        task_id = store.create_task("Test task")

        store.update_status(task_id, TaskStatus.PROCESSING)
        record = store.get_task(task_id)
        assert record.status == TaskStatus.PROCESSING
        assert record.started_at is not None

        store.update_status(task_id, TaskStatus.COMPLETED)
        record = store.get_task(task_id)
        assert record.status == TaskStatus.COMPLETED
        assert record.completed_at is not None

    def test_task_logs(self):
        from src.api.task_store import TaskStore

        store = TaskStore()
        task_id = store.create_task()
        store.add_log(task_id, "Step 1 completed")
        store.add_log(task_id, "Step 2 completed")

        record = store.get_task(task_id)
        assert len(record.logs) == 2

    def test_task_errors(self):
        from src.api.task_store import TaskStore, TaskStatus

        store = TaskStore()
        task_id = store.create_task()
        store.add_error(task_id, "Something went wrong")

        record = store.get_task(task_id)
        assert len(record.errors) == 1

    def test_set_failed(self):
        from src.api.task_store import TaskStore, TaskStatus

        store = TaskStore()
        task_id = store.create_task()
        store.set_failed(task_id, "Parse error")

        record = store.get_task(task_id)
        assert record.status == TaskStatus.FAILED
        assert "Parse error" in record.errors

    def test_set_result(self):
        from src.api.task_store import TaskStore, TaskStatus

        store = TaskStore()
        task_id = store.create_task()
        result = {"status": "completed", "pages": 42}
        store.set_result(task_id, result)

        record = store.get_task(task_id)
        assert record.status == TaskStatus.COMPLETED
        assert record.result["pages"] == 42

    def test_execution_plan(self):
        from src.api.task_store import TaskStore

        store = TaskStore()
        task_id = store.create_task()
        plan = [
            {"step_id": "s1", "tool_name": "mineru_parser", "status": "completed"},
            {"step_id": "s2", "tool_name": "verifier", "status": "pending"},
        ]
        store.set_execution_plan(task_id, plan)

        record = store.get_task(task_id)
        assert len(record.execution_plan) == 2
        assert record.total_steps == 2

    def test_list_tasks(self):
        from src.api.task_store import TaskStore

        store = TaskStore()
        ids = [store.create_task(f"Task {i}") for i in range(5)]
        tasks = store.list_tasks(limit=3)
        assert len(tasks) == 3

    def test_get_nonexistent_task(self):
        from src.api.task_store import TaskStore

        store = TaskStore()
        assert store.get_task("nonexistent") is None

    def test_cleanup_old_tasks(self):
        from src.api.task_store import TaskStore, TaskStatus
        import time

        store = TaskStore(max_tasks=5)

        # Create and complete a task
        old_id = store.create_task("Old task")
        store.update_status(old_id, TaskStatus.COMPLETED)
        # Manually set completed_at to past
        store._tasks[old_id].completed_at = time.time() - 7200  # 2 hours ago

        removed = store.cleanup(max_age_seconds=3600)
        assert removed == 1
        assert store.get_task(old_id) is None

    def test_max_tasks_eviction(self):
        from src.api.task_store import TaskStore

        store = TaskStore(max_tasks=3)
        ids = [store.create_task(f"Task {i}") for i in range(5)]
        tasks = store.list_tasks(limit=100)
        assert len(tasks) == 3  # Only 3 should survive

    def test_task_to_dict(self):
        from src.api.task_store import TaskStore

        store = TaskStore()
        task_id = store.create_task("Test")
        record = store.get_task(task_id)
        d = record.to_dict()

        assert isinstance(d, dict)
        assert d["task_id"] == task_id
        assert "status" in d
        assert "duration" in d

    def test_task_duration(self):
        from src.api.task_store import TaskStore
        import time

        store = TaskStore()
        task_id = store.create_task()
        record = store.get_task(task_id)
        record.started_at = 100.0
        record.completed_at = 105.5
        assert record.duration == 5.5


# ---------------------------------------------------------------------------
# API endpoint tests (using TestClient)
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    """Tests for FastAPI HTTP endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client (does NOT start MinerU models)."""
        from src.api.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_capabilities(self, client):
        resp = client.get("/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "supported_file_types" in data
        assert ".pdf" in data["supported_file_types"]

    def test_list_tasks_empty(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data

    def test_get_nonexistent_task(self, client):
        resp = client.get("/tasks/nonexistent_task_id")
        assert resp.status_code == 404

    def test_submit_task_no_file(self, client):
        resp = client.post(
            "/tasks",
            json={
                "task_description": "Parse test document",
                "file_url": "/nonexistent/path.pdf",
            },
        )
        # Should fail because file doesn't exist
        assert resp.status_code in (400, 500)

    def test_upload_unsupported_type(self, client):
        resp = client.post(
            "/tasks/upload",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
            data={"task_description": "Test"},
        )
        assert resp.status_code == 400

    def test_task_logs_nonexistent(self, client):
        resp = client.get("/tasks/nonexistent/logs")
        assert resp.status_code == 404

    def test_task_result_nonexistent(self, client):
        resp = client.get("/tasks/nonexistent/result")
        assert resp.status_code == 404
