"""
Task Planner Agent - Core orchestration module.
Responsible for: task analysis, decomposition, tool selection, execution monitoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class TaskType(Enum):
    """Supported document processing task types."""
    DOCUMENT_PARSE = "document_parse"
    TABLE_EXTRACT = "table_extract"
    CHART_ANALYSIS = "chart_analysis"
    FINANCIAL_REPORT = "financial_report"
    ENGINEERING_DRAWING = "engineering_drawing"
    LOW_QUALITY_OCR = "low_quality_ocr"
    CROSS_PAGE_MERGE = "cross_page_merge"
    STRUCTURED_EXPORT = "structured_export"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class SubTask:
    """A single step in the execution plan."""
    task_id: str
    task_type: TaskType
    description: str
    tool_name: str
    params: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task."""
    task_id: str
    original_request: str
    subtasks: list[SubTask]
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING


class TaskPlanner:
    """
    Central planner agent that:
    1. Analyzes incoming task requests
    2. Determines document type and processing difficulty
    3. Creates multi-step execution plans
    4. Selects appropriate tools for each step
    5. Monitors execution and handles failures
    """

    def __init__(self, llm_client=None, config: dict | None = None):
        self.llm = llm_client
        self.config = config or {}
        self._tool_registry: dict[str, Any] = {}

    def register_tool(self, name: str, tool_instance: Any) -> None:
        """Register a processing tool."""
        self._tool_registry[name] = tool_instance
        logger.info(f"Registered tool: {name}")

    async def analyze_task(self, request: str, file_info: dict | None = None) -> dict:
        """
        Analyze the task request and return assessment.

        Args:
            request: Natural language task description
            file_info: File metadata (name, type, size, pages, etc.)

        Returns:
            Task assessment with type, difficulty, estimated steps
        """
        prompt = f"""Analyze this document processing task and return a JSON assessment.

Task: {request}
File Info: {json.dumps(file_info or {}, ensure_ascii=False)}

Return JSON with:
- task_types: list of TaskType values needed
- difficulty: easy/medium/hard/extreme
- key_challenges: list of specific challenges
- recommended_tools: list of tool names
- estimated_subtasks: number
- preprocessing_needed: list of preprocessing steps
"""
        if self.llm:
            response = await self.llm.generate(prompt)
            return json.loads(response)
        else:
            # Fallback: basic heuristic analysis
            return self._heuristic_analysis(request, file_info)

    async def create_plan(self, request: str, file_info: dict | None = None) -> ExecutionPlan:
        """
        Create a complete execution plan for the task.

        Pipeline:
        1. Preprocess (image enhancement, format conversion)
        2. Extract (MinerU parse, OCR, layout analysis)
        3. Structure (table build, chart parse, cross-page merge)
        4. Verify (consistency check, LLM validation)
        5. Export (format output, generate report)
        """
        assessment = await self.analyze_task(request, file_info)

        subtasks = []
        task_id = f"task_{int(time.time())}"

        # Step 1: Preprocessing
        if assessment.get("preprocessing_needed"):
            subtasks.append(SubTask(
                task_id=f"{task_id}_preprocess",
                task_type=TaskType.DOCUMENT_PARSE,
                description="Preprocess document: image enhancement, format normalization",
                tool_name="preprocessor",
                params={"steps": assessment["preprocessing_needed"]},
            ))

        # Step 2: Core extraction with MinerU
        subtasks.append(SubTask(
            task_id=f"{task_id}_extract",
            task_type=TaskType.DOCUMENT_PARSE,
            description="Extract content using MinerU pipeline",
            tool_name="mineru_parser",
            params={"file_info": file_info},
            dependencies=[st.task_id for st in subtasks],
        ))

        # Step 3: Specialized processing based on task type
        for task_type in assessment.get("task_types", []):
            if task_type in ("table_extract", "financial_report"):
                subtasks.append(SubTask(
                    task_id=f"{task_id}_table_{len(subtasks)}",
                    task_type=TaskType.TABLE_EXTRACT,
                    description="Extract and structure tables",
                    tool_name="table_parser",
                    params={},
                    dependencies=[st.task_id for st in subtasks],
                ))
            elif task_type == "chart_analysis":
                subtasks.append(SubTask(
                    task_id=f"{task_id}_chart_{len(subtasks)}",
                    task_type=TaskType.CHART_ANALYSIS,
                    description="Analyze charts and extract data",
                    tool_name="chart_parser",
                    params={},
                    dependencies=[st.task_id for st in subtasks],
                ))

        # Step 4: Cross-page merge if needed
        if file_info and file_info.get("pages", 1) > 1:
            subtasks.append(SubTask(
                task_id=f"{task_id}_merge",
                task_type=TaskType.CROSS_PAGE_MERGE,
                description="Merge cross-page content and resolve references",
                tool_name="cross_page_merger",
                params={},
                dependencies=[st.task_id for st in subtasks],
            ))

        # Step 5: Verification
        subtasks.append(SubTask(
            task_id=f"{task_id}_verify",
            task_type=TaskType.STRUCTURED_EXPORT,
            description="Verify output quality and consistency",
            tool_name="verifier",
            params={},
            dependencies=[st.task_id for st in subtasks],
        ))

        plan = ExecutionPlan(
            task_id=task_id,
            original_request=request,
            subtasks=subtasks,
        )
        logger.info(f"Created plan {task_id} with {len(subtasks)} subtasks")
        return plan

    async def execute_plan(self, plan: ExecutionPlan) -> dict:
        """Execute an execution plan step by step."""
        plan.status = TaskStatus.RUNNING
        results = {}

        for subtask in plan.subtasks:
            # Wait for dependencies
            for dep_id in subtask.dependencies:
                dep = next((s for s in plan.subtasks if s.task_id == dep_id), None)
                if dep and dep.status != TaskStatus.COMPLETED:
                    subtask.status = TaskStatus.FAILED
                    subtask.error = f"Dependency {dep_id} not completed"
                    continue

            subtask.status = TaskStatus.RUNNING
            subtask.started_at = time.time()

            try:
                tool = self._tool_registry.get(subtask.tool_name)
                if not tool:
                    raise ValueError(f"Tool not found: {subtask.tool_name}")

                result = await tool.execute(subtask.params, context=results)
                subtask.result = result
                subtask.status = TaskStatus.COMPLETED
                results[subtask.task_id] = result
                logger.info(f"Completed subtask {subtask.task_id}")

            except Exception as e:
                subtask.error = str(e)
                subtask.status = TaskStatus.FAILED
                logger.error(f"Subtask {subtask.task_id} failed: {e}")
                # TODO: retry logic

            finally:
                subtask.completed_at = time.time()

        plan.status = TaskStatus.COMPLETED
        return results

    def _heuristic_analysis(self, request: str, file_info: dict | None) -> dict:
        """Fallback heuristic analysis when LLM is unavailable."""
        task_types = ["document_parse"]
        preprocessing = []

        request_lower = request.lower()
        if any(kw in request_lower for kw in ("表格", "报表", "财务", "table", "financial")):
            task_types.append("table_extract")
        if any(kw in request_lower for kw in ("图表", "chart", "图", "graph")):
            task_types.append("chart_analysis")
        if any(kw in request_lower for kw in ("模糊", "拍照", "手写", "blurry", "handwritten")):
            preprocessing.extend(["denoise", "enhance", "deskew"])

        return {
            "task_types": task_types,
            "difficulty": "medium",
            "key_challenges": [],
            "recommended_tools": ["mineru_parser"],
            "estimated_subtasks": len(task_types) + 2,
            "preprocessing_needed": preprocessing,
        }
