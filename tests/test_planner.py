"""
Tests for the Task Planner.
Tests: heuristic analysis, plan generation, task type detection.
"""

import pytest

from src.agents.planner import TaskPlanner, TaskType, TaskStatus, SubTask, ExecutionPlan


# ---------------------------------------------------------------------------
# TaskPlanner — heuristic analysis
# ---------------------------------------------------------------------------

class TestHeuristicAnalysis:
    """Test keyword-based heuristic task analysis."""

    def setup_method(self):
        self.planner = TaskPlanner(llm_client=None)

    @pytest.mark.asyncio
    async def test_basic_document_parse(self):
        result = await self.planner.analyze_task("Parse this document")
        assert "document_parse" in result["task_types"]
        assert "difficulty" in result

    @pytest.mark.asyncio
    async def test_financial_keyword_detection(self):
        result = await self.planner.analyze_task(
            "解析财务报表中的资产负债表数据",
            file_info={"name": "report.pdf"},
        )
        assert "table_extract" in result["task_types"]

    @pytest.mark.asyncio
    async def test_chart_keyword_detection(self):
        result = await self.planner.analyze_task(
            "Extract chart data from this research paper",
        )
        assert "chart_analysis" in result["task_types"]

    @pytest.mark.asyncio
    async def test_low_quality_keyword_detection(self):
        result = await self.planner.analyze_task(
            "OCR this blurry scanned contract",
        )
        assert any("denoise" in p or "enhance" in p
                    for p in result.get("preprocessing_needed", []))

    @pytest.mark.asyncio
    async def test_empty_request(self):
        result = await self.planner.analyze_task("")
        assert "task_types" in result
        assert isinstance(result["task_types"], list)


# ---------------------------------------------------------------------------
# Execution Plan creation
# ---------------------------------------------------------------------------

class TestPlanCreation:
    """Test multi-step execution plan generation."""

    def setup_method(self):
        self.planner = TaskPlanner(llm_client=None)

    @pytest.mark.asyncio
    async def test_basic_plan_has_extract_step(self):
        plan = await self.planner.create_plan("Parse this PDF")
        assert isinstance(plan, ExecutionPlan)
        tool_names = [st.tool_name for st in plan.subtasks]
        assert "mineru_parser" in tool_names

    @pytest.mark.asyncio
    async def test_financial_plan_has_table_step(self):
        plan = await self.planner.create_plan(
            "解析财务报表",
            file_info={"name": "report.pdf", "pages": 30},
        )
        tool_names = [st.tool_name for st in plan.subtasks]
        assert "table_parser" in tool_names

    @pytest.mark.asyncio
    async def test_multi_page_adds_cross_page_merge(self):
        plan = await self.planner.create_plan(
            "Parse this document",
            file_info={"name": "long.pdf", "pages": 50},
        )
        tool_names = [st.tool_name for st in plan.subtasks]
        assert "cross_page_merger" in tool_names

    @pytest.mark.asyncio
    async def test_single_page_no_cross_page(self):
        plan = await self.planner.create_plan(
            "Parse this document",
            file_info={"name": "short.pdf", "pages": 1},
        )
        tool_names = [st.tool_name for st in plan.subtasks]
        assert "cross_page_merger" not in tool_names

    @pytest.mark.asyncio
    async def test_plan_ends_with_verify(self):
        plan = await self.planner.create_plan("Parse document")
        last_tool = plan.subtasks[-1].tool_name
        assert last_tool == "verifier"

    @pytest.mark.asyncio
    async def test_subtask_dependencies_chain(self):
        plan = await self.planner.create_plan(
            "Extract financial tables",
            file_info={"name": "report.pdf", "pages": 10},
        )
        # Each subtask should depend on previous ones
        for i in range(1, len(plan.subtasks)):
            assert len(plan.subtasks[i].dependencies) > 0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TestDataModels:
    """Test SubTask and ExecutionPlan data models."""

    def test_subtask_duration_none_when_not_started(self):
        st = SubTask(
            task_id="t1",
            task_type=TaskType.DOCUMENT_PARSE,
            description="test",
            tool_name="mineru_parser",
            params={},
        )
        assert st.duration is None

    def test_subtask_duration_when_completed(self):
        st = SubTask(
            task_id="t1",
            task_type=TaskType.DOCUMENT_PARSE,
            description="test",
            tool_name="mineru_parser",
            params={},
            started_at=100.0,
            completed_at=105.5,
        )
        assert st.duration == 5.5

    def test_execution_plan_status_default(self):
        plan = ExecutionPlan(
            task_id="plan_1",
            original_request="test",
            subtasks=[],
        )
        assert plan.status == TaskStatus.PENDING

    def test_task_type_values(self):
        assert TaskType.DOCUMENT_PARSE.value == "document_parse"
        assert TaskType.TABLE_EXTRACT.value == "table_extract"
        assert TaskType.CHART_ANALYSIS.value == "chart_analysis"
        assert TaskType.FINANCIAL_REPORT.value == "financial_report"
