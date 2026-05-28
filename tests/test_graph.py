"""
Tests for the LangGraph Agent StateGraph.
Tests: graph compilation, analyze_task heuristic, plan_execution, routing, error handling.
"""

import pytest

from src.agents.graph import (
    AgentState,
    _heuristic_analyze,
    _merge_structured_content,
    route_after_execute,
    route_after_error,
    route_after_verify,
    create_agent_graph,
    MAX_RETRIES,
    QUALITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Heuristic analysis
# ---------------------------------------------------------------------------

class TestHeuristicAnalyze:
    """Test the fallback heuristic analysis function."""

    def test_basic_parse(self):
        result = _heuristic_analyze("Parse this document", {}, [])
        assert "document_parse" in result["task_types"]

    def test_financial_detection(self):
        result = _heuristic_analyze("解析财务报表", {}, ["pdf"])
        assert "table_extract" in result["task_types"]
        assert "financial_report" in result["task_types"]

    def test_chart_detection(self):
        result = _heuristic_analyze("Extract charts and graphs", {}, [])
        assert "chart_analysis" in result["task_types"]

    def test_low_quality_detection(self):
        result = _heuristic_analyze("OCR blurry scan", {}, ["image"])
        assert any("denoise" in p for p in result.get("preprocessing_needed", []))

    def test_multi_page_triggers_merge(self):
        result = _heuristic_analyze(
            "Parse document",
            {"pages": 20},
            ["pdf"],
        )
        assert "cross_page_merge" in result["task_types"]

    def test_single_page_no_merge(self):
        result = _heuristic_analyze(
            "Parse document",
            {"pages": 1},
            ["pdf"],
        )
        assert "cross_page_merge" not in result["task_types"]

    def test_difficulty_escalates_with_keywords(self):
        easy = _heuristic_analyze("Parse document", {}, [])
        hard = _heuristic_analyze("Parse financial report", {}, [])
        # Financial keywords should make it at least "hard"
        assert hard["difficulty"] in ("hard", "extreme")


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

class TestRouting:
    """Test conditional routing logic."""

    def test_route_after_execute_error(self):
        state = {"status": "error"}
        assert route_after_execute(state) == "error_handler"

    def test_route_after_execute_more_steps(self):
        state = {
            "status": "executing",
            "execution_plan": [{"step_id": "s1"}, {"step_id": "s2"}],
            "current_step_index": 0,
        }
        assert route_after_execute(state) == "execute_step"

    def test_route_after_execute_all_done(self):
        state = {
            "status": "executing",
            "execution_plan": [{"step_id": "s1"}],
            "current_step_index": 1,
        }
        assert route_after_execute(state) == "verify_result"

    def test_route_after_execute_no_plan(self):
        state = {
            "status": "executing",
            "execution_plan": None,
            "current_step_index": 0,
        }
        assert route_after_execute(state) == "verify_result"

    def test_route_after_error_failed(self):
        state = {"status": "failed"}
        assert route_after_error(state) == "format_output"

    def test_route_after_error_retrying(self):
        state = {"status": "retrying"}
        assert route_after_error(state) == "execute_step"

    def test_route_after_error_executing(self):
        state = {"status": "executing"}
        assert route_after_error(state) == "execute_step"

    def test_route_after_verify_passed(self):
        state = {
            "verification_result": {"passed": True, "_replan_count": 0},
        }
        assert route_after_verify(state) == "format_output"

    def test_route_after_verify_failed_can_replan(self):
        state = {
            "verification_result": {
                "passed": False,
                "_replan_count": 0,
                "retry_steps": ["step_001"],
            },
            "execution_plan": [
                {"step_id": "step_000", "status": "completed"},
                {"step_id": "step_001", "status": "completed"},
            ],
        }
        assert route_after_verify(state) == "execute_step"

    def test_route_after_verify_max_replan(self):
        state = {
            "verification_result": {
                "passed": False,
                "_replan_count": 2,
                "retry_steps": ["step_001"],
            },
        }
        assert route_after_verify(state) == "format_output"


# ---------------------------------------------------------------------------
# _merge_structured_content
# ---------------------------------------------------------------------------

class TestMergeStructuredContent:
    """Test the structured content merging function."""

    def test_empty_inputs(self):
        result = _merge_structured_content([], None)
        assert result["tables"] == []
        assert result["images"] == []

    def test_merges_raw_content(self):
        raw = {
            "content_list": [{"type": "text", "text": "hello"}],
            "tables": [{"table_index": 0}],
            "images": [],
            "markdown": "# Hello",
            "metadata": {"parser": "mineru"},
        }
        result = _merge_structured_content([], raw)
        assert result["markdown"] == "# Hello"
        assert len(result["tables"]) == 1

    def test_deduplicates_tables(self):
        step_results = [
            {"tables": [{"table_index": 0}, {"table_index": 1}]},
            {"tables": [{"table_index": 1}, {"table_index": 2}]},
        ]
        result = _merge_structured_content(step_results, None)
        indices = [t["table_index"] for t in result["tables"]]
        assert sorted(indices) == [0, 1, 2]

    def test_merges_charts(self):
        step_results = [{"charts": [{"type": "bar"}]}]
        result = _merge_structured_content(step_results, None)
        assert len(result["charts"]) == 1


# ---------------------------------------------------------------------------
# Graph creation and basic execution
# ---------------------------------------------------------------------------

class TestGraphCreation:
    """Test graph factory and compilation. All tests use mock tools only."""

    def test_creates_graph_with_tools(self):
        graph = create_agent_graph(tool_registry=self._make_mock_registry())
        assert graph is not None

    def test_creates_graph_without_llm(self):
        graph = create_agent_graph(
            tool_registry=self._make_mock_registry(), llm_client=None
        )
        assert graph is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_analyze_node_heuristic(self):
        """Test that analyze_task works without LLM (heuristic fallback).
        Run with: python -m pytest tests/ -m integration"""
        graph = create_agent_graph(tool_registry=self._make_mock_registry())

        result = await graph.ainvoke({
            "task_id": "test_analyze_001",
            "request": "解析财务报表，提取所有表格数据",
            "file_path": None,
            "file_info": {"name": "report.pdf", "suffix": ".pdf", "pages": 10},
        })

        assert result.get("assessment") is not None
        assert "task_types" in result["assessment"]
        assert "document_parse" in result["assessment"]["task_types"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_dry_run(self):
        """Test complete pipeline with no file (dry-run mode).
        Run with: python -m pytest tests/ -m integration"""
        graph = create_agent_graph(tool_registry=self._make_mock_registry())

        result = await graph.ainvoke({
            "task_id": "test_dry_run_001",
            "request": "Parse document and extract tables",
            "file_path": None,
            "file_info": {"name": "sample.pdf", "suffix": ".pdf", "pages": 1},
        })

        assert result.get("final_output") is not None
        assert result.get("status") in ("completed", "completed_with_errors")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_plan_creation(self):
        """Test that plan_execution creates a valid execution plan.
        Run with: python -m pytest tests/ -m integration"""
        graph = create_agent_graph(tool_registry=self._make_mock_registry())

        result = await graph.ainvoke({
            "task_id": "test_plan_001",
            "request": "解析财务报表并验证数值一致性",
            "file_path": None,
            "file_info": {"name": "finance.pdf", "suffix": ".pdf", "pages": 30},
        })

        plan = result.get("execution_plan")
        assert plan is not None
        assert len(plan) >= 2
        tool_names = [s["tool_name"] for s in plan]
        assert "mineru_parser" in tool_names
        assert "table_parser" in tool_names

    def test_quality_threshold_defaults(self):
        """Verify default constants are sensible."""
        assert 0.5 <= QUALITY_THRESHOLD <= 1.0
        assert MAX_RETRIES >= 1

    @staticmethod
    def _make_mock_registry() -> dict:
        """Create a lightweight mock tool registry — no real model loading."""
        from unittest.mock import AsyncMock

        default_result = {
            "source_file": "", "pages": 1, "content_list": [],
            "markdown": "# Mock", "tables": [], "images": [],
            "metadata": {"parser": "mock"},
        }
        return {
            "mineru_parser": AsyncMock(execute=AsyncMock(return_value=dict(default_result))),
            "table_parser": AsyncMock(execute=AsyncMock(return_value={
                **default_result, "tables": [{"table_index": 0}],
            })),
            "chart_analyzer": AsyncMock(execute=AsyncMock(return_value=dict(default_result))),
            "cross_page_merger": AsyncMock(execute=AsyncMock(return_value=dict(default_result))),
            "image_enhancer": AsyncMock(execute=AsyncMock(return_value={
                **default_result, "enhanced_image_bytes": b"",
            })),
            "verifier": AsyncMock(execute=AsyncMock(return_value={
                "quality_score": 0.85, "passed": True, "issues": [],
            })),
            "exporter": AsyncMock(execute=AsyncMock(return_value={
                "format": "json", "exported": True, "result_count": 1,
            })),
        }
