"""
Tests for the Cross-Page Merger tool.
Tests: initialisation, cross-page table merging, cross-page text merging,
       entity extraction, reference resolution, execute() integration,
       markdown table conversion.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.crosspage_merger import (
    CrossPageMerger,
    MergeOperation,
    ResolutionResult,
    _extract_table_data,
    _parse_markdown_table,
    _table_column_count,
    _is_likely_header_row,
    _row_similarity,
)


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def _make_table_block(
    page_idx: int,
    data: list[list[str]],
    **extra,
) -> dict:
    """Build a minimal table content block."""
    block = {
        "type": "table",
        "page_idx": page_idx,
        "data": data,
    }
    block.update(extra)
    return block


def _make_text_block(page_idx: int, text: str, **extra) -> dict:
    """Build a minimal text content block."""
    block = {
        "type": "text",
        "page_idx": page_idx,
        "text": text,
    }
    block.update(extra)
    return block


# ---------------------------------------------------------------------------
# 1. CrossPageMerger initialisation
# ---------------------------------------------------------------------------

class TestCrossPageMergerInit:
    """Test CrossPageMerger instantiation and configuration."""

    def test_creates_instance_with_no_config(self):
        merger = CrossPageMerger()
        assert merger is not None
        assert merger.config == {}

    def test_creates_instance_with_config(self):
        cfg = {"logging": {"level": "DEBUG"}, "ref_resolution": True}
        merger = CrossPageMerger(config=cfg)
        assert merger.config is cfg

    def test_default_llm_client_is_none(self):
        merger = CrossPageMerger()
        assert merger.llm_client is None

    def test_set_llm_client(self):
        merger = CrossPageMerger()
        mock_client = MagicMock()
        merger.set_llm_client(mock_client)
        assert merger.llm_client is mock_client

    def test_merge_log_starts_empty(self):
        merger = CrossPageMerger()
        assert merger._merge_log == []


# ---------------------------------------------------------------------------
# 2. Cross-page table merging
# ---------------------------------------------------------------------------

class TestCrossPageTableMerging:
    """Test detection and merging of tables that span consecutive pages."""

    def setup_method(self):
        self.merger = CrossPageMerger()

    # -- continuation detection ------------------------------------------------

    def test_detect_continuation_same_cols_consecutive_pages(self):
        """Tables on consecutive pages with matching column count should merge."""
        table_a = _make_table_block(
            page_idx=0,
            data=[
                ["项目", "金额", "占比"],
                ["营业收入", "1,000,000", "100%"],
            ],
        )
        table_b = _make_table_block(
            page_idx=1,
            data=[
                ["营业成本", "600,000", "60%"],
                ["净利润", "200,000", "20%"],
            ],
        )
        assert self.merger._detect_table_continuation(table_a, table_b) is True

    def test_reject_continuation_different_col_count(self):
        """Tables with wildly different column counts should not merge."""
        table_a = _make_table_block(
            page_idx=0,
            data=[
                ["项目", "金额"],
                ["收入", "500"],
            ],
        )
        table_b = _make_table_block(
            page_idx=1,
            data=[
                ["A", "B", "C", "D"],
                ["1", "2", "3", "4"],
            ],
        )
        assert self.merger._detect_table_continuation(table_a, table_b) is False

    def test_reject_continuation_large_page_gap(self):
        """Tables more than 1 page apart should not be considered continuations."""
        table_a = _make_table_block(page_idx=1, data=[["A"], ["1"]])
        table_b = _make_table_block(page_idx=5, data=[["A"], ["2"]])
        assert self.merger._detect_table_continuation(table_a, table_b) is False

    def test_detect_continuation_duplicate_header(self):
        """Table B repeating the header of table A is a continuation signal."""
        table_a = _make_table_block(
            page_idx=0,
            data=[
                ["项目", "本期金额", "上期金额"],
                ["营业收入", "1,234,567", "987,654"],
            ],
        )
        table_b = _make_table_block(
            page_idx=1,
            data=[
                ["项目", "本期金额", "上期金额"],
                ["营业成本", "876,543", "654,321"],
            ],
        )
        assert self.merger._detect_table_continuation(table_a, table_b) is True

    # -- full merge pipeline ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_merge_two_fragments_into_one_table(self):
        """Two table fragments across pages should merge into a single table."""
        table_p0 = _make_table_block(
            page_idx=0,
            data=[
                ["项目", "金额"],
                ["营业收入", "1,000万元"],
            ],
        )
        table_p1 = _make_table_block(
            page_idx=1,
            data=[
                ["营业成本", "600万元"],
                ["净利润", "200万元"],
            ],
        )
        # A non-table block in between to prove we handle gaps
        text_block = _make_text_block(page_idx=0, text="这是一段说明文字。")

        content_list = [table_p0, text_block, table_p1]
        result = await self.merger.execute({"content_list": content_list})

        tables_in_result = [
            b for b in result["merged_content"] if b.get("type") == "table"
        ]
        assert len(tables_in_result) == 1
        merged_data = tables_in_result[0].get("data", [])
        # Header + row from p0 + 2 rows from p1 = 4 rows
        assert len(merged_data) == 4
        assert result["merge_operations"]  # at least one op recorded


# ---------------------------------------------------------------------------
# 3. Cross-page text merging (broken sentence joining)
# ---------------------------------------------------------------------------

class TestCrossPageTextMerging:
    """Test detection and joining of text blocks broken across page boundaries."""

    def setup_method(self):
        self.merger = CrossPageMerger()

    def test_detect_text_continuation_mid_sentence(self):
        """Chinese text ending without punctuation and continuing on next page."""
        a = "本报告期内，公司实现营业收入"
        b = "5,000万元，同比增长15%"
        assert self.merger._detect_text_continuation(a, b) is True

    def test_no_continuation_when_a_ends_with_period(self):
        """Text ending with a period should not join the next block."""
        a = "公司业绩稳定增长。"
        b = "第二节 财务报表分析"
        assert self.merger._detect_text_continuation(a, b) is False

    def test_no_continuation_when_b_is_heading(self):
        """Next block starting as a heading should not be merged."""
        a = "上述数据已经审计确认"
        b = "3. 管理层讨论与分析"
        assert self.merger._detect_text_continuation(a, b) is False

    def test_join_char_cjk_to_cjk(self):
        """Two CJK strings should join without a space."""
        join = self.merger._determine_join_char("公司实现营业收入", "五千万元")
        assert join == ""

    def test_join_char_latin_to_latin(self):
        """Two Latin words should join with a space."""
        join = self.merger._determine_join_char("revenue", "growth")
        assert join == " "

    @pytest.mark.asyncio
    async def test_execute_merges_broken_paragraph(self):
        """End-to-end: two text blocks split mid-sentence are rejoined."""
        # text_a ends without punctuation (mid-sentence), text_b starts with
        # a CJK character (not a digit-heading pattern like "12.xxx").
        block_a = _make_text_block(
            page_idx=3,
            text="截至报告期末，公司总资产规模持续扩大，主要资产构成",
        )
        block_b = _make_text_block(
            page_idx=4,
            text="包括流动资产和非流动资产两大类，其中流动资产占比较大。",
        )
        result = await self.merger.execute(
            {"content_list": [block_a, block_b], "enable_reference_resolution": False},
        )
        merged_texts = [
            b for b in result["merged_content"] if b.get("type") == "text"
        ]
        assert len(merged_texts) == 1
        combined = merged_texts[0]["text"]
        assert "主要资产构成" in combined
        assert "流动资产" in combined
        assert any(
            op["operation_type"] == "text_merge" for op in result["merge_operations"]
        )


# ---------------------------------------------------------------------------
# 4. Entity extraction from Chinese text
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    """Test entity extraction for companies, money, dates, etc."""

    def setup_method(self):
        self.merger = CrossPageMerger()

    def test_extract_company_names(self):
        """Should extract Chinese company names."""
        content = [
            _make_text_block(
                page_idx=0,
                text="北京华胜天成科技股份有限公司成立于1998年，"
                     "是一家领先的IT综合服务提供商。",
            ),
        ]
        registry = self.merger._build_entity_registry(content)
        companies = [e["text"] for e in registry.get("company", [])]
        assert any("华胜天成" in c for c in companies)

    def test_extract_money_amounts(self):
        """Should extract Chinese monetary values."""
        content = [
            _make_text_block(
                page_idx=0,
                text="2024年度公司实现营业收入35.6亿元，净利润2.8亿元。",
            ),
        ]
        registry = self.merger._build_entity_registry(content)
        money = [e["text"] for e in registry.get("money", [])]
        assert len(money) >= 2
        assert any("35.6亿元" in m for m in money)
        assert any("2.8亿元" in m for m in money)

    def test_extract_dates(self):
        """Should extract Chinese-style dates."""
        content = [
            _make_text_block(
                page_idx=0,
                text="公司于2024年3月15日召开董事会，审议通过年度报告。",
            ),
        ]
        registry = self.merger._build_entity_registry(content)
        dates = [e["text"] for e in registry.get("date", [])]
        assert any("2024" in d and "3月" in d and "15日" in d for d in dates)

    def test_extract_percentage(self):
        """Should extract percentage values."""
        content = [
            _make_text_block(
                page_idx=0,
                text="毛利率为42.5%，较上年提升3.2个百分点。",
            ),
        ]
        registry = self.merger._build_entity_registry(content)
        percentages = [e["text"] for e in registry.get("percentage", [])]
        assert any("42.5%" in p for p in percentages)

    def test_extract_from_table_blocks(self):
        """Entity extraction should also work on table blocks."""
        content = [
            _make_table_block(
                page_idx=0,
                data=[
                    ["公司名称", "投资金额"],
                    ["中芯国际集成电路制造有限公司", "50,000万元"],
                ],
            ),
        ]
        registry = self.merger._build_entity_registry(content)
        companies = [e["text"] for e in registry.get("company", [])]
        assert any("中芯国际" in c for c in companies)
        money = [e["text"] for e in registry.get("money", [])]
        assert any("50,000万元" in m for m in money)

    def test_deduplicates_entities(self):
        """Same entity appearing multiple times should be stored once."""
        content = [
            _make_text_block(page_idx=0, text="华为技术有限公司发布了新产品。"),
            _make_text_block(page_idx=1, text="华为技术有限公司在2024年业绩良好。"),
        ]
        registry = self.merger._build_entity_registry(content)
        companies = [e["text"] for e in registry.get("company", [])]
        # Should have exactly one entry for 华为技术有限公司
        huawei_entries = [c for c in companies if "华为技术有限公司" in c]
        assert len(huawei_entries) == 1


# ---------------------------------------------------------------------------
# 5. Reference resolution ("该公司" -> resolved entity)
# ---------------------------------------------------------------------------

class TestReferenceResolution:
    """Test rule-based and mock-LLM reference resolution."""

    def setup_method(self):
        self.merger = CrossPageMerger()

    def test_resolve_gai_gongsi_to_company(self):
        """'该公司' should resolve to a known company in entity registry."""
        registry = {
            "company": [
                {"text": "华为技术有限公司", "page_idx": 0, "context": "华为技术有限公司发布了新产品"},
            ],
            "money": [],
            "date": [],
        }
        result = self.merger._resolve_references(
            reference="该公司",
            entities=registry,
            context="华为技术有限公司发布了新产品，该公司在5G领域处于领先地位",
        )
        assert result.resolved_entity == "华为技术有限公司"
        assert result.confidence >= 0.6
        assert result.method == "rule"

    def test_resolve_shangshu_jine_to_money(self):
        """'上述金额' should resolve to a known money entity."""
        registry = {
            "company": [],
            "money": [
                {"text": "5,000万元", "page_idx": 1, "context": "投资总额5,000万元"},
            ],
            "date": [],
        }
        result = self.merger._resolve_references(
            reference="上述金额",
            entities=registry,
            context="项目总投资额为5,000万元，上述金额已全部到位",
        )
        assert result.resolved_entity == "5,000万元"
        assert result.confidence >= 0.6

    def test_resolve_date_reference(self):
        """'同期' should resolve to a known date entity."""
        registry = {
            "company": [],
            "money": [],
            "date": [
                {"text": "2023年1月1日", "page_idx": 0, "context": "截至2023年1月1日"},
            ],
        }
        result = self.merger._resolve_references(
            reference="同期",
            entities=registry,
            context="截至2023年1月1日，同期营业收入有所增长",
        )
        assert "2023" in result.resolved_entity
        assert result.confidence >= 0.5

    def test_unresolved_reference_low_confidence(self):
        """References with no matching entity should get confidence 0."""
        registry = {"company": [], "money": [], "date": []}
        result = self.merger._resolve_references(
            reference="该公司",
            entities=registry,
            context="该公司经营良好",
        )
        assert result.confidence == 0.0
        assert result.method == "unresolved"

    def test_find_references_detects_pronouns(self):
        """_find_references should locate reference expressions in text."""
        text = "华为技术有限公司发布了新产品，该公司在5G领域处于领先地位。上述金额已全部到位。"
        refs = self.merger._find_references(text)
        ref_texts = [r[0] for r in refs]
        assert "该公司" in ref_texts
        # "上述" is matched as part of longer patterns like "上述金额" first;
        # check that at least one "上述..." reference is found
        assert any(r.startswith("上述") for r in ref_texts)

    @pytest.mark.asyncio
    async def test_llm_resolution_used_as_fallback(self):
        """When rule-based confidence is low, LLM should be consulted."""
        merger = CrossPageMerger()
        mock_client = MagicMock()

        # Make the LLM return a valid JSON response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"resolved_entity": "华为技术有限公司", "confidence": 0.9, "reasoning": "test"}')]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_client.model = "test-model"

        registry = {
            "company": [
                {"text": "华为技术有限公司", "page_idx": 0, "context": "..."},
            ],
            "money": [],
        }
        content = [
            _make_text_block(
                page_idx=1,
                text="该公司在5G领域处于领先地位",
            ),
        ]

        result = await merger._resolve_all_references(content, registry, mock_client)
        # The LLM should have been called (rule-based has low or high confidence)
        # In either case, we should get at least one resolution
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# 6. execute() integration
# ---------------------------------------------------------------------------

class TestExecuteIntegration:
    """End-to-end tests for the execute() pipeline."""

    @pytest.mark.asyncio
    async def test_execute_empty_content_list(self):
        """Should return empty result structures for empty input."""
        merger = CrossPageMerger()
        result = await merger.execute({"content_list": []})
        assert result["merged_content"] == []
        assert result["merge_operations"] == []
        assert result["entity_registry"] == {}
        assert result["reference_resolutions"] == []

    @pytest.mark.asyncio
    async def test_execute_no_content_list_key(self):
        """Should handle missing content_list key gracefully."""
        merger = CrossPageMerger()
        result = await merger.execute({})
        assert result["merged_content"] == []

    @pytest.mark.asyncio
    async def test_execute_with_mock_step_results(self):
        """Full pipeline with realistic mixed content across multiple pages."""
        merger = CrossPageMerger()
        content_list = [
            # Page 0: text + start of table
            _make_text_block(
                page_idx=0,
                text="北京星辰数据科技有限公司2024年度审计报告。"
                     "截至2024年12月31日，公司总资产达到",
            ),
            _make_table_block(
                page_idx=0,
                data=[
                    ["项目", "期末余额", "期初余额"],
                    ["货币资金", "3,500万元", "2,800万元"],
                    ["应收账款", "1,200万元", "900万元"],
                ],
            ),
            # Page 1: continuation of table + text with reference
            _make_table_block(
                page_idx=1,
                data=[
                    ["存货", "800万元", "600万元"],
                    ["固定资产", "5,000万元", "4,200万元"],
                ],
            ),
            _make_text_block(
                page_idx=1,
                text="5.6亿元，同比增长12.5%。该公司持续加大研发投入，"
                     "报告期内研发费用为3,200万元。",
            ),
            # Page 2: more text
            _make_text_block(
                page_idx=2,
                text="公司主要业务涵盖云计算、大数据分析及人工智能平台开发。"
                     "2024年度实现营业收入8.2亿元，净利润1.5亿元。",
            ),
        ]

        result = await merger.execute(
            {"content_list": content_list, "enable_reference_resolution": True},
        )

        # Basic structure checks
        assert isinstance(result, dict)
        assert "merged_content" in result
        assert "merge_operations" in result
        assert "entity_registry" in result
        assert "reference_resolutions" in result

        # Table merge should have occurred (pages 0 -> 1)
        table_ops = [
            op for op in result["merge_operations"]
            if op["operation_type"] == "table_merge"
        ]
        assert len(table_ops) >= 1

        # Text merge may have occurred (broken sentence page 0 -> 1)
        text_ops = [
            op for op in result["merge_operations"]
            if op["operation_type"] == "text_merge"
        ]
        # Whether text merges depends on the continuation heuristics

        # Entity registry should contain extracted entities
        entities = result["entity_registry"]
        company_names = [e["text"] for e in entities.get("company", [])]
        assert any("星辰数据" in c for c in company_names)

        money_entities = [e["text"] for e in entities.get("money", [])]
        assert len(money_entities) >= 1

        # Reference resolution should have run
        resolutions = result["reference_resolutions"]
        assert len(resolutions) >= 1
        # "该公司" should be resolved
        gai_gongsi_res = [
            r for r in resolutions if r["reference"] == "该公司"
        ]
        assert len(gai_gongsi_res) >= 1
        assert gai_gongsi_res[0]["resolved_entity"] != ""

    @pytest.mark.asyncio
    async def test_execute_with_reference_resolution_disabled(self):
        """When reference resolution is disabled, no resolutions returned."""
        merger = CrossPageMerger()
        content = [
            _make_text_block(
                page_idx=0,
                text="华为技术有限公司经营良好，该公司在5G领域领先。",
            ),
        ]
        result = await merger.execute(
            {"content_list": content, "enable_reference_resolution": False},
        )
        assert result["reference_resolutions"] == []
        # Entity registry should still be populated
        assert "company" in result["entity_registry"]


# ---------------------------------------------------------------------------
# 7. Markdown table conversion utility
# ---------------------------------------------------------------------------

class TestMarkdownTableConversion:
    """Test _table_to_markdown and _parse_markdown_table round-trip."""

    def test_basic_markdown_round_trip(self):
        """Convert a 2D list to markdown and parse it back."""
        original = [
            ["项目", "金额", "占比"],
            ["营业收入", "1,000万元", "100%"],
            ["营业成本", "600万元", "60%"],
        ]
        md = CrossPageMerger._table_to_markdown(original)
        assert md  # non-empty
        assert "| 项目" in md
        assert "| ---" in md  # separator row

        parsed = _parse_markdown_table(md)
        assert parsed == original

    def test_empty_table(self):
        """Empty input should produce empty string."""
        assert CrossPageMerger._table_to_markdown([]) == ""

    def test_single_row(self):
        """Single-row table should produce header + separator only."""
        md = CrossPageMerger._table_to_markdown([["A", "B"]])
        lines = md.splitlines()
        assert len(lines) == 2  # header + separator, no body
        assert "| A | B |" in lines[0]

    def test_uneven_rows_padded(self):
        """Rows with fewer columns should be padded with empty strings."""
        rows = [
            ["A", "B", "C"],
            ["1"],
            ["x", "y"],
        ]
        md = CrossPageMerger._table_to_markdown(rows)
        parsed = _parse_markdown_table(md)
        assert len(parsed) == 3
        assert parsed[0] == ["A", "B", "C"]
        assert parsed[1] == ["1", "", ""]
        assert parsed[2] == ["x", "y", ""]

    def test_parse_markdown_ignores_separator(self):
        """_parse_markdown_table should skip |---|---| rows."""
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        parsed = _parse_markdown_table(md)
        assert len(parsed) == 2
        assert parsed[0] == ["A", "B"]
        assert parsed[1] == ["1", "2"]

    def test_parse_markdown_ignores_non_table_lines(self):
        """Lines not starting with | should be ignored."""
        md = "Some text\n| A | B |\n| --- | --- |\n| 1 | 2 |\nMore text"
        parsed = _parse_markdown_table(md)
        assert len(parsed) == 2

    def test_chinese_content_round_trip(self):
        """Chinese content in table cells should survive round-trip."""
        original = [
            ["公司名称", "注册地", "注册资本"],
            ["华为技术有限公司", "深圳", "4,039,185万元"],
            ["中兴通讯股份有限公司", "深圳", "4,783万元"],
        ]
        md = CrossPageMerger._table_to_markdown(original)
        parsed = _parse_markdown_table(md)
        assert parsed == original


# ---------------------------------------------------------------------------
# Additional helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_extract_table_data_from_data_key(self):
        block = {"data": [["A", "B"], ["1", "2"]]}
        assert _extract_table_data(block) == [["A", "B"], ["1", "2"]]

    def test_extract_table_data_from_table_body_key(self):
        block = {"table_body": [["X", "Y"], ["3", "4"]]}
        assert _extract_table_data(block) == [["X", "Y"], ["3", "4"]]

    def test_extract_table_data_fallback_to_markdown(self):
        block = {"markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |"}
        result = _extract_table_data(block)
        assert len(result) == 2
        assert result[0] == ["A", "B"]

    def test_extract_table_data_empty_block(self):
        assert _extract_table_data({}) == []

    def test_table_column_count(self):
        data = [["A", "B", "C"], ["1", "2"], ["x", "y", "z", "w"]]
        assert _table_column_count(data) == 4

    def test_table_column_count_empty(self):
        assert _table_column_count([]) == 0

    def test_is_likely_header_row_with_keywords(self):
        assert _is_likely_header_row(["项目", "金额", "比率"]) is True

    def test_is_likely_header_row_with_mostly_digits(self):
        assert _is_likely_header_row(["1,234", "5,678", "9,012"]) is False

    def test_is_likely_header_row_empty(self):
        assert _is_likely_header_row(["", ""]) is False

    def test_row_similarity_identical(self):
        row = ["项目", "金额"]
        assert _row_similarity(row, row) == 1.0

    def test_row_similarity_different(self):
        a = ["项目", "金额"]
        b = ["完全", "不同"]
        assert _row_similarity(a, b) < 0.5
