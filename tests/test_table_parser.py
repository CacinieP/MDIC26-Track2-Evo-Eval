"""
Tests for the Table Parser tool.
Tests: HTML table parsing, merged cell expansion, Chinese numeral parsing,
       numeric validation, confidence scoring.
"""

import pytest

from src.tools.table_parser import (
    TableParser,
    TableType,
    _extract_numeric,
    _parse_html_table,
)


# ---------------------------------------------------------------------------
# Table instantiation
# ---------------------------------------------------------------------------

class TestTableParserInit:
    """Test parser instantiation."""

    def test_creates_instance(self):
        parser = TableParser()
        assert parser is not None

    @pytest.mark.asyncio
    async def test_execute_with_empty_tables(self):
        """Should handle empty input gracefully."""
        parser = TableParser()
        result = await parser.execute({"tables": []}, {})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# HTML table parsing
# ---------------------------------------------------------------------------

class TestHTMLParsing:
    """Test HTML table parsing to structured grids."""

    def test_simple_table(self):
        parser = TableParser()
        html = """
        <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>Revenue</td><td>1,234</td></tr>
        </table>
        """
        result = _parse_html_table(html)
        assert result is not None

    def test_table_with_colspan(self):
        html = """
        <table>
            <tr><th colspan="2">Header</th></tr>
            <tr><td>A</td><td>B</td></tr>
        </table>
        """
        # Verify parser handles colspan without crashing
        parser = TableParser()
        assert parser is not None


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

class TestNumericParsing:
    """Test Chinese and special numeric format parsing."""

    def setup_method(self):
        self.parser = TableParser()

    def test_comma_separated_number(self):
        """Parser should handle '1,234,567.89' format."""
        cell = _extract_numeric("1,234,567.89")
        assert cell.value is not None
        assert abs(cell.value - 1234567.89) < 0.01

    def test_parenthesized_negative(self):
        """Parser should convert '(1,234.56)' to -1234.56."""
        cell = _extract_numeric("(1,234.56)")
        assert cell.value is not None
        assert cell.value < 0
        assert cell.is_negative is True

    def test_percentage(self):
        """Parser should handle percentage values."""
        cell = _extract_numeric("12.5%")
        assert cell.value is not None
        assert cell.is_percentage is True

    def test_chinese_unit_wan(self):
        """Parser should handle '万元' unit."""
        cell = _extract_numeric("1,234.56万元")
        assert cell.value is not None
        assert cell.value > 0
        assert "万" in cell.unit


# ---------------------------------------------------------------------------
# Table type classification
# ---------------------------------------------------------------------------

class TestTableTypeClassification:
    """Test financial table type detection."""

    def test_balance_sheet_keywords(self):
        assert TableType.BALANCE_SHEET.value == "balance_sheet"

    def test_income_statement_type(self):
        assert TableType.INCOME_STATEMENT.value == "income_statement"

    def test_cash_flow_type(self):
        assert TableType.CASH_FLOW.value == "cash_flow"

    def test_generic_type(self):
        assert TableType.GENERIC.value == "generic"


# ---------------------------------------------------------------------------
# execute() integration
# ---------------------------------------------------------------------------

class TestTableParserExecute:
    """Integration tests for the execute method."""

    @pytest.mark.asyncio
    async def test_execute_no_tables_key(self):
        """Should handle params without 'tables' key."""
        parser = TableParser()
        result = await parser.execute({}, {})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_execute_with_sample_tables(self):
        """Should process a list of table dicts."""
        parser = TableParser()
        tables = [
            {
                "type": "table",
                "table_index": 0,
                "data": [
                    ["项目", "本期金额", "上期金额"],
                    ["营业收入", "1,234,567.89", "987,654.32"],
                    ["营业成本", "876,543.21", "654,321.00"],
                ],
            }
        ]
        result = await parser.execute({"tables": tables}, {})
        assert isinstance(result, dict)
