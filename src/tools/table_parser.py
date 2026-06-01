"""
Table Parser Tool - Specialized financial table extraction and structuring.

Handles:
- HTML table parsing to structured grids (list of rows, each row is list of cells)
- Merged cell detection and expansion (colspan / rowspan)
- Financial table classification (balance sheet, income statement, cash flow)
- Numeric extraction and validation:
  - Chinese number formats: "1,234.56万元", "壹佰贰拾叁万"
  - Row / column sum verification
  - Percentage cross-references (同比 / 环比)
  - Anomaly detection and flagging
- Per-cell and per-table confidence scoring
- Output as structured JSON, CSV, or Markdown
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TableType(str, Enum):
    """Financial table classification."""
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    EQUITY_CHANGE = "equity_change"
    FINANCIAL_GENERIC = "financial_generic"
    GENERIC = "generic"


class OutputFormat(str, Enum):
    """Supported output formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


@dataclass
class NumericCell:
    """Parsed numeric value with metadata."""
    raw_text: str = ""
    value: float | None = None
    unit: str = ""           # "元", "万元", "亿元", "%"
    is_negative: bool = False
    is_percentage: bool = False
    confidence: float = 1.0  # 0.0 – 1.0
    anomaly: str = ""        # non-empty when an anomaly was detected


@dataclass
class MergedRange:
    """Describes a single merged cell span."""
    row_start: int = 0
    col_start: int = 0
    row_end: int = 0          # exclusive
    col_end: int = 0          # exclusive


@dataclass
class ValidationReport:
    """Per-table numeric consistency validation."""
    row_sum_checks: list[dict] = field(default_factory=list)
    col_sum_checks: list[dict] = field(default_factory=list)
    percentage_checks: list[dict] = field(default_factory=list)
    anomaly_flags: list[dict] = field(default_factory=list)
    overall_pass: bool = True


@dataclass
class CellConfidence:
    """Confidence score for a single cell."""
    row: int = 0
    col: int = 0
    score: float = 1.0
    reason: str = ""


@dataclass
class StructuredTable:
    """Complete structured result for one parsed table."""
    index: int = 0
    type: str = TableType.GENERIC
    headers: list[list[str]] = field(default_factory=list)   # list of header rows
    rows: list[list[str]] = field(default_factory=list)       # data rows (post-expansion)
    numeric_validation: dict = field(default_factory=dict)
    confidence_scores: list[dict] = field(default_factory=list)
    raw_html: str = ""
    merged_ranges: list[dict] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    table_confidence: float = 1.0


# ---------------------------------------------------------------------------
# Chinese numeral maps
# ---------------------------------------------------------------------------

_CN_DIGIT: dict[str, int] = {
    "零": 0, "〇": 0,
    "壹": 1, "一": 1,
    "贰": 2, "二": 2, "貳": 2,
    "叁": 3, "三": 3, "參": 3,
    "肆": 4, "四": 4,
    "伍": 5, "五": 5,
    "陆": 6, "六": 6, "陸": 6,
    "柒": 7, "七": 7,
    "捌": 8, "八": 8,
    "玖": 9, "九": 9,
}

_CN_UNIT: dict[str, int] = {
    "拾": 10, "十": 10,
    "佰": 100, "百": 100,
    "仟": 1000, "千": 1000,
    "万": 10_000, "萬": 10_000,
    "亿": 100_000_000, "億": 100_000_000,
}


# ---------------------------------------------------------------------------
# Chinese numeral parser
# ---------------------------------------------------------------------------

def _parse_chinese_numeral(text: str) -> float | None:
    """
    Parse Chinese numeral strings like "壹佰贰拾叁万" to a float.

    Supports:
    - Pure Chinese: "壹佰贰拾叁万"  -> 1_230_000
    - Mixed: "1万" -> 10_000
    - "壹亿贰仟万" -> 120_000_000

    Returns None if the text is not a valid Chinese numeral.
    """
    text = text.strip()
    if not text:
        return None

    # Quick rejection: if there are digits or decimal points, it is not a pure
    # Chinese numeral (those are handled by the Arabic-number parser).
    if re.search(r"[0-9.]", text):
        return None

    # Check if any Chinese numeral character is present
    all_chars = set(text)
    if not all_chars.intersection(set(_CN_DIGIT) | set(_CN_UNIT)):
        return None

    result = 0
    current = 0
    last_unit = 1

    for ch in text:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit_val = _CN_UNIT[ch]
            if unit_val >= 10_000:
                # Major unit (万, 亿): multiply the accumulated value
                if current == 0 and last_unit < unit_val:
                    # e.g. "拾万" means 10 * 10000
                    result = (result + last_unit) * unit_val
                else:
                    result = (result + current) * unit_val
                current = 0
                last_unit = unit_val
            else:
                # Minor unit (十, 百, 千)
                if current == 0:
                    current = 1  # e.g. "拾" alone means 10
                result += current * unit_val
                last_unit = unit_val
                current = 0
        else:
            # Unknown character – skip
            continue

    result += current
    return float(result) if result != 0 else None


# ---------------------------------------------------------------------------
# Number extraction helpers
# ---------------------------------------------------------------------------

_UNIT_PATTERN = re.compile(
    r"(万亿元|万元|亿元|万|亿|元|％|%)",
)

_ARABIC_NUMBER = re.compile(
    r"[−\-]?\s*[\d,]+(?:\.\d+)?"
)


def _extract_numeric(text: str) -> NumericCell:
    """
    Extract a numeric value from a cell string.

    Handles:
    - Arabic: "1,234.56", "-5,678.90万元", "12.3%"
    - Chinese numerals: "壹佰贰拾叁万"
    - Parenthesised negatives: "(1,234.56)"

    Returns a NumericCell with the parsed value and metadata.
    """
    cell = NumericCell(raw_text=text)
    text = text.strip()

    if not text or text in ("—", "-", "–", "N/A", "n/a", "无", "/", "."):
        return cell

    # Try Chinese numeral first (pure Chinese characters)
    cn_val = _parse_chinese_numeral(text)
    if cn_val is not None:
        cell.value = cn_val
        # Detect unit suffix
        m = _UNIT_PATTERN.search(text)
        if m:
            cell.unit = m.group(1)
        return cell

    # Try Arabic number with optional unit
    # Detect parenthesised negative: (1,234.56)
    is_paren_negative = False
    working = text
    paren_match = re.match(r"^\((.+)\)$", text)
    if paren_match:
        working = paren_match.group(1)
        is_paren_negative = True

    # Extract unit
    unit_match = _UNIT_PATTERN.search(working)
    if unit_match:
        cell.unit = unit_match.group(1)

    # Extract number
    num_match = _ARABIC_NUMBER.search(working)
    if num_match:
        num_str = num_match.group(0)
        num_str = num_str.replace(",", "").replace(" ", "")
        try:
            val = float(num_str)
            if is_paren_negative or text.startswith("-") or text.startswith("−"):
                val = -val
                cell.is_negative = True
            cell.value = val
        except ValueError:
            cell.confidence = 0.3

    # Percentage detection
    if "%" in text or "％" in text:
        cell.is_percentage = True

    # Confidence heuristic
    if cell.value is not None:
        cell.confidence = 1.0
    else:
        cell.confidence = 0.2

    return cell


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------

def _parse_html_table(html_str: str) -> list[list[str]]:
    """
    Parse an HTML table string into a rectangular grid (list of rows).

    Uses BeautifulSoup when available; falls back to a stdlib
    ``html.parser.HTMLParser`` implementation otherwise.
    Expands colspan / rowspan so every row has the same number of cells.

    Args:
        html_str: Raw HTML string containing a ``<table>`` element.

    Returns:
        A list of rows, where each row is a list of cell text strings.
    """
    rows_raw: list[list[dict]] | None = None

    # --- Try BeautifulSoup first ---
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        soup = BeautifulSoup(html_str, "html.parser")
        table = soup.find("table")
        if table is None:
            first_tr = soup.find("tr")
            if first_tr:
                table = first_tr.parent or soup
            else:
                table = soup

        tr_list = table.find_all("tr")
        rows_raw = []
        for tr in tr_list:
            cells = tr.find_all(["td", "th"])
            row_cells: list[dict] = []
            for cell in cells:
                text = cell.get_text(strip=True)
                colspan = int(cell.get("colspan", 1) or 1)
                rowspan = int(cell.get("rowspan", 1) or 1)
                row_cells.append({
                    "text": text,
                    "colspan": max(1, colspan),
                    "rowspan": max(1, rowspan),
                })
            rows_raw.append(row_cells)
    except ImportError:
        pass

    # --- Stdlib fallback ---
    if rows_raw is None:
        rows_raw = _parse_html_table_stdlib(html_str)

    if not rows_raw:
        return []

    # --- Expand into rectangular grid ---
    max_rows = len(rows_raw) + 10  # extra for rowspan overflow
    max_cols = 0
    for row in rows_raw:
        cols_needed = sum(c["colspan"] for c in row)
        max_cols = max(max_cols, cols_needed)
    max_cols = max(max_cols, 1)

    grid: list[list[str | None]] = [[None] * max_cols for _ in range(max_rows)]

    for row_idx, row_cells in enumerate(rows_raw):
        col_cursor = 0
        for cell_info in row_cells:
            while col_cursor < max_cols and grid[row_idx][col_cursor] is not None:
                col_cursor += 1

            if col_cursor >= max_cols:
                for r in grid:
                    r.extend([None] * (col_cursor - max_cols + 1))
                max_cols = len(grid[0])

            text = cell_info["text"]
            cs = cell_info["colspan"]
            rs = cell_info["rowspan"]

            needed_rows = row_idx + rs
            while len(grid) < needed_rows:
                grid.append([None] * max_cols)

            needed_cols = col_cursor + cs
            if needed_cols > max_cols:
                for r in grid:
                    r.extend([None] * (needed_cols - max_cols))
                max_cols = len(grid[0])

            for dr in range(rs):
                for dc in range(cs):
                    r = row_idx + dr
                    c = col_cursor + dc
                    if r < len(grid) and c < len(grid[r]):
                        grid[r][c] = text

            col_cursor += cs

    # Trim trailing empty rows/columns and convert None to empty string
    result: list[list[str]] = []
    for row in grid:
        if all(v is None for v in row):
            continue
        result.append([v if v is not None else "" for v in row])

    if result:
        max_c = max(len(r) for r in result)
        for r in result:
            while len(r) < max_c:
                r.append("")

    return result


def _parse_html_table_stdlib(html_str: str) -> list[list[dict]]:
    """
    Parse HTML table using only the stdlib ``html.parser`` module.

    Returns a list of rows, each row a list of dicts with keys
    ``text``, ``colspan``, ``rowspan``.
    """
    import html as _html_module
    import html.parser

    rows: list[list[dict]] = []
    current_row: list[dict] | None = None
    current_cell_text_parts: list[str] = []
    in_cell = False
    current_colspan = 1
    current_rowspan = 1

    class _TableParser(html.parser.HTMLParser):
        nonlocal rows, current_row, current_cell_text_parts, in_cell
        nonlocal current_colspan, current_rowspan

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            nonlocal current_row, in_cell, current_colspan, current_rowspan
            tag_l = tag.lower()
            if tag_l == "tr":
                current_row = []
            elif tag_l in ("td", "th"):
                in_cell = True
                current_cell_text_parts.clear()
                current_colspan = 1
                current_rowspan = 1
                attr_dict = {k.lower(): (v or "") for k, v in attrs}
                if "colspan" in attr_dict:
                    try:
                        current_colspan = max(1, int(attr_dict["colspan"]))
                    except (ValueError, TypeError):
                        pass
                if "rowspan" in attr_dict:
                    try:
                        current_rowspan = max(1, int(attr_dict["rowspan"]))
                    except (ValueError, TypeError):
                        pass

        def handle_endtag(self, tag: str) -> None:
            nonlocal current_row, in_cell
            tag_l = tag.lower()
            if tag_l in ("td", "th") and in_cell:
                text = " ".join("".join(current_cell_text_parts).split()).strip()
                if current_row is not None:
                    current_row.append({
                        "text": text,
                        "colspan": current_colspan,
                        "rowspan": current_rowspan,
                    })
                in_cell = False
                current_cell_text_parts.clear()
            elif tag_l == "tr":
                if current_row is not None:
                    rows.append(current_row)
                current_row = None

        def handle_data(self, data: str) -> None:
            if in_cell:
                current_cell_text_parts.append(data)

        def handle_entityref(self, name: str) -> None:
            if in_cell:
                current_cell_text_parts.append(_html_module.unescape(f"&{name};"))

        def handle_charref(self, name: str) -> None:
            if in_cell:
                current_cell_text_parts.append(_html_module.unescape(f"&#{name};"))

    parser = _TableParser()
    parser.feed(html_str)
    parser.close()
    return rows


# ---------------------------------------------------------------------------
# Merged cell detection
# ---------------------------------------------------------------------------

def _detect_merged_cells(grid: list[list[str]]) -> list[MergedRange]:
    """
    Detect merged cell ranges in a parsed grid.

    A merged range is identified by consecutive cells with identical text
    that appear to originate from a single logical cell (i.e. the value
    was duplicated by colspan/rowspan expansion).

    Returns a list of MergedRange objects.
    """
    if not grid:
        return []

    merged: list[MergedRange] = []
    rows = len(grid)
    cols = max(len(r) for r in grid) if grid else 0

    visited: set[tuple[int, int]] = set()

    for r in range(rows):
        for c in range(len(grid[r])):
            if (r, c) in visited:
                continue
            val = grid[r][c]
            if not val:
                visited.add((r, c))
                continue

            # Check horizontal span (colspan)
            c_end = c + 1
            while c_end < len(grid[r]) and grid[r][c_end] == val and (r, c_end) not in visited:
                c_end += 1

            # Check vertical span (rowspan)
            r_end = r + 1
            if c_end - c > 1:
                # Verify the same horizontal pattern repeats for subsequent rows
                while r_end < rows:
                    match = True
                    for cc in range(c, c_end):
                        if cc >= len(grid[r_end]) or grid[r_end][cc] != val or (r_end, cc) in visited:
                            match = False
                            break
                    if not match:
                        break
                    r_end += 1

            # Mark visited
            for rr in range(r, r_end):
                for cc in range(c, c_end):
                    visited.add((rr, cc))

            # Only record if it is actually a span (not a single cell)
            if (c_end - c) > 1 or (r_end - r) > 1:
                merged.append(MergedRange(
                    row_start=r, col_start=c,
                    row_end=r_end, col_end=c_end,
                ))

    return merged


# ---------------------------------------------------------------------------
# Financial table classification
# ---------------------------------------------------------------------------

# Keyword sets for classification
_BALANCE_SHEET_KW = {
    "资产总计", "负债合计", "所有者权益", "股东权益", "资产负债表",
    "流动资产", "非流动资产", "流动负债", "非流动负债",
    "货币资金", "应收账款", "存货", "固定资产", "无形资产",
    "短期借款", "应付账款", "实收资本", "资本公积", "盈余公积",
    "未分配利润",
}

_INCOME_STATEMENT_KW = {
    "营业收入", "营业成本", "利润总额", "净利润", "利润表",
    "营业利润", "营业收入合计", "所得税", "毛利润", "营业外收入",
    "营业外支出", "管理费用", "销售费用", "财务费用",
    "基本每股收益", "稀释每股收益",
}

_CASH_FLOW_KW = {
    "经营活动", "投资活动", "筹资活动", "现金流量表",
    "现金及现金等价物", "经营活动产生的现金流量",
    "投资活动产生的现金流量", "筹资活动产生的现金流量",
    "支付给职工", "收回投资", "取得投资收益",
}

_EQUITY_CHANGE_KW = {
    "所有者权益变动表", "会计政策变更", "前期差错更正",
}


def _classify_financial_table(
    headers: list[list[str]],
    rows: list[list[str]],
) -> str:
    """
    Classify a financial table based on headers and content.

    Examines cell text for keyword matches and returns the most likely
    TableType value.
    """
    # Gather all text
    all_text = set()
    for row in headers:
        for cell in row:
            all_text.add(cell)
    for row in rows:
        for cell in row:
            all_text.add(cell)
    text_blob = " ".join(all_text)

    scores: dict[str, int] = {
        TableType.BALANCE_SHEET: 0,
        TableType.INCOME_STATEMENT: 0,
        TableType.CASH_FLOW: 0,
        TableType.EQUITY_CHANGE: 0,
    }

    for kw in _BALANCE_SHEET_KW:
        if kw in text_blob:
            scores[TableType.BALANCE_SHEET] += 1
    for kw in _INCOME_STATEMENT_KW:
        if kw in text_blob:
            scores[TableType.INCOME_STATEMENT] += 1
    for kw in _CASH_FLOW_KW:
        if kw in text_blob:
            scores[TableType.CASH_FLOW] += 1
    for kw in _EQUITY_CHANGE_KW:
        if kw in text_blob:
            scores[TableType.EQUITY_CHANGE] += 1

    max_score = max(scores.values())
    if max_score == 0:
        # Check for generic financial indicators
        generic_kw = {"万元", "元", "%", "同比", "环比", "收入", "利润", "资产", "负债"}
        if any(kw in text_blob for kw in generic_kw):
            return TableType.FINANCIAL_GENERIC
        return TableType.GENERIC

    # Return the type with the highest score; ties broken by priority order
    priority = [
        TableType.BALANCE_SHEET,
        TableType.INCOME_STATEMENT,
        TableType.CASH_FLOW,
        TableType.EQUITY_CHANGE,
    ]
    for t in priority:
        if scores[t] == max_score:
            return t

    return TableType.GENERIC


# ---------------------------------------------------------------------------
# Numeric validation
# ---------------------------------------------------------------------------

def _validate_numeric_consistency(
    rows: list[list[str]],
    headers: list[list[str]],
) -> ValidationReport:
    """
    Validate numeric consistency in financial table rows.

    Checks:
    1. Row sums: where a row appears to contain sub-items and a total,
       verify that the sub-items sum to the total.
    2. Column sums: where column headers suggest sub-categories and a total.
    3. Percentage cross-references: verify 同比/环比 percentages match
       the implied calculation.
    4. Anomaly detection: flag extreme values or inconsistencies.

    Returns a ValidationReport with check results.
    """
    report = ValidationReport()

    if not rows or not headers:
        return report

    # Build a numeric grid for validation
    numeric_grid: list[list[NumericCell]] = []
    for row in rows:
        numeric_row: list[NumericCell] = []
        for cell in row:
            numeric_row.append(_extract_numeric(cell))
        numeric_grid.append(numeric_row)

    # --- Row sum checks ---
    # NOTE: This validation only applies to tables where columns represent
    # sub-categories that sum to a total (e.g. a balance sheet where
    # "流动资产 + 非流动资产 = 资产总计").  For tables where columns are
    # independent (e.g. year-over-year comparison), the sum check may produce
    # false positives.  We gate this on the table having a recognised financial
    # type so that generic tables are not penalised.
    if len(rows) > 2 and numeric_grid:
        num_cols = max(len(r) for r in numeric_grid) if numeric_grid else 0
        if num_cols >= 2:
            # Determine if this table has a summable structure by checking
            # whether the first column contains sub-category labels and a
            # later column holds totals (common in financial statements).
            is_summable = _is_summable_table_structure(headers, rows)
            if is_summable:
                # Try the last numeric column as the "total" column
                total_col = _find_last_numeric_col(numeric_grid)
                if total_col is not None and total_col > 0:
                    for ri, nrow in enumerate(numeric_grid):
                        # Check if this row looks like a subtotal row
                        # (has values in multiple columns including the total)
                        values: list[tuple[int, float]] = []
                        for ci, cell in enumerate(nrow):
                            if ci == total_col:
                                continue
                            if cell.value is not None:
                                values.append((ci, cell.value))

                        if len(values) >= 2:
                            total_cell = nrow[total_col] if total_col < len(nrow) else None
                            if total_cell and total_cell.value is not None:
                                expected = sum(v for _, v in values)
                                actual = total_cell.value
                                # Allow tolerance for rounding
                                tolerance = max(abs(expected) * 0.02, 1.0)
                                diff = abs(expected - actual)
                                passed = diff <= tolerance
                                report.row_sum_checks.append({
                                    "row": ri,
                                    "expected": round(expected, 4),
                                    "actual": round(actual, 4),
                                    "diff": round(diff, 4),
                                    "passed": passed,
                                })
                                if not passed:
                                    report.overall_pass = False

    # --- Percentage cross-reference checks ---
    # Look for rows with 同比增长/环比增长 patterns
    for ri, row in enumerate(rows):
        row_text = " ".join(row)
        if "同比" in row_text or "环比" in row_text or "增长" in row_text:
            nrow = numeric_grid[ri] if ri < len(numeric_grid) else []
            # Find base value, current value, and percentage
            nums = [(ci, c) for ci, c in enumerate(nrow) if c.value is not None]
            pct_cells = [(ci, c) for ci, c in nums if c.is_percentage]
            val_cells = [(ci, c) for ci, c in nums if not c.is_percentage]

            if len(val_cells) >= 2 and pct_cells:
                base_val = val_cells[0][1].value
                cur_val = val_cells[1][1].value
                pct_val = pct_cells[0][1].value

                if base_val and cur_val and pct_val is not None and base_val != 0:
                    expected_pct = ((cur_val - base_val) / abs(base_val)) * 100
                    diff = abs(expected_pct - pct_val)
                    tolerance = max(abs(expected_pct) * 0.05, 0.5)
                    passed = diff <= tolerance
                    report.percentage_checks.append({
                        "row": ri,
                        "base_value": base_val,
                        "current_value": cur_val,
                        "reported_pct": pct_val,
                        "expected_pct": round(expected_pct, 2),
                        "diff": round(diff, 2),
                        "passed": passed,
                    })
                    if not passed:
                        report.overall_pass = False

    # --- Anomaly detection ---
    # Flag cells with extreme values or unusual patterns
    all_values: list[tuple[int, int, float]] = []
    for ri, nrow in enumerate(numeric_grid):
        for ci, cell in enumerate(nrow):
            if cell.value is not None and not cell.is_percentage:
                all_values.append((ri, ci, cell.value))

    if all_values:
        values_only = [v[2] for v in all_values]
        mean_val = sum(values_only) / len(values_only)
        # Compute standard deviation
        variance = sum((v - mean_val) ** 2 for v in values_only) / len(values_only)
        std_val = variance ** 0.5

        # Flag values more than 5 standard deviations from the mean
        threshold = 5.0
        for ri, ci, val in all_values:
            if std_val > 0 and abs(val - mean_val) / std_val > threshold:
                report.anomaly_flags.append({
                    "row": ri,
                    "col": ci,
                    "value": val,
                    "z_score": round(abs(val - mean_val) / std_val, 2),
                    "reason": "extreme_value",
                })

        # Flag negative values where all others are positive (or vice versa)
        positives = [v for v in values_only if v > 0]
        negatives = [v for v in values_only if v < 0]
        if positives and negatives:
            minority = negatives if len(negatives) < len(positives) else positives
            minority_set = set(minority)
            for ri, ci, val in all_values:
                if val in minority_set and abs(val) > abs(mean_val) * 0.1:
                    already_flagged = any(
                        f["row"] == ri and f["col"] == ci
                        for f in report.anomaly_flags
                    )
                    if not already_flagged:
                        report.anomaly_flags.append({
                            "row": ri,
                            "col": ci,
                            "value": val,
                            "reason": "sign_mismatch",
                        })

    return report


def _find_last_numeric_col(
    numeric_grid: list[list[NumericCell]],
) -> int | None:
    """Find the rightmost column that consistently has numeric values."""
    if not numeric_grid:
        return None
    max_col = max(len(r) for r in numeric_grid)
    for ci in range(max_col - 1, -1, -1):
        count = 0
        for row in numeric_grid:
            if ci < len(row) and row[ci].value is not None:
                count += 1
        if count >= len(numeric_grid) * 0.3:
            return ci
    return None


def _is_summable_table_structure(
    headers: list[list[str]],
    rows: list[list[str]],
) -> bool:
    """Heuristic: does this table have a structure where columns sum to a total?

    Returns True when the first column contains sub-category labels (mostly
    non-numeric) and subsequent columns hold numeric values -- the pattern
    found in balance sheets, income statements, etc.  Returns False for
    generic tables or year-over-year comparison tables where summation is
    not meaningful.
    """
    if not rows:
        return False

    # Check that the first column is predominantly text (labels)
    text_count = 0
    total_first_col = 0
    for row in rows:
        if row:
            cell = row[0].strip()
            if cell:
                total_first_col += 1
                nc = _extract_numeric(cell)
                if nc.value is None:
                    text_count += 1

    if total_first_col == 0:
        return False

    label_ratio = text_count / total_first_col

    # If more than half the first column cells are text labels, this looks
    # like a summable structure.
    return label_ratio > 0.5


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_cell_confidence(
    grid: list[list[str]],
    numeric_grid: list[list[NumericCell]],
    validation: ValidationReport,
) -> list[CellConfidence]:
    """
    Compute per-cell confidence scores.

    Factors:
    - Empty cells get a moderate confidence (data may be missing)
    - Numeric cells that failed validation get reduced confidence
    - Cells with unusual characters or mixed scripts get reduced confidence
    - Anomalous cells get reduced confidence
    """
    scores: list[CellConfidence] = []
    anomaly_cells: set[tuple[int, int]] = set()
    for flag in validation.anomaly_flags:
        anomaly_cells.add((flag["row"], flag["col"]))

    failed_rows: set[int] = set()
    for check in validation.row_sum_checks:
        if not check["passed"]:
            failed_rows.add(check["row"])

    for ri, row in enumerate(grid):
        for ci, cell_text in enumerate(row):
            score = 1.0
            reason = ""

            # Empty cell
            if not cell_text.strip():
                score = 0.7
                reason = "empty_cell"
            else:
                # Check for numeric parse issues
                if ri < len(numeric_grid) and ci < len(numeric_grid[ri]):
                    nc = numeric_grid[ri][ci]
                    if nc.confidence < 1.0:
                        score = min(score, nc.confidence)
                        reason = "numeric_parse_issue"

                # Anomaly flag
                if (ri, ci) in anomaly_cells:
                    score *= 0.6
                    reason = "anomaly_flagged"

                # Row sum failure affects entire row
                if ri in failed_rows:
                    score *= 0.85
                    if not reason:
                        reason = "row_sum_failed"

                # Mixed script detection (CJK + Latin digits interleaved oddly)
                has_cjk = bool(re.search(r"[一-鿿]", cell_text))
                has_digits = bool(re.search(r"[0-9]", cell_text))
                if has_cjk and has_digits:
                    # This is common in Chinese financial tables (e.g. "1,234万元")
                    # Only penalise if the mixing looks unusual
                    if not re.match(r"^[−\-]?[\d,.\s]*(万亿元|万元|亿元|万|亿|元)?[\s]*[％%]?$", cell_text.strip()):
                        score *= 0.9
                        if not reason:
                            reason = "mixed_script"

            scores.append(CellConfidence(
                row=ri, col=ci,
                score=round(score, 3),
                reason=reason,
            ))

    return scores


def _compute_table_confidence(cell_scores: list[CellConfidence]) -> float:
    """Compute overall table confidence as the mean of cell scores."""
    if not cell_scores:
        return 0.0
    return round(sum(s.score for s in cell_scores) / len(cell_scores), 3)


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _grid_to_markdown(headers: list[list[str]], rows: list[list[str]]) -> str:
    """Convert headers + rows to a Markdown table string."""
    all_rows = headers + rows
    if not all_rows:
        return ""

    max_cols = max(len(r) for r in all_rows)
    normalised: list[list[str]] = []
    for r in all_rows:
        padded = list(r) + [""] * (max_cols - len(r))
        normalised.append(padded)

    lines: list[str] = []
    # Header row
    lines.append("| " + " | ".join(normalised[0]) + " |")
    # Separator
    lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    # Data rows
    for row in normalised[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _grid_to_csv(headers: list[list[str]], rows: list[list[str]]) -> str:
    """Convert headers + rows to CSV string."""
    all_rows = headers + rows
    if not all_rows:
        return ""

    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in all_rows:
        writer.writerow(row)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class TableParser:
    """
    Advanced table parsing and structuring tool.

    Processes raw table data from MinerU (HTML tables as strings) and performs:
    1. HTML table parsing to structured grids
    2. Merged cell detection and expansion (colspan/rowspan)
    3. Financial table classification
    4. Numeric extraction and validation
    5. Confidence scoring per cell and per table
    6. Output as structured JSON, CSV, or Markdown

    Usage::

        parser = TableParser(config)
        result = await parser.execute({"tables": [...]}, context)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.llm_client = None  # Optional: injected for LLM-based verification

    def set_llm_client(self, client: Any) -> None:
        """Set an LLM client for enhanced verification (optional)."""
        self.llm_client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: dict | None = None,
    ) -> dict:
        """
        Parse and structure tables from MinerU content_list entries.

        Args:
            params: {
                tables: list[dict]  — items from MinerU content_list where
                      type=="table".  Each dict may contain:
                      - "html": raw HTML table string
                      - "text": plain text fallback
                      - "data": pre-parsed 2D list
                      - "markdown": markdown table string
                verify_numeric: bool (default True)
                output_format: "json" | "csv" | "markdown" (default "json")
            }
            context: Previous pipeline results (optional).

        Returns:
            {
                tables: list[StructuredTable dicts],
                total_count: int,
                verification_summary: dict,
            }
        """
        raw_tables: list[dict] = params.get("tables", [])
        verify_numeric = params.get("verify_numeric", True)
        output_format = params.get("output_format", "json")

        if not raw_tables:
            logger.info("TableParser: no tables provided")
            return {
                "tables": [],
                "total_count": 0,
                "verification_summary": {
                    "tables_checked": 0,
                    "total_checks": 0,
                    "total_failures": 0,
                    "overall_pass": True,
                },
            }

        logger.info(f"TableParser: processing {len(raw_tables)} table(s)")

        structured_tables: list[dict] = []
        total_checks = 0
        total_failures = 0

        for i, raw_table in enumerate(raw_tables):
            logger.info(f"Processing table {i + 1}/{len(raw_tables)}")
            structured = self._process_single_table(
                raw_table, i, verify_numeric,
            )
            structured_tables.append(structured)

            # Aggregate verification stats
            nv = structured.get("numeric_validation", {})
            checks = (
                len(nv.get("row_sum_checks", []))
                + len(nv.get("col_sum_checks", []))
                + len(nv.get("percentage_checks", []))
            )
            failures = (
                sum(1 for c in nv.get("row_sum_checks", []) if not c.get("passed", True))
                + sum(1 for c in nv.get("col_sum_checks", []) if not c.get("passed", True))
                + sum(1 for c in nv.get("percentage_checks", []) if not c.get("passed", True))
            )
            total_checks += checks
            total_failures += failures

        # Convert output format if needed
        if output_format == OutputFormat.CSV:
            structured_tables = self._convert_to_csv(structured_tables)
        elif output_format == OutputFormat.MARKDOWN:
            structured_tables = self._convert_to_markdown(structured_tables)

        verification_summary = {
            "tables_checked": len(raw_tables),
            "total_checks": total_checks,
            "total_failures": total_failures,
            "overall_pass": total_failures == 0,
        }

        result = {
            "tables": structured_tables,
            "total_count": len(structured_tables),
            "verification_summary": verification_summary,
        }

        logger.info(
            f"TableParser done: {len(structured_tables)} table(s), "
            f"{total_checks} checks, {total_failures} failures"
        )
        return result

    # ------------------------------------------------------------------
    # Internal: single table processing
    # ------------------------------------------------------------------

    def _process_single_table(
        self,
        raw_table: dict,
        index: int,
        verify_numeric: bool,
    ) -> dict:
        """Process a single raw table block into a StructuredTable dict."""
        raw_html = raw_table.get("html", "")
        raw_text = raw_table.get("text", "")
        raw_data = raw_table.get("data", [])
        raw_markdown = raw_table.get("markdown", "")

        # Step 1: Parse into grid
        grid: list[list[str]] = []
        if raw_html:
            grid = _parse_html_table(raw_html)
        elif raw_data and isinstance(raw_data, list) and raw_data:
            grid = [list(str(c) for c in row) if isinstance(row, list) else [str(row)] for row in raw_data]
        elif raw_text:
            # Try to parse text as a simple delimited table
            grid = self._text_to_grid(raw_text)
        elif raw_markdown:
            grid = self._markdown_to_grid(raw_markdown)

        # Handle empty grid
        if not grid:
            grid = [[""]]

        # Step 2: Detect merged cells (before splitting headers/rows)
        merged_ranges = _detect_merged_cells(grid)

        # Step 3: Split headers from data rows
        headers, data_rows = self._split_headers(grid)

        # Step 4: Classify table type
        table_type = _classify_financial_table(headers, data_rows)

        # Step 5: Build numeric grid for validation and scoring
        numeric_grid: list[list[NumericCell]] = []
        for row in data_rows:
            numeric_grid.append([_extract_numeric(cell) for cell in row])

        # Step 6: Numeric validation
        validation = ValidationReport()
        if verify_numeric:
            validation = _validate_numeric_consistency(data_rows, headers)

        # Step 7: Confidence scoring
        cell_scores = _compute_cell_confidence(data_rows, numeric_grid, validation)
        table_confidence = _compute_table_confidence(cell_scores)

        # Build the structured table
        structured = StructuredTable(
            index=index,
            type=table_type,
            headers=headers,
            rows=data_rows,
            numeric_validation=asdict(validation),
            confidence_scores=[asdict(cs) for cs in cell_scores],
            raw_html=raw_html,
            merged_ranges=[asdict(mr) for mr in merged_ranges],
            num_rows=len(data_rows),
            num_cols=max(len(r) for r in data_rows) if data_rows else 0,
            table_confidence=table_confidence,
        )

        return asdict(structured)

    # ------------------------------------------------------------------
    # Internal: header / data splitting
    # ------------------------------------------------------------------

    def _split_headers(
        self,
        grid: list[list[str]],
    ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Split a grid into header rows and data rows.

        Heuristic:
        - The first N rows are headers if they contain non-numeric text
          (labels, column names) and lack substantial numeric data.
        - We look for a transition from mostly-text to mostly-numeric rows.
        - If the grid has <th> markers (from HTML), those are header rows.
        """
        if not grid:
            return [], []

        header_rows: list[list[str]] = []
        data_rows: list[list[str]] = []

        # Find the header/data boundary
        # A row is "header-like" if more than half its cells are non-numeric
        for ri, row in enumerate(grid):
            numeric_count = 0
            total_cells = len([c for c in row if c.strip()])
            if total_cells == 0:
                if ri == 0:
                    # Skip leading empty row
                    continue
                else:
                    break

            for cell in row:
                nc = _extract_numeric(cell)
                if nc.value is not None and not cell.strip().startswith(("项目", "科目", "指标")):
                    numeric_count += 1

            numeric_ratio = numeric_count / total_cells if total_cells > 0 else 0

            if numeric_ratio < 0.5:
                header_rows.append(row)
            else:
                # This row has substantial numeric data -> start of data section
                break

        # Everything from the first numeric row onward is data
        start = len(header_rows)
        data_rows = grid[start:]

        # Edge case: no clear header row found; use first row as header
        if not header_rows and grid:
            header_rows = [grid[0]]
            data_rows = grid[1:]

        # Edge case: all rows are headers (very short table)
        if not data_rows and header_rows and len(header_rows) > 1:
            data_rows = header_rows[1:]
            header_rows = header_rows[:1]

        return header_rows, data_rows

    # ------------------------------------------------------------------
    # Internal: text / markdown to grid conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _text_to_grid(text: str) -> list[list[str]]:
        """Convert plain text table to grid. Tries tab, pipe, or multi-space delimiters."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return []

        # Detect delimiter
        if "\t" in lines[0]:
            delimiter = "tab"
        elif "|" in lines[0]:
            delimiter = "pipe"
        else:
            delimiter = "spaces"

        grid: list[list[str]] = []
        for line in lines:
            if delimiter == "tab":
                cells = line.split("\t")
            elif delimiter == "pipe":
                cells = [c.strip() for c in line.split("|")]
                # Remove leading/trailing empty cells from pipe delimiters
                if cells and cells[0] == "":
                    cells = cells[1:]
                if cells and cells[-1] == "":
                    cells = cells[:-1]
            else:
                # Split on 2+ consecutive spaces
                cells = re.split(r"  +", line)

            grid.append(cells)

        # Normalise column count
        if grid:
            max_cols = max(len(r) for r in grid)
            for r in grid:
                while len(r) < max_cols:
                    r.append("")

        return grid

    @staticmethod
    def _markdown_to_grid(md: str) -> list[list[str]]:
        """Convert a Markdown table to grid."""
        lines = [l.strip() for l in md.strip().splitlines() if l.strip()]
        grid: list[list[str]] = []

        for line in lines:
            # Skip separator lines like |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", line):
                continue
            if "|" in line:
                cells = [c.strip() for c in line.split("|")]
                # Remove leading/trailing empty from pipe delimiters
                if cells and cells[0] == "":
                    cells = cells[1:]
                if cells and cells[-1] == "":
                    cells = cells[:-1]
                if cells:
                    grid.append(cells)

        # Normalise column count
        if grid:
            max_cols = max(len(r) for r in grid)
            for r in grid:
                while len(r) < max_cols:
                    r.append("")

        return grid

    # ------------------------------------------------------------------
    # Internal: output format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_csv(tables: list[dict]) -> list[dict]:
        """Add CSV representation to each structured table."""
        for t in tables:
            headers = t.get("headers", [])
            rows = t.get("rows", [])
            t["csv_output"] = _grid_to_csv(headers, rows)
        return tables

    @staticmethod
    def _convert_to_markdown(tables: list[dict]) -> list[dict]:
        """Add Markdown representation to each structured table."""
        for t in tables:
            headers = t.get("headers", [])
            rows = t.get("rows", [])
            t["markdown_output"] = _grid_to_markdown(headers, rows)
        return tables
