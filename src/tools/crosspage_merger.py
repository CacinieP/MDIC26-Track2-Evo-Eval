"""
Cross-Page Merge and Reference Resolution Tool.

Handles:
1. Cross-page table detection and merging
   - Compare consecutive pages for table continuation patterns
   - Match by column count, header similarity, content type pattern
   - Handle tables split mid-row
   - Reconstruct complete table from fragments

2. Cross-page paragraph/text merging
   - Detect sentences/paragraphs broken across pages
   - Rejoin with proper spacing

3. Reference resolution (指代消解)
   - Detect pronouns and references: "该公司", "上述金额", "其", "本项目", etc.
   - Build entity registry from document context
   - Resolve references using LLM (if available) or rule-based fallback
   - Track resolution confidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResolutionResult:
    """Result of resolving a single reference expression."""
    reference: str = ""                    # the original reference text
    resolved_entity: str = ""              # what it resolves to
    confidence: float = 0.0                # 0.0 – 1.0
    method: str = "rule"                   # "rule" | "llm" | "context"
    context_snippet: str = ""              # surrounding text used for resolution
    page_idx: int = -1                     # page where the reference appeared


@dataclass
class MergeOperation:
    """Record of a single merge decision for traceability."""
    operation_type: str = ""               # "table_merge" | "text_merge"
    source_indices: list[int] = field(default_factory=list)
    source_page_indices: list[int] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference pattern library (Chinese + English)
# ---------------------------------------------------------------------------

# Pronouns and demonstrative references commonly found in Chinese documents
_PRONOUN_PATTERNS: list[re.Pattern] = [
    # Company / organisation references
    re.compile(r"该(?:公司|企业|单位|机构|组织|集团)"),
    re.compile(r"本(?:公司|企业|单位|机构|组织|集团|项目|期|年|月|季度|报告)"),
    re.compile(r"其(?:他|他)?(?:他|她|它)?(?:公司|企业|方)?"),
    # Amount / figure references
    re.compile(r"上述(?:金额|数额|数字|数据|金额|款项|费用|成本|收入|利润)"),
    re.compile(r"前述(?:金额|数额|数字|数据|款项|费用|成本|收入|利润)"),
    re.compile(r"该(?:金额|数额|数字|数据|款项|费用|成本|收入|利润|比例|比率)"),
    re.compile(r"以上(?:金额|数额|数字|数据|款项)"),
    # Project / contract references
    re.compile(r"该(?:项目|合同|协议|工程|产品|方案|计划)"),
    re.compile(r"上述(?:项目|合同|协议|工程|产品|方案|计划)"),
    # General pronouns
    re.compile(r"\b其\b"),
    re.compile(r"\b该\b"),
    re.compile(r"\b此\b"),
    re.compile(r"上述"),
    re.compile(r"前述"),
    re.compile(r"以下"),
    re.compile(r"如下"),
    # Time references
    re.compile(r"同(?:期|年|月|日)"),
    re.compile(r"上(?:年|期|季度|月)"),
    re.compile(r"报告(?:期|年度|期内)"),
]

# Entity extraction patterns (to build the entity registry)
_ENTITY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("company", re.compile(
        r"([一-鿿]+(?:有限公司|股份|集团|公司|企业|事务所|研究院|研究所|中心|基金))")),
    ("money", re.compile(
        r"([\d,，.]+\s*(?:万元|亿元|元|万|亿|人民币|美元|港币|欧元|日元|英镑))")),
    ("percentage", re.compile(r"([\d.]+%|百分之[\d.]+)")),
    ("date", re.compile(
        r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|"
        r"\d{4}\s*年\s*\d{1,2}\s*月|"
        r"\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2})")),
    ("project", re.compile(
        r"((?:[一-鿿]+)?(?:项目|工程|计划|方案|课题)[一-鿿]*)")),
    ("person", re.compile(
        r"((?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+|"
        r"[一-鿿]{2,4}(?:先生|女士|教授|博士|总|经理|主任))")),
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_table_data(block: dict) -> list[list[str]]:
    """Extract a 2D cell array from a content-list table block.

    MinerU table blocks may store data under different keys depending on
    how they were parsed.  This helper normalises the representation.
    """
    # Try the most common keys
    for key in ("data", "table_body", "cells", "rows", "table_data"):
        val = block.get(key)
        if isinstance(val, list) and val and isinstance(val[0], list):
            return val
    # If the block has a markdown representation, try to parse it
    md = block.get("markdown", "")
    if md:
        return _parse_markdown_table(md)
    return []


def _parse_markdown_table(md: str) -> list[list[str]]:
    """Parse a Markdown table string into a 2D list."""
    rows: list[list[str]] = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Skip separator rows like |---|---|
        if all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def _table_column_count(table_data: list[list[str]]) -> int:
    """Return the number of columns in a table (max across rows)."""
    if not table_data:
        return 0
    return max(len(row) for row in table_data)


def _is_likely_header_row(row: list[str]) -> bool:
    """Heuristic: is this row a table header?"""
    text = "".join(row).strip()
    if not text:
        return False
    # Header rows tend to have no digits or very few
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.4:
        return False
    # Common Chinese header keywords
    header_kw = {"项目", "类别", "名称", "合计", "小计", "合计", "序号",
                 "编号", "类型", "内容", "备注", "说明", "单位", "日期",
                 "金额", "数量", "比率", "年度", "本期", "上期"}
    joined = "".join(row)
    return any(kw in joined for kw in header_kw)


def _row_similarity(row_a: list[str], row_b: list[str]) -> float:
    """String similarity between two rows (SequenceMatcher ratio)."""
    text_a = "|".join(row_a)
    text_b = "|".join(row_b)
    return SequenceMatcher(None, text_a, text_b).ratio()


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class CrossPageMerger:
    """
    Cross-page merge and reference resolution tool.

    Usage::

        merger = CrossPageMerger(config)
        result = await merger.execute(
            {"content_list": [...], "enable_reference_resolution": True},
            context,
        )
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.llm_client: Any = None  # Injected externally or via params
        self._merge_log: list[MergeOperation] = []

    def set_llm_client(self, client: Any) -> None:
        """Set the LLM client for reference resolution."""
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
        Execute cross-page merge and reference resolution.

        Args:
            params: {
                content_list: list[dict]  -- MinerU content items with page_idx,
                enable_reference_resolution: bool (default True),
                llm_client: optional LLM client override,
            }
            context: Previous pipeline results.

        Returns:
            {
                merged_content: list[dict],
                merge_operations: list[dict],
                entity_registry: dict,
                reference_resolutions: list[dict],
            }
        """
        content_list: list[dict] = params.get("content_list", [])
        enable_ref = params.get("enable_reference_resolution", True)
        llm_override = params.get("llm_client")
        client = llm_override or self.llm_client

        self._merge_log = []

        if not content_list:
            logger.info("CrossPageMerger: empty content_list, nothing to do")
            return {
                "merged_content": [],
                "merge_operations": [],
                "entity_registry": {},
                "reference_resolutions": [],
            }

        logger.info(
            f"CrossPageMerger: processing {len(content_list)} content items"
        )

        # ---- Step 1: Sort by page index (stable) ----
        sorted_content = sorted(
            content_list,
            key=lambda x: x.get("page_idx", 0),
        )

        # ---- Step 2: Build entity registry (for reference resolution) ----
        entity_registry = self._build_entity_registry(sorted_content)
        logger.info(
            f"Entity registry built: "
            f"{sum(len(v) for v in entity_registry.values())} entities across "
            f"{len(entity_registry)} categories"
        )

        # ---- Step 3: Cross-page table merging ----
        merged, table_ops = self._merge_cross_page_tables(sorted_content)
        logger.info(f"Table merge: {len(table_ops)} operations performed")

        # ---- Step 4: Cross-page text merging ----
        merged, text_ops = self._merge_cross_page_text(merged)
        logger.info(f"Text merge: {len(text_ops)} operations performed")

        # ---- Step 5: Reference resolution ----
        reference_resolutions: list[ResolutionResult] = []
        if enable_ref:
            reference_resolutions = await self._resolve_all_references(
                merged, entity_registry, client,
            )
            logger.info(
                f"Reference resolution: "
                f"{len(reference_resolutions)} references resolved"
            )

        # Collect all merge operations
        all_ops = table_ops + text_ops

        return {
            "merged_content": merged,
            "merge_operations": [asdict(op) for op in all_ops],
            "entity_registry": entity_registry,
            "reference_resolutions": [asdict(r) for r in reference_resolutions],
        }

    # ==================================================================
    # TABLE MERGE LOGIC
    # ==================================================================

    def _merge_cross_page_tables(
        self,
        content_list: list[dict],
    ) -> tuple[list[dict], list[MergeOperation]]:
        """
        Detect and merge tables that span consecutive pages.

        Returns (merged_content, merge_operations).
        """
        if not content_list:
            return content_list, []

        # Separate table blocks from non-table blocks
        table_blocks: list[tuple[int, dict]] = []  # (index_in_list, block)
        for i, block in enumerate(content_list):
            if block.get("type") == "table":
                table_blocks.append((i, block))

        if len(table_blocks) < 2:
            return content_list, []

        # Group table blocks that are continuations of each other
        groups = self._group_table_fragments(table_blocks, content_list)

        operations: list[MergeOperation] = []
        merged_indices: set[int] = set()  # indices consumed into a merge
        merged_tables: dict[int, dict] = {}  # replacement for first index

        for group in groups:
            if len(group) <= 1:
                continue

            fragments = [content_list[idx] for idx in group]
            page_indices = [f.get("page_idx", -1) for f in fragments]
            merged_table = self._merge_table_fragments(fragments)

            op = MergeOperation(
                operation_type="table_merge",
                source_indices=group,
                source_page_indices=page_indices,
                reason=f"Table continued across pages {page_indices}",
                confidence=merged_table.get("_merge_confidence", 0.8),
                details={
                    "fragment_count": len(fragments),
                    "total_rows": len(_extract_table_data(merged_table)),
                    "page_indices": page_indices,
                },
            )
            operations.append(op)

            # Mark all but the first index as consumed
            merged_indices.update(group[1:])
            merged_tables[group[0]] = merged_table

            logger.info(
                f"Merged {len(fragments)} table fragments from pages "
                f"{page_indices} into one table"
            )

        # Rebuild content list: keep non-consumed items, replace first of
        # each group with the merged table
        result: list[dict] = []
        for i, block in enumerate(content_list):
            if i in merged_indices:
                continue
            if i in merged_tables:
                result.append(merged_tables[i])
            else:
                result.append(block)

        return result, operations

    def _group_table_fragments(
        self,
        table_blocks: list[tuple[int, dict]],
        content_list: list[dict],
    ) -> list[list[int]]:
        """
        Group table block indices that belong to the same logical table.

        Returns a list of groups, each group is a list of indices into
        *content_list*.
        """
        groups: list[list[int]] = []
        current_group: list[int] = [table_blocks[0][0]]

        for k in range(1, len(table_blocks)):
            prev_idx = table_blocks[k - 1][0]
            curr_idx = table_blocks[k][0]
            prev_block = content_list[prev_idx]
            curr_block = content_list[curr_idx]

            if self._detect_table_continuation(prev_block, curr_block):
                current_group.append(curr_idx)
            else:
                groups.append(current_group)
                current_group = [curr_idx]

        groups.append(current_group)
        return groups

    def _detect_table_continuation(self, table_a: dict, table_b: dict) -> bool:
        """
        Determine whether *table_b* is a continuation of *table_a*.

        Matching criteria:
        1. Pages are consecutive (or same page).
        2. Column counts match.
        3. Header similarity or content-type pattern consistency.
        4. No strong header row in table_b (suggesting it continues, not starts).
        """
        page_a = table_a.get("page_idx", -1)
        page_b = table_b.get("page_idx", -1)

        # Criterion 1: Pages must be consecutive (or on the same page)
        if page_a >= 0 and page_b >= 0 and abs(page_b - page_a) > 1:
            logger.debug(
                f"Table continuation rejected: page gap {page_a} -> {page_b}"
            )
            return False

        data_a = _extract_table_data(table_a)
        data_b = _extract_table_data(table_b)

        if not data_a or not data_b:
            # Fall back to column-count field if present
            cols_a = table_a.get("cols", table_a.get("columns", 0))
            cols_b = table_b.get("cols", table_b.get("columns", 0))
            if cols_a and cols_b and cols_a == cols_b:
                logger.debug(
                    "Table continuation accepted (matching column count from metadata)"
                )
                return True
            return False

        cols_a = _table_column_count(data_a)
        cols_b = _table_column_count(data_b)

        # Criterion 2: Column counts must match (within tolerance of 1)
        if abs(cols_a - cols_b) > 1:
            logger.debug(
                f"Table continuation rejected: column mismatch {cols_a} vs {cols_b}"
            )
            return False

        # Criterion 3: Header analysis
        has_header_a = data_a and _is_likely_header_row(data_a[0])
        has_header_b = data_b and _is_likely_header_row(data_b[0])

        # If table_b starts with a header row that differs from table_a's
        # header, it is likely a new table, not a continuation.
        if has_header_b and has_header_a and data_a and data_b:
            similarity = _row_similarity(data_a[0], data_b[0])
            if similarity > 0.7:
                # Duplicate header: table_b might repeat the header for
                # readability on a new page — still a continuation.
                logger.debug(
                    f"Table continuation accepted (duplicate header, sim={similarity:.2f})"
                )
                return True
            else:
                logger.debug(
                    f"Table continuation rejected (different header, sim={similarity:.2f})"
                )
                return False

        # If table_b has NO header, it likely continues table_a
        if not has_header_b:
            logger.debug("Table continuation accepted (no header in B)")
            return True

        # Criterion 4: Content type pattern consistency
        # Check if the cell content types (numeric vs text) are consistent
        if cols_a == cols_b and cols_a > 0:
            pattern_match = self._content_type_pattern_match(data_a, data_b)
            if pattern_match:
                logger.debug("Table continuation accepted (content pattern match)")
                return True

        # If on consecutive pages and same column count, accept with
        # moderate confidence even without strong signals.
        if page_a >= 0 and page_b >= 0 and abs(page_b - page_a) <= 1:
            if cols_a == cols_b and cols_a > 0:
                logger.debug(
                    "Table continuation accepted (consecutive pages, same cols)"
                )
                return True

        return False

    def _content_type_pattern_match(
        self,
        data_a: list[list[str]],
        data_b: list[list[str]],
    ) -> bool:
        """
        Check whether the content type pattern (numeric/text per column) is
        consistent between two tables.
        """
        if not data_a or not data_b:
            return False

        # Use the last few rows of A and the first few rows of B
        sample_a = data_a[-3:] if len(data_a) >= 3 else data_a
        sample_b = data_b[:3] if len(data_b) >= 3 else data_b

        def _column_type_profile(rows: list[list[str]]) -> list[float]:
            """Return per-column ratio of numeric cells."""
            if not rows:
                return []
            max_cols = max(len(r) for r in rows)
            profile = []
            for col in range(max_cols):
                numeric = 0
                total = 0
                for row in rows:
                    if col < len(row):
                        cell = row[col].strip().replace(",", "").replace("，", "")
                        total += 1
                        # Check if the cell looks numeric (including negative,
                        # percentage, Chinese units)
                        if re.match(
                            r"^[+-]?[\d.]+[%％]?$|"
                            r"^[\d.]+[万亿]?"
                            r"(?:元|万元|亿元|美元|港币)?$", cell
                        ):
                            numeric += 1
                profile.append(numeric / max(total, 1))
            return profile

        profile_a = _column_type_profile(sample_a)
        profile_b = _column_type_profile(sample_b)

        if not profile_a or not profile_b:
            return False

        # Compare profiles -- they should be roughly similar
        min_len = min(len(profile_a), len(profile_b))
        matching = sum(
            1 for i in range(min_len)
            if abs(profile_a[i] - profile_b[i]) < 0.4
        )
        return matching >= min_len * 0.6

    def _merge_table_fragments(self, fragments: list[dict]) -> dict:
        """
        Merge a list of table fragments into a single reconstructed table.

        Handles:
        - Duplicate header rows (removes repeated headers)
        - Mid-row splits (attempts recombination)
        - Preserves metadata from the first fragment
        """
        if not fragments:
            return {}
        if len(fragments) == 1:
            return fragments[0]

        # Extract 2D data from each fragment
        all_data: list[list[str]] = []
        first_fragment = fragments[0]
        first_data = _extract_table_data(first_fragment)

        # Keep header from the first fragment
        if first_data:
            header = first_data[0]
            all_data.append(header)
            # Add remaining rows from first fragment
            for row in first_data[1:]:
                all_data.append(row)

        # Process subsequent fragments
        for frag in fragments[1:]:
            frag_data = _extract_table_data(frag)
            if not frag_data:
                continue

            # Check if the first row is a duplicate header
            if all_data and frag_data:
                similarity = _row_similarity(all_data[0], frag_data[0])
                if similarity > 0.7 and _is_likely_header_row(frag_data[0]):
                    # Skip the duplicate header
                    logger.debug("Skipping duplicate header in table fragment")
                    frag_data = frag_data[1:]

            # Check for mid-row split: if the last row of the accumulated
            # data ends abruptly (fewer columns than expected), try to join
            # it with the first row of this fragment.
            if all_data and frag_data:
                expected_cols = _table_column_count(all_data)
                last_row = all_data[-1]
                first_row = frag_data[0]
                if len(last_row) < expected_cols and first_row:
                    # Possible mid-row split
                    joined = last_row + first_row
                    if len(joined) <= expected_cols + 1:
                        all_data[-1] = joined[:expected_cols]
                        frag_data = frag_data[1:]
                        logger.debug("Rejoined mid-row split in table")

            for row in frag_data:
                all_data.append(row)

        # Build the merged table dict, preserving metadata from the first
        # fragment and adding merge-specific metadata
        merged = dict(first_fragment)
        merged["data"] = all_data
        merged["rows"] = len(all_data)
        merged["cols"] = _table_column_count(all_data)
        merged["markdown"] = self._table_to_markdown(all_data)
        merged["_merge_confidence"] = 0.85
        merged["_merged_from_pages"] = [
            f.get("page_idx", -1) for f in fragments
        ]
        merged["_merged_from_fragments"] = len(fragments)

        # If there was a table_index, keep the first one
        # page_idx is preserved from first_fragment via dict(first_fragment)

        return merged

    # ==================================================================
    # TEXT MERGE LOGIC
    # ==================================================================

    def _merge_cross_page_text(
        self,
        content_list: list[dict],
    ) -> tuple[list[dict], list[MergeOperation]]:
        """
        Detect and merge text blocks broken across page boundaries.

        Supports merging chains of 3+ consecutive pages by first identifying
        all continuation relationships, grouping them into chains, and then
        merging each chain in a single pass.

        Returns (merged_content, merge_operations).
        """
        if not content_list:
            return content_list, []

        operations: list[MergeOperation] = []

        # Phase 1: Identify all continuation relationships between consecutive blocks
        # continuation_links[i] = i+1 means block i continues into block i+1
        continuation_links: dict[int, int] = {}
        for i in range(len(content_list) - 1):
            block_a = content_list[i]
            block_b = content_list[i + 1]

            if block_a.get("type") != "text" or block_b.get("type") != "text":
                continue

            text_a = block_a.get("text", "")
            text_b = block_b.get("text", "")

            if not text_a or not text_b:
                continue

            if self._detect_text_continuation(text_a, text_b):
                continuation_links[i] = i + 1

        if not continuation_links:
            return content_list, []

        # Phase 2: Group consecutive links into chains
        # A chain is a list of indices [start, start+1, ..., end] where
        # each consecutive pair is linked.
        visited: set[int] = set()
        chains: list[list[int]] = []

        for start_idx in sorted(continuation_links.keys()):
            if start_idx in visited:
                continue
            chain = [start_idx]
            visited.add(start_idx)
            current = start_idx
            while current in continuation_links:
                nxt = continuation_links[current]
                chain.append(nxt)
                visited.add(nxt)
                current = nxt
            if len(chain) >= 2:
                chains.append(chain)

        # Phase 3: Merge each chain into a single block
        merged_indices: set[int] = set()
        replacements: dict[int, dict] = {}

        for chain in chains:
            # chain = [i0, i1, i2, ...]  where i0 is the anchor
            anchor = chain[0]
            anchor_block = content_list[anchor]
            merged_text = anchor_block.get("text", "")
            page_indices = [anchor_block.get("page_idx", -1)]

            for idx in chain[1:]:
                block = content_list[idx]
                text_b = block.get("text", "")
                join_char = self._determine_join_char(merged_text, text_b)
                merged_text = merged_text + join_char + text_b
                page_indices.append(block.get("page_idx", -1))
                merged_indices.add(idx)

            merged_block = dict(anchor_block)
            merged_block["text"] = merged_text
            merged_block["_merged_from_pages"] = page_indices
            merged_block["_merge_confidence"] = 0.9
            replacements[anchor] = merged_block

            op = MergeOperation(
                operation_type="text_merge",
                source_indices=chain,
                source_page_indices=page_indices,
                reason=f"Text broken across pages {page_indices}",
                confidence=0.9,
                details={
                    "chain_length": len(chain),
                    "merged_length": len(merged_text),
                },
            )
            operations.append(op)

            logger.debug(
                f"Text merge: chain of {len(chain)} blocks across pages {page_indices}"
            )

        # Rebuild content list
        result: list[dict] = []
        for i, block in enumerate(content_list):
            if i in merged_indices:
                continue
            if i in replacements:
                result.append(replacements[i])
            else:
                result.append(block)

        return result, operations

    def _detect_text_continuation(self, text_a: str, text_b: str) -> bool:
        """
        Determine whether *text_b* is a continuation of *text_a*.

        Heuristics:
        - text_a ends mid-sentence (no period/punctuation)
        - text_b starts with lowercase or mid-sentence content
        - The last word of text_a is split (cut off mid-word)
        - For Chinese: text_a does not end with sentence-final punctuation
        """
        if not text_a or not text_b:
            return False

        # Chinese sentence-final punctuation
        sentence_enders = set("。！？；：…\n.!?;:")
        # Characters that suggest text_a ends properly (not mid-sentence)
        proper_endings = sentence_enders | set("）)】》」』\"'")

        a_last = text_a.rstrip()[-1] if text_a.rstrip() else ""
        b_first = text_b.lstrip()[0] if text_b.lstrip() else ""

        if not a_last or not b_first:
            return False

        # Strong signal: text_a does NOT end with sentence-final punctuation
        a_ends_properly = a_last in proper_endings

        # Strong signal: text_b starts with a connector / continuation
        continuation_starters = set("而并且或以及此外同时另外")
        b_starts_continuation = b_first in continuation_starters

        # Strong signal: the last word in text_a is cut off (Latin text)
        a_words = text_a.rstrip().split()
        if a_words and len(a_words[-1]) <= 2 and a_words[-1][-1:].isalpha():
            # Short trailing fragment suggests a cut word
            return True

        # Chinese: text_a ends without punctuation AND text_b starts with
        # non-heading content
        if not a_ends_properly:
            # Check if text_b looks like a heading (starts with # or a
            # number followed by a dot)
            b_is_heading = bool(re.match(r"^[#\d]+[.、]", text_b.lstrip()))
            if not b_is_heading:
                # Additional check: text_b should not start with a capital
                # letter after a period (English) or a paragraph indent
                if b_starts_continuation:
                    return True

        # Weak signal: the concatenation makes grammatical sense (very
        # rough — just check if there's no double punctuation)
        if a_last not in sentence_enders and b_first not in sentence_enders:
            # Reject headings: "3.", "#1", "1、" should never be treated as
            # continuations, even if the prior block lacked proper punctuation.
            b_is_heading = bool(re.match(r"^[#\d]+[.、]", text_b.lstrip()))
            if b_is_heading:
                return False
            # Check if text_b starts with lowercase (Latin), a digit
            # (e.g. "5,000万元..."), or a Chinese character.
            if b_first.islower() or b_first.isdigit() or ord(b_first) > 0x4E00:
                if not a_ends_properly:
                    return True

        return False

    def _determine_join_char(self, text_a: str, text_b: str) -> str:
        """Determine the appropriate character to join two text fragments."""
        a_last = text_a.rstrip()[-1] if text_a.rstrip() else ""
        b_first = text_b.lstrip()[0] if text_b.lstrip() else ""

        # If the last char of A is a CJK character and the first char of B
        # is also CJK, no separator is needed
        is_cjk_a = ord(a_last) >= 0x4E00 and ord(a_last) <= 0x9FFF if a_last else False
        is_cjk_b = ord(b_first) >= 0x4E00 and ord(b_first) <= 0x9FFF if b_first else False

        if is_cjk_a and is_cjk_b:
            return ""
        # If either side has trailing/leading whitespace, join with space
        if text_a.endswith(" ") or text_b.startswith(" "):
            return " "
        # Latin text typically needs a space
        if a_last.isalpha() and b_first.isalpha():
            return " "
        # Default: no separator
        return ""

    # ==================================================================
    # ENTITY REGISTRY
    # ==================================================================

    def _build_entity_registry(self, content_list: list[dict]) -> dict:
        """
        Build a registry of named entities from the full document content.

        Returns a dict keyed by entity category, each containing a list
        of entity records with text, page_idx, and context.
        """
        registry: dict[str, list[dict]] = {}

        for block in content_list:
            page_idx = block.get("page_idx", -1)
            text = self._extract_text_from_block(block)
            if not text:
                continue

            for entity_type, pattern in _ENTITY_PATTERNS:
                for match in pattern.finditer(text):
                    entity_text = match.group(1) if match.groups() else match.group(0)
                    if not entity_text or len(entity_text.strip()) < 2:
                        continue

                    record = {
                        "text": entity_text.strip(),
                        "page_idx": page_idx,
                        "context": self._extract_context(text, match.start(), 80),
                    }

                    if entity_type not in registry:
                        registry[entity_type] = []
                    # Deduplicate by text
                    existing_texts = {e["text"] for e in registry[entity_type]}
                    if entity_text.strip() not in existing_texts:
                        registry[entity_type].append(record)

        return registry

    # ==================================================================
    # REFERENCE RESOLUTION
    # ==================================================================

    async def _resolve_all_references(
        self,
        content_list: list[dict],
        entity_registry: dict,
        client: Any = None,
    ) -> list[ResolutionResult]:
        """
        Scan all content for references and resolve them.

        Uses LLM if available, otherwise falls back to rule-based resolution.
        """
        resolutions: list[ResolutionResult] = []

        for block in content_list:
            page_idx = block.get("page_idx", -1)
            text = self._extract_text_from_block(block)
            if not text:
                continue

            # Find all references in this text
            references = self._find_references(text)
            for ref_text, position in references:
                context = self._extract_context(text, position, 120)

                # Try rule-based resolution first
                result = self._resolve_references(
                    ref_text, entity_registry, context,
                )
                result.page_idx = page_idx

                # If rule-based confidence is low and LLM is available, try LLM
                if result.confidence < 0.6 and client is not None:
                    llm_result = await self._resolve_references_llm(
                        ref_text, entity_registry, context, page_idx, client,
                    )
                    if llm_result and llm_result.confidence > result.confidence:
                        result = llm_result

                resolutions.append(result)

        return resolutions

    def _find_references(self, text: str) -> list[tuple[str, int]]:
        """
        Find reference expressions in text.

        Returns list of (matched_text, position) tuples.
        """
        references: list[tuple[str, int]] = []
        seen_positions: set[int] = set()

        for pattern in _PRONOUN_PATTERNS:
            for match in pattern.finditer(text):
                pos = match.start()
                if pos not in seen_positions:
                    seen_positions.add(pos)
                    references.append((match.group(0), pos))

        # Sort by position
        references.sort(key=lambda x: x[1])
        return references

    def _resolve_references(
        self,
        reference: str,
        entities: dict,
        context: str,
    ) -> ResolutionResult:
        """
        Rule-based reference resolution.

        Uses pattern matching and context proximity to resolve references.
        """
        result = ResolutionResult(
            reference=reference,
            context_snippet=context[:200],
            method="rule",
        )

        # Strategy 1: Match reference type to entity category
        ref_lower = reference

        # Company references
        if any(kw in ref_lower for kw in ("该公司", "本公司", "该企业", "本企业")):
            companies = entities.get("company", [])
            if companies:
                # Find the most recently mentioned company in context
                best = self._find_nearest_entity(context, companies)
                if best:
                    result.resolved_entity = best["text"]
                    result.confidence = 0.7
                    return result

        # Money references
        if any(kw in ref_lower for kw in ("上述金额", "前述金额", "该金额",
                                           "上述数额", "该数额", "以上金额")):
            money_entities = entities.get("money", [])
            if money_entities:
                best = self._find_nearest_entity(context, money_entities)
                if best:
                    result.resolved_entity = best["text"]
                    result.confidence = 0.65
                    return result

        # Project references
        if any(kw in ref_lower for kw in ("该项目", "本项目", "该工程", "该计划")):
            projects = entities.get("project", [])
            if projects:
                best = self._find_nearest_entity(context, projects)
                if best:
                    result.resolved_entity = best["text"]
                    result.confidence = 0.7
                    return result

        # Date references
        if any(kw in ref_lower for kw in ("同期", "上年", "上期", "报告期", "报告年度")):
            dates = entities.get("date", [])
            if dates:
                best = self._find_nearest_entity(context, dates)
                if best:
                    result.resolved_entity = best["text"]
                    result.confidence = 0.6
                    return result

        # Strategy 2: Generic nearest-entity lookup
        # Collect all entities and find the closest match by context
        all_entities: list[dict] = []
        for category_items in entities.values():
            all_entities.extend(category_items)

        if all_entities:
            best = self._find_nearest_entity(context, all_entities)
            if best:
                result.resolved_entity = best["text"]
                result.confidence = 0.4
                result.method = "context"
                return result

        # Unresolved
        result.resolved_entity = ""
        result.confidence = 0.0
        result.method = "unresolved"
        return result

    async def _resolve_references_llm(
        self,
        reference: str,
        entities: dict,
        context: str,
        page_idx: int,
        client: Any,
    ) -> ResolutionResult | None:
        """
        Use LLM to resolve a reference when rule-based resolution fails.
        """
        import json

        # Build entity summary for the prompt
        entity_summary_parts: list[str] = []
        for category, items in entities.items():
            names = [e["text"] for e in items[:10]]  # Limit to first 10
            entity_summary_parts.append(f"{category}: {', '.join(names)}")
        entity_summary = "\n".join(entity_summary_parts)

        prompt = (
            "You are a reference resolution expert for Chinese documents.\n"
            "Resolve the following reference expression:\n\n"
            + "Reference: \"" + reference + "\"\n"
            + "Context: \"" + context[:300] + "\"\n\n"
            + "Known entities:\n" + entity_summary + "\n\n"
            + "Respond with a JSON object:\n"
            "{\n"
            '  "resolved_entity": "the entity this refers to",\n'
            '  "confidence": 0.0_to_1.0,\n'
            '  "reasoning": "brief explanation"\n'
            "}\n"
            "Respond with ONLY the JSON object, no other text."
        )

        try:
            response = await self._call_llm(client, prompt)
            if not response:
                return None

            # Try to parse JSON
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1).strip())
                else:
                    logger.warning(
                        f"Could not parse LLM reference resolution response: "
                        f"{response[:200]}"
                    )
                    return None

            return ResolutionResult(
                reference=reference,
                resolved_entity=parsed.get("resolved_entity", ""),
                confidence=float(parsed.get("confidence", 0.5)),
                method="llm",
                context_snippet=context[:200],
                page_idx=page_idx,
            )

        except Exception as exc:
            logger.warning(f"LLM reference resolution failed: {exc}")
            return None

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _extract_text_from_block(self, block: dict) -> str:
        """Extract plain text from a content block, regardless of type."""
        block_type = block.get("type", "")

        if block_type == "text":
            return block.get("text", "")

        if block_type == "table":
            # Convert table data to text
            data = _extract_table_data(block)
            if data:
                return " | ".join(" ".join(cell for cell in row) for row in data)
            return block.get("markdown", "")

        if block_type == "image":
            return block.get("alt_text", block.get("caption", ""))

        # Generic fallback
        return block.get("text", block.get("markdown", ""))

    def _extract_context(
        self,
        text: str,
        position: int,
        window: int = 80,
    ) -> str:
        """Extract a window of text around a position for context."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end]

    def _find_nearest_entity(
        self,
        context: str,
        candidates: list[dict],
    ) -> dict | None:
        """
        Find the entity most likely referenced by context.

        Uses string similarity and last-occurrence position.
        """
        if not candidates:
            return None

        best: dict | None = None
        best_score = -1.0

        for entity in candidates:
            entity_text = entity.get("text", "")
            if not entity_text:
                continue

            # Check if entity appears in the context
            if entity_text in context:
                # Weight by recency (position in context)
                pos = context.rfind(entity_text)
                recency = pos / max(len(context), 1)
                score = 1.0 + recency  # Prefer recent mentions
            else:
                # Use fuzzy matching
                ratio = SequenceMatcher(
                    None, context[-100:] if len(context) > 100 else context,
                    entity_text,
                ).ratio()
                score = ratio

            if score > best_score:
                best_score = score
                best = entity

        return best

    @staticmethod
    def _table_to_markdown(rows: list[list[str]]) -> str:
        """Convert a 2D list of strings into a Markdown table."""
        if not rows:
            return ""
        max_cols = max(len(r) for r in rows)
        normalised = []
        for r in rows:
            padded = list(r) + [""] * (max_cols - len(r))
            normalised.append(padded)

        header = "| " + " | ".join(normalised[0]) + " |"
        separator = "| " + " | ".join("---" for _ in range(max_cols)) + " |"
        body_lines = []
        for row in normalised[1:]:
            body_lines.append("| " + " | ".join(row) + " |")

        return "\n".join([header, separator] + body_lines)

    @staticmethod
    async def _call_llm(client: Any, prompt: str) -> str | None:
        """
        Call an LLM client with a text prompt.

        Supports Anthropic, OpenAI, and simple generate interfaces.
        """
        messages = [{"role": "user", "content": prompt}]

        # Anthropic-style
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            resp = await client.messages.create(
                model=getattr(client, "model", "claude-sonnet-4-20250514"),
                max_tokens=1024,
                messages=messages,
            )
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return None

        # OpenAI-style async
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = await client.chat.completions.create(
                model=getattr(client, "model", "gpt-4o"),
                max_tokens=1024,
                messages=messages,
            )
            return resp.choices[0].message.content if resp.choices else None

        # Simple generate interface
        if hasattr(client, "generate"):
            return await client.generate(prompt)

        logger.warning("LLM client has no recognised interface")
        return None
