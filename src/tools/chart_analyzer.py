"""
Chart Analyzer Tool - Chart/figure classification, data extraction, and description.

Handles:
- Chart classification (bar, line, pie, scatter, area, heatmap, etc.)
- Data extraction via LLM vision or OpenCV+OCR fallback
- Chart-to-table conversion
- Accessibility description generation
- Multiple charts per image and charts embedded in tables
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar_chart"
    STACKED_BAR = "stacked_bar_chart"
    GROUPED_BAR = "grouped_bar_chart"
    LINE = "line_chart"
    MULTI_LINE = "multi_line_chart"
    PIE = "pie_chart"
    DONUT = "donut_chart"
    SCATTER = "scatter_plot"
    AREA = "area_chart"
    STACKED_AREA = "stacked_area_chart"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    RADAR = "radar_chart"
    BUBBLE = "bubble_chart"
    WATERFALL = "waterfall_chart"
    TREEMAP = "treemap"
    FUNNEL = "funnel_chart"
    COMBO = "combo_chart"  # mixed types (e.g. bar + line)
    TABLE_WITH_CHART = "table_with_chart"
    UNKNOWN = "unknown"


@dataclass
class DataSeries:
    """A single data series extracted from a chart."""
    name: str = ""
    labels: list[str] = field(default_factory=list)
    values: list[float | int | str] = field(default_factory=list)
    color: str | None = None
    unit: str | None = None


@dataclass
class AnalyzedChart:
    """Complete analysis result for one chart."""
    index: int = 0
    chart_type: str = ChartType.UNKNOWN
    title: str = ""
    description: str = ""
    data_series: list[dict[str, Any]] = field(default_factory=list)
    data_table: list[list[str]] = field(default_factory=list)
    confidence: float = 0.0
    source_image: str = ""
    page_idx: int = -1
    axes: dict[str, Any] = field(default_factory=dict)
    legend: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """\
You are a chart analysis expert. Examine this image and classify the chart type.

Respond with a JSON object (and nothing else):
{
  "chart_type": one of: bar_chart, stacked_bar_chart, grouped_bar_chart, line_chart, multi_line_chart, pie_chart, donut_chart, scatter_plot, area_chart, stacked_area_chart, heatmap, histogram, box_plot, radar_chart, bubble_chart, waterfall_chart, treemap, funnel_chart, combo_chart, table_with_chart, unknown,
  "title": "the chart title if visible, empty string otherwise",
  "confidence": float between 0.0 and 1.0,
  "is_chart": true/false whether this image actually contains a chart,
  "chart_count": number of distinct charts visible in this image (integer >= 0)
}
"""

_DATA_EXTRACTION_PROMPT = """\
You are a data extraction expert specialising in reading chart data.

The image contains a chart of type: {chart_type}.

Extract ALL data from the chart and respond with a JSON object:
{{
  "title": "chart title",
  "x_axis_label": "x-axis label if present",
  "y_axis_label": "y-axis label if present",
  "x_axis_unit": "unit for x-axis values",
  "y_axis_unit": "unit for y-axis values",
  "legend": ["series1 name", "series2 name"],
  "series": [
    {{
      "name": "series name",
      "labels": ["label1", "label2", ...],
      "values": [v1, v2, ...],
      "color": "hex color if visible",
      "unit": "unit if different from axis"
    }}
  ],
  "notes": ["any footnotes, annotations, or source text visible"]
}}

Rules:
- Extract numeric values as numbers (int or float), not strings.
- If a value cannot be read precisely, use your best estimate.
- Include ALL data points visible in the chart.
- For pie charts, include the slice labels and their percentages/values.
- For scatter plots, extract individual (x, y) pairs.
- For heatmaps, provide the matrix with row/column labels.
"""

_DESCRIPTION_PROMPT = """\
Describe this chart for accessibility purposes. The chart is of type: {chart_type}.

Provide:
1. A one-sentence summary of what the chart shows.
2. The main trend or insight.
3. Key data points (highest, lowest, notable values).

Respond in plain text (not JSON). Keep the description under 200 words.
"""


# ---------------------------------------------------------------------------
# OpenCV fallback helpers
# ---------------------------------------------------------------------------

def _try_import_cv2():
    """Attempt to import cv2; return None if unavailable."""
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError:
        return None


def _try_import_pillow():
    """Attempt to import PIL; return None if unavailable."""
    try:
        from PIL import Image  # noqa: F401
        return Image
    except ImportError:
        return None


def _basic_chart_type_detection(image_path: str) -> tuple[str, float]:
    """
    Heuristic chart-type detection using image analysis.

    Uses colour distribution and shape heuristics to make a best guess.
    Returns (chart_type_name, confidence).
    """
    cv2 = _try_import_cv2()
    if cv2 is None:
        logger.warning("OpenCV not available; skipping heuristic classification")
        return ChartType.UNKNOWN, 0.0

    try:
        img = cv2.imread(image_path)
        if img is None:
            return ChartType.UNKNOWN, 0.0

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / (h * w)

        # Detect horizontal and vertical lines via Hough
        lines = cv2.HoughLinesP(
            edges, 1, 3.14159 / 180,
            threshold=80,
            minLineLength=min(h, w) * 0.15,
            maxLineGap=10,
        )

        h_lines, v_lines = 0, 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 5:
                    h_lines += 1
                elif abs(x1 - x2) < 5:
                    v_lines += 1

        # Heuristic rules (very rough)
        if h_lines > 10 and v_lines > 10 and edge_ratio > 0.08:
            return ChartType.HEATMAP, 0.3
        if h_lines > 5 and v_lines > 2:
            return ChartType.BAR, 0.35
        if h_lines > 5 and v_lines < 2:
            return ChartType.LINE, 0.3
        if edge_ratio < 0.03:
            return ChartType.PIE, 0.25
        if h_lines < 3 and v_lines < 3 and edge_ratio > 0.05:
            return ChartType.SCATTER, 0.25

        return ChartType.UNKNOWN, 0.1
    except Exception as exc:
        logger.warning(f"OpenCV chart detection failed: {exc}")
        return ChartType.UNKNOWN, 0.0


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class ChartAnalyzer:
    """
    Chart/figure analysis tool.

    Uses LLM-based vision analysis (Claude vision, Qwen-VL, or compatible)
    with OpenCV+OCR fallback when no LLM is available.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.llm_client = None  # Injected via execute() params or setter

    def set_llm_client(self, client: Any) -> None:
        """Set the LLM client for vision-based analysis."""
        self.llm_client = client

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: dict | None = None,
    ) -> dict:
        """
        Analyze charts/figures extracted from document images.

        Args:
            params: {
                images: list[{path: str, page_idx: int}],
                llm_client: optional LLM client override,
            }
            context: Previous pipeline results (unused but kept for interface consistency).

        Returns:
            {
                charts: list[dict],  # serialised AnalyzedChart objects
                total_count: int,
            }
        """
        images: list[dict] = params.get("images", [])
        llm_override = params.get("llm_client")
        client = llm_override or self.llm_client

        if not images:
            logger.info("ChartAnalyzer: no images provided")
            return {"charts": [], "total_count": 0}

        logger.info(f"ChartAnalyzer: processing {len(images)} image(s)")

        analyzed: list[AnalyzedChart] = []
        chart_index = 0

        for img_entry in images:
            image_path = img_entry.get("path", "")
            page_idx = img_entry.get("page_idx", -1)

            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            logger.info(f"Analyzing image: {image_path} (page {page_idx})")

            # Step 1: Classify chart type (also obtains chart_count from the
            # same LLM call, avoiding a duplicate request)
            classification_result = await self._classify_chart_type_full(
                image_path, client=client,
            )
            chart_type = classification_result.get("chart_type", ChartType.UNKNOWN)
            confidence = float(classification_result.get("confidence", 0.5))
            is_chart = classification_result.get("is_chart", True)
            if not is_chart:
                chart_type = ChartType.UNKNOWN
                confidence = 0.0

            logger.info(
                f"Classification: {chart_type} (confidence={confidence:.2f})"
            )

            if chart_type == ChartType.UNKNOWN and confidence < 0.1:
                logger.info("Image does not appear to contain a chart; skipping")
                continue

            # Step 2: Detect how many charts are in the image (reuse
            # classification result instead of making a second LLM call)
            chart_count = self._detect_chart_count(classification_result, image_path)
            logger.info(f"Detected {chart_count} chart(s) in image")

            # Step 3: Extract data
            data_result = await self._extract_chart_data(
                image_path, chart_type, client=client,
            )
            logger.info(
                f"Extracted {len(data_result.get('series', []))} data series"
            )

            # Step 4: Build data table from series
            series_objs = [
                DataSeries(
                    name=s.get("name", f"Series_{si+1}"),
                    labels=s.get("labels", []),
                    values=s.get("values", []),
                    color=s.get("color"),
                    unit=s.get("unit"),
                )
                for si, s in enumerate(data_result.get("series", []))
            ]
            data_table = self._chart_to_table(series_objs)

            # Step 5: Generate description
            description = await self._generate_description(
                image_path, chart_type, client=client,
            )

            chart = AnalyzedChart(
                index=chart_index,
                chart_type=chart_type,
                title=data_result.get("title", ""),
                description=description,
                data_series=[asdict(s) for s in series_objs],
                data_table=data_table,
                confidence=confidence,
                source_image=image_path,
                page_idx=page_idx,
                axes={
                    "x_axis_label": data_result.get("x_axis_label", ""),
                    "y_axis_label": data_result.get("y_axis_label", ""),
                    "x_axis_unit": data_result.get("x_axis_unit", ""),
                    "y_axis_unit": data_result.get("y_axis_unit", ""),
                },
                legend=data_result.get("legend", []),
                notes=data_result.get("notes", []),
            )
            analyzed.append(chart)
            chart_index += 1

        result = {
            "charts": [asdict(c) for c in analyzed],
            "total_count": len(analyzed),
        }
        logger.info(f"ChartAnalyzer: completed — {len(analyzed)} chart(s) analyzed")
        return result

    # -----------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------

    async def _classify_chart_type_full(
        self,
        image_path: str,
        client: Any = None,
    ) -> dict:
        """
        Classify the chart type and return the full classification dict.

        Tries LLM vision first; falls back to OpenCV heuristics.
        Returns a dict with keys: chart_type, confidence, is_chart, chart_count.
        """
        if client is not None:
            try:
                result = await self._llm_classify(image_path, client)
                if result:
                    return result
            except Exception as exc:
                logger.warning(f"LLM classification failed, falling back: {exc}")

        # Fallback: OpenCV heuristic
        chart_type, confidence = _basic_chart_type_detection(image_path)
        return {
            "chart_type": chart_type,
            "confidence": confidence,
            "is_chart": confidence > 0.1,
            "chart_count": 1,
        }

    async def _classify_chart_type(
        self,
        image_path: str,
        client: Any = None,
    ) -> tuple[str, float]:
        """
        Classify the chart type.

        Tries LLM vision first; falls back to OpenCV heuristics.
        Returns (chart_type: str, confidence: float).
        """
        result = await self._classify_chart_type_full(image_path, client)
        chart_type = result.get("chart_type", ChartType.UNKNOWN)
        confidence = float(result.get("confidence", 0.5))
        is_chart = result.get("is_chart", True)
        if not is_chart:
            return ChartType.UNKNOWN, 0.0
        return chart_type, confidence

    def _detect_chart_count(
        self,
        classification_result: dict,
        image_path: str,
    ) -> int:
        """Detect how many distinct charts exist in the image.

        Uses the cached classification result when available to avoid
        a redundant LLM call.
        """
        if classification_result and "chart_count" in classification_result:
            return int(classification_result["chart_count"])
        return 1

    # -----------------------------------------------------------------------
    # Data extraction
    # -----------------------------------------------------------------------

    async def _extract_chart_data(
        self,
        image_path: str,
        chart_type: str,
        client: Any = None,
    ) -> dict:
        """
        Extract structured data from the chart.

        Returns dict with keys: title, x_axis_label, y_axis_label,
        x_axis_unit, y_axis_unit, legend, series, notes.
        """
        if client is not None:
            try:
                result = await self._llm_extract_data(
                    image_path, chart_type, client,
                )
                if result:
                    return result
            except Exception as exc:
                logger.warning(f"LLM data extraction failed, falling back: {exc}")

        # Fallback: basic OCR for labels
        return await self._ocr_extract_labels(image_path)

    # -----------------------------------------------------------------------
    # Description generation
    # -----------------------------------------------------------------------

    async def _generate_description(
        self,
        image_path: str,
        chart_type: str,
        client: Any = None,
    ) -> str:
        """Generate an accessibility-friendly description of the chart."""
        if client is not None:
            try:
                description = await self._llm_describe(
                    image_path, chart_type, client,
                )
                if description:
                    return description
            except Exception as exc:
                logger.warning(f"LLM description failed: {exc}")

        return f"A {chart_type} extracted from the document."

    # -----------------------------------------------------------------------
    # Chart-to-table conversion
    # -----------------------------------------------------------------------

    def _chart_to_table(self, data_series: list[DataSeries]) -> list[list[str]]:
        """
        Convert extracted data series into a rectangular table (list of rows).

        The first row is a header: ["Label", "Series1", "Series2", ...].
        Subsequent rows contain label + values aligned by index.
        """
        if not data_series:
            return []

        # Determine the maximum label count to size the table
        max_len = max((len(s.labels) for s in data_series), default=0)
        if max_len == 0:
            # Try using values length if labels are empty
            max_len = max((len(s.values) for s in data_series), default=0)
        if max_len == 0:
            return []

        # Header row
        header = ["Label"] + [s.name or f"Series_{i+1}" for i, s in enumerate(data_series)]
        table = [header]

        for row_idx in range(max_len):
            # Use label from the first series that has one at this index
            label = ""
            for s in data_series:
                if row_idx < len(s.labels):
                    label = str(s.labels[row_idx])
                    break

            row = [label]
            for s in data_series:
                if row_idx < len(s.values):
                    row.append(str(s.values[row_idx]))
                else:
                    row.append("")

            table.append(row)

        return table

    # -----------------------------------------------------------------------
    # LLM interaction helpers
    # -----------------------------------------------------------------------

    async def _llm_classify(self, image_path: str, client: Any) -> dict | None:
        """Use LLM vision to classify a chart image."""
        import json

        image_b64 = self._encode_image(image_path)
        messages = self._build_vision_messages(
            _CLASSIFICATION_PROMPT,
            image_b64,
            image_path,
        )

        response = await self._call_llm(client, messages)
        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Attempt to extract JSON from markdown code fences
            import re
            match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse LLM classification response: {response[:200]}")
            return None

    async def _llm_extract_data(
        self, image_path: str, chart_type: str, client: Any,
    ) -> dict | None:
        """Use LLM vision to extract chart data."""
        import json

        prompt = _DATA_EXTRACTION_PROMPT.format(chart_type=chart_type)
        image_b64 = self._encode_image(image_path)
        messages = self._build_vision_messages(prompt, image_b64, image_path)

        response = await self._call_llm(client, messages)
        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse LLM data extraction response: {response[:200]}")
            return None

    async def _llm_describe(
        self, image_path: str, chart_type: str, client: Any,
    ) -> str | None:
        """Use LLM vision to generate a description."""
        prompt = _DESCRIPTION_PROMPT.format(chart_type=chart_type)
        image_b64 = self._encode_image(image_path)
        messages = self._build_vision_messages(prompt, image_b64, image_path)

        return await self._call_llm(client, messages)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read an image file and return a base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _build_vision_messages(prompt: str, image_b64: str, image_path: str | None = None) -> list[dict]:
        """
        Build a message payload compatible with common LLM vision APIs.

        Supports:
        - Anthropic Claude (content blocks)
        - OpenAI-compatible / Qwen-VL (image_url)
        """
        # Detect actual image format from the file extension; fall back to PNG
        media_type = None
        if image_path:
            media_type, _ = mimetypes.guess_type(str(image_path))
        if not media_type:
            media_type = "image/png"
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            },
        ]

    @staticmethod
    async def _call_llm(client: Any, messages: list[dict]) -> str | None:
        """
        Call an LLM client with the given messages.

        Supports clients with any of these interfaces:
        - client.messages.create(...)          — Anthropic SDK
        - client.chat.completions.create(...)  — OpenAI SDK
        - client.generate(prompt)              — Simple text-in/text-out
        """
        # Anthropic-style
        if hasattr(client, "messages") and hasattr(client.messages, "create"):
            resp = await client.messages.create(
                model=client.model if hasattr(client, "model") else "claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=messages,
            )
            # Extract text from response content blocks
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text
            return None

        # OpenAI-style async
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            # Convert Anthropic-format messages to OpenAI format
            oai_messages = ChartAnalyzer._convert_to_openai_messages(messages)
            resp = await client.chat.completions.create(
                model=getattr(client, "model", "gpt-4o"),
                max_tokens=4096,
                messages=oai_messages,
            )
            return resp.choices[0].message.content if resp.choices else None

        # Simple generate interface
        if hasattr(client, "generate"):
            # Flatten messages to a single prompt string
            text_parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part["text"])
                elif isinstance(content, str):
                    text_parts.append(content)
            combined_prompt = "\n".join(text_parts)
            return await client.generate(combined_prompt)

        logger.warning("LLM client has no recognised interface")
        return None

    @staticmethod
    def _convert_to_openai_messages(
        messages: list[dict],
    ) -> list[dict]:
        """
        Convert Anthropic-style vision messages to OpenAI-compatible format.

        Anthropic uses content blocks; OpenAI uses image_url with data URIs.
        """
        oai_messages: list[dict] = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                oai_messages.append({"role": msg["role"], "content": content})
                continue

            parts: list[dict] = []
            for block in content:
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image":
                    src = block.get("source", {})
                    b64 = src.get("data", "")
                    media = src.get("media_type", "image/png")
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media};base64,{b64}",
                        },
                    })
            oai_messages.append({"role": msg["role"], "content": parts})
        return oai_messages

    # -----------------------------------------------------------------------
    # OCR fallback
    # -----------------------------------------------------------------------

    async def _ocr_extract_labels(self, image_path: str) -> dict:
        """
        Fallback data extraction using OCR to read labels from the chart.

        Returns partial data with whatever labels/values can be extracted.
        """
        result: dict[str, Any] = {
            "title": "",
            "x_axis_label": "",
            "y_axis_label": "",
            "x_axis_unit": "",
            "y_axis_unit": "",
            "legend": [],
            "series": [],
            "notes": [],
        }

        # Try pytesseract
        try:
            import pytesseract
            from PIL import Image as PILImage

            img = PILImage.open(image_path)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            if lines:
                # First non-empty line is often the title
                result["title"] = lines[0]
                # Store remaining lines as notes
                result["notes"] = lines[1:5]

            logger.info(f"OCR extracted {len(lines)} text lines from {image_path}")
        except ImportError:
            logger.warning(
                "pytesseract/Pillow not available; OCR fallback skipped"
            )
        except Exception as exc:
            logger.warning(f"OCR extraction failed: {exc}")

        return result
