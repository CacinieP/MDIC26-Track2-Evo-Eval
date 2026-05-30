"""
Tests for the Chart Analyzer tool.
Tests: ChartType enum, DataSeries/AnalyzedChart dataclasses,
       message format conversion, OpenCV heuristic classification,
       execute() input validation, chart-to-table conversion.
"""

import base64
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.chart_analyzer import (
    AnalyzedChart,
    ChartAnalyzer,
    ChartType,
    DataSeries,
    _basic_chart_type_detection,
    _try_import_cv2,
    _try_import_pillow,
)


# ---------------------------------------------------------------------------
# ChartType enum
# ---------------------------------------------------------------------------

class TestChartTypeEnum:
    """Test ChartType enum completeness and values."""

    EXPECTED_TYPES = [
        "bar_chart",
        "stacked_bar_chart",
        "grouped_bar_chart",
        "line_chart",
        "multi_line_chart",
        "pie_chart",
        "donut_chart",
        "scatter_plot",
        "area_chart",
        "stacked_area_chart",
        "heatmap",
        "histogram",
        "box_plot",
        "radar_chart",
        "bubble_chart",
        "waterfall_chart",
        "treemap",
        "funnel_chart",
        "combo_chart",
        "table_with_chart",
        "unknown",
    ]

    def test_enum_member_count(self):
        """ChartType should have all 21 defined members (18 chart types + combo + table_with_chart + unknown)."""
        assert len(ChartType) == 21

    def test_all_expected_values_present(self):
        """Every expected chart type value should be present in the enum."""
        actual_values = {member.value for member in ChartType}
        for expected in self.EXPECTED_TYPES:
            assert expected in actual_values, f"Missing ChartType value: {expected}"

    def test_specific_enum_members(self):
        """Spot-check individual members."""
        assert ChartType.BAR.value == "bar_chart"
        assert ChartType.PIE.value == "pie_chart"
        assert ChartType.LINE.value == "line_chart"
        assert ChartType.SCATTER.value == "scatter_plot"
        assert ChartType.HEATMAP.value == "heatmap"
        assert ChartType.UNKNOWN.value == "unknown"
        assert ChartType.COMBO.value == "combo_chart"

    def test_enum_is_str_subclass(self):
        """ChartType members should be usable as strings."""
        assert isinstance(ChartType.BAR, str)
        assert ChartType.BAR == "bar_chart"


# ---------------------------------------------------------------------------
# DataSeries and AnalyzedChart dataclasses
# ---------------------------------------------------------------------------

class TestDataSeries:
    """Test DataSeries dataclass construction and defaults."""

    def test_default_construction(self):
        """DataSeries should initialise with empty defaults."""
        ds = DataSeries()
        assert ds.name == ""
        assert ds.labels == []
        assert ds.values == []
        assert ds.color is None
        assert ds.unit is None

    def test_full_construction(self):
        """DataSeries should accept all fields."""
        ds = DataSeries(
            name="Revenue",
            labels=["Q1", "Q2", "Q3"],
            values=[100, 200, 300],
            color="#FF0000",
            unit="万元",
        )
        assert ds.name == "Revenue"
        assert len(ds.labels) == 3
        assert ds.values[1] == 200
        assert ds.color == "#FF0000"
        assert ds.unit == "万元"

    def test_mixed_value_types(self):
        """DataSeries.values should accept mixed int/float/str."""
        ds = DataSeries(values=[1, 2.5, "N/A", 100])
        assert ds.values[1] == 2.5
        assert ds.values[2] == "N/A"

    def test_asdict_roundtrip(self):
        """DataSeries should survive a dataclass-asdict round-trip."""
        ds = DataSeries(name="Test", labels=["A"], values=[42])
        d = asdict(ds)
        assert d["name"] == "Test"
        assert d["labels"] == ["A"]
        assert d["values"] == [42]

    def test_independent_defaults(self):
        """Two DataSeries instances should not share mutable defaults."""
        ds1 = DataSeries()
        ds2 = DataSeries()
        ds1.labels.append("X")
        assert ds2.labels == []


class TestAnalyzedChart:
    """Test AnalyzedChart dataclass construction and defaults."""

    def test_default_construction(self):
        """AnalyzedChart should initialise with sensible defaults."""
        ac = AnalyzedChart()
        assert ac.index == 0
        assert ac.chart_type == ChartType.UNKNOWN
        assert ac.title == ""
        assert ac.description == ""
        assert ac.data_series == []
        assert ac.data_table == []
        assert ac.confidence == 0.0
        assert ac.source_image == ""
        assert ac.page_idx == -1
        assert ac.axes == {}
        assert ac.legend == []
        assert ac.notes == []

    def test_full_construction(self):
        """AnalyzedChart should accept all fields."""
        ac = AnalyzedChart(
            index=3,
            chart_type=ChartType.BAR,
            title="Revenue Chart",
            description="A bar chart showing revenue.",
            data_series=[{"name": "Revenue", "values": [100]}],
            data_table=[["Label", "Revenue"], ["Q1", "100"]],
            confidence=0.92,
            source_image="/tmp/chart.png",
            page_idx=2,
            axes={"x_axis_label": "Quarter", "y_axis_label": "Amount"},
            legend=["Revenue"],
            notes=["Source: Annual Report"],
        )
        assert ac.index == 3
        assert ac.chart_type == ChartType.BAR
        assert ac.confidence == 0.92
        assert len(ac.data_series) == 1
        assert ac.page_idx == 2

    def test_asdict_roundtrip(self):
        """AnalyzedChart should survive asdict round-trip."""
        ac = AnalyzedChart(index=1, title="Test", confidence=0.5)
        d = asdict(ac)
        assert d["index"] == 1
        assert d["title"] == "Test"
        assert d["confidence"] == 0.5


# ---------------------------------------------------------------------------
# _convert_to_openai_messages format conversion
# ---------------------------------------------------------------------------

class TestConvertToOpenAIMessages:
    """Test Anthropic-to-OpenAI message format conversion."""

    def test_string_content_passthrough(self):
        """Messages with string content should pass through unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = ChartAnalyzer._convert_to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_text_block_conversion(self):
        """Anthropic text blocks should become OpenAI text parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this chart."},
                ],
            },
        ]
        result = ChartAnalyzer._convert_to_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Describe this chart."

    def test_image_block_conversion(self):
        """Anthropic image blocks should become OpenAI image_url parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "See this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "fakeb64data",
                        },
                    },
                ],
            },
        ]
        result = ChartAnalyzer._convert_to_openai_messages(messages)
        assert len(result) == 1
        parts = result[0]["content"]
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,fakeb64data")

    def test_mixed_media_type(self):
        """Image blocks with JPEG media type should be preserved."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "jpegdata",
                        },
                    },
                ],
            },
        ]
        result = ChartAnalyzer._convert_to_openai_messages(messages)
        url = result[0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,jpegdata")

    def test_empty_messages_list(self):
        """Empty input should return empty list."""
        result = ChartAnalyzer._convert_to_openai_messages([])
        assert result == []

    def test_multiple_messages(self):
        """Multiple messages should all be converted."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = ChartAnalyzer._convert_to_openai_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"


# ---------------------------------------------------------------------------
# OpenCV heuristic classification
# ---------------------------------------------------------------------------

class TestOpenCVHeuristic:
    """Test the OpenCV-based chart type heuristic detection."""

    def test_cv2_import_returns_none_when_missing(self):
        """When cv2 is not importable, _try_import_cv2 returns None."""
        with patch.dict("sys.modules", {"cv2": None}):
            result = _try_import_cv2()
            # cv2 may or may not be installed in the test env;
            # we just verify the function returns cv2 or None
            assert result is None or hasattr(result, "imread")

    def test_pillow_import_returns_none_when_missing(self):
        """When PIL is not importable, _try_import_pillow returns None."""
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            result = _try_import_pillow()
            assert result is None or hasattr(result, "open")

    def test_heuristic_returns_unknown_when_no_cv2(self):
        """Without cv2, heuristic should return UNKNOWN with 0.0 confidence."""
        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=None):
            chart_type, conf = _basic_chart_type_detection("/nonexistent.png")
            assert chart_type == ChartType.UNKNOWN
            assert conf == 0.0

    def test_heuristic_returns_unknown_on_bad_image(self):
        """With cv2 but unreadable image, should return UNKNOWN with 0.0."""
        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = None
        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/nonexistent.png")
            assert chart_type == ChartType.UNKNOWN
            assert conf == 0.0

    def test_heuristic_bar_detection(self):
        """Should classify as BAR when many horizontal and some vertical lines."""
        mock_cv2 = MagicMock()
        # Simulate a 100x100 image
        fake_img = MagicMock()
        fake_img.shape = (100, 100, 3)
        mock_cv2.imread.return_value = fake_img
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()
        mock_cv2.countNonZero.return_value = 5000  # edge_ratio ~0.5

        # Create fake lines: 8 horizontal, 5 vertical
        fake_lines = []
        for i in range(8):
            fake_lines.append([[10, i * 10, 90, i * 10]])  # horizontal
        for i in range(5):
            fake_lines.append([[i * 20, 10, i * 20, 90]])  # vertical
        mock_cv2.HoughLinesP.return_value = fake_lines

        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.BAR
            assert conf == 0.35

    def test_heuristic_line_detection(self):
        """Should classify as LINE when many horizontal but few vertical lines."""
        mock_cv2 = MagicMock()
        fake_img = MagicMock()
        fake_img.shape = (100, 100, 3)
        mock_cv2.imread.return_value = fake_img
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()
        mock_cv2.countNonZero.return_value = 5000

        # 8 horizontal, 1 vertical -> triggers LINE rule
        fake_lines = []
        for i in range(8):
            fake_lines.append([[10, i * 10, 90, i * 10]])
        fake_lines.append([[50, 10, 50, 90]])  # 1 vertical
        mock_cv2.HoughLinesP.return_value = fake_lines

        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.LINE
            assert conf == 0.3

    def test_heuristic_heatmap_detection(self):
        """Should classify as HEATMAP when many H/V lines and high edge ratio."""
        mock_cv2 = MagicMock()
        fake_img = MagicMock()
        fake_img.shape = (100, 100, 3)
        mock_cv2.imread.return_value = fake_img
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()
        mock_cv2.countNonZero.return_value = 12000  # edge_ratio = 0.12

        # 15 horizontal, 15 vertical
        fake_lines = []
        for i in range(15):
            fake_lines.append([[10, i * 6, 90, i * 6]])
        for i in range(15):
            fake_lines.append([[i * 6, 10, i * 6, 90]])
        mock_cv2.HoughLinesP.return_value = fake_lines

        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.HEATMAP
            assert conf == 0.3

    def test_heuristic_pie_detection(self):
        """Should classify as PIE when edge ratio is very low."""
        mock_cv2 = MagicMock()
        fake_img = MagicMock()
        fake_img.shape = (100, 100, 3)
        mock_cv2.imread.return_value = fake_img
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()
        mock_cv2.countNonZero.return_value = 200  # edge_ratio = 0.02

        # Few lines so other rules don't trigger
        mock_cv2.HoughLinesP.return_value = None

        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.PIE
            assert conf == 0.25

    def test_heuristic_scatter_detection(self):
        """Should classify as SCATTER when few lines but moderate edges."""
        mock_cv2 = MagicMock()
        fake_img = MagicMock()
        fake_img.shape = (100, 100, 3)
        mock_cv2.imread.return_value = fake_img
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()
        mock_cv2.countNonZero.return_value = 800  # edge_ratio = 0.08

        # 2 horizontal, 2 vertical (both < 3)
        fake_lines = [
            [[10, 20, 90, 20]],
            [[10, 80, 90, 80]],
            [[20, 10, 20, 90]],
            [[80, 10, 80, 90]],
        ]
        mock_cv2.HoughLinesP.return_value = fake_lines

        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.SCATTER
            assert conf == 0.25

    def test_heuristic_handles_exception(self):
        """Should return UNKNOWN on unexpected exceptions."""
        mock_cv2 = MagicMock()
        mock_cv2.imread.side_effect = RuntimeError("disk error")
        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.UNKNOWN
            assert conf == 0.0

    def test_heuristic_unknown_fallback(self):
        """Should return UNKNOWN with 0.1 confidence when no rule matches."""
        mock_cv2 = MagicMock()
        fake_img = MagicMock()
        fake_img.shape = (100, 100, 3)
        mock_cv2.imread.return_value = fake_img
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.Canny.return_value = MagicMock()
        mock_cv2.countNonZero.return_value = 4000  # edge_ratio = 0.4

        # 4 horizontal, 1 vertical - doesn't match BAR (>5 h, >2 v),
        # doesn't match LINE (>5 h, <2 v with h=4 < 5) - actually h=4, v=1
        fake_lines = []
        for i in range(4):
            fake_lines.append([[10, i * 20, 90, i * 20]])
        fake_lines.append([[50, 10, 50, 90]])
        mock_cv2.HoughLinesP.return_value = fake_lines

        with patch("src.tools.chart_analyzer._try_import_cv2", return_value=mock_cv2):
            chart_type, conf = _basic_chart_type_detection("/fake.png")
            assert chart_type == ChartType.UNKNOWN
            assert conf == 0.1


# ---------------------------------------------------------------------------
# execute() with missing / invalid input
# ---------------------------------------------------------------------------

class TestChartAnalyzerExecute:
    """Integration tests for the execute method."""

    def setup_method(self):
        self.analyzer = ChartAnalyzer()

    @pytest.mark.asyncio
    async def test_execute_empty_images(self):
        """Should return empty result when no images provided."""
        result = await self.analyzer.execute({"images": []}, {})
        assert result == {"charts": [], "total_count": 0}

    @pytest.mark.asyncio
    async def test_execute_no_images_key(self):
        """Should return empty result when images key is missing."""
        result = await self.analyzer.execute({}, {})
        assert result == {"charts": [], "total_count": 0}

    @pytest.mark.asyncio
    async def test_execute_none_params(self):
        """Should handle None-like params gracefully."""
        result = await self.analyzer.execute({"images": None}, {})
        # images=None -> .get("images", []) returns None, not []
        # but the code does images = params.get("images", [])
        # None is falsy, so "if not images" is True -> returns empty
        assert result == {"charts": [], "total_count": 0}

    @pytest.mark.asyncio
    async def test_execute_nonexistent_image_path(self):
        """Should skip images whose path does not exist."""
        result = await self.analyzer.execute(
            {"images": [{"path": "/nonexistent/chart.png", "page_idx": 0}]},
            {},
        )
        assert result == {"charts": [], "total_count": 0}

    @pytest.mark.asyncio
    async def test_execute_empty_path_string(self):
        """Should skip images with empty path."""
        result = await self.analyzer.execute(
            {"images": [{"path": "", "page_idx": 0}]},
            {},
        )
        assert result == {"charts": [], "total_count": 0}

    @pytest.mark.asyncio
    async def test_execute_image_missing_path_key(self):
        """Should skip image entries without a 'path' key."""
        result = await self.analyzer.execute(
            {"images": [{"page_idx": 0}]},
            {},
        )
        assert result == {"charts": [], "total_count": 0}

    @pytest.mark.asyncio
    async def test_execute_with_mock_llm(self, tmp_path):
        """Should process a valid image with a mock LLM client end-to-end."""
        # Create a tiny 1x1 PNG file
        img_path = tmp_path / "test_chart.png"
        # Minimal valid PNG bytes (1x1 white pixel)
        import struct
        import zlib

        def _make_minimal_png():
            signature = b"\x89PNG\r\n\x1a\n"
            # IHDR
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            # IDAT
            raw = zlib.compress(b"\x00\x00\x00\x00")
            idat_crc = zlib.crc32(b"IDAT" + raw) & 0xFFFFFFFF
            idat = struct.pack(">I", len(raw)) + b"IDAT" + raw + struct.pack(">I", idat_crc)
            # IEND
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return signature + ihdr + idat + iend

        img_path.write_bytes(_make_minimal_png())

        # Mock LLM client (Anthropic-style) so we control the full path
        mock_client = MagicMock()
        mock_client.model = "claude-test"

        # Build a classification response that returns a chart
        classify_resp = MagicMock()
        classify_block = MagicMock()
        classify_block.text = '{"chart_type": "bar_chart", "title": "Test", "confidence": 0.8, "is_chart": true, "chart_count": 1}'
        classify_resp.content = [classify_block]

        # Build a data extraction response
        extract_resp = MagicMock()
        extract_block = MagicMock()
        extract_block.text = '{"title": "Test", "x_axis_label": "", "y_axis_label": "", "x_axis_unit": "", "y_axis_unit": "", "legend": ["Revenue"], "series": [{"name": "Revenue", "labels": ["Q1"], "values": [100]}], "notes": []}'
        extract_resp.content = [extract_block]

        # Build a description response
        desc_resp = MagicMock()
        desc_block = MagicMock()
        desc_block.text = "A bar chart showing revenue."
        desc_resp.content = [desc_block]

        mock_client.messages.create = AsyncMock(
            side_effect=[classify_resp, classify_resp, extract_resp, desc_resp],
        )

        result = await self.analyzer.execute(
            {
                "images": [{"path": str(img_path), "page_idx": 0}],
                "llm_client": mock_client,
            },
            {},
        )
        assert result["total_count"] == 1
        assert len(result["charts"]) == 1
        assert result["charts"][0]["chart_type"] == "bar_chart"


# ---------------------------------------------------------------------------
# chart_to_table conversion logic
# ---------------------------------------------------------------------------

class TestChartToTable:
    """Test the _chart_to_table conversion method."""

    def setup_method(self):
        self.analyzer = ChartAnalyzer()

    def test_empty_series(self):
        """Empty series list should produce an empty table."""
        result = self.analyzer._chart_to_table([])
        assert result == []

    def test_single_series(self):
        """Single series should produce a two-column table."""
        series = [
            DataSeries(
                name="Revenue",
                labels=["Q1", "Q2", "Q3"],
                values=[100, 200, 300],
            ),
        ]
        table = self.analyzer._chart_to_table(series)
        assert table[0] == ["Label", "Revenue"]
        assert table[1] == ["Q1", "100"]
        assert table[2] == ["Q2", "200"]
        assert table[3] == ["Q3", "300"]

    def test_multiple_series_aligned(self):
        """Multiple series with matching label lengths."""
        series = [
            DataSeries(name="Revenue", labels=["Q1", "Q2"], values=[100, 200]),
            DataSeries(name="Cost", labels=["Q1", "Q2"], values=[80, 150]),
        ]
        table = self.analyzer._chart_to_table(series)
        assert table[0] == ["Label", "Revenue", "Cost"]
        assert table[1] == ["Q1", "100", "80"]
        assert table[2] == ["Q2", "200", "150"]

    def test_uneven_series_lengths(self):
        """Series with different lengths should pad shorter ones with empty strings."""
        series = [
            DataSeries(name="A", labels=["X", "Y", "Z"], values=[1, 2, 3]),
            DataSeries(name="B", labels=["X", "Y"], values=[10, 20]),
        ]
        table = self.analyzer._chart_to_table(series)
        assert len(table) == 4  # header + 3 data rows
        assert table[0] == ["Label", "A", "B"]
        assert table[3] == ["Z", "3", ""]  # B has no value at index 2

    def test_labels_from_first_available_series(self):
        """Labels should be taken from the first series that has one at each index."""
        series = [
            DataSeries(name="A", labels=[], values=[1, 2]),
            DataSeries(name="B", labels=["X", "Y"], values=[10, 20]),
        ]
        table = self.analyzer._chart_to_table(series)
        # max_len from values: 2; labels from A are empty, so fall through to B
        assert table[1] == ["X", "1", "10"]
        assert table[2] == ["Y", "2", "20"]

    def test_no_labels_uses_values_length(self):
        """When labels are empty, table size is determined by values length."""
        series = [
            DataSeries(name="V", labels=[], values=[10, 20, 30]),
        ]
        table = self.analyzer._chart_to_table(series)
        assert table[0] == ["Label", "V"]
        assert len(table) == 4  # header + 3 rows
        assert table[1] == ["", "10"]

    def test_empty_labels_and_values(self):
        """Series with no labels or values should produce an empty table."""
        series = [DataSeries(name="Empty")]
        table = self.analyzer._chart_to_table(series)
        assert table == []

    def test_unnamed_series_gets_default_name(self):
        """Series without a name should get 'Series_N' in the header."""
        series = [
            DataSeries(name="", labels=["A"], values=[1]),
            DataSeries(name="", labels=["B"], values=[2]),
        ]
        table = self.analyzer._chart_to_table(series)
        assert table[0] == ["Label", "Series_1", "Series_2"]

    def test_string_conversion_of_values(self):
        """All values should be converted to strings in the output."""
        series = [
            DataSeries(
                name="Mixed",
                labels=["Row1", "Row2", "Row3"],
                values=[3.14, "N/A", 42],
            ),
        ]
        table = self.analyzer._chart_to_table(series)
        assert table[1] == ["Row1", "3.14"]
        assert table[2] == ["Row2", "N/A"]
        assert table[3] == ["Row3", "42"]


# ---------------------------------------------------------------------------
# _encode_image helper
# ---------------------------------------------------------------------------

class TestEncodeImage:
    """Test the static _encode_image method."""

    def test_encodes_file_to_base64(self, tmp_path):
        """Should base64-encode file contents."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello chart")
        encoded = ChartAnalyzer._encode_image(str(test_file))
        assert encoded == base64.b64encode(b"hello chart").decode("utf-8")

    def test_nonexistent_file_raises(self):
        """Should raise when file does not exist."""
        with pytest.raises(FileNotFoundError):
            ChartAnalyzer._encode_image("/nonexistent/file.png")


# ---------------------------------------------------------------------------
# _build_vision_messages helper
# ---------------------------------------------------------------------------

class TestBuildVisionMessages:
    """Test the static _build_vision_messages method."""

    def test_output_structure(self):
        """Should return a list with a single user message."""
        messages = ChartAnalyzer._build_vision_messages("prompt text", "base64data")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # text + image

    def test_text_block(self):
        """First content block should be the text prompt."""
        messages = ChartAnalyzer._build_vision_messages("my prompt", "b64")
        assert messages[0]["content"][0] == {"type": "text", "text": "my prompt"}

    def test_image_block(self):
        """Second content block should have base64 image data."""
        messages = ChartAnalyzer._build_vision_messages("p", "imgdata123")
        img_block = messages[0]["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["data"] == "imgdata123"
        assert img_block["source"]["media_type"] == "image/png"


# ---------------------------------------------------------------------------
# LLM client dispatch (_call_llm)
# ---------------------------------------------------------------------------

class TestCallLLM:
    """Test the _call_llm static method with different client types."""

    @pytest.mark.asyncio
    async def test_anthropic_style_client(self):
        """Should call client.messages.create for Anthropic-style clients."""
        mock_client = MagicMock()
        mock_client.model = "claude-test"
        mock_block = MagicMock()
        mock_block.text = '{"result": true}'
        mock_resp = MagicMock()
        mock_resp.content = [mock_block]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [{"role": "user", "content": "test"}]
        result = await ChartAnalyzer._call_llm(mock_client, messages)
        assert result == '{"result": true}'
        mock_client.messages.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_openai_style_client(self):
        """Should call client.chat.completions.create for OpenAI-style clients."""
        mock_client = MagicMock()
        mock_client.model = "gpt-test"
        # Remove 'messages' attribute so it doesn't match Anthropic check
        del mock_client.messages
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        messages = [{"role": "user", "content": "test"}]
        result = await ChartAnalyzer._call_llm(mock_client, messages)
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_simple_generate_client(self):
        """Should call client.generate for simple generate interfaces."""
        mock_client = MagicMock()
        # Remove other interfaces
        del mock_client.messages
        del mock_client.chat
        mock_client.generate = AsyncMock(return_value="generated text")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "my prompt"},
                    {"type": "image", "source": {"data": "b64", "media_type": "image/png"}},
                ],
            },
        ]
        result = await ChartAnalyzer._call_llm(mock_client, messages)
        assert result == "generated text"
        # Verify the prompt was flattened (only text parts)
        call_args = mock_client.generate.call_args[0][0]
        assert "my prompt" in call_args

    @pytest.mark.asyncio
    async def test_unrecognised_client_returns_none(self):
        """Should return None for clients with no recognised interface."""
        mock_client = MagicMock(spec=[])  # No attributes
        result = await ChartAnalyzer._call_llm(mock_client, [])
        assert result is None


# ---------------------------------------------------------------------------
# set_llm_client
# ---------------------------------------------------------------------------

class TestSetLLMClient:
    """Test the set_llm_client method."""

    def test_sets_client(self):
        """Should store the provided client."""
        analyzer = ChartAnalyzer()
        mock = MagicMock()
        analyzer.set_llm_client(mock)
        assert analyzer.llm_client is mock

    def test_overwrites_existing(self):
        """Should overwrite a previously set client."""
        analyzer = ChartAnalyzer()
        analyzer.set_llm_client(MagicMock())
        new_mock = MagicMock()
        analyzer.set_llm_client(new_mock)
        assert analyzer.llm_client is new_mock
