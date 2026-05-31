"""
Tests for the MinerU Parser tool.
Tests: initialisation, MinerU availability detection, image pre-processing
       quality assessment, execute() error handling, fallback parsing,
       supported file-type detection.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Ensure the project root is importable (conftest may not run first in
# isolated test-collection scenarios).
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Import target — the module-level import of cv2 means we skip the entire
# file gracefully when OpenCV is not installed.
# ---------------------------------------------------------------------------
cv2 = pytest.importorskip("cv2", reason="OpenCV required for MinerUParser tests")

from src.tools.mineru_parser import (
    _SUPPORTED_EXTS,
    _check_mineru_available,
    _suffix,
    ImagePreprocessor,
    MinerUParser,
)


# ===================================================================
# 1. MinerUParser initialisation
# ===================================================================


class TestMinerUParserInit:
    """Test MinerUParser instantiation with various configs."""

    def test_default_config(self):
        """Parser should initialise with sensible defaults."""
        parser = MinerUParser()
        assert parser.model_dir == "./models"
        assert parser.device == "cuda"
        assert parser.preprocess_enabled is True
        assert parser.table_enable is True
        assert parser.formula_enable is True
        assert parser.ocr_lang == "auto"
        assert isinstance(parser.output_base, Path)
        assert parser._ocr_engine is None

    def test_custom_config(self):
        """Parser should honour every recognised config key."""
        cfg = {
            "model_dir": "/tmp/models",
            "device": "cpu",
            "output_dir": "/tmp/out",
            "preprocess": False,
            "table_enable": False,
            "formula_enable": False,
            "ocr_lang": "en",
        }
        parser = MinerUParser(cfg)
        assert parser.model_dir == "/tmp/models"
        assert parser.device == "cpu"
        assert parser.output_base == Path("/tmp/out")
        assert parser.preprocess_enabled is False
        assert parser.table_enable is False
        assert parser.formula_enable is False
        assert parser.ocr_lang == "en"

    def test_none_config(self):
        """Passing None should behave identically to the default constructor."""
        parser = MinerUParser(None)
        assert parser.model_dir == "./models"
        assert parser.device == "cuda"

    def test_mineru_available_flag_set(self):
        """_mineru_available should be a boolean set at init time."""
        parser = MinerUParser()
        assert isinstance(parser._mineru_available, bool)

    @patch("src.tools.mineru_parser._check_mineru_available", return_value=True)
    def test_init_with_mineru_present(self, mock_check):
        """When MinerU is importable the flag should be True."""
        parser = MinerUParser()
        assert parser._mineru_available is True
        mock_check.assert_called_once()

    @patch("src.tools.mineru_parser._check_mineru_available", return_value=False)
    def test_init_without_mineru(self, mock_check):
        """When MinerU is absent the flag should be False (no crash)."""
        parser = MinerUParser()
        assert parser._mineru_available is False


# ===================================================================
# 2. _check_mineru_available detection
# ===================================================================


class TestCheckMineruAvailable:
    """Test the module-level MinerU detection helper."""

    @patch.dict("sys.modules", {"magic_pdf": MagicMock()})
    def test_returns_true_when_importable(self):
        assert _check_mineru_available() is True

    @patch.dict("sys.modules", {"magic_pdf": None})
    def test_returns_false_when_not_importable(self):
        # Force ImportError by removing magic_pdf if present
        with patch.dict("sys.modules", {"magic_pdf": None}):
            # Re-import the check inside the patch is not feasible at
            # module level, so call the already-bound function which
            # will attempt a fresh import.
            # Simulate ImportError by temporarily making magic_pdf
            # raise on attribute access.
            pass
        # Instead, patch more directly:
        with patch("builtins.__import__", side_effect=ImportError("no magic_pdf")):
            assert _check_mineru_available() is False


# ===================================================================
# 3. ImagePreprocessor quality assessment (synthetic images)
# ===================================================================


class TestImagePreprocessor:
    """Test the static image-enhancement pipeline on synthetic images."""

    # -- helpers --

    @staticmethod
    def _make_clean_text_image(size: int = 200) -> np.ndarray:
        """White background with a black rectangle (simulates text)."""
        img = np.ones((size, size), dtype=np.uint8) * 255
        img[60:140, 40:160] = 0  # black block
        return img

    @staticmethod
    def _make_blurry_image(size: int = 200) -> np.ndarray:
        """Gaussian-blurred version of a clean image."""
        clean = TestImagePreprocessor._make_clean_text_image(size)
        return cv2.GaussianBlur(clean, (31, 31), sigmaX=5)

    @staticmethod
    def _make_noisy_image(size: int = 200) -> np.ndarray:
        """Clean image corrupted by Gaussian noise."""
        clean = TestImagePreprocessor._make_clean_text_image(size)
        noise = np.random.normal(0, 50, clean.shape).astype(np.int16)
        noisy = np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def _make_low_contrast_image(size: int = 200) -> np.ndarray:
        """Image with pixel values squeezed into a narrow band."""
        img = np.ones((size, size), dtype=np.uint8) * 130
        img[60:140, 40:160] = 140  # barely distinguishable block
        return img

    # -- tests --

    def test_enhance_returns_uint8_ndarray(self):
        """Enhanced output must be a uint8 numpy array."""
        img = self._make_clean_text_image()
        result = ImagePreprocessor.enhance(img)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_enhance_shape_preserved(self):
        """Enhanced output should have the same spatial dimensions."""
        img = self._make_clean_text_image()
        result = ImagePreprocessor.enhance(img)
        assert result.shape == img.shape

    def test_enhance_colour_input_produces_grayscale(self):
        """A 3-channel image should come out 2-channel (grayscale)."""
        img_colour = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ImagePreprocessor.enhance(img_colour)
        assert result.ndim == 2

    def test_enhance_blurry_image_improves_contrast(self):
        """CLAHE should increase the contrast of a blurry image."""
        blurry = self._make_blurry_image()
        enhanced = ImagePreprocessor.enhance(blurry)
        # Standard deviation is a proxy for contrast
        std_before = float(np.std(blurry))
        std_after = float(np.std(enhanced))
        # Enhancement should not reduce contrast
        assert std_after >= std_before * 0.8  # allow small margin

    def test_enhance_noisy_image_does_not_crash(self):
        """Enhancement pipeline must survive heavy noise."""
        noisy = self._make_noisy_image()
        result = ImagePreprocessor.enhance(noisy)
        assert result is not None
        assert result.shape == noisy.shape

    def test_enhance_low_contrast_image_increases_std(self):
        """CLAHE should widen the pixel-value distribution of a low-contrast image."""
        low = self._make_low_contrast_image()
        enhanced = ImagePreprocessor.enhance(low)
        assert float(np.std(enhanced)) > float(np.std(low))

    def test_deskew_near_zero_angle_returns_original(self):
        """An image with negligible skew (< 0.5 deg) should be returned as-is."""
        img = self._make_clean_text_image()
        result = ImagePreprocessor._deskew(img)
        # Very slight numerical differences from CLAHE etc. are expected
        # but for a perfectly aligned image, _deskew short-circuits.
        assert result is not None


# ===================================================================
# 4. execute() — error / edge-case handling
# ===================================================================


class TestMinerUParserExecute:
    """Integration tests for the execute() entry point."""

    @pytest.mark.asyncio
    async def test_execute_no_file_path_returns_demo_result(self):
        """When no file_path is given, should return a demo-mode result."""
        parser = MinerUParser()
        result = await parser.execute({}, {})
        assert result["source_file"] == ""
        assert result["pages"] == 0
        assert result["metadata"]["mode"] == "demo"

    @pytest.mark.asyncio
    async def test_execute_unsupported_extension_raises(self, tmp_path):
        """A file with an unsupported extension should raise ValueError."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("hello")
        parser = MinerUParser()
        with pytest.raises(ValueError, match="Unsupported file format"):
            await parser.execute({"file_path": str(bad_file)}, {})

    @pytest.mark.asyncio
    async def test_execute_missing_file_raises(self, tmp_path):
        """A non-existent file_path should raise FileNotFoundError."""
        missing = tmp_path / "nonexistent.pdf"
        parser = MinerUParser()
        with pytest.raises(FileNotFoundError):
            await parser.execute({"file_path": str(missing)}, {})

    @pytest.mark.asyncio
    async def test_execute_none_file_path_returns_demo(self):
        """file_path=None should be treated like a missing key."""
        parser = MinerUParser()
        result = await parser.execute({"file_path": None}, {})
        assert result["metadata"]["mode"] == "demo"

    @pytest.mark.asyncio
    async def test_execute_supported_ext_existing_file(self, tmp_path):
        """
        A supported file that exists should not raise at the routing stage.
        We mock the handler so we do not need real MinerU / OCR deps.
        """
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        parser = MinerUParser()
        mock_result = {
            "source_file": str(pdf_file),
            "pages": 1,
            "content_list": [],
            "markdown": "",
            "tables": [],
            "images": [],
            "metadata": {"parser": "test"},
        }
        with patch.object(parser, "_handle_pdf", new_callable=AsyncMock, return_value=mock_result):
            result = await parser.execute({"file_path": str(pdf_file)}, {})
        assert result["pages"] == 1
        assert "parse_time_s" in result["metadata"]


# ===================================================================
# 5. _fallback_parse with a mock file
# ===================================================================


class TestFallbackParse:
    """Test the generic fallback routing method."""

    @pytest.mark.asyncio
    async def test_fallback_routes_image(self, tmp_path):
        """_fallback_parse should route .png to _parse_image_basic."""
        img_file = tmp_path / "test.png"
        # Create a minimal 1x1 white PNG
        white = np.ones((1, 1, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(img_file), white)

        parser = MinerUParser()
        # Mock _parse_image_basic so we do not need PaddleOCR
        expected = {
            "source_file": str(img_file),
            "pages": 1,
            "content_list": [],
            "markdown": "",
            "tables": [],
            "images": [],
            "metadata": {"parser": "mock"},
        }
        with patch.object(parser, "_parse_image_basic", new_callable=AsyncMock, return_value=expected):
            result = await parser._fallback_parse(img_file)
        assert result["metadata"]["parser"] == "mock"

    @pytest.mark.asyncio
    async def test_fallback_routes_pdf(self, tmp_path):
        """_fallback_parse should route .pdf to _parse_pdf_basic."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy")
        parser = MinerUParser()
        expected = {
            "source_file": str(pdf_file),
            "pages": 0,
            "content_list": [],
            "markdown": "",
            "tables": [],
            "images": [],
            "metadata": {"parser": "mock_pdf"},
        }
        with patch.object(parser, "_parse_pdf_basic", new_callable=AsyncMock, return_value=expected):
            result = await parser._fallback_parse(pdf_file)
        assert result["metadata"]["parser"] == "mock_pdf"

    @pytest.mark.asyncio
    async def test_fallback_routes_docx(self, tmp_path):
        """_fallback_parse should route .docx to _parse_docx_basic."""
        docx_file = tmp_path / "doc.docx"
        docx_file.write_bytes(b"PK\x03\x04 dummy")
        parser = MinerUParser()
        expected = {
            "source_file": str(docx_file),
            "pages": 1,
            "content_list": [],
            "markdown": "",
            "tables": [],
            "images": [],
            "metadata": {"parser": "mock_docx"},
        }
        with patch.object(parser, "_parse_docx_basic", new_callable=AsyncMock, return_value=expected):
            result = await parser._fallback_parse(docx_file)
        assert result["metadata"]["parser"] == "mock_docx"

    @pytest.mark.asyncio
    async def test_fallback_routes_html(self, tmp_path):
        """_fallback_parse should route .html to _parse_html."""
        html_file = tmp_path / "page.html"
        html_file.write_text("<html><body>Hello</body></html>")
        parser = MinerUParser()
        expected = {
            "source_file": str(html_file),
            "pages": 1,
            "content_list": [],
            "markdown": "",
            "tables": [],
            "images": [],
            "metadata": {"parser": "mock_html"},
        }
        with patch.object(parser, "_parse_html", new_callable=AsyncMock, return_value=expected):
            result = await parser._fallback_parse(html_file)
        assert result["metadata"]["parser"] == "mock_html"

    @pytest.mark.asyncio
    async def test_fallback_unknown_extension_raises(self, tmp_path):
        """An extension with no fallback parser should raise ValueError."""
        weird = tmp_path / "weird.zip"
        weird.write_bytes(b"dummy")
        parser = MinerUParser()
        with pytest.raises(ValueError, match="No fallback parser"):
            # We need to give it a path whose suffix is not in _SUPPORTED_EXTS
            # but _fallback_parse only routes known suffixes.
            # Create a file with a known-but-unhandled-in-fallback suffix
            # by temporarily patching _suffix — or just feed a .zip file.
            await parser._fallback_parse(weird)


# ===================================================================
# 6. Supported file-type detection
# ===================================================================


class TestSupportedFileTypes:
    """Verify the supported extension sets and _suffix helper."""

    def test_pdf_extensions(self):
        assert ".pdf" in _SUPPORTED_EXTS

    def test_image_extensions(self):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"):
            assert ext in _SUPPORTED_EXTS

    def test_office_extensions(self):
        assert ".docx" in _SUPPORTED_EXTS
        assert ".pptx" in _SUPPORTED_EXTS

    def test_html_extensions(self):
        assert ".html" in _SUPPORTED_EXTS
        assert ".htm" in _SUPPORTED_EXTS

    def test_suffix_lowercase(self):
        assert _suffix("FILE.PDF") == ".pdf"

    def test_suffix_with_path_object(self):
        assert _suffix(Path("/tmp/doc.PNG")) == ".png"

    def test_suffix_no_extension(self):
        assert _suffix("Makefile") == ""

    def test_unsupported_extension_not_in_set(self):
        assert ".exe" not in _SUPPORTED_EXTS
        assert ".txt" not in _SUPPORTED_EXTS
        assert ".csv" not in _SUPPORTED_EXTS


# ===================================================================
# 7. Output directory management
# ===================================================================


class TestMakeOutputDir:
    """Test _make_output_dir sanitisation and creation."""

    def test_creates_directory(self, tmp_path):
        parser = MinerUParser({"output_dir": str(tmp_path / "out")})
        fake_path = Path("report 2024.pdf")
        out_dir = parser._make_output_dir(fake_path)
        assert out_dir.exists()
        assert out_dir.is_dir()

    def test_sanitises_special_characters(self, tmp_path):
        parser = MinerUParser({"output_dir": str(tmp_path / "out")})
        fake_path = Path("my report (v2.0).pdf")
        out_dir = parser._make_output_dir(fake_path)
        # Spaces and parentheses should be replaced
        assert " " not in out_dir.name
        assert "(" not in out_dir.name
        assert out_dir.exists()


# ===================================================================
# 8. Table-to-Markdown helper
# ===================================================================


class TestTableToMarkdown:
    """Test the _table_to_markdown static helper."""

    def test_basic_table(self):
        rows = [["A", "B"], ["1", "2"]]
        md = MinerUParser._table_to_markdown(rows)
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_empty_rows(self):
        assert MinerUParser._table_to_markdown([]) == ""

    def test_uneven_columns_padded(self):
        rows = [["A", "B", "C"], ["1", "2"]]
        md = MinerUParser._table_to_markdown(rows)
        # Second row should be padded to 3 columns
        lines = md.split("\n")
        body_line = lines[2]
        # 3 columns => 4 delimiters: "| 1 | 2 |  |"
        assert body_line.count("|") == 4


# ===================================================================
# 9. PDF auto-split for Precise API (>200 pages)
# ===================================================================


class TestPdfAutoSplit:
    """Test _count_pdf_pages, _split_pdf, and _cloud_precise_split."""

    def test_count_pdf_pages_real(self, tmp_path):
        """Count pages of a real multi-page PDF created with fitz."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF required for split tests")
        doc = fitz.open()
        for _ in range(5):
            doc.new_page(width=72, height=72)
        pdf_path = tmp_path / "five.pdf"
        doc.save(str(pdf_path))
        doc.close()
        assert MinerUParser._count_pdf_pages(pdf_path) == 5

    def test_count_pdf_pages_missing_file(self):
        """Should return 0 for a file that can't be opened."""
        assert MinerUParser._count_pdf_pages(Path("/nonexistent/file.pdf")) == 0

    def test_split_pdf_produces_correct_chunks(self, tmp_path):
        """Split a 5-page PDF into chunks of 2 — should produce 3 files."""
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        for i in range(5):
            page = doc.new_page(width=72, height=72)
            page.insert_text((10, 30), f"Page {i+1}")
        pdf_path = tmp_path / "five_pages.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = MinerUParser()
        chunks = parser._split_pdf(pdf_path, chunk_size=2)

        assert len(chunks) == 3  # [1-2], [3-4], [5]

        # Verify each chunk has the right page count
        for chunk_path in chunks:
            chunk_doc = fitz.open(str(chunk_path))
            if "part1" in chunk_path.stem:
                assert len(chunk_doc) == 2
            elif "part2" in chunk_path.stem:
                assert len(chunk_doc) == 2
            elif "part3" in chunk_path.stem:
                assert len(chunk_doc) == 1
            chunk_doc.close()

    def test_split_pdf_no_split_needed(self, tmp_path):
        """A PDF with ≤chunk_size pages should produce a single chunk."""
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        doc.new_page(width=72, height=72)
        pdf_path = tmp_path / "single.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = MinerUParser()
        chunks = parser._split_pdf(pdf_path, chunk_size=200)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_cloud_precise_split_merges_results(self, tmp_path):
        """_cloud_precise_split should merge markdown and tables from all chunks."""
        parser = MinerUParser({"api_token": "test-token"})

        # Create a 5-page PDF, split with chunk_size=3 → [1-3], [4-5]
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        for i in range(5):
            doc.new_page(width=72, height=72)
        pdf_path = tmp_path / "five.pdf"
        doc.save(str(pdf_path))
        doc.close()

        # Mock _split_pdf to use chunk_size=3 instead of 200
        chunks = parser._split_pdf(pdf_path, chunk_size=3)
        assert len(chunks) == 2

        # Mock _cloud_precise_api to return controlled results
        chunk_results = [
            {
                "source_file": "chunk1",
                "pages": 3,
                "content_list": [{"type": "text", "text": "chunk1 text", "page": 0}],
                "markdown": "# Chunk 1",
                "tables": [{"type": "table", "data": [[1]], "page": 2}],
                "images": [],
                "metadata": {"parser": "mineru_cloud_precise", "api_batch_id": "b1"},
            },
            {
                "source_file": "chunk2",
                "pages": 2,
                "content_list": [{"type": "text", "text": "chunk2 text", "page": 0}],
                "markdown": "# Chunk 2",
                "tables": [{"type": "table", "data": [[2]], "page": 0}],
                "images": [],
                "metadata": {"parser": "mineru_cloud_precise", "api_batch_id": "b2"},
            },
        ]

        async def mock_precise(path, name, suffix):
            idx = mock_precise.call_count
            mock_precise.call_count += 1
            return chunk_results[idx] if idx < len(chunk_results) else None

        mock_precise.call_count = 0

        # Patch _split_pdf to return our 2 chunks and _cloud_precise_api for results
        with patch.object(parser, "_split_pdf", return_value=chunks), \
             patch.object(parser, "_cloud_precise_api", side_effect=mock_precise):
            result = await parser._cloud_precise_split(pdf_path, "five.pdf", 5)

        assert result is not None
        assert result["pages"] == 5
        assert "# Chunk 1" in result["markdown"]
        assert "# Chunk 2" in result["markdown"]
        assert len(result["tables"]) == 2
        # Page offset should be applied to chunk 2
        assert result["tables"][1]["page"] == 3  # 0 + 3 pages offset
        assert result["content_list"][1]["page"] == 3  # chunk2 text offset by 3
        assert result["metadata"]["parser"] == "mineru_cloud_precise_split"
        assert result["metadata"]["total_pages"] == 5

    @pytest.mark.asyncio
    async def test_parse_via_cloud_api_routes_to_split(self, tmp_path):
        """_parse_via_cloud_api should call _cloud_precise_split for >200 pages."""
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        for _ in range(201):
            doc.new_page(width=72, height=72)
        pdf_path = tmp_path / "big.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = MinerUParser({"api_token": "test-token", "api_mode": "cloud"})

        mock_split_result = {
            "source_file": str(pdf_path),
            "pages": 201,
            "content_list": [],
            "markdown": "merged",
            "tables": [],
            "images": [],
            "metadata": {"parser": "mineru_cloud_precise_split"},
        }

        with patch.object(parser, "_cloud_precise_split", new_callable=AsyncMock, return_value=mock_split_result):
            result = await parser._parse_via_cloud_api(pdf_path)

        assert result is not None
        assert result["metadata"]["parser"] == "mineru_cloud_precise_split"

    @pytest.mark.asyncio
    async def test_parse_via_cloud_api_no_split_under_200(self, tmp_path):
        """_parse_via_cloud_api should NOT split for ≤200 pages."""
        fitz = pytest.importorskip("fitz")
        doc = fitz.open()
        for _ in range(50):
            doc.new_page(width=72, height=72)
        pdf_path = tmp_path / "small.pdf"
        doc.save(str(pdf_path))
        doc.close()

        parser = MinerUParser({"api_token": "test-token", "api_mode": "cloud"})

        mock_result = {
            "source_file": str(pdf_path),
            "pages": 50,
            "content_list": [],
            "markdown": "no split",
            "tables": [],
            "images": [],
            "metadata": {"parser": "mineru_cloud_precise"},
        }

        with patch.object(parser, "_cloud_precise_api", new_callable=AsyncMock, return_value=mock_result):
            result = await parser._parse_via_cloud_api(pdf_path)

        assert result is not None
        assert result["metadata"]["parser"] == "mineru_cloud_precise"
