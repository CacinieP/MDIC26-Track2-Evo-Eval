"""
Image Enhancer Tool - Low-quality document image preprocessing.

Pipeline for blurry / noisy / handwritten / stamped documents:
1. Quality assessment  (blur, noise, contrast, skew, stamp detection)
2. Adaptive enhancement (CLAHE, denoise, deskew, binarize, stamp removal, upscale)
3. Multi-pass OCR with character-level voting (optional, when OCR engine supplied)
"""

from __future__ import annotations

import asyncio
import io
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# scikit-image imports (optional – some steps degrade gracefully)
# ---------------------------------------------------------------------------
try:
    from skimage.filters import threshold_sauvola
    from skimage.transform import rotate as sk_rotate
    from skimage import img_as_ubyte, img_as_float
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Structured result of image quality assessment."""
    blur_score: float = 0.0          # Laplacian variance – lower = blurrier
    is_blurry: bool = False
    noise_level: float = 0.0         # Estimated noise std-dev
    is_noisy: bool = False
    contrast: float = 0.0            # Michelson contrast
    is_low_contrast: bool = False
    skew_angle: float = 0.0          # Detected skew in degrees
    needs_deskew: bool = False
    has_stamps: bool = False
    stamp_pixel_ratio: float = 0.0   # Fraction of image covered by stamps
    estimated_dpi: float = 72.0
    needs_upscale: bool = False
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _StampDetectionResult:
    """Internal container for stamp/signature detection."""
    mask: np.ndarray | None = None
    stamp_pixel_ratio: float = 0.0
    has_stamps: bool = False


# ---------------------------------------------------------------------------
# Thresholds – tuned for typical document photos / scans
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, Any] = {
    "blur_variance_min": 100.0,       # Below this the image is considered blurry
    "noise_std_max": 30.0,            # Above this we apply heavy denoising
    "contrast_min": 0.2,              # Michelson contrast threshold
    "skew_max_deg": 0.5,              # Rotate if skew exceeds this
    "dpi_min": 200.0,                 # Upscale if estimated DPI is below this
    "stamp_area_ratio_min": 0.002,    # Minimum fraction to consider stamp present
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImageEnhancer:
    """
    Low-quality document image enhancer.

    Provides both sync and async entry-points.  Image processing is CPU-bound;
    the async ``execute`` method wraps the sync pipeline with
    ``run_in_executor`` so it can be awaited inside an async application
    without blocking the event loop.
    """

    # ----- construction -----------------------------------------------------

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.thresholds = {**_DEFAULT_THRESHOLDS, **self.config.get("thresholds", {})}

    # ----- public async API -------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: dict | None = None,
    ) -> dict:
        """
        Enhance a document image.

        Args:
            params: {
                image_path: str | Path,       # (mutually exclusive with image_bytes)
                image_bytes: bytes,           # raw image bytes
                target_dpi: int = 300,        # desired DPI for upscale
                enable_stamp_removal: bool = True,
                enable_binarization: bool = False,
                enable_deskew: bool = True,
                enable_denoise: bool = True,
            }
            context: Pipeline context (unused for now).

        Returns:
            {
                enhanced_image_bytes: bytes,
                quality_report: dict,
                preprocessing_steps_applied: list[str],
            }
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._execute_sync,
            params,
        )
        return result

    # ----- public sync API --------------------------------------------------

    def execute_sync(self, params: dict[str, Any]) -> dict:
        """Synchronous wrapper around the full pipeline."""
        return self._execute_sync(params)

    # ----- internal sync pipeline -------------------------------------------

    def _execute_sync(self, params: dict[str, Any]) -> dict:  # noqa: C901
        image = self._load_image(params)
        if image is None:
            raise ValueError("Could not load image from params")

        target_dpi = int(params.get("target_dpi", 300))
        enable_stamp_removal = params.get("enable_stamp_removal", True)
        enable_binarization = params.get("enable_binarization", False)
        enable_deskew = params.get("enable_deskew", True)
        enable_denoise = params.get("enable_denoise", True)

        # ---- Step 1: Quality assessment ----
        assessment: QualityReport = self._assess_quality(image)
        logger.info(
            "Quality assessment: blur={:.1f} noise={:.1f} contrast={:.3f} "
            "skew={:.2f}deg stamps={} dpi={:.0f}",
            assessment.blur_score,
            assessment.noise_level,
            assessment.contrast,
            assessment.skew_angle,
            assessment.has_stamps,
            assessment.estimated_dpi,
        )

        steps_applied: list[str] = []

        # ---- Step 2: Upscale if needed ----
        if assessment.needs_upscale:
            image = self._upscale(image, target_dpi, assessment.estimated_dpi)
            steps_applied.append(
                f"upscale_to_{target_dpi}dpi"
            )

        # ---- Step 3: Stamp/signature removal ----
        if enable_stamp_removal and assessment.has_stamps:
            image = self._remove_stamps(image)
            steps_applied.append("stamp_removal")

        # ---- Step 4: Deskew ----
        if enable_deskew and assessment.needs_deskew:
            image = self._deskew(image, assessment.skew_angle)
            steps_applied.append(
                f"deskew_{assessment.skew_angle:.2f}deg"
            )

        # ---- Step 5: Contrast enhancement (CLAHE) ----
        if assessment.is_low_contrast:
            image = self._apply_clahe(image)
            steps_applied.append("clahe_contrast")

        # ---- Step 6: Denoising ----
        if enable_denoise and assessment.is_noisy:
            image = self._denoise(image, assessment.noise_level)
            steps_applied.append("denoise")

        # ---- Step 7: Binarization (Sauvola) ----
        if enable_binarization:
            image = self._binarize(image)
            steps_applied.append("sauvola_binarization")

        # ---- Encode result ----
        enhanced_bytes = self._encode_image(image)

        return {
            "enhanced_image_bytes": enhanced_bytes,
            "quality_report": assessment.to_dict(),
            "preprocessing_steps_applied": steps_applied,
        }

    # =======================================================================
    # 1. IMAGE LOADING
    # =======================================================================

    @staticmethod
    def _load_image(params: dict[str, Any]) -> np.ndarray | None:
        """Load an image from file path or raw bytes.  Returns BGR ndarray."""
        image_path = params.get("image_path")
        image_bytes = params.get("image_bytes")

        if image_path is not None:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            buf = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if image_bytes is not None:
            buf = np.frombuffer(image_bytes, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return None

    # =======================================================================
    # 2. QUALITY ASSESSMENT
    # =======================================================================

    def _assess_quality(self, image: np.ndarray) -> QualityReport:
        """
        Evaluate image quality across multiple dimensions.

        All metrics are deterministic given the same input image.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        report = QualityReport(width=w, height=h)

        # -- Blur (Laplacian variance) --
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        report.blur_score = float(lap.var())
        report.is_blurry = report.blur_score < self.thresholds["blur_variance_min"]

        # -- Noise estimation (MAD of high-pass) --
        report.noise_level = self._estimate_noise(gray)
        report.is_noisy = report.noise_level > self.thresholds["noise_std_max"]

        # -- Contrast (Michelson) --
        report.contrast = self._michelson_contrast(gray)
        report.is_low_contrast = report.contrast < self.thresholds["contrast_min"]

        # -- Skew angle --
        report.skew_angle = self._detect_skew_angle(gray)
        report.needs_deskew = abs(report.skew_angle) > self.thresholds["skew_max_deg"]

        # -- Stamp / signature detection --
        stamp_result = self._detect_stamps(image)
        report.has_stamps = stamp_result.has_stamps
        report.stamp_pixel_ratio = stamp_result.stamp_pixel_ratio

        # -- Estimated DPI (heuristic: assumes A4 ~ 210x297 mm) --
        report.estimated_dpi = self._estimate_dpi(w, h)
        report.needs_upscale = report.estimated_dpi < self.thresholds["dpi_min"]

        return report

    # -- blur ----------------------------------------------------------------

    # (computed inline above)

    # -- noise ---------------------------------------------------------------

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """
        Estimate noise level using the Median Absolute Deviation (MAD)
        of the high-pass residual.  Robust to signal content.
        """
        # High-pass via difference from median-filtered version
        median_filtered = cv2.medianBlur(gray, 3)
        residual = gray.astype(np.float64) - median_filtered.astype(np.float64)
        # MAD scaled to approximate std-dev (Gaussian assumption: sigma ~ 1.4826 * MAD)
        mad = np.median(np.abs(residual - np.median(residual)))
        return float(mad * 1.4826)

    # -- contrast ------------------------------------------------------------

    @staticmethod
    def _michelson_contrast(gray: np.ndarray) -> float:
        """Michelson contrast: (I_max - I_min) / (I_max + I_min + eps)."""
        imin, imax = float(gray.min()), float(gray.max())
        denom = imax + imin
        if denom < 1e-6:
            return 0.0
        return (imax - imin) / denom

    # -- skew ----------------------------------------------------------------

    def _detect_skew_angle(self, gray: np.ndarray) -> float:
        """
        Detect document skew angle via Hough line transform.

        Returns angle in degrees (positive = clockwise).
        """
        # Adaptive threshold to get strong edges for line detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10,
        )

        # Morphological close to connect text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(gray.shape[1] // 30, 10), 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Hough lines
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=max(gray.shape[1] // 4, 50),
            maxLineGap=10,
        )

        if lines is None or len(lines) < 5:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-6:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            # Keep angles near horizontal (within +/- 45 deg)
            if abs(angle) < 45:
                angles.append(angle)

        if len(angles) < 3:
            return 0.0

        # Median angle is robust to outliers
        return float(np.median(angles))

    # -- stamp detection -----------------------------------------------------

    def _detect_stamps(self, image: np.ndarray) -> _StampDetectionResult:
        """
        Detect stamp/signature overlays via color-based segmentation.

        Stamps are typically red/orange, signatures blue/black-ink.
        We detect saturated colored regions that stand out from the
        grayscale document background.

        Returns a detection result with a binary mask.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # -- Red stamps (hue wraps around 0/180 in OpenCV HSV) --
        # Range 1: 0-10
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        # Range 2: 170-180
        mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # -- Blue ink / signatures --
        mask_blue = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))

        # -- Green stamps (less common but present in some locales) --
        mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

        combined_mask = cv2.bitwise_or(mask_red, mask_blue)
        combined_mask = cv2.bitwise_or(combined_mask, mask_green)

        # Morphological cleanup – remove speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # Close small gaps inside stamps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

        total_pixels = image.shape[0] * image.shape[1]
        stamp_pixels = int(cv2.countNonZero(combined_mask))
        ratio = stamp_pixels / total_pixels if total_pixels > 0 else 0.0

        has_stamps = ratio >= self.thresholds["stamp_area_ratio_min"]

        return _StampDetectionResult(
            mask=combined_mask,
            stamp_pixel_ratio=ratio,
            has_stamps=has_stamps,
        )

    # -- DPI estimation ------------------------------------------------------

    @staticmethod
    def _estimate_dpi(width: int, height: int) -> float:
        """
        Heuristic DPI estimation assuming the document is roughly A4
        (210 x 297 mm).  Uses the smaller dimension for a conservative
        estimate.
        """
        # A4 short edge in inches
        a4_short_inch = 210 / 25.4
        # Assume the image covers the short edge
        return min(width, height) / a4_short_inch

    # =======================================================================
    # 3. ENHANCEMENT OPERATIONS
    # =======================================================================

    # -- upscale -------------------------------------------------------------

    @staticmethod
    def _upscale(image: np.ndarray, target_dpi: int, current_dpi: float) -> np.ndarray:
        """
        Upscale image using Lanczos interpolation.
        """
        if current_dpi <= 0:
            return image
        scale = target_dpi / current_dpi
        if scale <= 1.05:
            # Already at or above target – skip
            return image
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # -- stamp removal -------------------------------------------------------

    def _remove_stamps(self, image: np.ndarray) -> np.ndarray:
        """
        Remove detected stamps/signatures by inpainting over colored regions.
        """
        stamp_result = self._detect_stamps(image)
        mask = stamp_result.mask
        if mask is None or not stamp_result.has_stamps:
            return image

        # Dilate mask slightly for cleaner inpainting boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)

        # cv2.inpaint works on BGR images; mask must be single-channel uint8
        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return result

    # -- deskew --------------------------------------------------------------

    def _deskew(self, image: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate image to correct skew.  Uses scikit-image for precise
        rotation if available, otherwise falls back to OpenCV.
        """
        if abs(angle_deg) < 0.01:
            return image

        if _HAS_SKIMAGE:
            return self._deskew_skimage(image, angle_deg)
        return self._deskew_opencv(image, angle_deg)

    @staticmethod
    def _deskew_skimage(image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Deskew via scikit-image (sub-pixel accuracy, preserves shape)."""
        # scikit-image rotate expects angle in degrees, counter-clockwise
        # Our detected angle is clockwise-positive, so negate
        # Use OpenCV warpAffine for color images (skimage rotate on uint8 is safe
        # but preserve_range can produce floats outside [0,1] for uint8 input).
        # For grayscale we can safely use skimage.
        if image.ndim == 3:
            # BGR: use OpenCV path for reliability
            return ImageEnhancer._deskew_opencv(image, angle_deg)
        # Grayscale uint8 – skimage handles this natively
        rotated = sk_rotate(image, -angle_deg, resize=False, order=3,
                            mode="edge", preserve_range=True)
        return np.clip(rotated, 0, 255).astype(np.uint8)

    @staticmethod
    def _deskew_opencv(image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Deskew via OpenCV rotation matrix."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        # Choose border color: white for documents
        return cv2.warpAffine(
            image, rot_mat, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # -- CLAHE ---------------------------------------------------------------

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Applied to luminance channel to preserve color.
        """
        if image.ndim == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

        # Convert to LAB, apply CLAHE on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -- denoise -------------------------------------------------------------

    @staticmethod
    def _denoise(image: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Adaptive denoising:
        - Light noise: bilateral filter (preserves edges well)
        - Heavy noise: non-local means (better noise suppression)
        """
        if image.ndim == 3:
            if noise_level > 50:
                # Heavy noise – use non-local means with color
                return cv2.fastNlMeansDenoisingColored(
                    image, None,
                    h=10, hForColorComponents=10,
                    templateWindowSize=7, searchWindowSize=21,
                )
            # Light noise – bilateral filter (faster, edge-preserving)
            return cv2.bilateralFilter(
                image, d=9, sigmaColor=75, sigmaSpace=75,
            )
        else:
            # Grayscale path
            if noise_level > 50:
                return cv2.fastNlMeansDenoising(
                    image, None, h=10,
                    templateWindowSize=7, searchWindowSize=21,
                )
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # -- binarization --------------------------------------------------------

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Sauvola adaptive thresholding for binarization.
        Produces a clean black/white document suitable for OCR.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if _HAS_SKIMAGE:
            return self._binarize_sauvola_skimage(gray)
        return self._binarize_adaptive_opencv(gray)

    @staticmethod
    def _binarize_sauvola_skimage(gray: np.ndarray) -> np.ndarray:
        """Sauvola thresholding via scikit-image."""
        float_img = img_as_float(gray)
        window_size = max(15, min(gray.shape) // 10)
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        thresh = threshold_sauvola(float_img, window_size=window_size, k=0.2)
        binary = (float_img > thresh).astype(np.uint8) * 255
        return binary

    @staticmethod
    def _binarize_adaptive_opencv(gray: np.ndarray) -> np.ndarray:
        """Fallback adaptive thresholding via OpenCV."""
        block_size = max(11, min(gray.shape) // 10)
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=10,
        )

    # =======================================================================
    # 4. MULTI-PASS OCR WITH VOTING
    # =======================================================================

    def multipass_ocr(
        self,
        image: np.ndarray,
        ocr_fn,  # Callable[[np.ndarray], str]
        param_sets: list[dict] | None = None,
    ) -> dict:
        """
        Run OCR with multiple preprocessing parameter sets and vote.

        Args:
            image: Input BGR image.
            ocr_fn: A callable that takes an ndarray and returns recognized text.
            param_sets: List of parameter dicts for different preprocessing
                        variants.  If None, uses 3 default sets.

        Returns:
            {
                "consensus_text": str,
                "per_char_confidence": list[float],
                "variant_texts": list[str],
            }
        """
        if param_sets is None:
            param_sets = [
                {},  # No extra preprocessing
                {"enable_binarization": True},
                {"enable_binarization": True, "enable_denoise": True},
            ]

        variant_texts: list[str] = []
        for pset in param_sets:
            merged = {**pset, "image_bytes": self._encode_image(image)}
            result = self._execute_sync(merged)
            enhanced = self._decode_image_bytes(result["enhanced_image_bytes"])
            if enhanced is not None:
                text = ocr_fn(enhanced)
            else:
                text = ""
            variant_texts.append(text)

        consensus, confidence = self._vote_characters(variant_texts)
        return {
            "consensus_text": consensus,
            "per_char_confidence": confidence,
            "variant_texts": variant_texts,
        }

    @staticmethod
    def _vote_characters(texts: list[str]) -> tuple[str, list[float]]:
        """
        Align multiple OCR outputs and pick the majority character at
        each position.  Returns consensus string and per-char confidence
        (fraction of votes agreeing).
        """
        if not texts:
            return "", []

        # Pad to same length
        max_len = max(len(t) for t in texts)
        padded = [t.ljust(max_len, "\x00") for t in texts]

        consensus_chars: list[str] = []
        confidence_list: list[float] = []

        for i in range(max_len):
            votes: dict[str, int] = {}
            for t in padded:
                ch = t[i]
                if ch == "\x00":
                    continue
                votes[ch] = votes.get(ch, 0) + 1

            if not votes:
                break

            best_char = max(votes, key=lambda c: votes[c])
            total_votes = sum(votes.values())
            confidence = votes[best_char] / len(texts) if len(texts) > 0 else 0.0

            consensus_chars.append(best_char)
            confidence_list.append(confidence)

        return "".join(consensus_chars), confidence_list

    # =======================================================================
    # 5. ENCODING / DECODING HELPERS
    # =======================================================================

    @staticmethod
    def _encode_image(image: np.ndarray, fmt: str = ".png") -> bytes:
        """Encode ndarray to bytes (PNG by default for lossless output)."""
        success, buf = cv2.imencode(fmt, image)
        if not success:
            raise RuntimeError(f"Failed to encode image as {fmt}")
        return buf.tobytes()

    @staticmethod
    def _decode_image_bytes(data: bytes) -> np.ndarray | None:
        """Decode bytes back to BGR ndarray."""
        buf = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
