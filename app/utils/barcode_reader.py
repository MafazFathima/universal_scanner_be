import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import logging
import time

# Primary decoder (best PDF417 support on Windows)
ZXING_AVAILABLE = False
try:
    import zxingcpp
    ZXING_AVAILABLE = True
except Exception:
    zxingcpp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BarcodeReader:
    @staticmethod
    def _extract_aamva_records(raw: str) -> dict:
        if not raw.startswith("@"):
            return {}
        records = {}
        for line in raw.splitlines():
            if len(line) >= 4:
                records[line[:3]] = line[3:]
        return records

    @staticmethod
    def _format_date_mdy(value: str):
        if not value or len(value) != 8 or not value.isdigit():
            return None
        return f"{value[4:8]}-{value[0:2]}-{value[2:4]}"

    @staticmethod
    def parse_aamva(raw: str):
        records = BarcodeReader._extract_aamva_records(raw)
        if not records:
            return None
        return {
            "license_number": records.get("DAQ"),
            "first_name": records.get("DAC"),
            "last_name": records.get("DCS"),
            "dob": records.get("DBB"),
            "expiry": records.get("DBA"),
            "issue_date": records.get("DBD"),
            "sex": records.get("DBC"),
            "address": {
                "street": records.get("DAG"),
                "city": records.get("DAI"),
                "state": records.get("DAJ"),
                "zip": records.get("DAK"),
            }
        }

    @staticmethod
    def _normalize_aamva_raw(raw: str) -> str:
        """
        Normalize AAMVA raw string with control markers to multiline text.
        Handles literal markers like <LF>, <CR>, <RS> as well as actual control chars.
        """
        # Replace literal markers (as seen from some decoders)
        normalized = raw.replace("<LF>", "\n").replace("<CR>", "\r").replace("<RS>", "\x1e")
        # Replace common control characters with line breaks where appropriate
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        # Record/segment separators -> newline
        normalized = normalized.replace("\x1e", "\n")  # RS
        normalized = normalized.replace("\x1d", "\n")  # GS
        return normalized

    @staticmethod
    def _build_frontend_payload(records: dict, normalized_raw: str, raw: str):
        if not records:
            return None
        dob_iso = BarcodeReader._format_date_mdy(records.get("DBB", ""))
        issue_iso = BarcodeReader._format_date_mdy(records.get("DBD", ""))
        expiry_iso = BarcodeReader._format_date_mdy(records.get("DBA", ""))
        height_raw = records.get("DAU")
        height_in = None
        if height_raw:
            digits = "".join(ch for ch in height_raw if ch.isdigit())
            if digits:
                height_in = digits
        return {
            "raw": raw,
            "raw_formatted": normalized_raw,
            "fields": records,
            "person": {
                "first_name": records.get("DAC"),
                "middle_name": records.get("DAD"),
                "last_name": records.get("DCS"),
                "suffix": records.get("DCU"),
                "sex": records.get("DBC"),
                "eye_color": records.get("DAY"),
                "hair_color": records.get("DAZ"),
                "height_in": height_in,
                "weight_lb": records.get("DAW"),
            },
            "document": {
                "license_number": records.get("DAQ"),
                "issue_date": issue_iso,
                "expiry_date": expiry_iso,
                "dob": dob_iso,
                "issuer_country": records.get("DCG"),
                "audit_number": records.get("DCF"),
                "document_discriminator": records.get("DCK"),
                "card_revision_date": records.get("DDB"),
                "hazmat_expiry": records.get("DCH"),
            },
            "address": {
                "street": records.get("DAG"),
                "city": records.get("DAI"),
                "state": records.get("DAJ"),
                "postal_code": (records.get("DAK") or "").strip(),
            },
        }

    @staticmethod
    def _try_decode_zxing(image: np.ndarray):
        """Full-ID to PDF417 pipeline: detect ROI -> tight crop -> enhance -> upscale -> multi-angle decode."""
        if not ZXING_AVAILABLE or zxingcpp is None:
            logger.error("Primary barcode decoder not available - cannot decode barcodes")
            return []
        allowed_types = {"PDF417"}
        fast_budget_sec = 2.2
        full_budget_sec = 7.0
        budget_warned = False

        def _read(img: np.ndarray, mode: str = "fast"):
            kwargs = {}
            try:
                if mode == "full" and "try_harder" in zxingcpp.read_barcodes.__code__.co_varnames:
                    kwargs["try_harder"] = True
            except Exception:
                pass
            try:
                if "try_rotate" in zxingcpp.read_barcodes.__code__.co_varnames:
                    kwargs["try_rotate"] = True
            except Exception:
                pass
            try:
                if "try_invert" in zxingcpp.read_barcodes.__code__.co_varnames:
                    kwargs["try_invert"] = True
            except Exception:
                pass
            try:
                results = zxingcpp.read_barcodes(img, **kwargs)
                if results:
                    return results
                # Fallback API in some builds can decode cases read_barcodes misses.
                if hasattr(zxingcpp, "read_barcode"):
                    one = zxingcpp.read_barcode(img, **kwargs)
                    if one is not None:
                        return [one]
                return []
            except TypeError:
                results = zxingcpp.read_barcodes(img)
                if results:
                    return results
                if hasattr(zxingcpp, "read_barcode"):
                    one = zxingcpp.read_barcode(img)
                    if one is not None:
                        return [one]
                return []
        def _normalize_results(results):
            decoded_items = []
            for r in results:
                text = getattr(r, "text", None)
                if text is None:
                    text = getattr(r, "data", None)
                if text is None:
                    raw_bytes = getattr(r, "bytes", None)
                    if raw_bytes is not None:
                        try:
                            text = bytes(raw_bytes).decode("utf-8", errors="ignore")
                        except Exception:
                            text = str(raw_bytes)
                if text is None:
                    continue
                if isinstance(text, (bytes, bytearray)):
                    text = text.decode("utf-8", errors="ignore")
                elif not isinstance(text, str):
                    text = str(text)
                text = text.strip()
                if not text:
                    continue
                fmt = getattr(r, "format", None)
                if hasattr(fmt, "name"):
                    btype = fmt.name
                elif fmt is not None:
                    btype = str(fmt)
                else:
                    btype = "UNKNOWN"
                decoded_items.append({"type": btype, "data": text})
            # PDF417 priority: return PDF417 entries if present.
            pdf417_items = [
                item for item in decoded_items
                if item["type"] in allowed_types or "PDF417" in item["type"]
            ]
            if pdf417_items:
                return pdf417_items
            # Some builds may return UNKNOWN format even when payload is AAMVA text.
            fallback_items = []
            for item in decoded_items:
                data = item.get("data", "")
                if data.startswith("@") or "ANSI " in data:
                    fallback_items.append(item)
            return fallback_items
        def _candidate_images(base: np.ndarray, mode: str = "fast"):
            candidates = []
            candidates.append(base)
            if len(base.shape) == 3:
                gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
            else:
                gray = base
            candidates.append(gray)
            # Fast mode: cheap transforms only.
            if mode == "fast":
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    contrast = clahe.apply(gray)
                    candidates.append(contrast)
                except Exception:
                    contrast = gray
                try:
                    bin_img = cv2.adaptiveThreshold(
                        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 31, 5
                    )
                    candidates.append(bin_img)
                except Exception:
                    pass
                try:
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    sharp = cv2.filter2D(gray, -1, kernel)
                    candidates.append(sharp)
                except Exception:
                    pass
                # Close-up PDF417 often needs more vertical pixels than horizontal.
                try:
                    anisotropic = cv2.resize(
                        contrast, None, fx=1.8, fy=3.0, interpolation=cv2.INTER_CUBIC
                    )
                    candidates.append(anisotropic)
                except Exception:
                    pass
                return candidates
            # Blur recovery pipeline: denoise + contrast boost + sharpen
            try:
                denoised = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
                candidates.append(denoised)
            except Exception:
                denoised = gray
            try:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                contrast = clahe.apply(denoised)
                candidates.append(contrast)
            except Exception:
                contrast = denoised
            try:
                blur = cv2.GaussianBlur(contrast, (0, 0), 1.2)
                unsharp = cv2.addWeighted(contrast, 1.8, blur, -0.8, 0)
                candidates.append(unsharp)
            except Exception:
                unsharp = contrast
            try:
                bin_img = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 5
                )
                candidates.append(bin_img)
            except Exception:
                pass
            # Additional threshold styles help noisy/low-contrast scans
            try:
                _, otsu = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                candidates.append(otsu)
            except Exception:
                pass
            try:
                inv = cv2.bitwise_not(unsharp)
                _, inv_otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                candidates.append(inv_otsu)
            except Exception:
                pass
            try:
                inv_bin = cv2.adaptiveThreshold(
                    unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 31, 7
                )
                candidates.append(inv_bin)
            except Exception:
                pass
            for scale in [1.8, 2.4]:
                try:
                    scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    candidates.append(scaled)
                except Exception:
                    pass
            try:
                scaled_unsharp = cv2.resize(unsharp, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
                candidates.append(scaled_unsharp)
            except Exception:
                pass
            try:
                anisotropic_full = cv2.resize(
                    unsharp, None, fx=2.0, fy=3.4, interpolation=cv2.INTER_CUBIC
                )
                candidates.append(anisotropic_full)
            except Exception:
                pass
            # Simple sharpen kernel can help dense PDF417
            try:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharp = cv2.filter2D(gray, -1, kernel)
                candidates.append(sharp)
            except Exception:
                pass
            return candidates
        def _tight_crop_pdf417(roi: np.ndarray):
            """Tighten ROI around barcode-like high vertical edge density."""
            try:
                if len(roi.shape) == 3:
                    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                else:
                    gray = roi
                grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                grad_x = cv2.convertScaleAbs(grad_x)
                blur = cv2.GaussianBlur(grad_x, (5, 5), 0)
                _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return roi
                h, w = roi.shape[:2]
                best = None
                best_score = 0.0
                for c in contours:
                    x, y, cw, ch = cv2.boundingRect(c)
                    if cw < 20 or ch < 8:
                        continue
                    aspect = float(cw) / float(max(1, ch))
                    area = float(cw * ch)
                    score = aspect * area
                    if aspect >= 2.0 and score > best_score:
                        best = (x, y, cw, ch)
                        best_score = score
                if best is None:
                    return roi
                x, y, cw, ch = best
                # Keep wider quiet-zone margins to avoid clipping PDF417 guards.
                pad_x = int(0.18 * cw)
                pad_y = int(0.35 * ch)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + cw + pad_x)
                y2 = min(h, y + ch + pad_y)
                tight = roi[y1:y2, x1:x2]
                return tight if tight.size > 0 else roi
            except Exception:
                return roi
        def _upscale_roi(roi: np.ndarray, min_width: int = 1200, min_height: int = 260):
            h, w = roi.shape[:2]
            if w >= min_width and h >= min_height:
                return roi
            scale_w = min_width / float(max(1, w))
            scale_h = min_height / float(max(1, h))
            scale = max(scale_w, scale_h)
            nw = int(round(w * scale))
            nh = int(round(h * scale))
            return cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_CUBIC)
        def _estimate_pdf417_band(base: np.ndarray):
            """Estimate horizontal band containing PDF417 using vertical edge energy."""
            try:
                if len(base.shape) == 3:
                    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
                else:
                    gray = base
                grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                energy = np.mean(np.abs(grad_x), axis=1)
                if energy.size < 20:
                    return None
                smooth = cv2.GaussianBlur(energy.reshape(-1, 1), (1, 21), 0).reshape(-1)
                peak = int(np.argmax(smooth))
                h = gray.shape[0]
                half = max(26, int(0.12 * h))
                y1 = max(0, peak - half)
                y2 = min(h, peak + half)
                if y2 - y1 < 28:
                    return None
                return (y1, y2)
            except Exception:
                return None
        def _detected_pdf417_regions(base: np.ndarray):
            regions = []
            try:
                if len(base.shape) == 3:
                    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
                else:
                    gray = base
                # Emphasize vertical transitions common in PDF417 bars.
                grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                grad_x = cv2.convertScaleAbs(grad_x)
                blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
                _, thresh = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                closed = cv2.erode(closed, None, iterations=1)
                closed = cv2.dilate(closed, None, iterations=1)
                contours, _ = cv2.findContours(
                    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    return regions
                h, w = base.shape[:2]
                min_area = 0.008 * float(h * w)
                for c in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
                    area = cv2.contourArea(c)
                    if area < min_area:
                        continue
                    rect = cv2.minAreaRect(c)
                    (cx, cy), (rw, rh), angle = rect
                    if rw <= 1 or rh <= 1:
                        continue
                    long_side = max(rw, rh)
                    short_side = min(rw, rh)
                    aspect = long_side / short_side if short_side > 0 else 0
                    if aspect < 2.0:
                        continue
                    # Normalize so extracted ROI is horizontal.
                    rot_angle = angle
                    crop_w, crop_h = int(round(rw)), int(round(rh))
                    if rw < rh:
                        rot_angle = angle + 90.0
                        crop_w, crop_h = int(round(rh)), int(round(rw))
                    rot_m = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
                    rotated = cv2.warpAffine(
                        base,
                        rot_m,
                        (base.shape[1], base.shape[0]),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    crop_w = max(20, crop_w)
                    crop_h = max(12, crop_h)
                    roi = cv2.getRectSubPix(rotated, (crop_w, crop_h), (cx, cy))
                    if roi is not None and roi.size > 0:
                        regions.append(roi)
            except Exception:
                return regions
            return regions
        def _region_images(base: np.ndarray, mode: str = "fast"):
            # Priority: detected/deskewed PDF417 candidate regions first.
            regions = []
            for r in _detected_pdf417_regions(base):
                tight = _tight_crop_pdf417(r)
                regions.append(_upscale_roi(tight))
            band = _estimate_pdf417_band(base)
            if band is not None:
                by1, by2 = band
                h, w = base.shape[:2]
                band_roi = base[max(0, by1 - int(0.06 * h)):min(h, by2 + int(0.06 * h)),
                                int(0.03 * w):int(0.98 * w)]
                if band_roi.size > 0:
                    regions.append(_upscale_roi(band_roi, min_width=1400, min_height=320))
            regions.append(base)
            h, w = base.shape[:2]
            if h < 240 or w < 320:
                return regions
            is_closeup = h <= int(0.78 * w)
            if mode == "fast":
                if is_closeup:
                    regions.append(base[int(0.45 * h):int(0.98 * h), int(0.04 * w):int(0.98 * w)])
                    regions.append(base[int(0.35 * h):int(0.90 * h), int(0.08 * w):int(0.96 * w)])
                    return regions[:4]
                regions.append(base[int(0.20 * h):int(0.85 * h), int(0.10 * w):int(0.90 * w)])
                regions.append(base[int(0.35 * h):int(0.70 * h), int(0.12 * w):int(0.92 * w)])
                regions.append(base[int(0.55 * h):int(0.92 * h), int(0.10 * w):int(0.95 * w)])
                return regions[:4]
            # Overlapping crops to recover barcodes occupying only part of the frame.
            if is_closeup:
                regions.append(base[int(0.40 * h):int(0.98 * h), int(0.04 * w):int(0.98 * w)])
                regions.append(base[int(0.30 * h):int(0.94 * h), int(0.08 * w):int(0.96 * w)])
                regions.append(base[int(0.48 * h):int(0.92 * h), int(0.12 * w):int(0.94 * w)])
                return regions[:5]
            y1, y2 = int(0.20 * h), int(0.85 * h)
            x1, x2 = int(0.10 * w), int(0.90 * w)
            regions.append(base[y1:y2, x1:x2])  # center crop
            regions.append(base[int(0.15 * h):int(0.45 * h), int(0.12 * w):int(0.92 * w)])
            regions.append(base[int(0.35 * h):int(0.70 * h), int(0.12 * w):int(0.92 * w)])
            regions.append(base[int(0.55 * h):int(0.92 * h), int(0.10 * w):int(0.95 * w)])
            return regions[:5]
        def _try_candidates(img: np.ndarray, label: str, mode: str = "fast", deadline: float = None):
            nonlocal budget_warned
            angle_candidates = [0, 180] if mode == "fast" else [0, 180, 90, 270]
            for ridx, region in enumerate(_region_images(img, mode=mode)):
                for idx, cand in enumerate(_candidate_images(region, mode=mode)):
                    if deadline is not None and time.perf_counter() > deadline:
                        if not budget_warned:
                            logger.info("Decode time budget reached; stopping further attempts")
                            budget_warned = True
                        return []
                    for angle in angle_candidates:
                        if deadline is not None and time.perf_counter() > deadline:
                            if not budget_warned:
                                logger.info("Decode time budget reached during angle attempts")
                                budget_warned = True
                            return []
                        rotated = cand if angle == 0 else np.rot90(cand, k=angle // 90)
                        # Ensure memory layout is contiguous for native decoder bindings.
                        rotated = np.ascontiguousarray(rotated)
                        results = _read(rotated, mode=mode)
                        decoded = _normalize_results(results)
                        if decoded:
                            logger.info(
                                f"Barcode decoded with primary decoder "
                                f"({label} region {ridx + 1}, variant {idx + 1}, angle {angle})"
                            )
                            return decoded
            return []
        # Stage 1: quick attempts.
        fast_deadline = time.perf_counter() + fast_budget_sec
        for label, frame in [("original", image)]:
            decoded = _try_candidates(frame, label, mode="fast", deadline=fast_deadline)
            if decoded:
                return decoded
        # Stage 2 (recovery): deeper pass with separate time budget.
        full_deadline = time.perf_counter() + full_budget_sec
        logger.debug("Fast pass failed; trying recovery pass on 90, 180, 270 rotations")
        for angle in [90, 180, 270]:
            rotated = np.rot90(image, k=angle // 90)
            decoded = _try_candidates(
                rotated, f"rotated {angle}", mode="full", deadline=full_deadline
            )
            if decoded:
                logger.info(f"Barcode decoded with primary decoder at {angle} rotation")
                return decoded
        logger.debug("No barcode found after rotation attempts")
        return []

    @staticmethod
    def _ensure_min_size(image: np.ndarray, min_width: int = 1200):
        """Upscale small images to improve PDF417 detection reliability."""
        height, width = image.shape[:2]
        if width >= min_width:
            return image
        scale = min_width / float(width)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        logger.info(f"Upscaling image from {width}x{height} to {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _ensure_min_height(image: np.ndarray, min_height: int = 900):
        """Upscale low-height images where PDF417 rows are too thin."""
        height, width = image.shape[:2]
        if height >= min_height:
            return image
        scale = min_height / float(max(1, height))
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        logger.info(f"Upscaling image height from {width}x{height} to {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _ensure_max_size(image: np.ndarray, max_width: int = 2400):
        """Downscale very large images to keep decode latency bounded."""
        height, width = image.shape[:2]
        if width <= max_width:
            return image
        scale = max_width / float(width)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        logger.info(f"Downscaling image from {width}x{height} to {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def validate_image(image_data: bytes) -> tuple:
        """Validate if the provided data is a valid image."""
        try:
            image = Image.open(BytesIO(image_data))
            image.verify()
            return True, "Valid image"
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

    @staticmethod
    def extract_barcodes(image_bytes: bytes):
        """Extract barcodes from image bytes"""
        try:
            if not ZXING_AVAILABLE:
                return {
                    "success": False,
                    "message": "Barcode decoder not available. Please contact support.",
                    "count": 0,
                    "barcodes": []
                }
            logger.info("Starting barcode extraction...")
            
            # Open and convert image
            image = Image.open(BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb_image = np.array(image)
            height, width = rgb_image.shape[:2]
            logger.info(f"Image size: {width}x{height}")
            # Downscale very large images first for speed.
            rgb_image = BarcodeReader._ensure_max_size(rgb_image)
            # Upscale tiny images more aggressively for low-res PDF417.
            target_min_width = 1600 if rgb_image.shape[1] <= 700 else 1200
            rgb_image = BarcodeReader._ensure_min_size(rgb_image, min_width=target_min_width)
            target_min_height = 1000 if rgb_image.shape[0] <= 700 else 850
            rgb_image = BarcodeReader._ensure_min_height(rgb_image, min_height=target_min_height)
            # Try decode with rotation handling
            decoded_objects = BarcodeReader._try_decode_zxing(rgb_image)
            decode_method = "universal_scanner"
            if not decoded_objects:
                logger.warning("No barcode detected")
                return {
                    "success": False,
                    "message": "Barcode not found",
                    "count": 0
                }
            # Process decoded barcodes
            barcodes = []
            for idx, obj in enumerate(decoded_objects):
                raw = obj["data"]
                btype = obj.get("type") or "UNKNOWN"
                normalized_raw = BarcodeReader._normalize_aamva_raw(raw)
                records = BarcodeReader._extract_aamva_records(normalized_raw)
                parsed = BarcodeReader.parse_aamva(normalized_raw)
                structured = BarcodeReader._build_frontend_payload(records, normalized_raw, raw)
                logger.info(f"Barcode {idx + 1}: type={btype}, decoded={len(raw)} chars")
                confidence = "96%"
                if structured:
                    barcodes.append({
                        "id": idx + 1,
                        "type": btype,
                        "method": decode_method,
                        "raw": structured.get("raw"),
                        "raw_formatted": structured.get("raw_formatted"),
                        "person": structured.get("person"),
                        "document": structured.get("document"),
                        "address": structured.get("address"),
                        "meta": {
                            "confidence": confidence
                        }
                    })
                else:
                    barcodes.append({
                        "id": idx + 1,
                        "type": btype,
                        "method": decode_method,
                        "raw": raw,
                        "raw_formatted": normalized_raw,
                        "person": None,
                        "document": None,
                        "address": None,
                        "meta": {
                            "confidence": confidence
                        }
                    })
            logger.info(f"Successfully extracted {len(barcodes)} barcode(s)")
            return {
                "success": True,
                "message": f"Detected {len(barcodes)} barcode(s)",
                "count": len(barcodes),
                "barcodes": barcodes
            }
        except Exception as e:
            logger.exception("Barcode extraction failed")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "count": 0,
                "barcodes": []
            }
# Test runner
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "barcode.png"
    
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        result = BarcodeReader.extract_barcodes(data)
        print(result)
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found")
