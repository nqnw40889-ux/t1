"""Read the fixed AI4I PNG form from visible glyphs, without a dataset lookup."""
from functools import lru_cache
import hashlib
import io
import json
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

from inference import DATA, FEATURES

SIZE = (800, 460)
ROWS = (130, 181, 232, 283, 334, 385)
LAYOUT_BOX = (40, 118, 400, 420)
UDI_BOX = (610, 63, 766, 105)
MAX_BYTES = 4 * 1024 * 1024


def ink(image, box):
    return np.asarray(image.crop(box).convert("L")) < 128


def glyphs(mask):
    occupied = mask.any(axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, occupied, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        glyph = mask[:, start:end]
        rows = np.flatnonzero(glyph.any(axis=1))
        trimmed = glyph[rows[0]:rows[-1] + 1]
        height, width = trimmed.shape
        yield f"{height}x{width}:{np.packbits(trimmed).tobytes().hex()}"


def layout_key(image):
    return hashlib.sha256(np.packbits(ink(image, LAYOUT_BOX)).tobytes()).hexdigest()


@lru_cache(maxsize=1)
def templates():
    return json.loads((DATA / "card_ocr_templates.json").read_text(encoding="utf-8"))


def read_card(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Chọn tệp PNG của bộ phiếu AI4I, tối đa 4 MB.")
    try:
        with Image.open(io.BytesIO(payload)) as raw:
            if raw.format != "PNG" or raw.size != SIZE:
                raise ValueError("Chọn ảnh PNG mẫu AI4I đúng kích thước 800 × 460.")
            image = raw.convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Không đọc được ảnh PNG.") from error
    reference = templates()
    if layout_key(image) != reference["layout"]:
        raise ValueError("Bố cục phiếu không khớp mẫu AI4I. Hãy dùng ảnh PNG gốc trong bộ phiếu.")

    def read_line(box, alphabet):
        return "".join(reference[alphabet].get(pattern, "?") for pattern in glyphs(ink(image, box)))

    header = read_line(UDI_BOX, "header")
    if not re.fullmatch(r"UDI[0-9]{5}", header) or not 1 <= int(header[3:]) <= 10_000:
        raise ValueError("Không đọc được mã UDI trên phiếu.")
    values = {}
    for index, (feature, y) in enumerate(zip(FEATURES, ROWS)):
        value = read_line((590, y - 6, 760, y + 35), "values")
        pattern = r"[LMH]" if index == 0 else (r"[0-9]+" if index in (3, 5) else r"[0-9]+(?:\.[0-9])?")
        if not re.fullmatch(pattern, value):
            raise ValueError(f"Không đọc được trường {feature} trên phiếu.")
        values[feature] = value
    return int(header[3:]), values


def read_photo(payload):
    """Fallback for photos or screenshots of the same form. Always review the output."""
    import os
    import shutil
    import subprocess

    from PIL import ImageOps

    if not payload or len(payload) > 8 * 1024 * 1024:
        raise ValueError("Ảnh phiếu đo tối đa 8 MB.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {"PNG", "JPEG"} or source.width * source.height > 16_000_000:
                raise ValueError("Dùng ảnh PNG hoặc JPG có tối đa 16 megapixel.")
            image = ImageOps.exif_transpose(source).convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Không đọc được ảnh phiếu đo.") from error
    executable = shutil.which("tesseract")
    if not executable:
        raise ValueError("Bộ đọc ảnh chụp cần Tesseract. Bạn có thể dùng PNG gốc trong bộ phiếu.")
    image.thumbnail((1600, 1600))
    image = ImageOps.autocontrast(image)
    data = io.BytesIO()
    image.save(data, "PNG")
    try:
        output = subprocess.run([executable, "stdin", "stdout", "-l", "eng", "--psm", "6"],
                                input=data.getvalue(), capture_output=True, check=True, timeout=12,
                                env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    except (subprocess.SubprocessError, OSError) as error:
        raise ValueError("OCR chưa đọc được ảnh. Hãy dùng ảnh thẳng, rõ chữ hoặc tải PNG gốc.") from error
    words = output.stdout.decode("utf-8", errors="replace")
    udi = re.search(r"\bUDI\s*([0-9]{1,5})\b", words, re.IGNORECASE)
    values = {name: "" for name in FEATURES}
    type_match = re.search(r"^\s*Type\s*[:=]?\s*([LMH])\b", words, re.MULTILINE | re.IGNORECASE)
    if type_match:
        values["Type"] = type_match[1].upper()
    for name in FEATURES[1:]:
        label = name.split(" [")[0]
        for line in words.splitlines():
            if line.strip().casefold().startswith(label.casefold()):
                match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*$", line)
                if match:
                    values[name] = match[1].replace(",", ".")
                break
    if not any(values.values()):
        raise ValueError("OCR chưa tìm thấy thông số. Hãy chụp đúng mẫu phiếu và lấy rõ cả tên trường.")
    return int(udi[1]) if udi else None, values
