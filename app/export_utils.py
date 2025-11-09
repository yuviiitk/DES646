## app/export_utils.py

import io
import math
from typing import List, Optional

from PIL import Image, ImageOps
from fpdf import FPDF


def make_grid_image(paths: List[str], columns: int = 3, pad: int = 8, bg=(245, 245, 245)) -> Image.Image:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(i.width for i in imgs), max(i.height for i in imgs)
    rows = math.ceil(len(imgs) / columns)
    grid = Image.new("RGB", (columns * w + (columns + 1) * pad, rows * h + (rows + 1) * pad), color=bg)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, columns)
        x, y = pad + c * (w + pad), pad + r * (h + pad)
        grid.paste(ImageOps.contain(im, (w, h)), (x, y))
    return grid


def make_pdf_bytes(paths: List[str], captions: Optional[List[str]] = None, columns: int = 3) -> bytes:
    pdf = FPDF(unit="pt", format="A4")
    page_w, page_h = 595, 842  # approx points
    margin = 36
    cell_w = (page_w - margin * 2) / columns
    cell_h = cell_w

    pdf.set_auto_page_break(auto=True, margin=margin)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for i, p in enumerate(paths):
        if i % columns == 0 and i != 0:
            pdf.ln(12)
        x = margin + (i % columns) * cell_w
        y = pdf.get_y()
        pdf.image(p, x=x, y=y, w=cell_w, h=cell_h)
        if captions:
            pdf.set_xy(x, y + cell_h + 4)
            pdf.cell(cell_w, 14, txt=str(captions[i]), ln=0, align="C")
        # move cursor appropriately
        if (i % columns) == (columns - 1):
            pdf.ln(cell_h + 24)
        else:
            pdf.set_xy(x + cell_w + 0, y)

    return pdf.output(dest="S").encode("latin-1")
