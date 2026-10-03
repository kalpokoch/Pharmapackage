"""Shared python-docx scaffolding for the manuscript builders.

`make_builder()` returns a fresh IEEE-styled Document together with the paragraph, caption and
table helpers bound to it, so `build_docx.py` and `build_manuscript.py` cannot drift apart.

Numbering is handled by `Numbering`, which hands out table (roman) and figure (arabic) labels in
order, and citations by `Citations`, which assigns reference numbers on first use and emits the
list in citation order -- IEEE style. Nothing is hard-coded, so inserting a table or figure
renumbers everything after it automatically.
"""
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

NEEDS = RGBColor(0xC0, 0x00, 0x00)
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]


class Numbering:
    """Monotonic table/figure labels. `table_label()` -> 'I', 'II', ...; `fig_label()` -> '1', '2', ..."""

    def __init__(self):
        self.n_tables = 0
        self.n_figures = 0

    def table_label(self):
        self.n_tables += 1
        return ROMAN[self.n_tables]

    def fig_label(self):
        self.n_figures += 1
        return str(self.n_figures)

    def peek_tables(self, k):
        """The labels the next k tables will receive, without consuming them."""
        return [ROMAN[self.n_tables + i] for i in range(1, k + 1)]

    def peek_figures(self, k):
        return [str(self.n_figures + i) for i in range(1, k + 1)]


class Citations:
    """IEEE numeric citations: numbers assigned on first use, list emitted in that order."""

    def __init__(self, database):
        self.db = database
        self.order = []

    def cite(self, *keys):
        nums = []
        for k in keys:
            if k not in self.db:
                raise KeyError(f"unknown reference key: {k!r}")
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        nums.sort()
        if len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1)):
            return f"[{nums[0]}]-[{nums[-1]}]"
        return ", ".join(f"[{n}]" for n in nums)

    def entries(self):
        return [(i + 1, self.db[k]) for i, k in enumerate(self.order)]


def make_builder():
    """A fresh styled Document plus helpers bound to it: (doc, helpers_dict)."""
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Inches(1))
    sec.top_margin = sec.bottom_margin = Inches(1)

    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(10)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.line_spacing = 1.1
    for name, size in (("Heading 1", 12), ("Heading 2", 10.5)):
        h = doc.styles[name]
        h.font.name = "Times New Roman"
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        h.font.size = Pt(size)
        h.font.bold = name == "Heading 1"
        h.font.italic = name == "Heading 2"
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)

    def para(parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, italic=False):
        """parts: str or list of str / (str, 'needs'|'b'|'i'|'sub'|'sup')."""
        p = doc.add_paragraph()
        p.alignment = align
        for part in ([parts] if isinstance(parts, str) else parts):
            txt, kind = (part, None) if isinstance(part, str) else part
            r = p.add_run(txt)
            r.italic = italic or kind == "i"
            r.bold = kind == "b"
            if kind == "needs":
                r.font.color.rgb = NEEDS
                r.bold = True
            if kind == "sub":
                r.font.subscript = True
            if size:
                r.font.size = Pt(size)
        return p


    def caption(label, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        r = p.add_run(label)
        r.font.size = Pt(8)
        r.font.small_caps = True
        r2 = p.add_run(text)
        r2.font.size = Pt(8)
        return p


    def fig_caption(label, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(label)
        r.font.size = Pt(8)
        r2 = p.add_run(text)
        r2.font.size = Pt(8)


    def set_cell_border(cell, **kw):
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tcPr.append(borders)
        for edge, val in kw.items():
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), "6")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            borders.append(el)


    def table(header, rows, widths, notes=None, bold_cells=None, group_rows=None):
        """IEEE-style three-rule table. rows: list of lists of str. bold_cells: set of (r, c)."""
        t = doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        bold_cells = bold_cells or set()
        group_rows = group_rows or set()
        for ri, row in enumerate([header] + rows):
            for ci, val in enumerate(row):
                c = t.cell(ri, ci)
                c.width = Inches(widths[ci])
                c.text = ""
                p = c.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                r = p.add_run(val)
                r.font.size = Pt(7.5)
                if ri == 0 or (ri - 1, ci) in bold_cells:
                    r.bold = True
                if "PENDING" in val or "NEEDS" in val:
                    r.font.color.rgb = NEEDS
                if (ri - 1) in group_rows and ri > 0:
                    r.italic = True
                if ri == 0:
                    set_cell_border(c, top="single", bottom="single")
                if ri == len(rows):
                    set_cell_border(c, bottom="single")
        for ci, w in enumerate(widths):
            for cell in t.columns[ci].cells:
                cell.width = Inches(w)
        if notes:
            for n in notes:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                r = p.add_run(n)
                r.font.size = Pt(7)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t


    def f(v, d=3):
        return f"{v:.{d}f}"

    return doc, {"doc": doc, "para": para, "caption": caption, "fig_caption": fig_caption,
                 "table": table, "set_cell_border": set_cell_border, "f": f,
                 "Inches": Inches, "WD_ALIGN_PARAGRAPH": WD_ALIGN_PARAGRAPH, "NEEDS": NEEDS}
