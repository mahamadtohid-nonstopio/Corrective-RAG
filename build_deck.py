"""Generate the lightning-talk deck for 'Retrieval Isn't Enough: A Field Guide to Corrective RAG'.

Reads NonStop-Template.pptx to inherit the master's design (dark background,
teal accent, Arial), clears the existing sample slides, then builds 10 new
slides suited to an engineering-audience lightning talk.

Run:
    ragenvironment/Scripts/python.exe build_deck.py
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / "NonStop-Template.pptx"
OUTPUT = BASE / "RAG-CRAG-lightning-talk.pptx"

# Theme colors pulled from the template's master:
INK          = RGBColor(0xFF, 0xFF, 0xFF)  # primary text on dark bg
DIM          = RGBColor(0xAD, 0xAD, 0xAD)  # secondary text
TEAL         = RGBColor(0x00, 0x96, 0x88)  # accent1 — primary brand
ORANGE       = RGBColor(0xFF, 0xAB, 0x40)  # accent4 — highlight
CYAN         = RGBColor(0x4D, 0xD0, 0xE1)  # accent5 — secondary highlight
YELLOW       = RGBColor(0xEE, 0xFF, 0x41)  # accent6 — punch color
CARD_BG      = RGBColor(0x2A, 0x2A, 0x2A)  # slightly lighter than page bg
CARD_EDGE    = RGBColor(0x40, 0x40, 0x40)
DIAGRAM_BOX  = RGBColor(0x30, 0x30, 0x30)
GREEN_OK     = RGBColor(0x66, 0xBB, 0x6A)
RED_BAD      = RGBColor(0xEF, 0x53, 0x50)

TITLE_FONT = "Arial"
BODY_FONT  = "Arial"
MONO_FONT  = "Consolas"


# ─── helpers ──────────────────────────────────────────────────────────────

def clear_slides(prs):
    """Delete every existing slide so we start with a blank deck that still
    inherits the master's design."""
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        rId = sldId.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        prs.part.drop_rel(rId)
        sldIdLst.remove(sldId)


def set_run(run, text, *, size=18, bold=False, italic=False, color=INK, font=BODY_FONT):
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_paragraph(tf, text, *, size=18, bold=False, italic=False, color=INK,
                  font=BODY_FONT, align=PP_ALIGN.LEFT, space_before=0, space_after=6,
                  bullet=False, level=0, first=False):
    if first and tf.paragraphs and not tf.paragraphs[0].text:
        p = tf.paragraphs[0]
        p.text = ""
    else:
        p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    p.level = level
    if bullet:
        text = "•  " + text
    run = p.add_run()
    set_run(run, text, size=size, bold=bold, italic=italic, color=color, font=font)
    return p


def add_text_box(slide, left, top, width, height, *, fill=None, edge=None):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    if fill is None:
        box.fill.background()
    else:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    if edge is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = edge
        box.line.width = Pt(0.75)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return box, tf


def add_flow_box(slide, left, top, width, height, text, *, fill=DIAGRAM_BOX,
                 edge=TEAL, text_color=INK, size=13, bold=False):
    box, tf = add_text_box(slide, left, top, width, height, fill=fill, edge=edge)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = tf.paragraphs[0].add_run()
    set_run(run, text, size=size, bold=bold, color=text_color, font=BODY_FONT)
    return box


def add_arrow(slide, x1, y1, x2, y2, color=TEAL, weight=1.5):
    line = slide.shapes.add_connector(1, x1, y1, x2, y2)  # STRAIGHT
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    # add arrowhead
    lnEl = line.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tailEnd = lnEl.find(qn("a:tailEnd"))
    if tailEnd is None:
        from lxml import etree
        tailEnd = etree.SubElement(lnEl, qn("a:tailEnd"))
    tailEnd.set("type", "triangle")
    tailEnd.set("w", "med")
    tailEnd.set("h", "med")
    return line


def add_slide_number_footer(slide, index, total, *, tag=""):
    left = Inches(0.35)
    top = Inches(5.25)
    box, tf = add_text_box(slide, left, top, Inches(9.3), Inches(0.3))
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = tf.paragraphs[0].add_run()
    set_run(run, f"{index:02d} / {total:02d}   ·   Retrieval Isn't Enough" +
            (f"   ·   {tag}" if tag else ""),
            size=9, color=DIM, font=BODY_FONT)


# ─── slide constructors ───────────────────────────────────────────────────

def slide_title(prs, total):
    layout = prs.slide_layouts[10]  # BLANK
    slide = prs.slides.add_slide(layout)

    # Tag ribbon
    _, tag_tf = add_text_box(slide, Inches(0.5), Inches(0.7), Inches(4), Inches(0.3))
    run = tag_tf.paragraphs[0].add_run()
    set_run(run, "LIGHTNING TALK  ·  RAG SERIES", size=10, color=TEAL, bold=True)

    # Title
    _, t_tf = add_text_box(slide, Inches(0.5), Inches(1.4), Inches(9), Inches(1.8))
    p1 = t_tf.paragraphs[0]
    p1.alignment = PP_ALIGN.LEFT
    r1 = p1.add_run()
    set_run(r1, "Retrieval Isn't Enough", size=52, bold=True, color=INK, font=TITLE_FONT)

    # Subtitle
    _, s_tf = add_text_box(slide, Inches(0.5), Inches(2.8), Inches(9), Inches(0.8))
    p2 = s_tf.paragraphs[0]
    p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run()
    set_run(r2, "A Field Guide to Corrective RAG", size=26, italic=True, color=DIM, font=TITLE_FONT)

    # Author + context
    _, a_tf = add_text_box(slide, Inches(0.5), Inches(4.5), Inches(9), Inches(0.6))
    p3 = a_tf.paragraphs[0]
    r3 = p3.add_run()
    set_run(r3, "Mahamadtohid Naikwadi  ·  NonStop io", size=13, color=INK, font=BODY_FONT)
    p4 = a_tf.add_paragraph()
    r4 = p4.add_run()
    set_run(r4, "Healthcare Knowledge Navigator — RAG + CRAG on Gemini/NVIDIA + Pinecone",
            size=11, color=DIM, italic=True, font=BODY_FONT)

    # Accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(3.7),
                                 Inches(0.6), Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    add_slide_number_footer(slide, 1, total, tag="Title")


def slide_problem(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    # Section eyebrow
    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "THE PROBLEM", size=10, color=TEAL, bold=True)

    # Big statement
    _, t_tf = add_text_box(slide, Inches(0.5), Inches(0.75), Inches(9), Inches(1.2))
    r = t_tf.paragraphs[0].add_run()
    set_run(r, "RAG fails silently — and confidently.", size=32, bold=True, color=INK, font=TITLE_FONT)

    # Two columns: symptom + example
    # Left: symptoms
    add_paragraph_column(
        slide, Inches(0.5), Inches(2.05), Inches(4.3), Inches(2.7),
        header="What goes wrong",
        items=[
            "Retrieval returns off-topic chunks — LLM still answers.",
            "No mechanism to grade whether context actually helps.",
            "Confidence is guessed after the fact, or ignored.",
            "Corpus gaps are indistinguishable from lazy queries.",
        ],
    )

    # Right: real example
    add_example_card(
        slide, Inches(5.1), Inches(2.05), Inches(4.4), Inches(2.7),
        title="Real example — this build",
        question="Who is the captain of the Indian cricket team?",
        retrieved="asthma guidelines · anticoagulation · sepsis protocols",
        answer="Shubman Gill is the current Test/ODI captain ... [2] [4]",
        verdict="MODERATE  · 0.60",
        verdict_color=ORANGE,
    )

    add_slide_number_footer(slide, 2, total, tag="Problem")


def add_paragraph_column(slide, left, top, width, height, *, header, items):
    _, tf = add_text_box(slide, left, top, width, height)
    r = tf.paragraphs[0].add_run()
    set_run(r, header, size=11, bold=True, color=TEAL, font=BODY_FONT)
    tf.paragraphs[0].space_after = Pt(10)
    for item in items:
        add_paragraph(tf, item, size=14, color=INK, bullet=True, space_after=8)


def add_example_card(slide, left, top, width, height, *,
                     title, question, retrieved, answer, verdict, verdict_color):
    box, tf = add_text_box(slide, left, top, width, height,
                           fill=CARD_BG, edge=CARD_EDGE)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.2); tf.margin_right = Inches(0.2)
    tf.margin_top = Inches(0.15); tf.margin_bottom = Inches(0.15)

    r = tf.paragraphs[0].add_run()
    set_run(r, title, size=10, bold=True, color=TEAL, font=BODY_FONT)
    tf.paragraphs[0].space_after = Pt(8)

    for label, body, body_color, body_italic, body_font in [
        ("QUERY",     question,  INK,       True,  BODY_FONT),
        ("RETRIEVED", retrieved, DIM,       False, MONO_FONT),
        ("CRAG SAID", answer,    INK,       True,  BODY_FONT),
    ]:
        p = tf.add_paragraph()
        r = p.add_run(); set_run(r, label, size=9, bold=True, color=DIM, font=BODY_FONT)
        p.space_before = Pt(6); p.space_after = Pt(2)
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); set_run(r2, body, size=12, italic=body_italic,
                                    color=body_color, font=body_font)
        p2.space_after = Pt(0)

    p = tf.add_paragraph()
    p.space_before = Pt(10)
    r = p.add_run()
    set_run(r, f"→ Verdict: {verdict}", size=13, bold=True, color=verdict_color, font=BODY_FONT)


def slide_rag_arch(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "STANDARD RAG", size=10, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(0.75), Inches(9), Inches(0.9))
    r = t.paragraphs[0].add_run()
    set_run(r, "One pass. No feedback loop.", size=28, bold=True, color=INK)

    # Flow: [Question] → [Embed] → [Retrieve k] → [Context] → [LLM] → [Answer]
    boxes = [
        ("Question",       0.65),
        ("Embed",          2.10),
        ("Retrieve  top-k",3.60),
        ("Context",        5.55),
        ("LLM (generate)", 7.00),
        ("Answer + cites", 8.55),
    ]
    # Actually 6 boxes across ~9 inches: keep it to 5 in a single row for readability
    boxes = [
        ("Question",             0.6),
        ("Embed + Retrieve",     2.35),
        ("Context k",            4.35),
        ("LLM  (generate)",      5.95),
        ("Answer  +  cites",     7.85),
    ]
    y = Inches(2.15)
    h = Inches(0.85)
    w = Inches(1.55)
    xs = []
    for label, left_in in boxes:
        left = Inches(left_in)
        add_flow_box(slide, left, y, w, h, label, size=12, bold=True)
        xs.append((left, left + w))
    # arrows between consecutive boxes
    for (a_l, a_r), (b_l, _b_r) in zip(xs, xs[1:]):
        add_arrow(slide, a_r, y + h / 2, b_l, y + h / 2)

    # Callouts under the flow
    _, notes = add_text_box(slide, Inches(0.5), Inches(3.35), Inches(9), Inches(1.5))
    add_paragraph(notes, "What's missing", size=11, bold=True, color=TEAL, first=True)
    add_paragraph(notes, "No grader — every retrieved chunk is treated as gospel.",
                  size=13, bullet=True, color=INK)
    add_paragraph(notes, "No fallback — corpus miss = LLM invents or refuses awkwardly.",
                  size=13, bullet=True, color=INK)
    add_paragraph(notes, "No topic gate — cricket, coffee, C++ all get \"answered.\"",
                  size=13, bullet=True, color=INK)

    add_slide_number_footer(slide, 3, total, tag="RAG Architecture")


def slide_where_rag_breaks(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "FAILURE MODES", size=10, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(0.75), Inches(9), Inches(0.9))
    r = t.paragraphs[0].add_run()
    set_run(r, "Three ways plain RAG lies to you.", size=28, bold=True, color=INK)

    modes = [
        ("Confident hallucination",
         "Retrieval hits low-relevance chunks; the LLM stitches them together anyway.",
         "The confidence chip says HIGH. The answer is fabricated."),
        ("Silent off-topic drift",
         "Questions outside the corpus scope get answered from partial matches.",
         "Cricket + a medical corpus = a fluent cricket answer."),
        ("Underused corpus",
         "Weak queries retrieve weak context; strong reformulations never happen.",
         "\"Metformin?\" retrieves less than \"first-line for T2DM with CKD stage 3.\""),
    ]
    y = 1.85
    for title, why, ex in modes:
        box, tf = add_text_box(slide, Inches(0.5), Inches(y), Inches(9.0), Inches(0.95),
                               fill=CARD_BG, edge=CARD_EDGE)
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = Inches(0.2); tf.margin_top = Inches(0.1)
        r = tf.paragraphs[0].add_run()
        set_run(r, title, size=15, bold=True, color=ORANGE, font=BODY_FONT)
        tf.paragraphs[0].space_after = Pt(4)
        p2 = tf.add_paragraph(); r2 = p2.add_run()
        set_run(r2, why, size=12, color=INK)
        p3 = tf.add_paragraph(); r3 = p3.add_run()
        set_run(r3, ex, size=11, italic=True, color=DIM)
        y += 1.05

    add_slide_number_footer(slide, 4, total, tag="Failure Modes")


def slide_enter_crag(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(1.3), Inches(9), Inches(0.4))
    r = eb.paragraphs[0].add_run()
    set_run(r, "THE FIX", size=12, color=TEAL, bold=True)

    # Big thesis
    _, t = add_text_box(slide, Inches(0.5), Inches(1.8), Inches(9), Inches(2.0))
    p = t.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    set_run(r, "Retrieve → ", size=52, color=DIM, font=TITLE_FONT, bold=True)
    r2 = p.add_run()
    set_run(r2, "Grade → Correct → ", size=52, color=TEAL, font=TITLE_FONT, bold=True)
    r3 = p.add_run()
    set_run(r3, "Answer.", size=52, color=INK, font=TITLE_FONT, bold=True)

    _, s = add_text_box(slide, Inches(0.5), Inches(3.5), Inches(9), Inches(0.6))
    r = s.paragraphs[0].add_run()
    set_run(r, "Yan et al., 2024 — Corrective Retrieval Augmented Generation",
            size=14, italic=True, color=DIM)

    add_slide_number_footer(slide, 5, total, tag="Thesis")


def slide_crag_arch(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "CRAG ARCHITECTURE", size=10, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(0.75), Inches(9), Inches(0.6))
    r = t.paragraphs[0].add_run()
    set_run(r, "Grade first. Correct if needed. Then generate.",
            size=22, bold=True, color=INK)

    # Layout of shapes on a 10x5.62 canvas.
    # Row 1: Retrieve → Grade → Decide
    # Branches from Decide:
    #   generate (top)     → Generate → Answer
    #   supplement/web_only (bottom) → Rewrite → Web Search → Generate

    W  = Inches(1.65); H = Inches(0.62)
    y_top = Inches(1.75)
    y_gen = Inches(1.75)      # top branch keeps y
    y_bot = Inches(3.35)      # bottom branch

    # Row: retrieve, grade, decide
    x_ret = Inches(0.4)
    x_grd = Inches(2.25)
    x_dec = Inches(4.10)
    add_flow_box(slide, x_ret, y_top, W, H, "Retrieve  (score)")
    add_flow_box(slide, x_grd, y_top, W, H, "Grade each doc")
    add_flow_box(slide, x_dec, y_top, W, H, "Decide", fill=DIAGRAM_BOX, edge=ORANGE)
    add_arrow(slide, x_ret + W, y_top + H / 2, x_grd, y_top + H / 2)
    add_arrow(slide, x_grd + W, y_top + H / 2, x_dec, y_top + H / 2)

    # Top branch: generate → answer
    x_gen1 = Inches(6.10)
    x_ans  = Inches(7.95)
    add_flow_box(slide, x_gen1, y_gen, W, H, "Generate", edge=CYAN)
    add_flow_box(slide, x_ans,  y_gen, Inches(1.7), H, "Answer + cites",
                 edge=CYAN, fill=DIAGRAM_BOX, bold=True)
    add_arrow(slide, x_dec + W, y_top + H / 2, x_gen1, y_gen + H / 2, color=CYAN)
    add_arrow(slide, x_gen1 + W, y_gen + H / 2, x_ans, y_gen + H / 2, color=CYAN)

    # Label the top branch
    _, lbl = add_text_box(slide, x_gen1, y_gen - Inches(0.35), Inches(3.5), Inches(0.3))
    lbl.paragraphs[0].alignment = PP_ALIGN.CENTER
    r = lbl.paragraphs[0].add_run()
    set_run(r, "relevance ≥ threshold  →  generate", size=10, color=CYAN, italic=True)

    # Bottom branch: rewrite → web search → generate (merges into answer)
    x_rw = Inches(4.10); x_web = Inches(6.10); x_gen2 = Inches(7.95)  # gen2 not needed, we can merge
    add_flow_box(slide, x_rw, y_bot, W, H, "Rewrite query", edge=ORANGE)
    add_flow_box(slide, x_web, y_bot, W, H, "Web search  (Tavily)", edge=ORANGE)
    # Route from decide down to rewrite
    add_arrow(slide, x_dec + W / 2, y_top + H,
              x_rw + W / 2, y_bot, color=ORANGE)
    add_arrow(slide, x_rw + W, y_bot + H / 2, x_web, y_bot + H / 2, color=ORANGE)
    # Web search back up to Generate (merges with top branch)
    add_arrow(slide, x_web + W, y_bot + H / 2, x_gen1 + W / 2, y_gen + H, color=ORANGE)

    # Label the bottom branch
    _, lbl2 = add_text_box(slide, x_rw, y_bot + H + Inches(0.05), Inches(3.5), Inches(0.3))
    r = lbl2.paragraphs[0].add_run()
    set_run(r, "irrelevant  →  correct with web fallback", size=10, color=ORANGE, italic=True)

    # Legend / callouts
    _, notes = add_text_box(slide, Inches(0.5), Inches(4.5), Inches(9), Inches(0.6))
    add_paragraph(notes,
        "Grader is one batched LLM call for k passages.  "
        "Rewriter runs only when relevance is low.  "
        "Confidence = f(top retrieval score).",
        size=11, color=DIM, italic=True, first=True)

    add_slide_number_footer(slide, 6, total, tag="CRAG Architecture")


def slide_three_impls(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "IMPLEMENTATIONS", size=10, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(0.75), Inches(9), Inches(0.6))
    r = t.paragraphs[0].add_run()
    set_run(r, "Three ways to build CRAG in production.", size=22, bold=True, color=INK)

    impls = [
        {
            "name": "A · Paper-faithful",
            "sub":  "Yan et al., 2024",
            "flow": "retrieve → grade → rewrite → web → generate → LLM-audit confidence",
            "pros": ["Most rigorous", "Independent auditor"],
            "cons": ["4–5 LLM calls / query", "No streaming — blocking wait"],
            "color": DIM,
        },
        {
            "name": "B · Streamed + heuristic",
            "sub":  "what we built",
            "flow": "retrieve (with score) → topic gate → grade → maybe correct → stream generate",
            "pros": ["2–3 LLM calls", "TTFT < 1s (real streaming)", "Score-based topic gate is free"],
            "cons": ["Confidence is heuristic, not LLM-audited"],
            "color": TEAL,
        },
        {
            "name": "C · Domain-restricted CRAG",
            "sub":  "B + Tavily allowlist",
            "flow": "same as B, web search locked to PubMed / NIH / WHO / Cochrane",
            "pros": ["Safe web fallback for regulated corpora", "No extra LLM cost"],
            "cons": ["Fewer web hits", "Requires curated domain list per vertical"],
            "color": CYAN,
        },
    ]
    y = 1.60
    w = Inches(3.0)
    h = Inches(3.4)
    for i, impl in enumerate(impls):
        x = Inches(0.35 + i * 3.15)
        box, tf = add_text_box(slide, x, Inches(y), w, h, fill=CARD_BG, edge=impl["color"])
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = Inches(0.18); tf.margin_top = Inches(0.14)
        r = tf.paragraphs[0].add_run()
        set_run(r, impl["name"], size=15, bold=True, color=impl["color"])
        p = tf.add_paragraph(); r = p.add_run()
        set_run(r, impl["sub"], size=10, italic=True, color=DIM)
        p.space_after = Pt(8)

        p = tf.add_paragraph(); r = p.add_run()
        set_run(r, "FLOW", size=9, bold=True, color=DIM)
        p.space_before = Pt(4); p.space_after = Pt(2)
        p = tf.add_paragraph(); r = p.add_run()
        set_run(r, impl["flow"], size=10, color=INK, font=MONO_FONT)

        p = tf.add_paragraph(); r = p.add_run()
        set_run(r, "PROS", size=9, bold=True, color=GREEN_OK)
        p.space_before = Pt(10); p.space_after = Pt(2)
        for pro in impl["pros"]:
            p = tf.add_paragraph(); r = p.add_run()
            set_run(r, "+  " + pro, size=11, color=INK)

        p = tf.add_paragraph(); r = p.add_run()
        set_run(r, "CONS", size=9, bold=True, color=RED_BAD)
        p.space_before = Pt(8); p.space_after = Pt(2)
        for con in impl["cons"]:
            p = tf.add_paragraph(); r = p.add_run()
            set_run(r, "–  " + con, size=11, color=INK)

    add_slide_number_footer(slide, 7, total, tag="Implementations")


def slide_comparison(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "COST & LATENCY", size=10, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(0.75), Inches(9), Inches(0.6))
    r = t.paragraphs[0].add_run()
    set_run(r, "The numbers that actually decide it.", size=22, bold=True, color=INK)

    # Comparison table
    rows = [
        ["",                          "A · Paper-faithful", "B · Streamed heuristic",   "C · Domain-restricted"],
        ["LLM calls / happy query",   "3–4",                "2",                        "2"],
        ["LLM calls / correction",    "4–5",                "3",                        "3"],
        ["LLM calls / off-topic",     "3",                  "0",                        "0"],
        ["Time-to-first-word",        "5–10 s (blocking)",  "0.4–0.8 s",                "0.4–0.8 s"],
        ["Wall-clock happy query",    "8–15 s",             "4–8 s",                    "4–8 s"],
        ["Confidence signal",         "LLM self-audit",     "top-cosine × 1.2",         "top-cosine × 1.2"],
        ["Off-topic behaviour",       "Answers anyway",     "Refuses for free",         "Refuses for free"],
    ]

    left = Inches(0.5); top = Inches(1.6); width = Inches(9.0); height = Inches(3.3)
    tbl = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, height).table

    # Column widths
    tbl.columns[0].width = Inches(2.5)
    for c in range(1, 4):
        tbl.columns[c].width = Inches((9.0 - 2.5) / 3)

    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = ""
            para = cell.text_frame.paragraphs[0]
            run = para.add_run()
            if r_idx == 0:
                run.text = value
                run.font.name = BODY_FONT
                run.font.size = Pt(11)
                run.font.bold = True
                run.font.color.rgb = TEAL if c_idx == 2 else DIM
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
            else:
                run.text = value
                run.font.name = BODY_FONT if c_idx == 0 else MONO_FONT
                run.font.size = Pt(11) if c_idx == 0 else Pt(10)
                run.font.bold = (c_idx == 0)
                run.font.color.rgb = DIM if c_idx == 0 else INK
                if c_idx == 2:  # Highlight column B
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(0x00, 0x35, 0x30)
                else:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
            para.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            cell.margin_left = Inches(0.1); cell.margin_right = Inches(0.1)
            cell.margin_top = Inches(0.05); cell.margin_bottom = Inches(0.05)

    _, foot = add_text_box(slide, Inches(0.5), Inches(4.95), Inches(9), Inches(0.3))
    r = foot.paragraphs[0].add_run()
    set_run(r, "Column B is what shipped. See next slide for why.",
            size=10, italic=True, color=DIM)

    add_slide_number_footer(slide, 8, total, tag="Numbers")


def slide_recommendation(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(0.35), Inches(9), Inches(0.3))
    r = eb.paragraphs[0].add_run()
    set_run(r, "RECOMMENDATION", size=10, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(0.7), Inches(9), Inches(0.9))
    r = t.paragraphs[0].add_run()
    set_run(r, "Ship Architecture B.", size=36, bold=True, color=INK)

    _, s = add_text_box(slide, Inches(0.5), Inches(1.5), Inches(9), Inches(0.5))
    r = s.paragraphs[0].add_run()
    set_run(r,
            "40–50% fewer LLM calls than paper-faithful. TTFT under a second. Off-topic questions refused for free.",
            size=14, italic=True, color=DIM)

    # Three big-number tiles
    tiles = [
        ("−50%",  "LLM cost per query",  TEAL),
        ("< 1 s", "Time-to-first-word",  ORANGE),
        ("0",     "Calls when off-topic", CYAN),
    ]
    y = Inches(2.3); h = Inches(1.8)
    w = Inches(2.85)
    for i, (num, label, color) in enumerate(tiles):
        x = Inches(0.5 + i * 3.05)
        box, tf = add_text_box(slide, x, y, w, h, fill=CARD_BG, edge=color)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); set_run(r, num, size=48, bold=True, color=color, font=TITLE_FONT)
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r = p2.add_run(); set_run(r, label, size=12, color=DIM)

    _, why = add_text_box(slide, Inches(0.5), Inches(4.4), Inches(9), Inches(0.9))
    add_paragraph(why, "Why B beats A and C in this build",
                  size=11, bold=True, color=TEAL, first=True)
    add_paragraph(why,
        "Retrieval score IS a topic gate — no LLM classifier needed.",
        size=12, color=INK, bullet=True)
    add_paragraph(why,
        "Streaming inverts the latency problem: user reads while model finishes.",
        size=12, color=INK, bullet=True)
    add_paragraph(why,
        "Add C's Tavily allowlist only if your vertical demands regulated sources.",
        size=12, color=INK, bullet=True)

    add_slide_number_footer(slide, 9, total, tag="Recommendation")


def slide_demo(prs, total):
    layout = prs.slide_layouts[10]
    slide = prs.slides.add_slide(layout)

    _, eb = add_text_box(slide, Inches(0.5), Inches(1.5), Inches(9), Inches(0.4))
    r = eb.paragraphs[0].add_run()
    set_run(r, "NEXT", size=12, color=TEAL, bold=True)

    _, t = add_text_box(slide, Inches(0.5), Inches(2.0), Inches(9), Inches(1.5))
    r = t.paragraphs[0].add_run()
    set_run(r, "Live demo.", size=64, bold=True, color=INK)
    p = t.add_paragraph()
    r = p.add_run()
    set_run(r, "Then your questions.", size=32, color=DIM, italic=True)

    _, s = add_text_box(slide, Inches(0.5), Inches(4.7), Inches(9), Inches(0.5))
    r = s.paragraphs[0].add_run()
    set_run(r, "Healthcare Knowledge Navigator  ·  RAG + CRAG  ·  Gemini · NVIDIA · Pinecone",
            size=11, color=DIM, italic=True)

    add_slide_number_footer(slide, 10, total, tag="Demo")


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    prs = Presentation(str(TEMPLATE))
    clear_slides(prs)

    total = 10
    slide_title(prs, total)
    slide_problem(prs, total)
    slide_rag_arch(prs, total)
    slide_where_rag_breaks(prs, total)
    slide_enter_crag(prs, total)
    slide_crag_arch(prs, total)
    slide_three_impls(prs, total)
    slide_comparison(prs, total)
    slide_recommendation(prs, total)
    slide_demo(prs, total)

    prs.save(str(OUTPUT))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
