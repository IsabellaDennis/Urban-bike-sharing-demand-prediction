import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Colors
    BG_DARK = RGBColor(15, 23, 42)        # Slate 900
    BG_LIGHT = RGBColor(248, 250, 252)    # Clean off-white
    CARD_BG = RGBColor(255, 255, 255)     # White card
    CARD_BG_ALT = RGBColor(241, 245, 249) # Light slate
    BORDER_COLOR = RGBColor(226, 232, 240)
    
    PRIMARY_BLUE = RGBColor(2, 132, 199)  # Ocean Blue
    ACCENT_TEAL = RGBColor(13, 148, 136)  # Teal
    ACCENT_AMBER = RGBColor(217, 119, 6)  # Warm Amber
    ACCENT_ROSE = RGBColor(225, 29, 72)   # Soft Red
    
    TEXT_MAIN = RGBColor(15, 23, 42)      # Main text
    TEXT_MUTED = RGBColor(100, 116, 139)  # Secondary text
    TEXT_LIGHT = RGBColor(255, 255, 255)  # White text

    TOTAL_SLIDES = 15

    def set_bg(slide, color):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_text, slide_num, stage_badge=None):
        box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(10.5), Inches(0.95))
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_chip = tf.paragraphs[0]
        chip_str = category_text.upper()
        if stage_badge:
            chip_str += f"  •  {stage_badge.upper()}"
        p_chip.text = chip_str
        p_chip.font.name = "Arial"
        p_chip.font.size = Pt(10)
        p_chip.font.bold = True
        p_chip.font.color.rgb = PRIMARY_BLUE
        p_chip.space_after = Pt(2)
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN

        # Slide counter
        num_box = slide.shapes.add_textbox(Inches(11.4), Inches(0.35), Inches(1.1), Inches(0.4))
        ntf = num_box.text_frame
        np = ntf.paragraphs[0]
        np.text = f"{slide_num:02d} / {TOTAL_SLIDES:02d}"
        np.font.name = "Arial"
        np.font.size = Pt(11)
        np.font.bold = True
        np.font.color.rgb = TEXT_MUTED
        np.alignment = PP_ALIGN.RIGHT

        # Divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = BORDER_COLOR
        line.line.color.rgb = BORDER_COLOR

    def set_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = notes_text

    # =============================================================
    # SLIDE 1: TITLE SLIDE
    # =============================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, BG_DARK)
    
    card1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(11.333), Inches(5.9))
    card1.fill.solid()
    card1.fill.fore_color.rgb = RGBColor(30, 41, 59)
    card1.line.color.rgb = RGBColor(51, 65, 85)
    card1.line.width = Pt(1.5)

    tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(1.3), Inches(10.333), Inches(3.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "MCA MINI PROJECT PRESENTATION"
    p0.font.name = "Arial"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(56, 189, 248)
    p0.space_after = Pt(10)
    
    p1 = tf1.add_paragraph()
    p1.text = "Urban Bike Sharing Demand Forecasting"
    p1.font.name = "Arial"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT
    p1.space_after = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "Predicting Station-Level Hourly Bike Demand Using Machine Learning"
    p2.font.name = "Calibri"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    
    meta_s1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(4.7), Inches(4.9), Inches(1.5))
    meta_s1.fill.solid()
    meta_s1.fill.fore_color.rgb = RGBColor(15, 23, 42)
    meta_s1.line.color.rgb = PRIMARY_BLUE
    meta_s1.line.width = Pt(1.2)
    mtf = meta_s1.text_frame
    mtf.word_wrap = True
    mtf.margin_left = mtf.margin_top = Inches(0.2)
    
    mp0 = mtf.paragraphs[0]
    mp0.text = "STUDENT DETAILS"
    mp0.font.name = "Arial"
    mp0.font.size = Pt(10)
    mp0.font.bold = True
    mp0.font.color.rgb = RGBColor(56, 189, 248)
    mp0.space_after = Pt(4)
    
    mp1 = mtf.add_paragraph()
    mp1.text = "Isabella Dennis (Roll No: 30)"
    mp1.font.name = "Arial"
    mp1.font.size = Pt(18)
    mp1.font.bold = True
    mp1.font.color.rgb = TEXT_LIGHT
    
    mp2 = mtf.add_paragraph()
    mp2.text = "Department of Computer Applications"
    mp2.font.name = "Calibri"
    mp2.font.size = Pt(12)
    mp2.font.color.rgb = RGBColor(148, 163, 184)

    guide_s1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(4.7), Inches(4.9), Inches(1.5))
    guide_s1.fill.solid()
    guide_s1.fill.fore_color.rgb = RGBColor(15, 23, 42)
    guide_s1.line.color.rgb = ACCENT_TEAL
    guide_s1.line.width = Pt(1.2)
    gtf = guide_s1.text_frame
    gtf.word_wrap = True
    gtf.margin_left = gtf.margin_top = Inches(0.2)
    
    gp0 = gtf.paragraphs[0]
    gp0.text = "PROJECT GUIDE"
    gp0.font.name = "Arial"
    gp0.font.size = Pt(10)
    gp0.font.bold = True
    gp0.font.color.rgb = ACCENT_TEAL
    gp0.space_after = Pt(4)
    
    gp1 = gtf.add_paragraph()
    gp1.text = "Mr. Sibu Skaria"
    gp1.font.name = "Arial"
    gp1.font.size = Pt(18)
    gp1.font.bold = True
    gp1.font.color.rgb = TEXT_LIGHT
    
    gp2 = gtf.add_paragraph()
    gp2.text = "Assistant Professor, Department of Computer Applications"
    gp2.font.name = "Calibri"
    gp2.font.size = Pt(12)
    gp2.font.color.rgb = RGBColor(148, 163, 184)

    set_notes(s1, """Good morning respected guide Mr. Sibu Skaria, external examiners, and panel members.
My name is Isabella Dennis, Roll Number 30, from the Department of Computer Applications.
Today, I am presenting my MCA Mini Project on 'Urban Bike Sharing Demand Forecasting: Predicting Station-Level Hourly Bike Demand Using Machine Learning'.
In this project, we address a critical logistical problem in modern smart cities: forecasting hourly bicycle departures at each docking station across a regional network using historical transaction data.
Notice that our target is explicitly formulated as station-level hourly departure demand, which provides a direct and actionable metric for transit operators to maintain system balance.""")

    # =============================================================
    # SLIDE 2: PROBLEM STATEMENT & OBJECTIVES (COMBINED SINGLE SLIDE)
    # =============================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, BG_LIGHT)

    # 1. Top Pill Badge: THE CHALLENGE & THE GOALS
    pill = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.38), Inches(3.2), Inches(0.34))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(14, 116, 144)  # Cyan/Teal
    pill.line.fill.background()
    ptf = pill.text_frame
    ptf.margin_top = Inches(0.04)
    pp = ptf.paragraphs[0]
    pp.text = "THE CHALLENGE & THE GOALS"
    pp.font.name = "Arial"
    pp.font.size = Pt(9)
    pp.font.bold = True
    pp.font.color.rgb = TEXT_LIGHT
    pp.alignment = PP_ALIGN.CENTER

    # 2. Main Title: Problem Statement & Project Objectives
    tb_title = s2.shapes.add_textbox(Inches(0.8), Inches(0.78), Inches(9.5), Inches(0.5))
    ttf = tb_title.text_frame
    ttf.word_wrap = True
    ttf.margin_left = ttf.margin_top = ttf.margin_right = ttf.margin_bottom = 0
    tp = ttf.paragraphs[0]
    tp.text = "Problem Statement & Project Objectives"
    tp.font.name = "Arial"
    tp.font.size = Pt(21)
    tp.font.bold = True
    tp.font.color.rgb = RGBColor(15, 23, 42)

    # Slide counter at top right
    s2_num = s2.shapes.add_textbox(Inches(11.4), Inches(0.38), Inches(1.1), Inches(0.35))
    s2_ntf = s2_num.text_frame
    s2_np = s2_ntf.paragraphs[0]
    s2_np.text = f"02 / {TOTAL_SLIDES:02d}"
    s2_np.font.name = "Arial"
    s2_np.font.size = Pt(11)
    s2_np.font.bold = True
    s2_np.font.color.rgb = TEXT_MUTED
    s2_np.alignment = PP_ALIGN.RIGHT

    # -------------------------------------------------------------
    # LEFT COLUMN: THE CHALLENGE (PROBLEM STATEMENT)
    # -------------------------------------------------------------
    col_l_left = Inches(0.8)
    col_w = Inches(5.65)

    # Heading: Problem Statement
    tb_prob_hdr = s2.shapes.add_textbox(col_l_left, Inches(1.35), col_w, Inches(0.3))
    phtf = tb_prob_hdr.text_frame
    phtf.margin_left = phtf.margin_top = phtf.margin_right = phtf.margin_bottom = 0
    php = phtf.paragraphs[0]
    php.text = "THE CHALLENGE: PROBLEM STATEMENT"
    php.font.name = "Arial"
    php.font.size = Pt(11)
    php.font.bold = True
    php.font.color.rgb = RGBColor(14, 116, 144)

    # Diagram Card: Station Imbalance, Illustrated
    dia_card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_l_left, Inches(1.72), col_w, Inches(2.0))
    dia_card.fill.solid()
    dia_card.fill.fore_color.rgb = CARD_BG
    dia_card.line.color.rgb = RGBColor(226, 232, 240)
    dia_card.line.width = Pt(1.5)

    tb_dia_title = s2.shapes.add_textbox(col_l_left + Inches(0.2), Inches(1.85), col_w - Inches(0.4), Inches(0.28))
    dtf = tb_dia_title.text_frame
    dtf.margin_left = dtf.margin_top = dtf.margin_right = dtf.margin_bottom = 0
    dp = dtf.paragraphs[0]
    dp.text = "Station Imbalance, Illustrated"
    dp.font.name = "Arial"
    dp.font.size = Pt(11)
    dp.font.bold = True
    dp.font.color.rgb = RGBColor(15, 23, 42)

    # Sub-box 1: Station A (Shortage: 0 bikes)
    b_stA = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_l_left + Inches(0.2), Inches(2.2), Inches(2.3), Inches(0.95))
    b_stA.fill.solid()
    b_stA.fill.fore_color.rgb = RGBColor(254, 242, 242) # soft rose/peach
    b_stA.line.color.rgb = RGBColor(252, 165, 165)
    b_stA.line.width = Pt(1)
    atf = b_stA.text_frame
    atf.margin_top = Inches(0.06)
    atf.margin_left = atf.margin_right = Inches(0.1)
    ap0 = atf.paragraphs[0]
    ap0.text = "STATION A (Residential)"
    ap0.font.name = "Arial"
    ap0.font.size = Pt(7.5)
    ap0.font.bold = True
    ap0.font.color.rgb = RGBColor(185, 28, 28)
    ap0.alignment = PP_ALIGN.CENTER
    ap1 = atf.add_paragraph()
    ap1.text = "0 BIKES"
    ap1.font.name = "Arial"
    ap1.font.size = Pt(13)
    ap1.font.bold = True
    ap1.font.color.rgb = RGBColor(185, 28, 28)
    ap1.alignment = PP_ALIGN.CENTER
    ap2 = atf.add_paragraph()
    ap2.text = "SHORTAGE: Riders cannot start trips"
    ap2.font.name = "Calibri"
    ap2.font.size = Pt(8)
    ap2.font.color.rgb = RGBColor(127, 29, 29)
    ap2.alignment = PP_ALIGN.CENTER

    # Arrow shape in center
    arr = s2.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, col_l_left + Inches(2.6), Inches(2.55), Inches(0.45), Inches(0.25))
    arr.fill.solid()
    arr.fill.fore_color.rgb = RGBColor(217, 119, 6) # amber
    arr.line.fill.background()

    # Sub-box 2: Station B (Surplus: FULL)
    b_stB = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_l_left + Inches(3.15), Inches(2.2), Inches(2.3), Inches(0.95))
    b_stB.fill.solid()
    b_stB.fill.fore_color.rgb = RGBColor(240, 253, 244) # soft green
    b_stB.line.color.rgb = RGBColor(134, 239, 172)
    b_stB.line.width = Pt(1)
    btf2 = b_stB.text_frame
    btf2.margin_top = Inches(0.06)
    btf2.margin_left = btf2.margin_right = Inches(0.1)
    bp0 = btf2.paragraphs[0]
    bp0.text = "STATION B (Transit Hub)"
    bp0.font.name = "Arial"
    bp0.font.size = Pt(7.5)
    bp0.font.bold = True
    bp0.font.color.rgb = RGBColor(21, 128, 61)
    bp0.alignment = PP_ALIGN.CENTER
    bp1 = btf2.add_paragraph()
    bp1.text = "FULL"
    bp1.font.name = "Arial"
    bp1.font.size = Pt(13)
    bp1.font.bold = True
    bp1.font.color.rgb = RGBColor(21, 128, 61)
    bp1.alignment = PP_ALIGN.CENTER
    bp2 = btf2.add_paragraph()
    bp2.text = "SURPLUS: Docks cannot accept returns"
    bp2.font.name = "Calibri"
    bp2.font.size = Pt(8)
    bp2.font.color.rgb = RGBColor(20, 83, 45)
    bp2.alignment = PP_ALIGN.CENTER

    # Caption under diagram
    tb_dia_sub = s2.shapes.add_textbox(col_l_left + Inches(0.2), Inches(3.22), col_w - Inches(0.4), Inches(0.45))
    cptf = tb_dia_sub.text_frame
    cptf.word_wrap = True
    cptf.margin_left = cptf.margin_top = cptf.margin_right = cptf.margin_bottom = 0
    cpp = cptf.paragraphs[0]
    cpp.text = "Anticipating hourly station-level demand enables operators to redistribute bikes proactively before shortages or surpluses occur."
    cpp.font.name = "Calibri"
    cpp.font.size = Pt(9)
    cpp.font.italic = True
    cpp.font.color.rgb = RGBColor(71, 85, 105)

    # 4 Numbered Problem Statements (List items 1 to 4)
    prob_items = [
        ("1", "Uneven Demand", "Bike-sharing demand varies sharply across stations and across the hours of the day."),
        ("2", "Shortages", "Popular stations frequently run out of bikes during peak commute windows."),
        ("3", "Surpluses", "Other stations fill up with returned bikes and cannot accept more."),
        ("4", "Manual Redistribution", "Moving bikes between stations by hand is slow, costly and reactive."),
        ("5", "Multiple Drivers", "Demand is shaped by temporal patterns, station characteristics and weather.")
    ]

    p_top_start = Inches(3.82)
    p_step = Inches(0.58)
    for idx, (num_s, title_s, desc_s) in enumerate(prob_items):
        item_top = p_top_start + idx * p_step
        
        # Circle number badge
        c_num = s2.shapes.add_shape(MSO_SHAPE.OVAL, col_l_left + Inches(0.05), item_top + Inches(0.02), Inches(0.36), Inches(0.36))
        c_num.fill.solid()
        c_num.fill.fore_color.rgb = RGBColor(14, 116, 144) # Teal
        c_num.line.fill.background()
        cntf = c_num.text_frame
        cntf.margin_left = cntf.margin_top = cntf.margin_right = cntf.margin_bottom = 0
        cnp = cntf.paragraphs[0]
        cnp.text = num_s
        cnp.font.name = "Arial"
        cnp.font.size = Pt(11)
        cnp.font.bold = True
        cnp.font.color.rgb = TEXT_LIGHT
        cnp.alignment = PP_ALIGN.CENTER

        # Text block
        tb_pt = s2.shapes.add_textbox(col_l_left + Inches(0.52), item_top, col_w - Inches(0.58), Inches(0.68))
        pt_tf = tb_pt.text_frame
        pt_tf.word_wrap = True
        pt_tf.margin_left = pt_tf.margin_top = pt_tf.margin_right = pt_tf.margin_bottom = 0
        pp0 = pt_tf.paragraphs[0]
        pp0.text = title_s
        pp0.font.name = "Arial"
        pp0.font.size = Pt(10)
        pp0.font.bold = True
        pp0.font.color.rgb = RGBColor(15, 23, 42)
        pp0.space_after = Pt(1)

        pp1 = pt_tf.add_paragraph()
        pp1.text = desc_s
        pp1.font.name = "Calibri"
        pp1.font.size = Pt(9)
        pp1.font.color.rgb = RGBColor(71, 85, 105)

    # -------------------------------------------------------------
    # RIGHT COLUMN: THE GOALS (PROJECT OBJECTIVES)
    # -------------------------------------------------------------
    col_r_left = Inches(6.85)

    # Heading: Project Objectives
    tb_obj_hdr = s2.shapes.add_textbox(col_r_left, Inches(1.35), col_w, Inches(0.3))
    ohtf = tb_obj_hdr.text_frame
    ohtf.margin_left = ohtf.margin_top = ohtf.margin_right = ohtf.margin_bottom = 0
    ohp = ohtf.paragraphs[0]
    ohp.text = "THE GOALS: PROJECT OBJECTIVES"
    ohp.font.name = "Arial"
    ohp.font.size = Pt(11)
    ohp.font.bold = True
    ohp.font.color.rgb = RGBColor(30, 41, 59)

    # 5 Numbered Objective Cards
    obj_cards = [
        ("1", "Forecast Demand", "Predict station-level bike-sharing demand using machine learning models."),
        ("2", "Analyze Patterns", "Study demand variation across time, stations and environmental conditions."),
        ("3", "Leverage Historical Data", "Use historical trip and weather information as predictors."),
        ("4", "Flag Imbalances", "Identify potential shortage and surplus situations before they occur."),
        ("5", "Support Redistribution", "Enable smarter, data-driven bike redistribution through an interactive app.")
    ]

    o_top_start = Inches(1.72)
    o_step = Inches(1.0)
    for idx, (num_s, title_s, desc_s) in enumerate(obj_cards):
        card_top = o_top_start + idx * o_step

        # Card container
        c_box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col_r_left, card_top, col_w, Inches(0.92))
        c_box.fill.solid()
        c_box.fill.fore_color.rgb = CARD_BG
        c_box.line.color.rgb = RGBColor(226, 232, 240)
        c_box.line.width = Pt(1.2)

        # Number circle inside card
        c_num = s2.shapes.add_shape(MSO_SHAPE.OVAL, col_r_left + Inches(0.18), card_top + Inches(0.24), Inches(0.42), Inches(0.42))
        c_num.fill.solid()
        c_num.fill.fore_color.rgb = RGBColor(30, 41, 59) # Dark Navy
        c_num.line.fill.background()
        cntf = c_num.text_frame
        cntf.margin_left = cntf.margin_top = cntf.margin_right = cntf.margin_bottom = 0
        cnp = cntf.paragraphs[0]
        cnp.text = num_s
        cnp.font.name = "Arial"
        cnp.font.size = Pt(12)
        cnp.font.bold = True
        cnp.font.color.rgb = TEXT_LIGHT
        cnp.alignment = PP_ALIGN.CENTER

        # Text block
        tb_ot = s2.shapes.add_textbox(col_r_left + Inches(0.72), card_top + Inches(0.14), col_w - Inches(0.85), Inches(0.68))
        ot_tf = tb_ot.text_frame
        ot_tf.word_wrap = True
        ot_tf.margin_left = ot_tf.margin_top = ot_tf.margin_right = ot_tf.margin_bottom = 0
        op0 = ot_tf.paragraphs[0]
        op0.text = title_s
        op0.font.name = "Arial"
        op0.font.size = Pt(10.5)
        op0.font.bold = True
        op0.font.color.rgb = RGBColor(15, 23, 42)
        op0.space_after = Pt(2)

        op1 = ot_tf.add_paragraph()
        op1.text = desc_s
        op1.font.name = "Calibri"
        op1.font.size = Pt(9.5)
        op1.font.color.rgb = RGBColor(71, 85, 105)

    # Footer (MACE format matching department requirements)
    tb_foot_l = s2.shapes.add_textbox(Inches(0.8), Inches(6.88), Inches(4.5), Inches(0.3))
    fltf = tb_foot_l.text_frame
    fltf.margin_left = fltf.margin_top = fltf.margin_right = fltf.margin_bottom = 0
    flp = fltf.paragraphs[0]
    flp.text = "Department of Computer Applications | MACE"
    flp.font.name = "Calibri"
    flp.font.size = Pt(9.5)
    flp.font.color.rgb = RGBColor(100, 116, 139)

    tb_foot_c = s2.shapes.add_textbox(Inches(4.5), Inches(6.88), Inches(5.0), Inches(0.3))
    fctf = tb_foot_c.text_frame
    fctf.margin_left = fctf.margin_top = fctf.margin_right = fctf.margin_bottom = 0
    fcp = fctf.paragraphs[0]
    fcp.text = "Mar Athanasius College of Engineering, Kothamangalam"
    fcp.font.name = "Calibri"
    fcp.font.size = Pt(9.5)
    fcp.font.color.rgb = RGBColor(100, 116, 139)
    fcp.alignment = PP_ALIGN.CENTER

    tb_foot_r = s2.shapes.add_textbox(Inches(10.0), Inches(6.88), Inches(2.5), Inches(0.3))
    frtf = tb_foot_r.text_frame
    frtf.margin_left = frtf.margin_top = frtf.margin_right = frtf.margin_bottom = 0
    frp = frtf.paragraphs[0]
    frp.text = "Slide 2 of 15"
    frp.font.name = "Calibri"
    frp.font.size = Pt(9.5)
    frp.font.color.rgb = RGBColor(100, 116, 139)
    frp.alignment = PP_ALIGN.RIGHT

    set_notes(s2, """The main problem is that bike demand varies between different stations and different times of the day.
During peak hours, some stations may run out of bikes, while other stations may become full.
This creates shortage at some stations and surplus at others.
However, Manual redistribution of bikes is slow, costly and time taken.
Demand is influenced by factors such as time, station characteristics and weather.
So, my project aims to forecast station-level demand and support efficient bike redistribution.

The first objective is to forecast bike demand at the station level using machine learning.
The second objective is to analyze demand patterns across time, stations and environmental conditions.
I also use historical trip and weather information as predictors.
Another objective is to identify possible shortage and surplus situations in advance.
Finally, the system will support data-driven bike redistribution through an interactive application.""")

    # =============================================================
    # SLIDE 3: LITERATURE REVIEW (PRESERVED AS DEDICATED SLIDE)
    # =============================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3, BG_LIGHT)
    add_header(s3, "Literature Review", "Related Works", 3)
    
    papers = [
        {
            "num": "PAPER 1",
            "author": "Aydın et al. (2023)",
            "title": "Prediction bike-sharing demand with gradient boosting methods",
            "says": "Predicts hourly bike demand in two cities (Konya and Washington D.C.) using weather and time features to help balance bike fleets.",
            "algos": "XGBoost, LightGBM, CatBoost",
            "best": "CatBoost and Gradient Boosting",
            "color": PRIMARY_BLUE
        },
        {
            "num": "PAPER 2",
            "author": "Karunanithi et al. (2024)",
            "title": "A predictive analytics approach for forecasting bike rental demand",
            "says": "Compares different machine learning models to forecast hourly rental demand and prevent stations from becoming empty or full.",
            "algos": "Random Forest, Gradient Boosting, Decision Tree, MLP",
            "best": "Random Forest and Gradient Boosting",
            "color": ACCENT_TEAL
        },
        {
            "num": "PAPER 3",
            "author": "Yao et al. (2024)",
            "title": "GSTGAT: Gated spatiotemporal graph attention network",
            "says": "Uses the SF Bay Area dataset to study how demand at one bike station influences neighboring stations across the city network.",
            "algos": "GSTGAT (Graph Attention Network), GRU",
            "best": "GSTGAT model",
            "color": ACCENT_AMBER
        }
    ]

    for i, p in enumerate(papers):
        left_pos = Inches(0.8 + i * 4.0)
        c = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.55), Inches(3.733), Inches(5.5))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR
        c.line.width = Pt(1.5)

        strip = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, left_pos, Inches(1.55), Inches(3.733), Inches(0.1))
        strip.fill.solid()
        strip.fill.fore_color.rgb = p["color"]
        strip.line.fill.background()

        ctf = c.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_right = Inches(0.2)
        ctf.margin_top = Inches(0.2)

        p0 = ctf.paragraphs[0]
        p0.text = p["num"]
        p0.font.name = "Arial"
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = p["color"]
        p0.space_after = Pt(2)

        p1 = ctf.add_paragraph()
        p1.text = p["author"]
        p1.font.name = "Arial"
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN
        p1.space_after = Pt(2)

        p2 = ctf.add_paragraph()
        p2.text = f'"{p["title"]}"'
        p2.font.name = "Calibri"
        p2.font.size = Pt(10.5)
        p2.font.italic = True
        p2.font.color.rgb = PRIMARY_BLUE
        p2.space_after = Pt(8)

        p3 = ctf.add_paragraph()
        p3.text = "WHAT THE PAPER SAYS:"
        p3.font.name = "Arial"
        p3.font.size = Pt(9.5)
        p3.font.bold = True
        p3.font.color.rgb = TEXT_MUTED
        p3.space_after = Pt(2)

        p4 = ctf.add_paragraph()
        p4.text = p["says"]
        p4.font.name = "Calibri"
        p4.font.size = Pt(10.5)
        p4.font.color.rgb = TEXT_MAIN
        p4.space_after = Pt(8)

        p5 = ctf.add_paragraph()
        p5.text = "RELEVANT ALGORITHMS:"
        p5.font.name = "Arial"
        p5.font.size = Pt(9.5)
        p5.font.bold = True
        p5.font.color.rgb = TEXT_MUTED
        p5.space_after = Pt(2)

        p6 = ctf.add_paragraph()
        p6.text = p["algos"]
        p6.font.name = "Calibri"
        p6.font.size = Pt(10.5)
        p6.font.color.rgb = TEXT_MAIN
        p6.space_after = Pt(10)

        p7 = ctf.add_paragraph()
        p7.text = "MODEL FOUND AS BEST:"
        p7.font.name = "Arial"
        p7.font.size = Pt(9.5)
        p7.font.bold = True
        p7.font.color.rgb = p["color"]
        p7.space_after = Pt(2)

        p8 = ctf.add_paragraph()
        p8.text = p["best"]
        p8.font.name = "Arial"
        p8.font.size = Pt(11)
        p8.font.bold = True
        p8.font.color.rgb = TEXT_MAIN

    set_notes(s3, """In our literature review on Slide 3, we studied three relevant papers.
• Paper 1 by Aydın et al. (2023) evaluated gradient boosting methods across two cities, showing CatBoost and XGBoost capture non-linear hourly demand effectively.
• Paper 2 by Karunanithi et al. (2024) compared multiple machine learning models and proved that tree ensemble methods significantly outperform standard neural networks on structured tabular time features.
• Paper 3 by Yao et al. (2024) analyzed spatial network relationships on this exact SF Bay Area dataset.
• Research Foundation: These studies directly justify our algorithm selection of Random Forest, XGBoost, and CatBoost, while highlighting our unique contributions in full zero-demand grid expansion and strict chronological leakage audits.""")

    # =============================================================
    # SLIDE 4: ABOUT THE DATASET (STAGE 1: RAW DATA)
    # =============================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4, BG_LIGHT)
    add_header(s4, "About the Dataset", "Dataset Information", 4, "Stage 1: Raw Data (669,959 Rows, 12 Columns)")

    c4 = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.5))
    c4.fill.solid()
    c4.fill.fore_color.rgb = CARD_BG
    c4.line.color.rgb = BORDER_COLOR
    c4.line.width = Pt(1.5)

    tf4 = c4.text_frame
    tf4.word_wrap = True
    tf4.margin_left = tf4.margin_right = Inches(0.25)
    tf4.margin_top = Inches(0.25)

    p = tf4.paragraphs[0]
    p.text = "RAW DATASET SUMMARY"
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    info_list = [
        ("Dataset Source:", "SF Bay Area Bike Share Dataset (trip.csv)"),
        ("Total Raw Records:", "669,959 trip departure transactions"),
        ("Total Raw Columns:", "12 columns (receipts, timestamps, dock IDs, user type)"),
        ("Time Span:", "Aug 29, 2013 to Aug 31, 2015 (733 days / ~2 full years)"),
        ("Hourly Timestamps Used:", "17,583 hourly intervals across the 2-year timeline"),
        ("Active Stations:", "70 unique docking stations across the region"),
        ("Target Variable:", "Hourly demand = number of bike trips starting from a station during an hour."),
        ("Why Hourly Departures?:", "Trip duration had extreme values (from 60 seconds up to several months due to unreturned bikes). Forecasting hourly departures is clean, stable, and directly useful for operators.")
    ]

    for label, val in info_list:
        p = tf4.add_paragraph()
        p.text = f"• {label} "
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN
        r = p.add_run()
        r.text = val
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.bold = False
        r.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(4)

    img1 = "reports/images/fig_3_1_shape_dtypes.png"
    if os.path.exists(img1):
        s4.shapes.add_picture(img1, Inches(6.6), Inches(1.55), width=Inches(5.9), height=Inches(2.6))

    img2 = "reports/images/fig_3_3_first_10_records.png"
    if os.path.exists(img2):
        s4.shapes.add_picture(img2, Inches(6.6), Inches(4.35), width=Inches(5.9), height=Inches(2.7))

    set_notes(s4, """Slide 4 introduces Stage 1 of our pipeline: Raw Data Collection.
• Dataset Overview: We use the real-world San Francisco Bay Area Bike Share dataset from Kaggle, specifically trip.csv.
• Key Specifications: It records 669,959 raw trip transactions spanning August 29, 2013 to August 31, 2015. That represents 733 days, or two full years, covering 17,583 continuous hourly timestamps across 70 unique docking stations.
• Why Hourly Departures as Target?: Trip durations in raw data have extreme outliers ranging from 60 seconds to over 200 days due to unreturned bikes. Meanwhile, trip.csv only logs departure events, not live dock occupancy. Formulating the target as hourly departures provides a clean, stable, and operationally actionable metric.
• Visuals: Figure 3.1 on the right shows the dataset shape (669,959 rows by 12 columns) and data types. Figure 3.3 displays the first 10 raw ride receipts.
• Stage 1 Summary: We begin with 669,959 raw checkout records and 12 columns.""")

    # =============================================================
    # SLIDE 5: EXPLORATORY DATA ANALYSIS (EDA)
    # =============================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5, BG_LIGHT)
    add_header(s5, "Exploratory Data Analysis (EDA)", "Data Insights", 5, "Quality & Commuter Behavior")

    c5_1 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.55), Inches(5.6), Inches(2.6))
    c5_1.fill.solid()
    c5_1.fill.fore_color.rgb = CARD_BG
    c5_1.line.color.rgb = BORDER_COLOR
    c5_1.line.width = Pt(1.5)

    tf5_1 = c5_1.text_frame
    tf5_1.word_wrap = True
    tf5_1.margin_left = tf5_1.margin_right = Inches(0.2)
    tf5_1.margin_top = Inches(0.2)

    p = tf5_1.paragraphs[0]
    p.text = "DATA QUALITY AUDIT (669,959 TRIPS)"
    p.font.name = "Arial"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(4)

    q_points = [
        "• Missing Values: Only 6,619 in zip_code (~0.98%), an optional input field. All other 11 columns have 0 missing values.",
        "• Duplicate Rows: Exactly 0 duplicate rows detected in the entire dataset.",
        "• Active Network: 70 unique docking stations across the Bay Area.",
        "• User Categories: ~84% Annual Subscribers (daily office commuters) and ~16% Casual Customers (tourists/leisure)."
    ]
    for qp in q_points:
        p = tf5_1.add_paragraph()
        p.text = qp
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(2)

    c5_2 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.35), Inches(5.6), Inches(2.7))
    c5_2.fill.solid()
    c5_2.fill.fore_color.rgb = CARD_BG
    c5_2.line.color.rgb = BORDER_COLOR
    c5_2.line.width = Pt(1.5)

    tf5_2 = c5_2.text_frame
    tf5_2.word_wrap = True
    tf5_2.margin_left = tf5_2.margin_right = Inches(0.2)
    tf5_2.margin_top = Inches(0.2)

    p = tf5_2.paragraphs[0]
    p.text = "RIDER BEHAVIOR PATTERNS"
    p.font.name = "Arial"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = ACCENT_TEAL
    p.space_after = Pt(4)

    b_points = [
        "• Weekdays (Mon–Fri): Two sharp commuter rush hour peaks — Morning (8:00 AM) and Evening (5:00 PM).",
        "• Weekends (Sat–Sun): Rush hour peaks vanish completely. Demand forms a gentle recreational curve peaking between 12:00 PM and 3:00 PM.",
        "• Station Volumes: SF Caltrain (Townsend at 4th) recorded the highest demand with ~49,000+ trip departures, while quiet suburban stations had under 1,000 trips.",
        "• Practical Insight: Demand changes significantly between stations and across different hours, showing the need for demand forecasting."
    ]
    for bp in b_points:
        p = tf5_2.add_paragraph()
        p.text = bp
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(3)

    if os.path.exists("reports/images/hourly_demand.png"):
        s5.shapes.add_picture("reports/images/hourly_demand.png", Inches(6.6), Inches(1.55), width=Inches(5.9), height=Inches(2.65))

    if os.path.exists("reports/images/weekday_demand.png"):
        s5.shapes.add_picture("reports/images/weekday_demand.png", Inches(6.6), Inches(4.35), width=Inches(5.9), height=Inches(2.7))

    set_notes(s5, """Slide 5 presents our Exploratory Data Analysis, examining data quality and rider behavior patterns.
• Data Quality Audit (Top Card):
The raw data is remarkably high quality. Only 6,619 records had missing values in zip_code (~0.98%), which is an optional personal field. All other 11 columns have zero missing values. Duplicate check revealed exactly zero duplicates across all 669k records. User breakdown shows that 84% of riders are Annual Subscribers (daily office commuters) and 16% are Casual Customers (tourists).
• Rider Behavior Patterns (Bottom Card):
On weekdays, demand displays a sharp twin commuter rush hour peak at 8:00 AM and 5:00 PM. On weekends, the commuter spikes vanish completely, replaced by a gentle recreational midday curve between 12:00 PM and 3:00 PM. Station demand varies drastically: SF Caltrain recorded over 49,000 trips, while quiet suburban stations had fewer than 1,000 trips over two years.
• Visuals: The top graph shows the twin-peak weekday commute curve vs the weekend curve. The bottom graph confirms higher total demand on workdays.
• Key Insight: Demand fluctuates heavily by hour, day of week, and station, proving the necessity of automated machine learning forecasts.""")

    # =============================================================
    # SLIDE 6: PREPROCESSING - THE ZERO-DEMAND PROBLEM & GRID EXPANSION
    # =============================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6, BG_LIGHT)
    add_header(s6, "Data Preprocessing: Solving the Zero-Demand Problem", "Data Preparation", 6, "Stage 2: Grid Expansion (1,230,810 Rows)")

    prep_steps = [
        {
            "step": "STEP 1: HOURLY ROUNDING",
            "title": "Rounding Datetime to Hour",
            "what": "WHAT IS IT?\nRound the exact start datetime down to the beginning of that hour:\n2013-08-29 09:08:24  →  2013-08-29 09:00:00\n2013-08-29 09:35:10  →  2013-08-29 09:00:00",
            "why": "WHY DO WE USE IT?\nIndividual trips have different seconds. Rounding allows us to group and count all trips that started in that hour.",
            "color": PRIMARY_BLUE
        },
        {
            "step": "STEP 2: THE PROBLEM",
            "title": "The Zero-Demand Problem",
            "what": "WHAT IS IT?\nIn raw logs, if nobody rents a bike at Station #12 at 3:00 AM, NO row is recorded.",
            "why": "WHY IS THIS DANGEROUS?\nIf we only train on hours when rides occurred, the model never learns when demand is 0! It will severely over-predict nighttime demand.",
            "color": ACCENT_ROSE
        },
        {
            "step": "STEP 3: COMPLETE TIMELINE",
            "title": "Grid Expansion (70 × 17,583)",
            "what": "WHY MULTIPLY 70 × 17,583?\nStations exist even when 0 rides occur. In raw logs, quiet hours have no record. To teach the model when demand is 0, we create a record for EVERY station at EVERY hour across the 2 years:\n70 stations × 17,583 hours = 1,230,810 station-hour records.",
            "why": "ZERO-FILLING:\nHours with trips get their actual count; quiet hours get demand = 0.",
            "color": ACCENT_TEAL
        },
        {
            "step": "STEP 4: EVENT CALENDAR",
            "title": "Adding Special Events",
            "what": "WHAT IS IT?\nExtracted holiday and festival dates from trip.csv into a calendar lookup dictionary.",
            "why": "WHY DO WE USE IT?\nEvery newly added 0-demand station-hour row correctly inherits its holiday label (e.g. Halloween, Christmas).",
            "color": ACCENT_AMBER
        }
    ]

    for i, s in enumerate(prep_steps):
        s_left = Inches(0.8 + i * 2.98)
        sc = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, s_left, Inches(1.55), Inches(2.82), Inches(4.3))
        sc.fill.solid()
        sc.fill.fore_color.rgb = CARD_BG
        sc.line.color.rgb = BORDER_COLOR
        sc.line.width = Pt(1.5)

        st = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, s_left, Inches(1.55), Inches(2.82), Inches(0.1))
        st.fill.solid()
        st.fill.fore_color.rgb = s["color"]
        st.line.fill.background()

        stf = sc.text_frame
        stf.word_wrap = True
        stf.margin_left = stf.margin_right = Inches(0.16)
        stf.margin_top = Inches(0.18)

        p = stf.paragraphs[0]
        p.text = s["step"]
        p.font.name = "Arial"
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = s["color"]
        p.space_after = Pt(2)

        p1 = stf.add_paragraph()
        p1.text = s["title"]
        p1.font.name = "Arial"
        p1.font.size = Pt(11.5)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN
        p1.space_after = Pt(4)

        p2 = stf.add_paragraph()
        p2.text = s["what"]
        p2.font.name = "Calibri"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_MAIN
        p2.space_after = Pt(5)

        p3 = stf.add_paragraph()
        p3.text = s["why"]
        p3.font.name = "Calibri"
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = RGBColor(51, 65, 85)

    bot = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.0), Inches(11.733), Inches(1.0))
    bot.fill.solid()
    bot.fill.fore_color.rgb = CARD_BG_ALT
    bot.line.color.rgb = PRIMARY_BLUE
    bot.line.width = Pt(1.2)
    btf = bot.text_frame
    btf.word_wrap = True
    btf.margin_left = Inches(0.25)
    btf.margin_top = Inches(0.15)
    
    bp0 = btf.paragraphs[0]
    bp0.text = "STAGE 2 SUMMARY (ROW COUNT PROGRESSION):"
    bp0.font.name = "Arial"
    bp0.font.size = Pt(10.5)
    bp0.font.bold = True
    bp0.font.color.rgb = PRIMARY_BLUE

    bp1 = btf.add_paragraph()
    bp1.text = "Expanded raw 669,959 individual trips into 1,230,810 station-hour records (70 stations × 17,583 hourly timestamps). Features: 3 base columns (station_id, hour_timestamp, demand). Inactive hours now have demand = 0, teaching the model when stations are idle."
    bp1.font.name = "Calibri"
    bp1.font.size = Pt(10.5)
    bp1.font.bold = True
    bp1.font.color.rgb = TEXT_MAIN

    set_notes(s6, """Slide 6 describes Stage 2: Data Preprocessing and Solving the Zero-Demand Problem.
• Step 1 - Hourly Rounding: Exact ride timestamps contain arbitrary minutes and seconds. We floor every start datetime down to the beginning of the hour (e.g., 2013-08-29 09:08:24 becomes 2013-08-29 09:00:00). This groups all rides within each 60-minute window.
• Step 2 - The Zero-Demand Problem: In raw logs, if nobody rents a bike at Station 12 at 3:00 AM, no row is recorded. If we only train on rows where rides occurred, the model never learns when demand is zero and will severely over-predict nighttime demand.
• Step 3 - Grid Expansion: To fix this, we generate a complete Cartesian product of 70 stations multiplied by 17,583 hours = exactly 1,230,810 station-hour records. For active hours, we record actual trip counts; for quiet hours, we explicitly assign demand = 0.
• Step 4 - Event Calendar: We extracted city holidays and festivals from trip.csv into a lookup table so that every newly created station-hour row inherits its holiday status (e.g., Thanksgiving, Christmas).
• Stage 2 Summary: Row count expanded from 669,959 trips into 1,230,810 station-hour records across 3 base features: station_id, hour_timestamp, and demand.""")

    # =============================================================
    # SLIDE 7: RAW DATASET COLUMN AUDIT (MOVED AFTER PREPROCESSING)
    # =============================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7, BG_LIGHT)
    add_header(s7, "Raw Dataset Columns (Why Kept or Removed)", "Data Preprocessing", 7, "Stage 3: Column Selection (12 Raw Columns Filtered)")

    tbl_s7 = s7.shapes.add_table(13, 5, Inches(0.8), Inches(1.5), Inches(11.733), Inches(4.7))
    t7 = tbl_s7.table
    t7.columns[0].width = Inches(2.0)
    t7.columns[1].width = Inches(1.4)
    t7.columns[2].width = Inches(2.9)
    t7.columns[3].width = Inches(1.7)
    t7.columns[4].width = Inches(3.733)

    h7 = ["Column Name", "Data Type", "Meaning", "Example Value", "Decision & Simple Reason"]
    for c_idx, h in enumerate(h7):
        cell = t7.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    rows7 = [
        ("id", "Integer", "Unique trip receipt number", "4576", "Removed: Just a receipt number for one ride"),
        ("duration", "Integer", "Trip duration in seconds", "63", "Removed: Future info (we only know duration after ride ends)"),
        ("start_date", "Object (String)", "Time when bike unlocked", "8/29/2013 09:08", "Transformed: Floored to hour_timestamp to group trips"),
        ("start_station_name", "Object (String)", "Origin station name", "San Jose City Hall", "Kept: Station name for display and reports"),
        ("start_station_id", "Integer", "Origin station number", "10", "Kept: Target station ID to predict demand for"),
        ("end_date", "Object (String)", "Time when bike returned", "8/29/2013 09:09", "Removed: Future info (unknown before ride happens)"),
        ("end_station_name", "Object (String)", "Destination station name", "San Jose City Hall", "Removed: Future info (destination is unknown at start)"),
        ("end_station_id", "Integer", "Destination station number", "10", "Removed: Destination is unknown when predicting departures"),
        ("bike_id", "Integer", "Bicycle serial number", "520", "Removed: Individual bike ID is not needed for station-level demand prediction"),
        ("subscription_type", "Object (String)", "Subscriber or Customer", "Subscriber", "Removed: Belongs to individual rider, not hourly station total"),
        ("zip_code", "Object (String)", "Rider postal code", "94127 (6,619 missing)", "Removed: Has missing values and is personal to rider"),
        ("special_event", "Object (String)", "Holiday or event flag", "No Event, Halloween", "Kept: Tells if the day is a holiday across the city")
    ]

    for r_idx, row in enumerate(rows7):
        for c_idx, val in enumerate(row):
            cell = t7.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 0 else CARD_BG_ALT
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(8.5)
            if c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = PRIMARY_BLUE
            elif "Removed" in val:
                p.font.color.rgb = ACCENT_ROSE
            elif "Kept" in val:
                p.font.color.rgb = RGBColor(16, 149, 103)
                p.font.bold = True
            elif "Transformed" in val:
                p.font.color.rgb = PRIMARY_BLUE
                p.font.bold = True
            else:
                p.font.color.rgb = TEXT_MAIN

    bot7 = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.3), Inches(11.733), Inches(0.75))
    bot7.fill.solid()
    bot7.fill.fore_color.rgb = CARD_BG_ALT
    bot7.line.color.rgb = PRIMARY_BLUE
    bot7.line.width = Pt(1.2)
    b7tf = bot7.text_frame
    b7tf.word_wrap = True
    b7tf.margin_left = Inches(0.2)
    b7tf.margin_top = Inches(0.1)
    
    b7p = b7tf.paragraphs[0]
    b7p.text = "STAGE 3 SUMMARY: 12 raw columns filtered down to core station/time identifiers + target variable demand across all 1,230,810 rows."
    b7p.font.name = "Calibri"
    b7p.font.size = Pt(10.5)
    b7p.font.bold = True
    b7p.font.color.rgb = TEXT_MAIN

    set_notes(s7, """Slide 7 details Stage 3: Raw Dataset Column Selection, auditing why each of the 12 raw columns was kept, transformed, or removed.
• id: Removed, as it is simply an individual ride transaction receipt with no predictive power.
• duration: Removed, because it is future leakage—we only know duration after a trip finishes, which is unknown when predicting upcoming departures!
• start_date: Transformed, by flooring it to the hour to group rides into hourly departure buckets.
• start_station_name & start_station_id: Kept, as station identifiers.
• end_date, end_station_name, end_station_id: Removed, because destination is unknown when predicting departures at the start of an hour. Including them would introduce future target leakage.
• bike_id: Removed, because individual bike serial numbers do not affect aggregate station-level demand.
• subscription_type & zip_code: Removed, as they are individual rider attributes rather than station totals, and zip code has missing values.
• special_event: Kept, to identify holidays.
• Stage 3 Summary: Across all 1,230,810 rows, the 12 raw columns are filtered down to core station/time base identifiers.""")

    # =============================================================
    # SLIDE 8: FEATURE ENGINEERING (62 CANDIDATE FEATURES - BALANCED CARDS)
    # =============================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8, BG_LIGHT)
    add_header(s8, "Feature Engineering (62 Candidate Features)", "Creating Model Inputs", 8, "Stage 4: Feature Extraction (1,230,810 Rows, 62 Features)")

    eb = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.8))
    eb.fill.solid()
    eb.fill.fore_color.rgb = RGBColor(238, 242, 255)
    eb.line.color.rgb = PRIMARY_BLUE
    eb.line.width = Pt(1.2)
    etf = eb.text_frame
    etf.word_wrap = True
    etf.margin_left = Inches(0.2)
    etf.margin_top = Inches(0.08)
    
    ep0 = etf.paragraphs[0]
    ep0.text = "ILLUSTRATIVE EXAMPLE (Not actual model output):"
    ep0.font.name = "Arial"
    ep0.font.size = Pt(9.5)
    ep0.font.bold = True
    ep0.font.color.rgb = PRIMARY_BLUE
    
    ep1 = etf.add_paragraph()
    ep1.text = "Imagine forecasting demand for Tuesday, Oct 15, 2013 at 8:00 AM at Station #2 (San Jose Train Station). Below are the 5 feature groups with key examples:"
    ep1.font.name = "Calibri"
    ep1.font.size = Pt(10.5)
    ep1.font.bold = True
    ep1.font.color.rgb = TEXT_MAIN

    f_groups = [
        ("1. Calendar & Flags (14)", PRIMARY_BLUE, [
            ("hour (0–23):", "Direct hour of day; captures daily commute cycles."),
            ("weekday (0–6):", "Separates Mon–Fri workdays from weekend leisure."),
            ("morning_peak (1):", "Flags 7:00–9:00 AM morning rush hour departures."),
            ("is_weekend (0/1):", "Binary flag isolating Saturday & Sunday behavior.")
        ], "Captures daily commuter routines and peak hours."),
        ("2. Cyclical Sine/Cos (8)", ACCENT_TEAL, [
            ("The Clock Gap:", "23:00 and 00:00 are 1h apart, but arithmetic sees 23!"),
            ("hour_sin & cos:", "Maps 24 hours onto a continuous 360° circular face."),
            ("weekday_sin & cos:", "Connects Sunday night smoothly into Monday morning."),
            ("month_sin & cos:", "Connects December into January for yearly seasons.")
        ], "Preserves seamless continuous time without jumps."),
        ("3. Interactions (4)", ACCENT_AMBER, [
            ("weekday_hour:", "is_weekday × hour: gives weight to workday rush hours."),
            ("weekend_hour:", "is_weekend × hour: captures weekend midday leisure curves."),
            ("month_hour:", "month × hour: adapts rush hour intensity by season."),
            ("peak_station:", "Flags rush hours specifically at busy transit hubs.")
        ], "Separates busy Tuesday 8 AM from calm Sunday 8 AM."),
        ("4. Past Demand Lags (13)", PRIMARY_BLUE, [
            ("lag_1h (12 trips):", "Departures in previous 1h (captures immediate momentum)."),
            ("lag_2h (4 trips):", "Departures 2 hours ago (shows rising or falling trend)."),
            ("lag_24h (14 trips):", "Departures yesterday at same hour (daily baseline)."),
            ("lag_168h (15 trips):", "Departures same hour last week (weekly habit baseline).")
        ], "Uses recent and recurring past trips as direct anchors."),
        ("5. Rolling Averages (15)", ACCENT_ROSE, [
            ("rolling_mean_3h (5.7):", "Moving average over past 3h; filters single spikes."),
            ("rolling_mean_24h (2.1):", "Past full day's average hourly trip rate."),
            ("rolling_max_24h (16):", "Maximum single-hour surge recorded in the last 24h."),
            ("lag_1h_ratio_24h (5.71):", "Surge ratio showing station running hotter than normal.")
        ], "Detects sudden real-time spikes above daily baseline.")
    ]

    for i, (title, col, items, takeaway) in enumerate(f_groups):
        f_left = Inches(0.8 + i * 2.38)
        fc = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, f_left, Inches(2.45), Inches(2.26), Inches(4.7))
        fc.fill.solid()
        fc.fill.fore_color.rgb = CARD_BG
        fc.line.color.rgb = BORDER_COLOR
        fc.line.width = Pt(1.5)

        fst = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, f_left, Inches(2.45), Inches(2.26), Inches(0.1))
        fst.fill.solid()
        fst.fill.fore_color.rgb = col
        fst.line.fill.background()

        ftf = fc.text_frame
        ftf.word_wrap = True
        ftf.margin_left = ftf.margin_right = Inches(0.12)
        ftf.margin_top = Inches(0.16)

        p = ftf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(4)

        for label, desc in items:
            p = ftf.add_paragraph()
            p.text = f"• {label} "
            p.font.name = "Arial"
            p.font.size = Pt(8.5)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
            r = p.add_run()
            r.text = desc
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            r.font.bold = False
            r.font.color.rgb = RGBColor(51, 65, 85)
            p.space_after = Pt(2)

        p_end = ftf.add_paragraph()
        p_end.text = f"Takeaway: {takeaway}"
        p_end.font.name = "Calibri"
        p_end.font.size = Pt(8.5)
        p_end.font.italic = True
        p_end.font.bold = True
        p_end.font.color.rgb = PRIMARY_BLUE

    set_notes(s8, """Slide 8 covers Stage 4: Feature Engineering, where we extract 62 candidate features across 1,230,810 rows.
Consider an illustrative scenario: predicting demand for Tuesday, Oct 15, 2013 at 8:00 AM at Station #2 (San Jose Train Station).
• Card 1 - Calendar & Flags (14 features): Captures hour (8), weekday (Tuesday), is_weekend (0), and morning_peak (1). This informs the model that a busy Tuesday morning rush is occurring.
• Card 2 - Cyclical Sine/Cosine (8 features): Solves the clock gap problem. In linear numbers, 11 PM (23) and 12 AM (0) are 23 units apart, but in reality they are 1 hour apart! Sine and cosine place 24 hours on a continuous 360-degree circle so time flows seamlessly without artificial boundary jumps.
• Card 3 - Temporal Interactions (4 features): weekday_hour (1 × 8 = 8) gives heavy weight to workday rush hours, while weekend_hour (0 × 8 = 0) isolates Sunday morning traffic.
• Card 4 - Past Demand Lags (13 features): lag_1h captures immediate 1-hour momentum (12 trips at 7 AM). lag_24h captures demand yesterday at 8 AM (14 trips). lag_168h captures demand last Tuesday at 8 AM (15 trips). Past demand provides strong baseline anchors.
• Card 5 - Rolling Window Statistics (15 features): rolling_mean_3h (5.7 trips/hr) filters single-hour spikes. rolling_mean_24h (2.1 trips/hr) provides a daily baseline. lag_1h_ratio_24h (5.71) indicates the station is running much hotter than normal.
• Stage 4 Summary: 1,230,810 rows by 62 candidate features generated.""")

    # =============================================================
    # SLIDE 9: FINAL FEATURE LEAKAGE AUDIT (STAGE 5: 54 FEATURES)
    # =============================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9, BG_LIGHT)
    add_header(s9, "Feature Audit: Removing Future Information", "Data Leakage Check", 9, "Stage 5: Leakage Audit (62 → 54 Features)")

    c9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.5))
    c9.fill.solid()
    c9.fill.fore_color.rgb = CARD_BG
    c9.line.color.rgb = BORDER_COLOR
    c9.line.width = Pt(1.5)

    tf9 = c9.text_frame
    tf9.word_wrap = True
    tf9.margin_left = tf9.margin_right = Inches(0.25)
    tf9.margin_top = Inches(0.25)

    p = tf9.paragraphs[0]
    p.text = "WHAT IS DATA LEAKAGE?"
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(4)

    p1 = tf9.add_paragraph()
    p1.text = "Data leakage happens when a feature accidentally uses information that would not be available in real life at prediction time."
    p1.font.name = "Calibri"
    p1.font.size = Pt(10.5)
    p1.font.color.rgb = TEXT_MAIN
    p1.space_after = Pt(10)

    p2 = tf9.add_paragraph()
    p2.text = "WHY DID WE REMOVE 8 FEATURES?"
    p2.font.name = "Arial"
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_ROSE
    p2.space_after = Pt(4)

    audit_pts = [
        "1. Removed 3 Demand Deltas: Features like demand_change_1h used the formula (current_demand - lag). But when we make a prediction at 8:00 AM, we do not yet know the demand between 8:00 AM and 9:00 AM!",
        "2. Removed 5 Global Averages: Features like historical_mean averaged the entire 2-year dataset. That meant 2013 rows were using future numbers from 2015!"
    ]
    for apt in audit_pts:
        p = tf9.add_paragraph()
        p.text = apt
        p.font.name = "Calibri"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    p3 = tf9.add_paragraph()
    p3.text = "RESULT: 62 candidate features - 8 removed = 54 final features. All 54 features strictly use information available before the prediction time."
    p3.font.name = "Calibri"
    p3.font.size = Pt(10.5)
    p3.font.bold = True
    p3.font.color.rgb = RGBColor(16, 149, 103)

    tbl_s9 = s9.shapes.add_table(4, 3, Inches(6.6), Inches(1.55), Inches(5.9), Inches(4.2))
    t9 = tbl_s9.table
    t9.columns[0].width = Inches(1.8)
    t9.columns[1].width = Inches(2.1)
    t9.columns[2].width = Inches(2.0)

    h9 = ["Removed Features", "Why It Was Cheating", "Real-Life Truth"]
    for c_idx, h in enumerate(h9):
        cell = t9.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    rows9 = [
        ("3 Demand Deltas\n(demand_change_1h, etc.)", "Used the current hour's target answer in the formula", "At 8:00 AM, you cannot know the demand at 9:00 AM!"),
        ("5 Global Averages\n(historical_mean, etc.)", "Used future 2015 data to calculate averages for 2013", "In 2013, you cannot see data from 2015!"),
        ("Strict Lags Checked\n(lag_1h, 24h, 168h)", "Verified lags only look backwards (t-1, t-24, t-168)", "Completely valid because past records already happened.")
    ]
    for r_idx, row in enumerate(rows9):
        for c_idx, val in enumerate(row):
            cell = t9.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 0 else CARD_BG_ALT
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_MAIN

    badge9 = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(5.95), Inches(5.9), Inches(1.1))
    badge9.fill.solid()
    badge9.fill.fore_color.rgb = RGBColor(236, 253, 245)
    badge9.line.color.rgb = RGBColor(16, 185, 129)
    badge9.line.width = Pt(1.2)
    b9tf = badge9.text_frame
    b9tf.word_wrap = True
    b9tf.margin_left = Inches(0.2)
    b9tf.margin_top = Inches(0.12)
    bp = b9tf.paragraphs[0]
    bp.text = "STAGE 5 SUMMARY: AUDIT PASSED"
    bp.font.name = "Arial"
    bp.font.size = Pt(10)
    bp.font.bold = True
    bp.font.color.rgb = RGBColor(16, 149, 103)
    bp1 = b9tf.add_paragraph()
    bp1.text = "1,230,810 rows with 54 final features. All features strictly use information available prior to the prediction timestamp."
    bp1.font.name = "Calibri"
    bp1.font.size = Pt(9.5)
    bp1.font.color.rgb = TEXT_MAIN

    set_notes(s9, """Slide 9 presents Stage 5: our Feature Leakage Audit.
Data leakage happens when a feature accidentally uses information that would not be available in real life at prediction time. If uncorrected, the model achieves artificial 99% accuracy in training but fails completely in deployment.
We audited all 62 features and removed 8 non-causal features:
• 1. Removed 3 Demand Deltas: Features like demand_change_1h used the formula (current_demand - lag_1h). But when predicting at 8:00 AM, current demand is the target answer we are trying to predict! Using it was cheating.
• 2. Removed 5 Global Averages: Features like historical_mean averaged demand over the entire two-year dataset (2013 to 2015). That meant 2013 records were using future numbers from 2015!
• Strict Lags Verified: We confirmed that lag_1h, lag_24h, and lag_168h look strictly backwards in time.
• Stage 5 Summary: 62 candidate features minus 8 removed equals 54 final features. All 54 features strictly use information available prior to the prediction timestamp.""")

    # =============================================================
    # SLIDE 10: CHRONOLOGICAL SPLIT (STAGE 6)
    # =============================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10, BG_LIGHT)
    add_header(s10, "Splitting Data by Time (70% / 15% / 15%)", "Train / Validation / Test", 10, "Stage 6: Chronological Partition")

    wb = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.9))
    wb.fill.solid()
    wb.fill.fore_color.rgb = RGBColor(254, 242, 242)
    wb.line.color.rgb = ACCENT_ROSE
    wb.line.width = Pt(1.2)
    wtf = wb.text_frame
    wtf.word_wrap = True
    wtf.margin_left = Inches(0.2)
    wtf.margin_top = Inches(0.1)

    wp0 = wtf.paragraphs[0]
    wp0.text = "WHY WE DO NOT USE RANDOM SHUFFLING IN TIME SERIES:"
    wp0.font.name = "Arial"
    wp0.font.size = Pt(10)
    wp0.font.bold = True
    wp0.font.color.rgb = ACCENT_ROSE

    wp1 = wtf.add_paragraph()
    wp1.text = "If you randomly shuffle time data, the model uses Wednesday to predict Tuesday, or June to predict May! It memorizes future answers to predict the past. We must split strictly by date."
    wp1.font.name = "Calibri"
    wp1.font.size = Pt(10.5)
    wp1.font.color.rgb = TEXT_MAIN

    splits = [
        {
            "tag": "TRAIN SET (70%)",
            "rows": "861,560 rows × 54 features",
            "dates": "Aug 29, 2013 to Jan 24, 2015",
            "role": "The model learns normal patterns, rush hours, and station demand levels.",
            "color": PRIMARY_BLUE
        },
        {
            "tag": "VALIDATION SET (15%)",
            "rows": "184,590 rows × 54 features",
            "dates": "Jan 24, 2015 to May 14, 2015",
            "role": "Used to tune settings (tree depth, number of trees) without touching the test set.",
            "color": ACCENT_AMBER
        },
        {
            "tag": "TEST SET (15%)",
            "rows": "184,660 rows × 54 features",
            "dates": "May 14, 2015 to Aug 31, 2015",
            "role": "Kept untouched. Tests how well models perform on completely new, future months.",
            "color": ACCENT_TEAL
        }
    ]

    for i, sp in enumerate(splits):
        s_left = Inches(0.8 + i * 4.0)
        sc = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, s_left, Inches(2.55), Inches(3.733), Inches(4.5))
        sc.fill.solid()
        sc.fill.fore_color.rgb = CARD_BG
        sc.line.color.rgb = BORDER_COLOR
        sc.line.width = Pt(1.5)

        s_strip = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, s_left, Inches(2.55), Inches(3.733), Inches(0.1))
        s_strip.fill.solid()
        s_strip.fill.fore_color.rgb = sp["color"]
        s_strip.line.fill.background()

        sctf = sc.text_frame
        sctf.word_wrap = True
        sctf.margin_left = sctf.margin_right = Inches(0.2)
        sctf.margin_top = Inches(0.25)

        p = sctf.paragraphs[0]
        p.text = sp["tag"]
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = sp["color"]
        p.space_after = Pt(4)

        p = sctf.add_paragraph()
        p.text = sp["rows"]
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(4)

        p = sctf.add_paragraph()
        p.text = sp["dates"]
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = PRIMARY_BLUE
        p.space_after = Pt(14)

        p = sctf.add_paragraph()
        p.text = "WHAT THIS SET DOES:"
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(2)

        p = sctf.add_paragraph()
        p.text = sp["role"]
        p.font.name = "Calibri"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_MAIN

    set_notes(s10, """Slide 10 explains Stage 6: our Chronological Data Partitioning strategy.
• Why No Random Shuffling?: In standard tabular machine learning, random train_test_split is common. In time-series forecasting, this is strictly forbidden! Random shuffling allows the model to use Wednesday to predict Tuesday, or June to predict May. It memorizes future answers to predict the past.
• Strict Chronological Splits:
1. Training Set (70%): 861,560 rows from Aug 29, 2013 to Jan 24, 2015. The model learns baseline traffic, rush hour dynamics, and station profiles across 17 full months.
2. Validation Set (15%): 184,590 rows from Jan 24, 2015 to May 14, 2015. Used exclusively to tune hyperparameters like tree depth and learning rate without ever touching the test set.
3. Test Set (15%): 184,660 rows from May 14, 2015 to Aug 31, 2015. Kept strictly locked until final evaluation to test true real-world generalization on unseen future months.
• Stage 6 Summary: 54 causal features evaluated across Train (861k), Validation (184k), and Test (184k) rows.""")

    # =============================================================
    # SLIDE 11: MODEL SELECTION - HOW EACH MODEL WORKS
    # =============================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11, BG_LIGHT)
    add_header(s11, "Model Selection: How Each Model Works", "Algorithms Used", 11)

    models_s11 = [
        {
            "tag": "MODEL 1: RANDOM FOREST",
            "name": "Random Forest",
            "how": "HOW DOES IT WORK?\nBuilds many decision trees at the same time in parallel. Takes the average prediction of all trees.",
            "example": "EXAMPLE:\nTree 1 says: 14 trips\nTree 2 says: 16 trips\nTree 3 says: 15 trips\nAverage: (14 + 16 + 15) / 3 = 15 trips",
            "why": "WHY WE USE IT?\nAveraging many independent trees reduces errors and handles noisy data well.",
            "color": PRIMARY_BLUE
        },
        {
            "tag": "MODEL 2: XGBOOST",
            "name": "XGBoost",
            "how": "HOW DOES IT WORK?\nBuilds trees one after another in a line. Each new tree tries to correct the previous tree's mistake.",
            "example": "EXAMPLE:\nReal demand = 15 trips.\nTree 1 guesses 11 (mistake = +4).\nTree 2 tries to predict the +4 mistake!\nRepeats across 600 trees.",
            "why": "WHY WE USE IT?\nVery good at learning difficult patterns like rush hour spikes.",
            "color": ACCENT_TEAL
        },
        {
            "tag": "MODEL 3: CATBOOST",
            "name": "CatBoost",
            "how": "HOW DOES IT WORK?\nCatBoost builds trees one after another, similar to XGBoost. It uses ordered boosting to improve how the model learns.",
            "example": "ORDERED BOOSTING:\nCalculates residuals without target leakage, preventing overfitting on repetitive time data.",
            "why": "WHY WE USE IT?\nProvides another boosting model for comparison and gives fast predictions.",
            "color": ACCENT_AMBER
        }
    ]

    for i, m in enumerate(models_s11):
        m_left = Inches(0.8 + i * 4.0)
        c = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, m_left, Inches(1.55), Inches(3.733), Inches(5.5))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR
        c.line.width = Pt(1.5)

        st = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, m_left, Inches(1.55), Inches(3.733), Inches(0.1))
        st.fill.solid()
        st.fill.fore_color.rgb = m["color"]
        st.line.fill.background()

        ctf = c.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_right = Inches(0.22)
        ctf.margin_top = Inches(0.22)

        p = ctf.paragraphs[0]
        p.text = m["tag"]
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = m["color"]
        p.space_after = Pt(2)

        p1 = ctf.add_paragraph()
        p1.text = m["name"]
        p1.font.name = "Arial"
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN
        p1.space_after = Pt(8)

        p2 = ctf.add_paragraph()
        p2.text = m["how"]
        p2.font.name = "Calibri"
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MAIN
        p2.space_after = Pt(8)

        p3 = ctf.add_paragraph()
        p3.text = m["example"]
        p3.font.name = "Calibri"
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = RGBColor(51, 65, 85)
        p3.space_after = Pt(8)

        p4 = ctf.add_paragraph()
        p4.text = m["why"]
        p4.font.name = "Calibri"
        p4.font.size = Pt(10)
        p4.font.bold = True
        p4.font.color.rgb = PRIMARY_BLUE

    set_notes(s11, """Slide 11 details our Model Selection: how each of the three tree-based algorithms works.
• Model 1 - Random Forest (Bagging):
Random Forest builds hundreds of decision trees simultaneously in parallel. Each tree votes independently, and we take their average. For example, if Tree 1 predicts 14 trips, Tree 2 predicts 16, and Tree 3 predicts 15, the average prediction is 15 trips. Averaging independent trees reduces variance and handles noisy data well.
• Model 2 - XGBoost (Gradient Boosting):
XGBoost builds decision trees sequentially in a line, where each new tree directly predicts and corrects the previous tree's residual error. For example, if actual demand is 15 and Tree 1 predicts 11, the residual error is +4. Tree 2 is trained specifically to predict that +4 mistake! Repeating this over 600 trees makes it exceptionally powerful at learning sharp commuter rush spikes.
• Model 3 - CatBoost (Ordered Boosting):
CatBoost is also a sequential boosting algorithm. It uses ordered boosting to prevent target leakage during tree building. It provides an independent boosting benchmark and offers extremely fast prediction speeds during real-world inference.""")

    # =============================================================
    # SLIDE 12: RESULTS COMPARISON (RF FIRST, XGBOOST, CATBOOST)
    # =============================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_bg(s12, BG_LIGHT)
    add_header(s12, "Model Results on Test Set (184,660 Rows)", "Performance Comparison", 12)

    tbl_s12 = s12.shapes.add_table(4, 5, Inches(0.8), Inches(1.55), Inches(11.733), Inches(2.6))
    t12 = tbl_s12.table
    w12 = [Inches(2.8), Inches(2.0), Inches(2.3), Inches(2.3), Inches(2.333)]
    for idx, w in enumerate(w12):
        t12.columns[idx].width = w

    h12 = ["Model", "Features", "Test MAE", "Test RMSE", "Test R²"]
    for c_idx, h in enumerate(h12):
        cell = t12.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial"
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_LIGHT

    rows12 = [
        ("Random Forest", "54", "0.439206", "0.908465", "0.731390"),
        ("XGBoost", "54", "0.430550", "0.898213", "0.737418"),
        ("CatBoost", "54", "0.436126", "0.897665", "0.737738")
    ]

    for r_idx, row in enumerate(rows12):
        for c_idx, val in enumerate(row):
            cell = t12.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 0 else CARD_BG_ALT
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(10.5)
            if c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = PRIMARY_BLUE
            elif c_idx == 2 and "0.430550" in val:
                p.font.bold = True
            elif c_idx in [3, 4] and ("0.897665" in val or "0.737738" in val):
                p.font.bold = True
            else:
                p.font.color.rgb = TEXT_MAIN

    perf_cards = [
        ("1. Random Forest Performance", PRIMARY_BLUE, [
            "• Test MAE: 0.439206 | RMSE: 0.908465 | R²: 0.731390",
            "• Demonstrates solid performance, explaining about 73% of the demand variation.",
            "• Validates that bagging decision trees provides a reliable baseline on tabular data."
        ]),
        ("2. XGBoost Performance", ACCENT_TEAL, [
            "• Test MAE: 0.430550 | RMSE: 0.898213 | R²: 0.737418",
            "• Achieved the lowest MAE (0.430550) among all three models.",
            "• This means it has the lowest average absolute error on test predictions."
        ]),
        ("3. CatBoost Performance", ACCENT_AMBER, [
            "• Test MAE: 0.436126 | RMSE: 0.897665 | R²: 0.737738",
            "• Achieved the lowest RMSE (0.897665) and highest R² (0.737738).",
            "• Its lower RMSE means slightly lower error when larger errors are given more weight."
        ])
    ]

    for i, (title, col, bullets) in enumerate(perf_cards):
        c_left = Inches(0.8 + i * 4.0)
        c = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, c_left, Inches(4.45), Inches(3.733), Inches(2.6))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = BORDER_COLOR
        c.line.width = Pt(1.5)

        st = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, c_left, Inches(4.45), Inches(3.733), Inches(0.1))
        st.fill.solid()
        st.fill.fore_color.rgb = col
        st.line.fill.background()

        ctf = c.text_frame
        ctf.word_wrap = True
        ctf.margin_left = ctf.margin_right = Inches(0.2)
        ctf.margin_top = Inches(0.2)

        p = ctf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(4)

        for b in bullets:
            p = ctf.add_paragraph()
            p.text = b
            p.font.name = "Calibri"
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_MAIN
            p.space_after = Pt(2)

    set_notes(s12, """Slide 12 presents our final Model Results evaluated on 184,660 completely unseen test rows.
• Performance Table:
- Random Forest: Test MAE = 0.439206 | RMSE = 0.908465 | R² = 0.731390
- XGBoost: Test MAE = 0.430550 | RMSE = 0.898213 | R² = 0.737418
- CatBoost: Test MAE = 0.436126 | RMSE = 0.897665 | R² = 0.737738
• Detailed Analysis by Card:
1. Random Forest: Delivers solid baseline performance, explaining about 73.1% of demand variation. It confirms that bagging decision trees creates a reliable, stable baseline on tabular time features.
2. XGBoost: Achieved the lowest MAE (0.430550) among all three models. This means it has the lowest average absolute error on hourly trip predictions.
3. CatBoost: Achieved the lowest RMSE (0.897665) and highest R² (0.737738). Lower RMSE means better performance when penalizing large surge mistakes.
• Metric Interpretation: MAE of ~0.43 means our forecast is off by less than half a bike per station-hour on average. R² of ~0.74 means the models explain nearly 74% of the total variation in demand on unseen future data.""")

    # =============================================================
    # SLIDE 13: WHAT INFLUENCED THE PREDICTIONS? (TOP 5 FEATURES)
    # =============================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_bg(s13, BG_LIGHT)
    add_header(s13, "What Influenced the Predictions? (Top 5 Features)", "Model Explanation", 13)

    c13 = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.55), Inches(5.6), Inches(5.5))
    c13.fill.solid()
    c13.fill.fore_color.rgb = CARD_BG
    c13.line.color.rgb = BORDER_COLOR
    c13.line.width = Pt(1.5)

    tf13 = c13.text_frame
    tf13.word_wrap = True
    tf13.margin_left = tf13.margin_right = Inches(0.25)
    tf13.margin_top = Inches(0.25)

    p = tf13.paragraphs[0]
    p.text = "TOP 5 FEATURES ACROSS ALL MODELS"
    p.font.name = "Arial"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(8)

    top_list = [
        ("1. same_hour_previous_week:", "Demand at this station exactly 7 days ago at the same hour."),
        ("2. lag_168h:", "Demand 168 hours ago (same 7-day past demand)."),
        ("3. peak_hour:", "Whether the current hour is morning rush (7–9 AM) or evening rush (4–7 PM)."),
        ("4. lag_1h:", "How many bike trips started in the last 1 hour. Shows if the station is busy right now."),
        ("5. lag_24h / same_hour_prev_day:", "Demand yesterday at this same hour.")
    ]

    for name, desc in top_list:
        p = tf13.add_paragraph()
        p.text = f"• {name} "
        p.font.name = "Arial"
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN
        r = p.add_run()
        r.text = desc
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.bold = False
        r.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(6)

    p_take = tf13.add_paragraph()
    p_take.text = "Takeaway: Past demand and peak-hour information were important for the model predictions."
    p_take.font.name = "Calibri"
    p_take.font.size = Pt(10.5)
    p_take.font.bold = True
    p_take.font.color.rgb = RGBColor(16, 149, 103)

    if os.path.exists("reports/images/feature_importance_comparison.png"):
        s13.shapes.add_picture("reports/images/feature_importance_comparison.png", Inches(6.6), Inches(1.55), width=Inches(5.9), height=Inches(2.65))

    if os.path.exists("reports/images/actual_vs_predicted_catboost.png"):
        s13.shapes.add_picture("reports/images/actual_vs_predicted_catboost.png", Inches(6.6), Inches(4.35), width=Inches(5.9), height=Inches(2.7))

    set_notes(s13, """Slide 13 examines Feature Importance: what influenced the predictions across all models.
• Top 5 Features Across All Models:
1. same_hour_previous_week: Demand at this exact station 7 days ago at the same hour was the single most powerful predictor.
2. lag_168h: Demand 168 hours ago, reinforcing identical weekly commuter travel habits.
3. peak_hour: Morning rush (7 to 9 AM) and evening rush (4 to 7 PM) binary flags.
4. lag_1h: Departures in the immediately preceding hour, capturing real-time station momentum.
5. lag_24h / same_hour_prev_day: Demand yesterday at the same hour.
• Visual Insights:
The top chart confirms that all three algorithms place the heaviest weight on 7-day past demand and rush hour flags.
The bottom chart displays actual vs predicted departures for CatBoost on the test set. Notice that prediction errors are tightly and symmetrically clustered around zero, confirming consistent calibration without systematic bias.""")

    # =============================================================
    # SLIDE 14: NEXT PHASE PLAN
    # =============================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_bg(s14, BG_LIGHT)
    add_header(s14, "Next Phase: Implementation Plan", "Future Work", 14)

    c14 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.55), Inches(11.733), Inches(5.5))
    c14.fill.solid()
    c14.fill.fore_color.rgb = CARD_BG
    c14.line.color.rgb = BORDER_COLOR
    c14.line.width = Pt(1.5)

    tf14 = c14.text_frame
    tf14.word_wrap = True
    tf14.margin_left = tf14.margin_right = Inches(0.35)
    tf14.margin_top = Inches(0.3)

    p = tf14.paragraphs[0]
    p.text = "WHAT WE PLAN TO DO IN THE NEXT PHASE:"
    p.font.name = "Arial"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_BLUE
    p.space_after = Pt(12)

    next_steps = [
        ("1. Build Streamlit UI:", "Create an interactive and clean web interface using Python Streamlit."),
        ("2. Connect Trained Model:", "Load the trained model into the application backend to generate instant predictions."),
        ("3. Station and Time Selection:", "Allow users to pick a specific station, date, and hour from simple dropdowns."),
        ("4. Display Predicted Demand:", "Show the predicted number of bike checkouts clearly for that hour."),
        ("5. Show Demand Visualizations:", "Display simple charts showing historical trends and daily peak patterns."),
        ("6. Redistribution Support:", "Provide demand-based redistribution support to assist transit managers in planning."),
        ("7. Test the Complete Application:", "Run end-to-end testing with different inputs to ensure reliable behavior.")
    ]

    for title, desc in next_steps:
        p = tf14.add_paragraph()
        p.text = f"{title} "
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN
        r = p.add_run()
        r.text = desc
        r.font.name = "Calibri"
        r.font.size = Pt(11)
        r.font.bold = False
        r.font.color.rgb = RGBColor(51, 65, 85)
        p.space_after = Pt(8)

    set_notes(s14, """Slide 14 outlines our roadmap for the Next Phase of implementation:
1. Build Streamlit UI: Develop an interactive, user-friendly web interface using Python Streamlit.
2. Connect Trained Model: Load the serialized CatBoost/XGBoost model into the backend pipeline for instant inference.
3. Station and Time Dropdowns: Allow transit managers to select any of the 70 stations, a target date, and an hour from simple dropdowns.
4. Display Predicted Demand: Instantly display the forecasted number of bike checkouts for that station-hour.
5. Interactive Visualizations: Show 24-hour historical curves and expected peak rush hour demand.
6. Redistribution Support: Provide proactive rebalancing recommendations so transit operators can move bikes before shortages occur.
7. End-to-End Testing: Perform comprehensive testing across edge cases, holidays, and high-traffic stations to ensure production reliability.""")

    # =============================================================
    # SLIDE 15: THANK YOU
    # =============================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_bg(s15, BG_DARK)

    card15 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.0), Inches(10.333), Inches(5.5))
    card15.fill.solid()
    card15.fill.fore_color.rgb = RGBColor(30, 41, 59)
    card15.line.color.rgb = RGBColor(51, 65, 85)
    card15.line.width = Pt(1.5)

    tf15 = card15.text_frame
    tf15.word_wrap = True
    tf15.margin_top = Inches(0.5)

    p = tf15.paragraphs[0]
    p.text = "THANK YOU!"
    p.font.name = "Arial"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = TEXT_LIGHT
    p.alignment = PP_ALIGN.CENTER
    p.space_after = Pt(10)

    p1 = tf15.add_paragraph()
    p1.text = "Urban Bike Sharing Demand Forecasting"
    p1.font.name = "Arial"
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(56, 189, 248)
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(26)

    det15 = s15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.8), Inches(4.0), Inches(7.733), Inches(1.8))
    det15.fill.solid()
    det15.fill.fore_color.rgb = RGBColor(15, 23, 42)
    det15.line.color.rgb = PRIMARY_BLUE
    det15.line.width = Pt(1.2)

    dtf15 = det15.text_frame
    dtf15.word_wrap = True
    dtf15.margin_top = Inches(0.25)

    dp0 = dtf15.paragraphs[0]
    dp0.text = "STUDENT & GUIDE DETAILS"
    dp0.font.name = "Arial"
    dp0.font.size = Pt(10)
    dp0.font.bold = True
    dp0.font.color.rgb = RGBColor(56, 189, 248)
    dp0.alignment = PP_ALIGN.CENTER
    dp0.space_after = Pt(4)

    dp1 = dtf15.add_paragraph()
    dp1.text = "Isabella Dennis  (Roll No: 30)   |   Project Guide: Mr. Sibu Skaria"
    dp1.font.name = "Arial"
    dp1.font.size = Pt(13)
    dp1.font.bold = True
    dp1.font.color.rgb = TEXT_LIGHT
    dp1.alignment = PP_ALIGN.CENTER
    dp1.space_after = Pt(4)

    dp2 = dtf15.add_paragraph()
    dp2.text = "Department of Computer Applications   •   Mar Athanasius College of Engineering, Kothamangalam"
    dp2.font.name = "Calibri"
    dp2.font.size = Pt(11)
    dp2.font.color.rgb = RGBColor(148, 163, 184)
    dp2.alignment = PP_ALIGN.CENTER

    set_notes(s15, """Slide 15 concludes our presentation.
• Summary of Achievements:
We successfully resolved the zero-demand problem by expanding 669k trips into a continuous 1.23 million row grid, engineered 54 strictly causal features with zero leakage, and proved that tree ensemble models explain approximately 74% of demand variation on unseen future test data.
I would like to express my sincere gratitude to my project guide, Mr. Sibu Skaria, for his continuous guidance, and to the Department of Computer Applications.
Thank you, respected panel members. I am now ready to take your questions and discussion.""")

    out = "Urban_Bike_Sharing_Demand_Forecasting.pptx"
    try:
        prs.save(out)
        print(f"Updated 15-slide presentation saved to {out}")
    except PermissionError:
        import subprocess
        subprocess.run(["taskkill", "/F", "/IM", "POWERPNT.EXE"], capture_output=True)
        prs.save(out)
        print(f"PowerPoint process terminated and updated 15-slide presentation saved to {out}")

if __name__ == "__main__":
    create_presentation()
