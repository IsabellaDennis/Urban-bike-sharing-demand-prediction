import re

# New implementation for Slide 2 in generate_presentation.py matching the user's friend's slide design
SLIDE_2_PY = '''    # =============================================================
    # SLIDE 2: PROBLEM STATEMENT & OBJECTIVES (FRIEND'S SLIDE DESIGN)
    # =============================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, BG_LIGHT)

    # 1. Top Pill Badge: MCA MINI PROJECT | INTRODUCTION
    pill = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(3.4), Inches(0.38))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(14, 116, 144)  # Cyan/Teal
    pill.line.fill.background()
    ptf = pill.text_frame
    ptf.margin_top = Inches(0.04)
    pp = ptf.paragraphs[0]
    pp.text = "MCA MINI PROJECT  |  INTRODUCTION"
    pp.font.name = "Arial"
    pp.font.size = Pt(9.5)
    pp.font.bold = True
    pp.font.color.rgb = TEXT_LIGHT
    pp.alignment = PP_ALIGN.CENTER

    # 2. Main Title: Problem Statement & Objectives
    tb_title = s2.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(10.0), Inches(0.55))
    ttf = tb_title.text_frame
    ttf.word_wrap = True
    ttf.margin_left = ttf.margin_top = ttf.margin_right = ttf.margin_bottom = 0
    tp = ttf.paragraphs[0]
    tp.text = "Problem Statement & Objectives"
    tp.font.name = "Arial"
    tp.font.size = Pt(22)
    tp.font.bold = True
    tp.font.color.rgb = RGBColor(15, 23, 42)

    # Slide counter at top right
    s2_num = s2.shapes.add_textbox(Inches(11.4), Inches(0.4), Inches(1.1), Inches(0.35))
    s2_ntf = s2_num.text_frame
    s2_np = s2_ntf.paragraphs[0]
    s2_np.text = f"02 / {TOTAL_SLIDES:02d}"
    s2_np.font.name = "Arial"
    s2_np.font.size = Pt(11)
    s2_np.font.bold = True
    s2_np.font.color.rgb = TEXT_MUTED
    s2_np.alignment = PP_ALIGN.RIGHT

    # 3. Top Row: 2 Big Cards (Problem on Left, Objective on Right)
    top_cards = [
        {
            "icon_char": "!",
            "icon_color": RGBColor(107, 33, 168), # Purple
            "title": "Problem",
            "text": "Urban bike-sharing networks suffer severe spatial and temporal imbalances during commuter rush hours. One-way trips deplete residential stations in the morning and cause dock congestion at commercial transit hubs. Furthermore, raw ride logs omit unrecorded quiet hours, making manual rebalancing inefficient and causing naive models to severely over-predict nighttime demand.",
            "left": Inches(0.8),
            "width": Inches(5.72)
        },
        {
            "icon_char": "→",
            "icon_color": RGBColor(15, 118, 110), # Dark Teal
            "title": "Objective",
            "text": "Develop an automated Machine Learning forecasting system that predicts station-level hourly bike departure demand across the network, resolving the zero-demand problem, enforcing strict causal leakage prevention, and providing actionable predictions for proactive fleet rebalancing.",
            "left": Inches(6.8),
            "width": Inches(5.72)
        }
    ]

    for tc in top_cards:
        # Card container
        card_box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, tc["left"], Inches(1.5), tc["width"], Inches(2.1))
        card_box.fill.solid()
        card_box.fill.fore_color.rgb = CARD_BG
        card_box.line.color.rgb = RGBColor(226, 232, 240)
        card_box.line.width = Pt(1.5)

        # Circular Icon
        circle = s2.shapes.add_shape(MSO_SHAPE.OVAL, tc["left"] + Inches(0.28), Inches(1.72), Inches(0.46), Inches(0.46))
        circle.fill.solid()
        circle.fill.fore_color.rgb = tc["icon_color"]
        circle.line.fill.background()
        ctf = circle.text_frame
        ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
        cp = ctf.paragraphs[0]
        cp.text = tc["icon_char"]
        cp.font.name = "Arial"
        cp.font.size = Pt(15)
        cp.font.bold = True
        cp.font.color.rgb = TEXT_LIGHT
        cp.alignment = PP_ALIGN.CENTER

        # Title text next to icon
        tb_hdr = s2.shapes.add_textbox(tc["left"] + Inches(0.85), Inches(1.72), tc["width"] - Inches(1.1), Inches(0.46))
        htf = tb_hdr.text_frame
        htf.margin_left = htf.margin_top = htf.margin_right = htf.margin_bottom = 0
        hp = htf.paragraphs[0]
        hp.text = tc["title"]
        hp.font.name = "Arial"
        hp.font.size = Pt(16)
        hp.font.bold = True
        hp.font.color.rgb = RGBColor(15, 23, 42)

        # Description text below
        tb_body = s2.shapes.add_textbox(tc["left"] + Inches(0.28), Inches(2.3), tc["width"] - Inches(0.56), Inches(1.15))
        btf = tb_body.text_frame
        btf.word_wrap = True
        btf.margin_left = btf.margin_top = btf.margin_right = btf.margin_bottom = 0
        bp = btf.paragraphs[0]
        bp.text = tc["text"]
        bp.font.name = "Calibri"
        bp.font.size = Pt(10)
        bp.font.color.rgb = RGBColor(51, 65, 85)

    # 4. Mid Section Heading: Key Project Objectives
    tb_sub = s2.shapes.add_textbox(Inches(0.8), Inches(3.78), Inches(6.0), Inches(0.35))
    stf = tb_sub.text_frame
    stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = 0
    sp = stf.paragraphs[0]
    sp.text = "Key Project Objectives"
    sp.font.name = "Arial"
    sp.font.size = Pt(13.5)
    sp.font.bold = True
    sp.font.color.rgb = RGBColor(15, 23, 42)

    # 5. Bottom Row: 4 Equal Cards
    bot_objectives = [
        {
            "tag": "EDA",
            "tag_color": RGBColor(107, 33, 168), # Purple
            "title": "Explore & Analyze",
            "desc": "Audit data quality and identify weekday rush-hour peaks (8 AM & 5 PM) versus weekend midday curves on 669k trips."
        },
        {
            "tag": "GRID",
            "tag_color": RGBColor(15, 118, 110), # Teal
            "title": "Expand Hourly Grid",
            "desc": "Construct a continuous citywide grid of 1,230,810 station-hour records (70 stations × 17,583 hours) with zero-demand handling."
        },
        {
            "tag": "FE",
            "tag_color": RGBColor(30, 41, 59), # Dark Navy
            "title": "Extract Features",
            "desc": "Engineer 54 strictly causal features (cyclical sine/cos, lags, rolling averages) and remove non-causal leakage."
        },
        {
            "tag": "ML",
            "tag_color": RGBColor(124, 58, 237), # Indigo/Purple
            "title": "Benchmark & Deploy",
            "desc": "Evaluate Random Forest, XGBoost, and CatBoost chronologically; provide redistribution support for transit operators."
        }
    ]

    for i, bo in enumerate(bot_objectives):
        b_left = Inches(0.8 + i * 3.0)
        b_width = Inches(2.733)
        b_card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, b_left, Inches(4.2), b_width, Inches(2.35))
        b_card.fill.solid()
        b_card.fill.fore_color.rgb = CARD_BG
        b_card.line.color.rgb = RGBColor(226, 232, 240)
        b_card.line.width = Pt(1.5)

        # Centered circle tag icon
        circle_tag = s2.shapes.add_shape(MSO_SHAPE.OVAL, b_left + Inches(1.1), Inches(4.38), Inches(0.52), Inches(0.52))
        circle_tag.fill.solid()
        circle_tag.fill.fore_color.rgb = bo["tag_color"]
        circle_tag.line.fill.background()
        ctf = circle_tag.text_frame
        ctf.margin_left = ctf.margin_top = ctf.margin_right = ctf.margin_bottom = 0
        cp = ctf.paragraphs[0]
        cp.text = bo["tag"]
        cp.font.name = "Arial"
        cp.font.size = Pt(8.5)
        cp.font.bold = True
        cp.font.color.rgb = TEXT_LIGHT
        cp.alignment = PP_ALIGN.CENTER

        # Subtitle
        tb_st = s2.shapes.add_textbox(b_left + Inches(0.15), Inches(5.0), b_width - Inches(0.3), Inches(0.35))
        stf = tb_st.text_frame
        stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = 0
        stp = stf.paragraphs[0]
        stp.text = bo["title"]
        stp.font.name = "Arial"
        stp.font.size = Pt(11)
        stp.font.bold = True
        stp.font.color.rgb = RGBColor(15, 23, 42)
        stp.alignment = PP_ALIGN.CENTER

        # Description
        tb_sd = s2.shapes.add_textbox(b_left + Inches(0.18), Inches(5.38), b_width - Inches(0.36), Inches(1.05))
        sdtf = tb_sd.text_frame
        sdtf.word_wrap = True
        sdtf.margin_left = sdtf.margin_top = sdtf.margin_right = sdtf.margin_bottom = 0
        sdp = sdtf.paragraphs[0]
        sdp.text = bo["desc"]
        sdp.font.name = "Calibri"
        sdp.font.size = Pt(9.5)
        sdp.font.color.rgb = RGBColor(71, 85, 105)
        sdp.alignment = PP_ALIGN.CENTER

    # 6. Footer (MACE format matching friend's slide)
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
    frp.text = "Slide 2 of 14"
    frp.font.name = "Calibri"
    frp.font.size = Pt(9.5)
    frp.font.color.rgb = RGBColor(100, 116, 139)
    frp.alignment = PP_ALIGN.RIGHT

    set_notes(s2, """Slide 2 establishes our Problem Statement and Key Project Objectives, structured according to our department guidelines.

• Top Left Card &mdash; Problem:
In urban bike-sharing systems, commuters travel one-way during morning and evening rush hours. This causes severe spatial imbalance: residential stations run out of bikes in the morning, while commercial and transit hubs overflow. Fleet managers cannot reposition bikes effectively without knowing upcoming hourly departure demand in advance. Furthermore, raw logs only record active trips—missing quiet hours completely—which causes naive models to severely over-predict nighttime demand. Hence, our target is defined as hourly trip departures starting from each station.

• Top Right Card &mdash; Objective:
Our primary objective is to develop an automated Machine Learning forecasting system that predicts station-level hourly bike departure demand across the network, resolving the zero-demand problem, enforcing strict causal leakage prevention, and providing actionable predictions for proactive fleet rebalancing.

• Bottom Row &mdash; 4 Key Project Objectives:
1. Explore & Analyze (EDA): Audit data quality and identify weekday rush-hour peaks (8 AM & 5 PM) versus weekend midday curves on 669k trips.
2. Expand Hourly Grid (GRID): Construct a continuous citywide grid of 1,230,810 station-hour records (70 stations × 17,583 hours) with explicit zero-demand handling.
3. Extract Causal Features (FE): Engineer 54 strictly causal features (cyclical sine/cos, lags, rolling averages) and eliminate future information leakage.
4. Benchmark & Deploy (ML): Evaluate Random Forest, XGBoost, and CatBoost under strict chronological time splits and support proactive station rebalancing.""")
'''

def update_presentation_py():
    with open("generate_presentation.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Find Slide 2 in generate_presentation.py and replace it
    start_marker = "# SLIDE 2: PROBLEM STATEMENT"
    # Find start
    idx1 = content.find("# =============================================================\n    # SLIDE 2:")
    idx2 = content.find("# =============================================================\n    # SLIDE 3:")

    if idx1 != -1 and idx2 != -1:
        new_content = content[:idx1] + SLIDE_2_PY + "\n" + content[idx2:]
        with open("generate_presentation.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        print("generate_presentation.py Slide 2 updated successfully!")
    else:
        print("Could not locate Slide 2 markers in generate_presentation.py")

update_presentation_py()
