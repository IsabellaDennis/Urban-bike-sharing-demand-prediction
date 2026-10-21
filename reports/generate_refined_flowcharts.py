"""
generate_refined_flowcharts.py
Generates Figures 3.20 - 3.23 with:
- Substantial, well-proportioned box heights (NO slim / compressed ribbons)
- Clearly visible, prominent arrow shafts with sharp, crisp arrowheads
- Generous vertical spacing between all containers in the Project Pipeline
- Zero collision between arrows and container titles / text
- Defensible descriptions (Interactive Demand Inference, Pickle serialization)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Typography
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

reports_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(reports_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Color Scheme
COLOR_START_EDGE = "#0D9488"  # Teal
COLOR_START_FILL = "#E6FFFA"
COLOR_PROC_EDGE  = "#1E40AF"  # Blue
COLOR_PROC_FILL  = "#EEF2FF"
COLOR_BOOST_EDGE = "#D97706"  # Amber
COLOR_BOOST_FILL = "#FEF3C7"
COLOR_CHAMP_EDGE = "#059669"  # Emerald
COLOR_CHAMP_FILL = "#D1FAE5"
COLOR_TEXT       = "#0F172A"  # Dark navy/slate
COLOR_ARROW      = "#1E293B"  # Deep slate for clear, prominent contrast

def draw_arrow(ax, p_from, p_to, lw=1.8, mutation_scale=12):
    ax.annotate(
        "", xy=p_to, xytext=p_from,
        arrowprops=dict(
            arrowstyle="-|>",
            color=COLOR_ARROW,
            lw=lw,
            mutation_scale=mutation_scale,
            shrinkA=0,
            shrinkB=0
        )
    )

def save_diagram(fig, basename):
    for target_dir in [reports_dir, images_dir]:
        pdf_path = os.path.join(target_dir, f"{basename}.pdf")
        png_path = os.path.join(target_dir, f"{basename}.png")
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.08)
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f"Generated {basename}.pdf and {basename}.png successfully.")

# -----------------------------------------------------------------------------
# 1. Random Forest Flowchart (diagram_3)
# -----------------------------------------------------------------------------
def make_random_forest():
    # Width 6.8 in, Height 9.6 in - gives substantial box thickness and long arrows
    fig, ax = plt.subplots(figsize=(6.8, 9.6), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    items = [
        ("start", "START", 0.048, True),
        ("proc", "Load Historical Trip Dataset", 0.042, False),
        ("proc", "Preprocess Data and Build Station-Hour Matrix", 0.042, False),
        ("proc", "Train using Random Forest Regressor", 0.042, True),
        ("big", "1. Draw bootstrap samples with replacement (N = 861,560)\n2. For each decision tree:\n    a) Randomly select feature subset (max_features = sqrt)\n    b) Build regression tree (max_depth = 15, min_samples_leaf = 2)\n3. Repeat for N = 200 de-correlated trees\n4. Aggregate predictions by arithmetic averaging", 0.165, False),
        ("proc", "Test on Chronological Test Partition (15%)", 0.042, False),
        ("proc", "Evaluate Model Performance (MAE, RMSE, $R^2$)", 0.042, False),
        ("proc", "Input New Station and Temporal Features", 0.042, False),
        ("proc", "Predict Hourly Station-Level Departure Demand", 0.042, False),
        ("start", "END", 0.048, True)
    ]
    
    total_h = sum(item[2] for item in items)
    gap = (0.92 - total_h) / (len(items) - 1)
    cur_y = 0.96
    box_coords = []
    
    for kind, text, h, bold in items:
        center_y = cur_y - h / 2.0
        pad = 0.005
        
        if kind == "start":
            box = patches.FancyBboxPatch(
                (0.18, cur_y - h), 0.64, h,
                boxstyle=f"round,pad={pad},rounding_size=0.018",
                ec=COLOR_START_EDGE, fc=COLOR_START_FILL, lw=1.8
            )
            ax.add_patch(box)
            ax.text(0.50, center_y, text, ha='center', va='center',
                    fontsize=12, fontweight='bold', color=COLOR_TEXT)
        elif kind == "proc":
            box = patches.FancyBboxPatch(
                (0.06, cur_y - h), 0.88, h,
                boxstyle=f"round,pad={pad},rounding_size=0.014",
                ec=COLOR_PROC_EDGE, fc=COLOR_PROC_FILL, lw=1.6
            )
            ax.add_patch(box)
            weight = 'bold' if bold else 'normal'
            ax.text(0.50, center_y, text, ha='center', va='center',
                    fontsize=11.2, fontweight=weight, color=COLOR_TEXT)
        else: # big box
            box = patches.FancyBboxPatch(
                (0.04, cur_y - h), 0.92, h,
                boxstyle=f"round,pad={pad},rounding_size=0.016",
                ec=COLOR_BOOST_EDGE, fc="#FFFBEB", lw=1.8
            )
            ax.add_patch(box)
            ax.text(0.08, center_y, text, ha='left', va='center',
                    fontsize=10.5, fontfamily='serif', color=COLOR_TEXT, linespacing=1.38)
            
        top_y = cur_y + pad
        bot_y = (cur_y - h) - pad
        box_coords.append((top_y, bot_y))
        cur_y -= (h + gap)
        
    for i in range(len(box_coords) - 1):
        y_from = box_coords[i][1]
        y_to = box_coords[i+1][0]
        draw_arrow(ax, (0.50, y_from), (0.50, y_to), lw=1.8, mutation_scale=12)
        
    save_diagram(fig, "diagram_3")

# -----------------------------------------------------------------------------
# 2. XGBoost Flowchart (diagram_4)
# -----------------------------------------------------------------------------
def make_xgboost():
    # Width 6.8 in, Height 11.2 in - ensures substantial box thickness for all 13 steps
    fig, ax = plt.subplots(figsize=(6.8, 11.2), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    steps = [
        ("START", COLOR_START_EDGE, COLOR_START_FILL, 0.044, True),
        ("Load Historical Trip Dataset", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Preprocess Data & Engineer 54 Features", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Initialize XGBoost Regressor", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, True),
        ("Make Initial Prediction (Base Demand = 0.5443)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Build Decision Tree (max_depth = 10, reg_lambda = 1.0)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Calculate Gradient & Residual Prediction Errors", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Build Next Tree to Reduce Errors (learning_rate = 0.04)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Repeat for 600 Boosting Rounds", COLOR_BOOST_EDGE, COLOR_BOOST_FILL, 0.042, True),
        ("Final XGBoost Model Ensemble", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, True),
        ("Evaluate on Test Partition (MAE, RMSE, $R^2$)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("Predict Hourly Station-Level Departure Demand", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.038, False),
        ("END", COLOR_START_EDGE, COLOR_START_FILL, 0.044, True)
    ]
    
    total_h = sum(s[3] for s in steps)
    gap = (0.92 - total_h) / (len(steps) - 1)
    cur_y = 0.96
    box_coords = []
    
    for text, edge, fill, h, bold in steps:
        center_y = cur_y - h / 2.0
        is_term = (text in ["START", "END"])
        w = 0.60 if is_term else 0.88
        x = (1.0 - w) / 2.0
        pad = 0.005
        
        box = patches.FancyBboxPatch(
            (x, cur_y - h), w, h,
            boxstyle=f"round,pad={pad},rounding_size=0.014",
            ec=edge, fc=fill, lw=1.8 if bold else 1.5
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(0.50, center_y, text, ha='center', va='center',
                fontsize=11.5 if is_term else 10.6,
                fontweight=weight, color=COLOR_TEXT)
        
        top_y = cur_y + pad
        bot_y = (cur_y - h) - pad
        box_coords.append((top_y, bot_y))
        cur_y -= (h + gap)
        
    for i in range(len(box_coords) - 1):
        y_from = box_coords[i][1]
        y_to = box_coords[i+1][0]
        draw_arrow(ax, (0.50, y_from), (0.50, y_to), lw=1.8, mutation_scale=12)
        
    save_diagram(fig, "diagram_4")

# -----------------------------------------------------------------------------
# 3. CatBoost Flowchart (diagram_5)
# -----------------------------------------------------------------------------
def make_catboost():
    # Width 6.8 in, Height 10.8 in - substantial box thickness and comfortable padding
    fig, ax = plt.subplots(figsize=(6.8, 10.8), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    steps = [
        ("START", COLOR_START_EDGE, COLOR_START_FILL, 0.046, True),
        ("Load Historical Trip Dataset", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("Preprocess Data & Construct Hourly Station Grid", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("Initialize CatBoost Regressor", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, True),
        ("Prepare Data Using Ordered Boosting Permutations", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("Build Symmetric Oblivious Decision Tree (depth = 8)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("Update Model Using Regularized Errors (l2_leaf_reg = 3.0)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("Repeat for 600 Iterations (learning_rate = 0.05)", COLOR_BOOST_EDGE, COLOR_BOOST_FILL, 0.046, True),
        ("Final CatBoost Model Ensemble", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, True),
        ("Evaluate on Test Partition (MAE, RMSE, $R^2$)", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("Predict Hourly Station-Level Departure Demand", COLOR_PROC_EDGE, COLOR_PROC_FILL, 0.042, False),
        ("END", COLOR_START_EDGE, COLOR_START_FILL, 0.046, True)
    ]
    
    total_h = sum(s[3] for s in steps)
    gap = (0.92 - total_h) / (len(steps) - 1)
    cur_y = 0.96
    box_coords = []
    
    for text, edge, fill, h, bold in steps:
        center_y = cur_y - h / 2.0
        is_term = (text in ["START", "END"])
        w = 0.60 if is_term else 0.88
        x = (1.0 - w) / 2.0
        pad = 0.005
        
        box = patches.FancyBboxPatch(
            (x, cur_y - h), w, h,
            boxstyle=f"round,pad={pad},rounding_size=0.014",
            ec=edge, fc=fill, lw=1.8 if bold else 1.5
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(0.50, center_y, text, ha='center', va='center',
                fontsize=11.5 if is_term else 10.6,
                fontweight=weight, color=COLOR_TEXT)
        
        top_y = cur_y + pad
        bot_y = (cur_y - h) - pad
        box_coords.append((top_y, bot_y))
        cur_y -= (h + gap)
        
    for i in range(len(box_coords) - 1):
        y_from = box_coords[i][1]
        y_to = box_coords[i+1][0]
        draw_arrow(ax, (0.50, y_from), (0.50, y_to), lw=1.8, mutation_scale=12)
        
    save_diagram(fig, "diagram_5")

# -----------------------------------------------------------------------------
# 4. Project Workflow Diagram (fig_3_17_project_architecture)
# -----------------------------------------------------------------------------
def make_project_pipeline():
    # Width 8.8 in, Height 10.6 in
    # Provides clear visible gaps (spaces) between Level containers
    # Zero text collisions with arrows
    fig, ax = plt.subplots(figsize=(8.8, 10.8), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # -------------------------------------------------------------
    # 3 LEVEL CONTAINERS WITH GENEROUS VISIBLE MARGINS IN BETWEEN
    # -------------------------------------------------------------
    # Level 1: [0.720, 0.985] (h = 0.265)
    # Gap between L1 bottom (0.720) and L2 top (0.670) = 0.050 (~0.54 in space)
    # Level 2: [0.370, 0.670] (h = 0.300)
    # Gap between L2 bottom (0.370) and L3 top (0.320) = 0.050 (~0.54 in space)
    # Level 3: [0.020, 0.320] (h = 0.300)
    
    l1 = patches.FancyBboxPatch((0.02, 0.720), 0.96, 0.265, boxstyle="round,pad=0.008,rounding_size=0.020",
                                ec="#2563EB", fc="#EFF6FF", lw=1.8)
    ax.add_patch(l1)
    ax.text(0.50, 0.960, "LEVEL 1: DATA PREPARATION", ha='center', va='center',
            fontsize=12, fontweight='bold', color="#1E3A8A")
    
    l2 = patches.FancyBboxPatch((0.02, 0.370), 0.96, 0.300, boxstyle="round,pad=0.008,rounding_size=0.020",
                                ec="#059669", fc="#ECFDF5", lw=1.8)
    ax.add_patch(l2)
    # Centered at 0.50 with zero collision since connector enters Feature Engineering at x=0.20
    ax.text(0.50, 0.648, "LEVEL 2: MODEL BUILDING", ha='center', va='center',
            fontsize=12, fontweight='bold', color="#065F46")
    
    l3 = patches.FancyBboxPatch((0.02, 0.020), 0.96, 0.300, boxstyle="round,pad=0.008,rounding_size=0.020",
                                ec="#D97706", fc="#FFFBEB", lw=1.8)
    ax.add_patch(l3)
    # Centered at 0.50 with zero collision since model convergence enters Test Evaluation at x=0.20
    ax.text(0.50, 0.298, "LEVEL 3: EVALUATION & DEPLOYMENT", ha='center', va='center',
            fontsize=12, fontweight='bold', color="#92400E")
    
    def draw_node(x, y, w, h, text, fontsize=9.4, ec="#1E293B", fc="#FFFFFF", bold=True):
        pad = 0.004
        b = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.014",
                                   ec=ec, fc=fc, lw=1.4)
        ax.add_patch(b)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2.0, y + h/2.0, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=COLOR_TEXT)
        return {
            'left': x - pad, 'right': x + w + pad,
            'top': y + h + pad, 'bot': y - pad,
            'cx': x + w/2.0, 'cy': y + h/2.0
        }
    
    # LEVEL 1 NODES (thick, well-proportioned boxes: h = 0.054)
    n_trip = draw_node(0.08, 0.865, 0.24, 0.054, "trip.csv\n(669,959 Trips)")
    n_clean = draw_node(0.38, 0.865, 0.24, 0.054, "Data Cleaning &\nOutlier Filtering")
    n_demand = draw_node(0.68, 0.865, 0.24, 0.054, "Hourly Station-\nLevel Demand")
    
    n_grid = draw_node(0.08, 0.745, 0.24, 0.054, "Complete Station-\nHour Grid")
    n_zero = draw_node(0.38, 0.745, 0.24, 0.054, "Zero-Demand\nFilling")
    n_eda  = draw_node(0.68, 0.745, 0.24, 0.054, "Exploratory Data\nAnalysis (EDA)")
    
    # Level 1 Arrows (Row 1)
    draw_arrow(ax, (n_trip['right'], n_trip['cy']), (n_clean['left'], n_clean['cy']))
    draw_arrow(ax, (n_clean['right'], n_clean['cy']), (n_demand['left'], n_demand['cy']))
    
    # Stepped connector from Hourly Demand down to Complete Station-Hour Grid
    y_corridor1 = 0.825
    ax.plot([n_demand['cx'], n_demand['cx']], [n_demand['bot'], y_corridor1], color=COLOR_ARROW, lw=1.8)
    ax.plot([n_demand['cx'], n_grid['cx']], [y_corridor1, y_corridor1], color=COLOR_ARROW, lw=1.8)
    draw_arrow(ax, (n_grid['cx'], y_corridor1), (n_grid['cx'], n_grid['top']))
    
    # Level 1 Arrows (Row 2)
    draw_arrow(ax, (n_grid['right'], n_grid['cy']), (n_zero['left'], n_zero['cy']))
    draw_arrow(ax, (n_zero['right'], n_zero['cy']), (n_eda['left'], n_eda['cy']))
    
    # LEVEL 2 NODES
    n_fe = draw_node(0.08, 0.548, 0.24, 0.052, "Feature Engineering\n(54 Features)")
    n_leak = draw_node(0.38, 0.548, 0.24, 0.052, "Data Leakage\nAudit")
    n_split = draw_node(0.68, 0.548, 0.24, 0.052, "Chronological Split\n(70% - 15% - 15%)")
    
    # Connect EDA down to Feature Engineering (traverses open gap between L1 and L2 at y=0.695)
    y_inter_1_2 = 0.695
    ax.plot([n_eda['cx'], n_eda['cx']], [n_eda['bot'], y_inter_1_2], color=COLOR_ARROW, lw=1.8)
    ax.plot([n_eda['cx'], n_fe['cx']], [y_inter_1_2, y_inter_1_2], color=COLOR_ARROW, lw=1.8)
    # Drops straight into Feature Engineering on the left (x=0.20), perfectly avoiding Level 2 title at x=0.50!
    draw_arrow(ax, (n_fe['cx'], y_inter_1_2), (n_fe['cx'], n_fe['top']))
    
    # Level 2 Arrows (Row 1)
    draw_arrow(ax, (n_fe['right'], n_fe['cy']), (n_leak['left'], n_leak['cy']))
    draw_arrow(ax, (n_leak['right'], n_leak['cy']), (n_split['left'], n_split['cy']))
    
    # Row 2 (3 Models)
    n_rf = draw_node(0.08, 0.398, 0.24, 0.052, "Random Forest\n(200 Trees)", fontsize=9.3)
    n_xgb = draw_node(0.38, 0.398, 0.24, 0.052, "Tuned XGBoost\n(600 Rounds)", fontsize=9.3, ec="#059669", fc="#F0FDF4")
    n_cat = draw_node(0.68, 0.398, 0.24, 0.052, "CatBoost\n(600 Iterations)", fontsize=9.3)
    
    # Branching from Chronological Split down to 3 models
    y_split_bar = 0.495
    ax.plot([n_split['cx'], n_split['cx']], [n_split['bot'], y_split_bar], color=COLOR_ARROW, lw=1.8)
    ax.plot([n_rf['cx'], n_cat['cx']], [y_split_bar, y_split_bar], color=COLOR_ARROW, lw=1.8)
    draw_arrow(ax, (n_rf['cx'], y_split_bar), (n_rf['cx'], n_rf['top']))
    draw_arrow(ax, (n_xgb['cx'], y_split_bar), (n_xgb['cx'], n_xgb['top']))
    draw_arrow(ax, (n_cat['cx'], y_split_bar), (n_cat['cx'], n_cat['top']))
    
    # LEVEL 3 NODES
    # Row 1: Test Evaluation, Metric Comparison, Selected Model
    n_test = draw_node(0.08, 0.208, 0.24, 0.052, "Test Evaluation\n(184,660 Records)", fontsize=9.2)
    n_comp = draw_node(0.38, 0.208, 0.24, 0.052, "Metric Comparison\n(MAE, RMSE, $R^2$)", fontsize=9.2)
    n_champ_sel = draw_node(0.68, 0.208, 0.24, 0.052, "Selected Model\n(Tuned XGBoost)", fontsize=9.2)
    
    # Converge 3 candidate models cleanly into Test Evaluation:
    # 1. Straight vertical connectors downward from all 3 model boxes
    y_eval_bar = 0.345
    ax.plot([n_rf['cx'], n_rf['cx']], [n_rf['bot'], y_eval_bar], color=COLOR_ARROW, lw=1.6)
    ax.plot([n_xgb['cx'], n_xgb['cx']], [n_xgb['bot'], y_eval_bar], color=COLOR_ARROW, lw=1.6)
    ax.plot([n_cat['cx'], n_cat['cx']], [n_cat['bot'], y_eval_bar], color=COLOR_ARROW, lw=1.6)
    
    # 2. Neat horizontal collection line in the gap between Level 2 and Level 3
    ax.plot([n_rf['cx'], n_cat['cx']], [y_eval_bar, y_eval_bar], color=COLOR_ARROW, lw=1.6)
    
    # 3. One downward arrow entering Test Evaluation box
    draw_arrow(ax, (n_test['cx'], y_eval_bar), (n_test['cx'], n_test['top']))
    
    # Row 1 horizontal arrows
    draw_arrow(ax, (n_test['right'], n_test['cy']), (n_comp['left'], n_comp['cy']))
    draw_arrow(ax, (n_comp['right'], n_comp['cy']), (n_champ_sel['left'], n_champ_sel['cy']))
    
    # Row 2: Champion Metrics Box (concise verified test results without repeating model name)
    n_champ = draw_node(0.16, 0.118, 0.68, 0.050, r"Test Results: $\mathbf{R^2 = 0.7374}$ | MAE = 0.4305 | RMSE = 0.8982",
                        fontsize=9.8, ec=COLOR_CHAMP_EDGE, fc=COLOR_CHAMP_FILL)
    
    # Step down from Champion Selection (at 0.80) to Champion Box (center 0.50)
    y_c2 = 0.186
    ax.plot([n_champ_sel['cx'], n_champ_sel['cx']], [n_champ_sel['bot'], y_c2], color=COLOR_ARROW, lw=1.8)
    ax.plot([n_champ_sel['cx'], n_champ['cx']], [y_c2, y_c2], color=COLOR_ARROW, lw=1.8)
    draw_arrow(ax, (n_champ['cx'], y_c2), (n_champ['cx'], n_champ['top']))
    
    # Row 3: Deployment & Redistribution Dual Outcomes
    n_deploy = draw_node(0.06, 0.035, 0.42, 0.046, "Streamlit Dashboard Deployment\n(Interactive Demand Inference)", fontsize=9.2)
    n_redist = draw_node(0.52, 0.035, 0.42, 0.046, "Feature Importance &\nFleet Redistribution Support", fontsize=9.2)
    
    y_champ_bar = 0.095
    ax.plot([n_champ['cx'], n_champ['cx']], [n_champ['bot'], y_champ_bar], color=COLOR_ARROW, lw=1.8)
    ax.plot([n_deploy['cx'], n_redist['cx']], [y_champ_bar, y_champ_bar], color=COLOR_ARROW, lw=1.8)
    draw_arrow(ax, (n_deploy['cx'], y_champ_bar), (n_deploy['cx'], n_deploy['top']))
    draw_arrow(ax, (n_redist['cx'], y_champ_bar), (n_redist['cx'], n_redist['top']))
    
    save_diagram(fig, "fig_3_17_project_architecture")

if __name__ == "__main__":
    print("Generating perfected flowcharts with robust box thickness and zero text collision...")
    make_random_forest()
    make_xgboost()
    make_catboost()
    make_project_pipeline()
    print("All diagrams generated successfully!")
