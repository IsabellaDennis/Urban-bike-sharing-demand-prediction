import re

SLIDE_2_HTML = """      <!-- SLIDE 2: PROBLEM STATEMENT & OBJECTIVES (FRIEND'S SLIDE DESIGN) -->
      <section class="slide" id="slide-2" data-notes="Slide 2 establishes our Problem Statement and Key Project Objectives, structured according to our department guidelines.  • Top Left Card — Problem: In urban bike-sharing systems, commuters travel one-way during morning and evening rush hours. This causes severe spatial imbalance: residential stations run out of bikes in the morning, while commercial and transit hubs overflow. Fleet managers cannot reposition bikes effectively without knowing upcoming hourly departure demand in advance. Furthermore, raw logs only record active trips—missing quiet hours completely—which causes naive models to severely over-predict nighttime demand. Hence, our target is defined as hourly trip departures starting from each station.  • Top Right Card — Objective: Our primary objective is to develop an automated Machine Learning forecasting system that predicts station-level hourly bike departure demand across the network, resolving the zero-demand problem, enforcing strict causal leakage prevention, and providing actionable predictions for proactive fleet rebalancing.  • Bottom Row — 4 Key Project Objectives: 1. Explore &amp; Analyze (EDA): Audit data quality and identify weekday rush-hour peaks (8 AM &amp; 5 PM) versus weekend midday curves on 669k trips. 2. Expand Hourly Grid (GRID): Construct a continuous citywide grid of 1,230,810 station-hour records (70 stations × 17,583 hours) with explicit zero-demand handling. 3. Extract Causal Features (FE): Engineer 54 strictly causal features (cyclical sine/cos, lags, rolling averages) and eliminate future information leakage. 4. Benchmark &amp; Deploy (ML): Evaluate Random Forest, XGBoost, and CatBoost under strict chronological time splits and support proactive station rebalancing.">
        
        <!-- Header -->
        <div style="margin-bottom: 14px;">
          <div style="background: #0e7490; color: #fff; display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 10px; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px;">
            MCA MINI PROJECT &nbsp;|&nbsp; INTRODUCTION
          </div>
          <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <h2 style="font-size: 25px; font-weight: 800; color: #0f172a;">Problem Statement &amp; Objectives</h2>
            <div class="slide-num-tag">SLIDE 02 / 14</div>
          </div>
        </div>

        <!-- Body -->
        <div class="slide-body" style="flex-direction: column; gap: 14px;">
          
          <!-- Top Row: 2 Big Cards -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 18px;">
            
            <!-- Left Card: Problem -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column;">
              <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: #6b21a8; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 800;">!</div>
                <h3 style="font-size: 19px; font-weight: 800; color: #0f172a; margin: 0;">Problem</h3>
              </div>
              <p style="font-size: 12px; line-height: 1.55; color: #334155; margin: 0;">
                Urban bike-sharing networks suffer severe spatial and temporal imbalances during commuter rush hours. One-way trips deplete residential stations in the morning and cause dock congestion at commercial transit hubs. Furthermore, raw ride logs omit unrecorded quiet hours, making manual rebalancing inefficient and causing naive models to severely over-predict nighttime demand.
              </p>
            </div>

            <!-- Right Card: Objective -->
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column;">
              <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
                <div style="width: 38px; height: 38px; border-radius: 50%; background: #0f766e; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 800;">&rarr;</div>
                <h3 style="font-size: 19px; font-weight: 800; color: #0f172a; margin: 0;">Objective</h3>
              </div>
              <p style="font-size: 12px; line-height: 1.55; color: #334155; margin: 0;">
                Develop an automated Machine Learning forecasting system that predicts station-level hourly bike departure demand across the network, resolving the zero-demand problem, enforcing strict causal leakage prevention, and providing actionable predictions for proactive fleet rebalancing.
              </p>
            </div>

          </div>

          <!-- Section Subheading -->
          <div style="font-size: 15px; font-weight: 800; color: #0f172a; margin-top: 4px;">
            Key Project Objectives
          </div>

          <!-- Bottom Row: 4 Cards -->
          <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; flex: 1;">
            
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column; align-items: center; text-align: center;">
              <div style="width: 44px; height: 44px; border-radius: 50%; background: #6b21a8; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; margin-bottom: 10px;">EDA</div>
              <div style="font-size: 13.5px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Explore &amp; Analyze</div>
              <p style="font-size: 11px; line-height: 1.45; color: #475569; margin: 0;">
                Audit data quality and identify weekday rush-hour peaks (8 AM &amp; 5 PM) versus weekend midday curves on 669k trips.
              </p>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column; align-items: center; text-align: center;">
              <div style="width: 44px; height: 44px; border-radius: 50%; background: #0f766e; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 800; margin-bottom: 10px;">GRID</div>
              <div style="font-size: 13.5px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Expand Hourly Grid</div>
              <p style="font-size: 11px; line-height: 1.45; color: #475569; margin: 0;">
                Construct a continuous citywide grid of 1,230,810 station-hour records (70 stations &times; 17,583 hours) with zero-demand handling.
              </p>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column; align-items: center; text-align: center;">
              <div style="width: 44px; height: 44px; border-radius: 50%; background: #1e293b; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; margin-bottom: 10px;">FE</div>
              <div style="font-size: 13.5px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Extract Features</div>
              <p style="font-size: 11px; line-height: 1.45; color: #475569; margin: 0;">
                Engineer 54 strictly causal features (cyclical sine/cos, lags, rolling averages) and remove non-causal leakage.
              </p>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column; align-items: center; text-align: center;">
              <div style="width: 44px; height: 44px; border-radius: 50%; background: #7c3aed; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800; margin-bottom: 10px;">ML</div>
              <div style="font-size: 13.5px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">Benchmark &amp; Deploy</div>
              <p style="font-size: 11px; line-height: 1.45; color: #475569; margin: 0;">
                Evaluate Random Forest, XGBoost, and CatBoost chronologically; provide redistribution support for transit operators.
              </p>
            </div>

          </div>

          <!-- MACE Footer Bar -->
          <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10.5px; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 8px; margin-top: 4px;">
            <div>Department of Computer Applications | MACE</div>
            <div>Mar Athanasius College of Engineering, Kothamangalam</div>
            <div>Slide 2 of 14</div>
          </div>

        </div>
      </section>"""

def update_build_html():
    with open("build_html_presentation.py", "r", encoding="utf-8") as f:
        content = f.read()

    idx1 = content.find("<!-- SLIDE 2:")
    idx2 = content.find("<!-- SLIDE 3:")

    if idx1 != -1 and idx2 != -1:
        new_content = content[:idx1] + SLIDE_2_HTML + "\n\n      " + content[idx2:]
        with open("build_html_presentation.py", "w", encoding="utf-8") as f:
            f.write(new_content)
        print("build_html_presentation.py Slide 2 updated successfully!")
    else:
        print("Could not find Slide 2 markers in build_html_presentation.py")

update_build_html()
