import gradio as gr
import pandas as pd
import pickle
import numpy as np
import spaces

# 1. Load the Model
with open("player_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function


def format_compact(value):
    """Format a euro amount into a short K/M/B style string, e.g. 2.84M."""
    if value >= 1_000_000_000:
        return f"€{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"€{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"€{value / 1_000:.1f}K"
    else:
        return f"€{value:.0f}"


@spaces.GPU(duration=10)
def predict_market_value(age, height_in_cm, years_to_contract_expiry,
                         international_caps, international_goals,
                         position, sub_position, foot,
                         country_of_citizenship, competition):

    # Pack inputs into a DataFrame
    # The column names must match the training columns exactly
    input_df = pd.DataFrame([[
        age, height_in_cm, years_to_contract_expiry,
        international_caps, international_goals,
        position, sub_position, foot,
        country_of_citizenship, competition
    ]],
        columns=[
        'age', 'height_in_cm', 'years_to_contract_expiry',
        'international_caps', 'international_goals',
        'position', 'sub_position', 'foot',
        'country_of_citizenship', 'current_club_domestic_competition_id'
    ])

    # Predict (model was trained on log1p(market_value_in_eur))
    log_prediction = model.predict(input_df)[0]
    prediction_eur = np.clip(np.expm1(log_prediction), 0, None)

    full = f"€ {prediction_eur:,.0f}"
    compact = format_compact(prediction_eur)

    result_html = f"""
    <div style="text-align:center; padding: 18px;">
        <div style="font-size: 14px; color: #9ca3af; margin-bottom: 4px;">
            Estimated Market Value
        </div>
        <div style="font-size: 34px; font-weight: 700; color: #f59e0b;">
            {compact}
        </div>
        <div style="font-size: 15px; color: #9ca3af; margin-top: 4px;">
            {full}
        </div>
    </div>
    """
    return result_html


def clear_all():
    placeholder = """
    <div style="text-align:center; padding: 18px; color:#9ca3af;">
        Fill in the form and click <b>Predict Market Value</b>
    </div>
    """
    return (24, 180, 2.0, 0, 0, "Midfield", "Central Midfield", "right",
            "Germany", "GB1", placeholder)


# 3. Reference option lists

POSITIONS = ["Attack", "Defender", "Midfield", "Goalkeeper", "Missing"]
SUB_POSITIONS = [
    "Centre-Forward", "Second Striker", "Left Winger", "Right Winger",
    "Attacking Midfield", "Central Midfield", "Defensive Midfield",
    "Left Midfield", "Right Midfield", "Left-Back", "Right-Back",
    "Centre-Back", "Goalkeeper",
]
COUNTRIES = ["Germany", "England", "France", "Spain", "Italy", "Brazil",
             "Argentina", "Netherlands", "Portugal", "Turkey", "Other"]
LEAGUES = [
    ("Premier League (England)", "GB1"),
    ("La Liga (Spain)", "ES1"),
    ("Serie A (Italy)", "IT1"),
    ("Bundesliga (Germany)", "L1"),
    ("Ligue 1 (France)", "FR1"),
    ("Süper Lig (Turkey)", "TR1"),
    ("Eredivisie (Netherlands)", "NL1"),
    ("Primeira Liga (Portugal)", "PO1"),
    ("Jupiler Pro League (Belgium)", "BE1"),
    ("Other", "Other"),
]

# Handy presets so users don't have to fill everything from scratch
EXAMPLES = [
    [24, 182, 2.5, 0, 0, "Midfield", "Central Midfield", "right", "Germany", "L1"],
    [29, 175, 1.0, 80, 30, "Attack", "Centre-Forward", "right", "Argentina", "ES1"],
    [19, 178, 3.5, 0, 0, "Defender", "Centre-Back", "left", "England", "GB1"],
    [33, 188, 0.5, 45, 0, "Goalkeeper", "Goalkeeper", "right", "Germany", "L1"],
]

# 4. The App Interface

with gr.Blocks(title="Player Market Value Predictor") as app:
    gr.Markdown(
        """
        # ⚽ Football Player Market Value Predictor
        Estimate a player's market value (in EUR) from their profile and career stats,
        using a Random Forest model trained on Transfermarkt data.
        Fill in the details below, or click an example to auto-fill the form.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### 👤 Player Profile")
                age = gr.Slider(15, 45, value=24, step=1, label="Age",
                                info="Player's current age in years")
                height_in_cm = gr.Slider(
                    150, 210, value=180, step=1, label="Height (cm)")
                foot = gr.Radio(["right", "left", "both", "Unknown"],
                                value="right", label="Preferred Foot")
                country_of_citizenship = gr.Dropdown(
                    COUNTRIES, value="Germany", label="Country of Citizenship")

            with gr.Group():
                gr.Markdown("#### 🏟️ Position & Club")
                position = gr.Dropdown(
                    POSITIONS, value="Midfield", label="Position")
                sub_position = gr.Dropdown(SUB_POSITIONS, value="Central Midfield",
                                           label="Sub-position",
                                           info="More specific role on the pitch")
                competition = gr.Dropdown(
                    LEAGUES, value="GB1", label="Current Club's League")
                years_to_contract_expiry = gr.Slider(
                    0, 6, value=2, step=0.5, label="Years to Contract Expiry",
                    info="Time remaining on current contract")

            with gr.Group():
                gr.Markdown("#### 🌍 International Career")
                international_caps = gr.Number(
                    label="International Caps", value=0, minimum=0,
                    info="Appearances for national team")
                international_goals = gr.Number(
                    label="International Goals", value=0, minimum=0)

            gr.Examples(
                examples=EXAMPLES,
                inputs=[age, height_in_cm, years_to_contract_expiry,
                        international_caps, international_goals,
                        position, sub_position, foot,
                        country_of_citizenship, competition],
                label="Try an example player",
            )

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button(
                    "Predict Market Value", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("#### 💰 Prediction")
            output = gr.HTML(
                value="""
                <div style="text-align:center; padding: 18px; color:#9ca3af;">
                    Fill in the form and click <b>Predict Market Value</b>
                </div>
                """
            )
            gr.Markdown(
                """
                <div style="font-size: 12px; color: #9ca3af;">
                Note: This is a statistical estimate based on historical data and should
                not be treated as an official valuation.
                </div>
                """
            )

    inputs_list = [age, height_in_cm, years_to_contract_expiry,
                   international_caps, international_goals,
                   position, sub_position, foot,
                   country_of_citizenship, competition]

    submit_btn.click(fn=predict_market_value,
                     inputs=inputs_list, outputs=output)
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=inputs_list + [output],
    )

if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft(primary_hue="orange"))
