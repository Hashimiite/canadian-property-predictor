import os

import joblib
import pandas as pd
import streamlit as st

import charts
from model import forecast

st.set_page_config(page_title="Canadian House Price Forecast", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');
    [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not(text):not(tspan),
    [data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]) {
        font-family: 'Geist', system-ui, sans-serif !important;
    }
    [data-testid="stAppViewContainer"] p.lede > strong:not(text),
    [data-testid="stSidebar"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"],
    [data-testid="stSidebar"] input { font-family: 'Geist Mono', ui-monospace, monospace !important; }
    .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1180px; }
    h1 { font-weight: 600 !important; letter-spacing: -0.035em; font-size: 2.6rem !important; margin-bottom: 0.2rem !important; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em; }
    .lede { color: #9a968f; font-size: 1.15rem; line-height: 1.6; max-width: 46em; margin: 0.4rem 0 1.4rem; }
    .lede strong { color: #ece9e4; font-weight: 500; font-size: 1.05rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #2b2926; }
    [data-testid="stSidebar"] h2 { font-size: 1.05rem !important; margin-top: 0.4rem; }
    [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label { color: #ece9e4 !important; font-weight: 500; }
    [data-testid="stCaptionContainer"] { color: #9a968f !important; }
    [data-testid="stExpander"] { border: 1px solid #2b2926 !important; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; }
    #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_latest_model():
    """Load the most recent model file"""
    model_files = [f for f in os.listdir(".") if f.endswith(".joblib") and f.startswith("housing_model_")]
    if not model_files:
        return None
    # Trusted input: this is the model file model.py writes in this project, never user uploaded
    return joblib.load(sorted(model_files)[-1])


def calculate_economic_impacts(interest_rate_change, crime_change, population_growth, economic_outlook):
    """Scenario assumptions that shift each year's predicted growth"""
    return {
        "interest_rate_impact": -0.02 * interest_rate_change,
        "crime_impact": -0.001 * crime_change,
        "population_growth_impact": 0.005 * population_growth,
        "economic_outlook_impact": 0.01 * economic_outlook,
    }


def generate_predictions_over_years(province, start_year, end_year, economic_factors, model_data, error=0.0):
    """Compound the model's predicted yearly growth from the region's latest price, shifted by the scenario.

    error is the model's typical yearly growth miss. Treating yearly misses as independent, the band widens
    with the square root of the number of years ahead.
    """
    code = model_data["label_encoder"].transform([province])[0]

    def adjust(g):
        g = g + economic_factors["interest_rate_impact"] + economic_factors["crime_impact"]
        return g * (1 + economic_factors["population_growth_impact"]) * (1 + economic_factors["economic_outlook_impact"])

    rows = forecast(model_data["model"], model_data["history"][province], code, end_year, adjust)
    last_year = max(model_data["history"][province])
    out = []
    for year, price, g in rows:
        spread = (year - last_year) ** 0.5
        if year >= start_year:
            out.append({
                "Year": year, "Predicted_Value": price, "Growth_Rate": g * 100,
                "Low": price * (1 - error) ** spread, "High": price * (1 + error) ** spread,
            })
    return pd.DataFrame(out)


def main():
    model_data = load_latest_model()
    if not model_data or "history" not in model_data:
        st.error("No trained model found. Run `python model.py` to train one, then reload this page.")
        st.stop()

    history_all = model_data["history"]
    last_year = max(max(h) for h in history_all.values())
    best = model_data.get("model_name", "Model")
    growth_error = model_data["comparison"][best].get("growth_mae_pts", 0) / 100

    with st.sidebar:
        st.header("Forecast settings")
        regions = ["Canada"] + sorted(p for p in history_all if p != "Canada")
        province = st.selectbox("Region", regions)
        col_a, col_b = st.columns(2)
        start_year = col_a.number_input("From", min_value=last_year + 1, max_value=2099, value=last_year + 1, step=1)
        end_year = col_b.number_input("To", min_value=last_year + 2, max_value=2100, value=last_year + 11, step=1,
                                      help="Forecasts compound year by year and can run as far as 2100.")
        if end_year <= start_year:
            end_year = start_year + 1
            st.caption(f"The end year moved to {end_year} so it comes after the start year.")

        st.header("Scenario")
        st.caption("These assumptions shift each year's predicted growth. They come from the scenario you set, not from the data.")
        interest_rate_change = st.slider("Interest rate change (points)", -5.0, 5.0, 0.0, 0.25)
        crime_change = st.slider("Crime rate change (%)", -20.0, 20.0, 0.0, 1.0)
        population_growth = st.slider("Population growth (%)", -5.0, 10.0, 1.5, 0.5)
        economic_outlook = st.slider("Economic outlook", -5, 5, 0, 1, help="From -5 (recession) to +5 (boom)")

    factors = calculate_economic_impacts(interest_rate_change, crime_change, population_growth, economic_outlook)
    predictions = generate_predictions_over_years(province, start_year, end_year, factors, model_data, growth_error)
    history = history_all[province]
    today, final = history[last_year], predictions.iloc[-1]
    years = int(final["Year"]) - last_year
    yearly = ((final["Predicted_Value"] / today) ** (1 / years) - 1) * 100

    st.title(f"House prices in {province}")
    st.markdown(
        f"<p class='lede'>The average new house was about <strong>${today:,.0f}</strong> in {last_year}. "
        f"Under this scenario the model expects about <strong>${final['Predicted_Value']:,.0f}</strong> by {int(final['Year'])}, "
        f"which works out to {yearly:.1f}% a year. Allowing for the model's typical yearly error, "
        f"the {int(final['Year'])} price most likely lands between ${final['Low']:,.0f} and ${final['High']:,.0f}.</p>",
        unsafe_allow_html=True,
    )
    st.pyplot(charts.forecast_band(history, predictions))

    with st.expander("Year by year forecast"):
        table = predictions.rename(columns={"Predicted_Value": "Forecast", "Growth_Rate": "Growth"})
        st.dataframe(
            table[["Year", "Forecast", "Low", "High", "Growth"]],
            hide_index=True, use_container_width=True,
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "Forecast": st.column_config.NumberColumn(format="$%.0f"),
                "Low": st.column_config.NumberColumn("Typical low", format="$%.0f"),
                "High": st.column_config.NumberColumn("Typical high", format="$%.0f"),
                "Growth": st.column_config.NumberColumn("Yearly growth", format="%.2f%%"),
            },
        )

    st.subheader("How the model performs")
    st.caption(
        f"Both models were tuned and trained on 1990 to 2017 and tested on 2018 onward, where {best} scored better. "
        "It predicts each year's growth from recent growth, so these scores describe one year ahead accuracy. "
        "Longer forecasts compound those predictions and grow less certain the further out they go."
    )
    tab_compare, tab_ab, tab_fit, tab_features = st.tabs(
        ["Model comparison", "A/B tests", "Actual vs predicted", "Feature importance"])
    with tab_compare:
        st.pyplot(charts.model_comparison(model_data["comparison"]))
    with tab_ab:
        st.caption(
            "Each model's current settings (A) were tested against settings tuned by grid search with year by year "
            "cross validation on 1990 to 2017 (B). A one sided paired Wilcoxon signed-rank test compares their price "
            "errors on the same 2018 onward rows. B ships only if its errors are significantly smaller (p < 0.05)."
        )
        st.pyplot(charts.ab_tests(model_data["ab_tests"]))
        st.dataframe(
            pd.DataFrame([
                {"Model": name, "A: current": t["mae_a"], "B: tuned": t["mae_b"], "p": t["p_value"],
                 "Ships": "B (tuned)" if t["winner"] == "B" else "A (current)",
                 "Tuned settings": ", ".join(f"{k}={v}" for k, v in t["params_b"].items())}
                for name, t in model_data["ab_tests"].items()
            ]),
            hide_index=True, use_container_width=True,
            column_config={
                "A: current": st.column_config.NumberColumn("A: current MAE", format="$%.0f"),
                "B: tuned": st.column_config.NumberColumn("B: tuned MAE", format="$%.0f"),
                "p": st.column_config.NumberColumn("p value", format="%.4f"),
            },
        )
    with tab_fit:
        st.pyplot(charts.actual_vs_predicted(model_data["test_actual"], model_data["test_predicted"], best))
    with tab_features:
        st.pyplot(charts.feature_importance(model_data["feature_names"], model_data["model"].feature_importances_, best))
    st.caption("Data: Statistics Canada new housing price index, used under the Open Government Licence (Canada).")


if __name__ == "__main__":
    main()
