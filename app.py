import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import charts
from model import forecast

st.set_page_config(page_title="Canadian Property Value Predictor", layout="wide")

@st.cache_resource
def load_latest_model():
    """Load the most recent model file"""
    model_files = [f for f in os.listdir('.') if f.endswith('.joblib') and f.startswith('housing_model_')]
    if not model_files:
        return None
    model = sorted(model_files)[-1]
    try:
        model_data = joblib.load(model)
        return model_data, model
    except:
        return None, None

def calculate_economic_impacts(interest_rate_change, crime_change, population_growth, economic_outlook):
    """Calculate impacts impact on housing prices"""
    interest_rate_impact = -0.02 * interest_rate_change
    crime_impact = -0.001 * crime_change
    population_growth_impact = 0.005 * population_growth
    economic_outlook_impact = 0.01 * economic_outlook
    return {
        'interest_rate_impact': interest_rate_impact,
        'crime_impact': crime_impact,
        'population_growth_impact': population_growth_impact,
        'economic_outlook_impact': economic_outlook_impact
    }

def generate_predictions_over_years(province, start_year, end_year, economic_factors, model_data):
    """Compound the model's predicted yearly growth from the province's latest price, shifted by the scenario."""
    code = model_data["label_encoder"].transform([province])[0]

    def adjust(g):
        g = g + economic_factors["interest_rate_impact"] + economic_factors["crime_impact"]
        return g * (1 + economic_factors["population_growth_impact"]) * (1 + economic_factors["economic_outlook_impact"])

    rows = forecast(model_data["model"], model_data["history"][province], code, end_year, adjust)
    return pd.DataFrame(
        [{"Year": y, "Predicted_Value": price, "Growth_Rate": g * 100} for y, price, g in rows if y >= start_year]
    )


def create_line_chart(prediction_data, province, value_label, history=None):
    """Create the line chart"""
    
    fig = go.Figure()
    if history:
        years = [y for y in sorted(history) if y >= 1990]
        fig.add_trace(go.Scatter(
            x=years,
            y=[history[y] for y in years],
            mode='lines',
            name='Since 1990',
            line=dict(color='#9a9a9a', width=2),
            hovertemplate='Year: %{x}<br>Estimated price: $%{y:,.0f}<extra></extra>'
        ))
    fig.add_trace(go.Scatter(
        x=prediction_data['Year'],
        y=prediction_data['Predicted_Value'],
        mode='lines+markers',
        name=f'{value_label}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate='Year: %{x}<br>' + f'{value_label}: $%{{y:,.0f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{value_label} Forecast for {province}',
        xaxis_title='Year',
        yaxis_title=f'{value_label} (CAD)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        showlegend=bool(history),
        legend=dict(orientation='h', y=1.08),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_yaxes(
        tickprefix='$',
        tickformat=',.0f'
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    return fig

def main():
    model_info = load_latest_model()
    # Initialize variables
    model_data = None
    data_type = 'house_prices'

    if model_info and model_info[0]:
        model_data, model_file = model_info
        data_type = model_data.get('data_type', 'house_prices')
    if not model_data or "history" not in model_data:
        st.error("No trained model found. Run `python model.py` first.")
        st.stop()
    
    # Determine value label based on data type
    if 'house' in str(data_type).lower():
        value_label = "House Price"
        unit = "CAD"
        title = "Canadian House Price Predictor"
    else:
        value_label = "Property Value"
        unit = "CAD"
        title = "Canadian Property Value Predictor"
    
    st.title(title)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Prediction Settings")
        # Province selection
        province = st.selectbox(
            "Select Province",
            ["Canada", "Ontario", "Quebec", "British Columbia", "Alberta",
             "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
             "Newfoundland and Labrador", "Prince Edward Island"]
        )
        # Year range selection
        first_forecast_year = max(max(h) for h in model_data["history"].values()) + 1
        st.markdown("Forecast Period")
        col1a, col1b = st.columns(2)
        with col1a:
            start_year = st.number_input(
                "Start Year",
                min_value=first_forecast_year,
                max_value=2099,
                value=first_forecast_year,
                step=1
            )
        with col1b:
            end_year = st.number_input(
                "End Year",
                min_value=first_forecast_year + 1,
                max_value=2100,
                value=first_forecast_year + 10,
                step=1,
                help="Forecasts compound year by year and can run as far as 2100."
            )
            
        if end_year <= start_year:
            st.warning("End year must be greater than start year. Adjusting...")
            end_year = start_year + 1
        
        # Economic factors
        st.markdown("---")
        st.markdown("Economic Factors")
        interest_rate_change = st.slider(
            "Interest Rate Change (%)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="Higher rates typically lower property values."
        )
        crime_change = st.slider(
            "Crime Rate Change (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Higher crime typically lowers property values."
        )
        population_growth = st.slider(
            "Population Growth (%)",
            min_value=-5.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help="Higher growth typically increases property values."
        )
        economic_outlook = st.slider(
            "Economic Outlook",
            min_value=-5,
            max_value=5,
            value=0,
            step=1,
            help="General economic outlook from -5 (recession) to +5 (boom)"
        )
        predict_button = st.button("Generate Forecast", type="primary", use_container_width=True)
    
    with col2:
        st.subheader(" Forecast Over Time")
        # Placeholder for charts and predictions
        chart_placeholder = st.empty()# Placeholder for chart
        details_placeholder = st.empty()
        if predict_button:
            with st.spinner("Generating forecast..."):
                economic_factors = calculate_economic_impacts(
                    interest_rate_change,
                    crime_change,
                    population_growth,
                    economic_outlook
                )
                predictions_df = generate_predictions_over_years( # Generate predictions for the year range
                    province,  
                    start_year, 
                    end_year,
                    economic_factors,
                    model_data,
                )
                end_year_prediction = predictions_df[predictions_df['Year'] == end_year].iloc[0]
                start_year_prediction = predictions_df[predictions_df['Year'] == start_year].iloc[0]
                fig = create_line_chart(predictions_df, province, value_label, model_data['history'][province])
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                with details_placeholder.container():
                    st.markdown("---")
                    st.subheader("Forecast Summary")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        total_change = end_year_prediction['Predicted_Value'] - start_year_prediction['Predicted_Value']
                        pct_change = (total_change / start_year_prediction['Predicted_Value']) * 100
                        st.metric(
                            f"{value_label} ({end_year})",
                            f"${end_year_prediction['Predicted_Value']:,.0f}",
                            delta=f"{pct_change:.1f}% from {start_year}"
                        )
                    with col_b:
                        annual_growth_rate = ((end_year_prediction['Predicted_Value'] / start_year_prediction['Predicted_Value']) ** (1/(end_year - start_year)) - 1) * 100
                        st.metric(
                            "Annual Growth Rate",
                            f"{annual_growth_rate:.1f}%",
                            delta="per year"
                        )
                    with col_c:
                        compound_effect = ((1 + economic_factors['interest_rate_impact'] + # Show compound impact
                                          economic_factors['crime_impact'] + 
                                          economic_factors['population_growth_impact'] + 
                                          economic_factors['economic_outlook_impact']) - 1) * 100
                        st.metric(
                            "Economic Impact",
                            f"{compound_effect:.1f}%"
                        )
                    # Show detailed table
                    with st.expander("View Detailed Forecast Table"):
                        display_df = predictions_df.copy()
                        display_df['Predicted_Value'] = display_df['Predicted_Value'].apply(lambda x: f"${x:.0f}")
                        display_df['Growth_Rate'] = display_df['Growth_Rate'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    # Show factors breakdown
                    with st.expander("View Economic Factors Breakdown"):
                        factors_df = pd.DataFrame({
                            'Factor': ['Interest Rate Change', 'Crime Rate Change', 'Population Growth', 'Economic Outlook'],
                            'Impact (%)': [
                                economic_factors['interest_rate_impact'] * 100,
                                economic_factors['crime_impact'] * 100,
                                economic_factors['population_growth_impact'] * 100,
                                economic_factors['economic_outlook_impact'] * 100
                            ],
                            'Effect': [
                                'Negative' if economic_factors['interest_rate_impact'] < 0 else 'Positive',
                                'Negative' if economic_factors['crime_impact'] < 0 else 'Positive',
                                'Positive' if economic_factors['population_growth_impact'] > 0 else 'Negative',
                                'Positive' if economic_factors['economic_outlook_impact'] > 0 else 'Negative'
                            ]
                        })
                        st.dataframe(factors_df, use_container_width=True, hide_index=True)
        else: # Show instruction
            with chart_placeholder.container():
                st.info("Adjust the settings and click 'Generate Forecast' to update this chart")
                preview_factors = calculate_economic_impacts(
                    interest_rate_change, crime_change, population_growth, economic_outlook
                )
                preview = generate_predictions_over_years(province, start_year, end_year, preview_factors, model_data)
                preview_fig = create_line_chart(preview, province, value_label, model_data["history"][province])
                st.plotly_chart(preview_fig, use_container_width=True)
                st.caption(f"Prices since 1990 and the forecast to {end_year} at the current settings")
            with details_placeholder.container():
                st.caption("Adjust the sliders to see how different economic factors affect property values over time.")

    # Model evaluation drawn with Matplotlib and Seaborn
    if model_data and "comparison" in model_data:
        st.markdown("---")
        st.subheader("How the model performs")
        best = model_data.get("model_name", "Model")
        st.caption(
            f"Both models trained on 1990 to 2017 and were tested on 2018 onward; {best} scored better. "
            "It predicts each year's growth from recent growth, so these scores describe one year ahead accuracy. "
            "Longer forecasts compound those predictions and grow less certain the further out they go."
        )
        tab_compare, tab_fit, tab_features = st.tabs(["Model comparison", "Actual vs predicted", "Feature importance"])
        with tab_compare:
            st.pyplot(charts.model_comparison(model_data["comparison"]))
        with tab_fit:
            st.pyplot(charts.actual_vs_predicted(model_data["test_actual"], model_data["test_predicted"], best))
        with tab_features:
            st.pyplot(charts.feature_importance(model_data["feature_names"], model_data["model"].feature_importances_, best))
                
if __name__ == "__main__":
    main()