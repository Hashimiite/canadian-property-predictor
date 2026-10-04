import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import charts

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

def create_prediction_features(province, year, economic_factors, model_data=None):
    """Create features"""
    features = {}
    # bbest scenario: Getting province encoding from model 
    if model_data and 'label_encoder' in model_data:
        try:
            province_code = model_data['label_encoder'].transform([province])[0]
        except:
            # Fallback encoding
            province_codes = {
                'Ontario': 0, 'Quebec': 1, 'British Columbia': 2, 'Alberta': 3,
                'Manitoba': 4, 'Saskatchewan': 5, 'Nova Scotia': 6, 'New Brunswick': 7,
                'Newfoundland and Labrador': 8, 'Prince Edward Island': 9, 'Canada': 10
            }
            province_code = province_codes.get(province, 0)
    else:
       # Fallback encoding
        province_codes = {
            'Ontario': 0, 'Quebec': 1, 'British Columbia': 2, 'Alberta': 3,
            'Manitoba': 4, 'Saskatchewan': 5, 'Nova Scotia': 6, 'New Brunswick': 7,
            'Newfoundland and Labrador': 8, 'Prince Edward Island': 9, 'Canada': 10
        }
        province_code = province_codes.get(province, 0)
        
    # Base features
    features['YEAR'] = year
    features['TIME_TREND'] = year - 2015
    features['PROVINCE_CODE'] = province_code
    # Base values for each province (2024 estimates)
    base_values = {
        'Ontario': 850000,
        'Quebec': 480000,
        'British Columbia': 950000,
        'Alberta': 520000,
        'Manitoba': 370000,
        'Saskatchewan': 320000,
        'Nova Scotia': 380000,
        'New Brunswick': 320000,
        'Newfoundland and Labrador': 340000,
        'Prince Edward Island': 350000
    }
    # Get base value,growth for province and then apply econ factors
    base_value = base_values.get(province, 400000)
    base_growth = 0.03  # 3% base growth
    adjusted_growth = base_growth + economic_factors['interest_rate_impact'] + economic_factors['crime_impact']
    growth_multiplier = (1 + economic_factors['population_growth_impact']) * (1 + economic_factors['economic_outlook_impact'])
    adjusted_growth *= growth_multiplier
    
    # calculate year
    current_year = pd.Timestamp.now().year
    # Calculate value based on year difference from current year
    years_from_now = year - current_year
    adjusted_value = base_value * ((1 + adjusted_growth) ** years_from_now)
    features['VALUE_LAG1'] = adjusted_value * 0.97  # Previous year value (3% less)
    features['GROWTH_RATE'] = adjusted_growth
    
    # Add any additional features the model expects if available if you have extra csv's
    if model_data and 'feature_names' in model_data:
        for feature in model_data['feature_names']:
            if feature not in features:
                features[feature] = 0
    return features, adjusted_value, adjusted_growth

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

def generate_predictions_over_years(province, start_year, end_year, economic_factors, model_data=None, model=None, scaler=None):
    """Generate predictions for years"""
    predictions = []
    current_year = pd.Timestamp.now().year
    base_growth = 0.03 
    adjusted_growth = base_growth + economic_factors['interest_rate_impact'] + economic_factors['crime_impact']
    growth_multiplier = (1 + economic_factors['population_growth_impact']) * (1 + economic_factors['economic_outlook_impact'])
    final_growth_rate = adjusted_growth * growth_multiplier
    base_values = {
        'Ontario': 850000,
        'Quebec': 480000,
        'British Columbia': 950000,
        'Alberta': 520000,
        'Manitoba': 370000,
        'Saskatchewan': 320000,
        'Nova Scotia': 380000,
        'New Brunswick': 320000,
        'Newfoundland and Labrador': 340000,
        'Prince Edward Island': 350000
    }
    base_value = base_values.get(province, 400000)
    for year in range(start_year, end_year + 1):
        years_from_now = year - current_year
        simple_prediction = base_value * ((1 + final_growth_rate) ** years_from_now)
        # If model is available, use it
        if model_data and model and scaler:
            features, _, _ = create_prediction_features(province, year, economic_factors, model_data)
            try:
                if 'feature_names' in model_data:
                    features_dict = {}
                    for feature in model_data['feature_names']:
                        if feature in features:
                            features_dict[feature] = features[feature]
                        else:
                            features_dict[feature] = 0
                    
                    features_df = pd.DataFrame([features_dict])
                    features_scaled = scaler.transform(features_df)
                    prediction = model.predict(features_scaled)[0]
                else:
                    prediction = simple_prediction
            except:
                prediction = simple_prediction
        else:
            prediction = simple_prediction
        
        predictions.append({
            'Year': year,
            'Predicted_Value': prediction,
            'Growth_Rate': final_growth_rate * 100
        })
    
    return pd.DataFrame(predictions)

def create_line_chart(prediction_data, province, value_label):
    """Create the line chart"""
    
    fig = go.Figure()
    
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
        showlegend=False,
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
    model = None
    scaler = None
    data_type = 'house_prices'
    
    if model_info and model_info[0]:
        model_data, model_file = model_info
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        data_type = model_data.get('data_type', 'house_prices')
    
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
            ["Ontario", "Quebec", "British Columbia", "Alberta",
             "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
             "Newfoundland and Labrador", "Prince Edward Island"]
        )
        # Year range selection
        current_year = pd.Timestamp.now().year
        st.markdown("Forecast Period")
        col1a, col1b = st.columns(2)
        with col1a:
            start_year = st.number_input(
                "Start Year",
                min_value=current_year,
                max_value=current_year + 10,
                value=current_year,
                step=1
            )
        with col1b:
            end_year = st.number_input(
                "End Year",
                min_value=current_year + 1,
                max_value=current_year + 10,
                value=current_year + 5,
                step=1
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
                    model, 
                    scaler
                )
                end_year_prediction = predictions_df[predictions_df['Year'] == end_year].iloc[0]
                start_year_prediction = predictions_df[predictions_df['Year'] == start_year].iloc[0]
                fig = create_line_chart(predictions_df, province, value_label) # Create and display line chart
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
                st.info("Configure prediction settings and click 'Generate Forecast' to see results")
                # Show example chart
                example_data = pd.DataFrame({
                    'Year': [2024, 2025, 2026, 2027, 2028],
                    'Predicted_Value': [850000, 875500, 901765, 928818, 956682]
                })
                example_fig = create_line_chart(example_data, "Example Province", value_label)
                st.plotly_chart(example_fig, use_container_width=True)
                st.caption("Example forecast showing typical growth pattern")
            with details_placeholder.container():
                st.caption("Adjust the sliders to see how different economic factors affect property values over time.")

    # Model evaluation drawn with Matplotlib and Seaborn
    if model_data and "comparison" in model_data:
        st.markdown("---")
        st.subheader("How the model performs")
        best = model_data.get("model_name", "Model")
        st.caption(
            f"{best} was selected after comparing models on the same held out 20% of the data. "
            "It predicts each year's price from the previous year, so these scores describe one year ahead accuracy."
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