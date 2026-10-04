# Canadian property predictor

A machine learning model that estimates average new house prices for each Canadian province, with a Streamlit app for exploring forecasts under different economic scenarios.

## How it works

`model.py` reads the Statistics Canada new housing price index (1981 onward), converts the index into estimated prices in Canadian dollars and engineers features such as the previous year's value, the growth rate, the decade and a time trend. It trains a Random Forest and a Gradient Boosting model on the same 80/20 split, checks both with 5 fold cross validation and saves whichever scores better, along with the scaler, encoders and metrics.

| Model | R² (held out) | Mean absolute error | Cross validated R² |
|---|---|---|---|
| Random Forest | 0.968 | $11,179 | 0.990 |
| **Gradient Boosting** (selected) | **0.981** | **$8,070** | **0.996** |

The previous year's price carries most of the predictive weight, so these scores describe one year ahead accuracy rather than long range forecasting. The model covers Canada and ten provinces from 1982 to 2025.

`app.py` loads the saved model and lets you pick a province, then adjust interest rates, crime, population growth and the economic outlook to see how the price forecast changes over the coming years. A "How the model performs" section draws the evaluation charts live.

## Charts

`charts.py` builds every evaluation chart with Matplotlib and Seaborn. Training saves them to `figures/` and the app renders the same functions live.

![Estimated new house prices by province](figures/price_trends.png)

![Random Forest vs Gradient Boosting](figures/model_comparison.png)

| Actual vs predicted | Feature importance |
|---|---|
| ![Actual vs predicted](figures/actual_vs_predicted.png) | ![Feature importance](figures/feature_importance.png) |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain, run `python model.py`. It compares both models, saves the better one as `housing_model_*.joblib` (which the app picks up automatically) and redraws the charts in `figures/`.

## Data

All tables in `tables/` come from Statistics Canada and are used under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada):

- New housing price index (`housingvalueinindex.csv`)
- Crime severity index (`crime.csv`)
- Population estimates (`population.csv`)
- Farm land and building values (`landvalue.csv`)

The financial market statistics table (`interest.csv`, about 95 MB) is left out of this repository because training doesn't read it. Download it from Statistics Canada's financial market statistics table if you want the full data survey that `model.py` prints.
