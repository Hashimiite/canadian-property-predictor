# Canadian property predictor

A machine learning model that forecasts average new house prices for Canada and each province, with a Streamlit app that projects prices from 2026 to any year up to 2100 under different economic scenarios.

![The Streamlit app showing prices since 1990 and the forecast to 2036](figures/app.jpg)

## How it works

`model.py` reads the Statistics Canada new housing price index, converts it into estimated prices in Canadian dollars and keeps every year from 1990 to 2025. Instead of predicting a price directly, the models predict each year's growth from the previous years' growth, the region and the year. Forecasts then compound those predictions year by year from each region's latest price, which lets them run well past the range the models were trained on.

Two models were tuned and trained on 1990 to 2017 and tested on 2018 to 2025, so the test years are ones the models never saw.

### Hyperparameter tuning

Each model's settings are tuned with a grid search (18 combinations per model) scored by cross validation on the training years. The folds are expanding windows that never split a year: each one trains on earlier years and validates on the next block of five, so tuning never peeks at the 2018 to 2025 test years.

### A/B tests

Tuned settings don't ship just because they scored well in cross validation. Each model's current settings (A) are tested against its tuned settings (B) on the same 88 held out rows with a one sided paired Wilcoxon signed-rank test on the absolute price errors. B replaces A only if its errors are significantly smaller (p < 0.05):

| Model | A: current MAE | B: tuned MAE | p value | Ships | Tuned settings |
|---|---|---|---|---|---|
| Random Forest | $17,144 | $15,882 | 0.015 | **B (tuned)** | `max_depth=None, max_features=0.6, min_samples_leaf=1` |
| Gradient Boosting | $17,464 | $17,301 | 0.392 | A (current) | `learning_rate=0.01, max_depth=4, n_estimators=300` |

![A/B tests: current vs tuned settings for each model](figures/ab_tests.png)

The shipped variant of each model is then compared:

| Model | R² (one year ahead price) | Mean absolute error | Growth error |
|---|---|---|---|
| **Random Forest, tuned** (selected) | **0.958** | **$15,882** | **2.75 points** |
| Gradient Boosting | 0.948 | $17,464 | 3.04 points |

The tuned Random Forest scored better, so it was refit on all years from 1990 to 2025 and saved. Its forecast for Canada is about $640,000 in 2026 and $832,000 by 2035, which is roughly 3% growth a year. Forecasts further out compound the same way and become less certain the further they go.

`app.py` loads the saved model and shows prices since 1990 next to the forecast, with a shaded range for the model's typical yearly error. Pick a region in the sidebar, choose any range from 2026 to 2100 (ten years by default) and adjust interest rates, crime, population growth and the economic outlook to shift each year's growth. The chart and summary update as you change settings, and a "How the model performs" section draws the evaluation charts and A/B test results live. The app uses the same dark palette and Geist type as [hashimjama.dev](https://hashimjama.dev).

![House prices since 1990 and the ten year forecast for every region](figures/forecast.png)

## Charts

`charts.py` builds every chart with Matplotlib (and Seaborn on top of it), including the app's forecast chart. Training saves them to `figures/` and the app renders the same functions live.

![Estimated new house prices by province](figures/price_trends.png)

![Tuned Random Forest vs Gradient Boosting](figures/model_comparison.png)

| Actual vs predicted (2018 to 2025) | Feature importance |
|---|---|
| ![Actual vs predicted](figures/actual_vs_predicted.png) | ![Feature importance](figures/feature_importance.png) |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain, run `python model.py`. It tunes both models, A/B tests tuned against current settings, compares the shipped variants on the recent test years, refits the better one on all data, saves it as `housing_model_*.joblib` (which the app picks up automatically) and redraws the charts in `figures/`.

## Data

All tables in `tables/` come from Statistics Canada and are used under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada):

- New housing price index (`housingvalueinindex.csv`)
- Crime severity index (`crime.csv`)
- Population estimates (`population.csv`)
- Farm land and building values (`landvalue.csv`)

The financial market statistics table (`interest.csv`, about 95 MB) is left out of this repository because training doesn't read it. Download it from Statistics Canada's financial market statistics table if you want the full data survey that `model.py` prints.
