# Canadian property predictor

A machine learning model that estimates average new house prices for each Canadian province, with a Streamlit app for exploring forecasts under different economic scenarios.

## How it works

`model.py` reads the Statistics Canada new housing price index (1981 onward), converts the index into estimated prices in Canadian dollars and engineers features such as the previous year's value, the growth rate, the decade and a time trend. It then trains a Random Forest regressor (200 trees, depth 10) and saves the model, scaler, encoders and metrics to a `.joblib` file.

On the held out test set the saved model reached:

| Metric | Value |
|---|---|
| R² | 0.97 |
| Mean absolute error | about $11,000 |
| Root mean squared error | about $25,000 |

The model covers Canada and ten provinces from 1982 to 2025.

`app.py` loads the latest model and lets you pick a province, then adjust interest rates, crime, population growth and the economic outlook to see how the price forecast changes over the coming years.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, run `python model.py`. It writes a new `housing_model_*.joblib` file that the app picks up automatically.

## Data

All tables in `tables/` come from Statistics Canada and are used under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada):

- New housing price index (`housingvalueinindex.csv`)
- Crime severity index (`crime.csv`)
- Population estimates (`population.csv`)
- Farm land and building values (`landvalue.csv`)

The financial market statistics table (`interest.csv`, about 95 MB) is left out of this repository because training doesn't read it. Download it from Statistics Canada's financial market statistics table if you want the full data survey that `model.py` prints.
