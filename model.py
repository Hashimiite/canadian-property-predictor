import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

import charts
import warnings
warnings.filterwarnings('ignore')

def analyze_data_sources():#getting data
    print("📊 Analyzing available data sources...")
    data_dir = 'tables'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    data_source_info = []
    
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            # Read just headers and first few rows
            df_sample = pd.read_csv(file_path, nrows=5)
            row_count = sum(1 for _ in open(file_path, 'r')) - 1  # Count rows
            sample_data = {} #get sample for formatting issues
            for col in df_sample.columns:
                sample_data[col] = df_sample[col].iloc[0] if len(df_sample) > 0 else "N/A"
            info = {
                'file': csv_file,
                'rows': row_count,
                'columns': len(df_sample.columns),
                'sample_columns': df_sample.columns.tolist()[:5],
                'size_mb': os.path.getsize(file_path) / (1024*1024),
                'sample_data': sample_data
            }
            # Check for key indicators
            info['has_value'] = 'VALUE' in df_sample.columns
            info['has_date'] = any(col in ['REF_DATE', 'DATE', 'YEAR'] for col in df_sample.columns)
            info['has_geo'] = any(col in ['GEO', 'GEOGRAPHY', 'PROVINCE'] for col in df_sample.columns)
            # Check datta type
            if 'housing' in csv_file.lower() or 'index' in csv_file.lower():
                info['type'] = 'housing_index'
            elif 'land' in csv_file.lower():
                info['type'] = 'land_value'
            elif 'population' in csv_file.lower():
                info['type'] = 'population'
            elif 'interest' in csv_file.lower():
                info['type'] = 'interest_rate'
            elif 'crime' in csv_file.lower():
                info['type'] = 'crime'
            else:
                info['type'] = 'other'
            data_source_info.append(info)
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
    
    print(f"\nFound {len(data_source_info)} data sources:")
    for info in data_source_info:
        print(f"  📁 {info['file']}:")
        print(f"     Type: {info['type']}")
        print(f"     Rows: {info['rows']:,}")
        print(f"     Columns: {info['columns']} (sample: {', '.join(info['sample_columns'][:3])}...)")
        if 'VALUE' in info['sample_data']:
            print(f"     Sample VALUE: {info['sample_data']['VALUE']}")
        if 'REF_DATE' in info['sample_data']:
            print(f"     Sample REF_DATE: {info['sample_data']['REF_DATE']}")
        if 'GEO' in info['sample_data']:
            print(f"     Sample GEO: {info['sample_data']['GEO']}")
    return pd.DataFrame(data_source_info) 

def load_housing_index_with_conversion():
    """Converting housing index to prices"""
    print("Converting housing index to estimated house prices")
    
    try:
        file_path = os.path.join('tables', 'housingvalueinindex.csv')
        df = pd.read_csv(file_path)
        # Filter for house and land
        if 'New housing price indexes' in df.columns:
            df = df[df['New housing price indexes'] == 'Total (house and land)']
        df = df[['REF_DATE', 'GEO', 'VALUE']].rename(columns={
            'REF_DATE': 'YEAR',
            'GEO': 'PROVINCE',
            'VALUE': 'PRICE_INDEX'
        })
        
        # Extract year
        df['YEAR'] = df['YEAR'].astype(str).str.extract(r'(\d{4})').astype(int)
        df['PRICE_INDEX'] = pd.to_numeric(df['PRICE_INDEX'], errors='coerce')
        
        # Filter provinces
        provinces = [
            'Canada', 'Newfoundland and Labrador', 'Prince Edward Island', 
            'Nova Scotia', 'New Brunswick', 'Quebec', 'Ontario',
            'Manitoba', 'Saskatchewan', 'Alberta', 'British Columbia'
        ]
        df = df[df['PROVINCE'].isin(provinces)] # Average by year and province
        df = df.groupby(['PROVINCE', 'YEAR'], as_index=False)['PRICE_INDEX'].mean()
        
        # Convert index to estimated house prices Base year 2016 index = 100 
        base_year = 2016
        base_price_canada = 500000  # 2016 average house price
        canada_2016_idx = df[(df['PROVINCE'] == 'Canada') & (df['YEAR'] == base_year)]
        if not canada_2016_idx.empty:
            canada_2016_index = canada_2016_idx['PRICE_INDEX'].iloc[0]
            conversion_factor = base_price_canada / canada_2016_index if canada_2016_index > 0 else 5000
        else:
            # Use median of recent years
            recent = df[(df['PROVINCE'] == 'Canada') & (df['YEAR'] >= 2010)]
            if not recent.empty:
                median_index = recent['PRICE_INDEX'].median()
                conversion_factor = base_price_canada / median_index if median_index > 0 else 5000
            else:
                conversion_factor = 5000
        
        print(f" Conversion factor: ${conversion_factor:,.0f} per index point")
        
        #Convert to dollars
        df['TARGET_VALUE'] = df['PRICE_INDEX'] * conversion_factor
        recent_year = df['YEAR'].max()
        recent_indices = df[df['YEAR'] == recent_year].set_index('PROVINCE')['PRICE_INDEX']
        if 'Canada' in recent_indices.index and len(recent_indices) > 1:
            canada_recent = recent_indices['Canada']
            for province in df['PROVINCE'].unique():
                if province != 'Canada' and province in recent_indices.index:
                    prov_idx = recent_indices[province]
                    if canada_recent > 0:
                        adjustment = prov_idx / canada_recent
                        df.loc[df['PROVINCE'] == province, 'TARGET_VALUE'] *= adjustment
        print(f"  Converted {len(df)} rows, value range: ${df['TARGET_VALUE'].min():,.0f} to ${df['TARGET_VALUE'].max():,.0f}")
        print(f"  Mean house price: ${df['TARGET_VALUE'].mean():,.0f}")
        return df, 'house_prices'
        
    except Exception as e:
        print(f"Failed to convert housing index: {e}")
        return pd.DataFrame(), None
    
FIRST_YEAR = 1990
TEST_FROM = 2018  # train on 1990 to 2017, test on 2018 onward
FORECAST_CHART_END = 2036
FEATURES = ['YEAR', 'PROVINCE_CODE', 'GROWTH_LAG1', 'GROWTH_LAG2', 'GROWTH_AVG3']
AB_ALPHA = 0.05  # tuned settings ship only if they beat the current ones at this significance level

# Variant A: the hand picked settings each model has used so far
BASELINES = {
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=42
    ),
}
# Variant B: the best settings from these grids, found with cross validation on the training years only
PARAM_GRIDS = {
    "Random Forest": {
        "max_depth": [3, 6, None],
        "min_samples_leaf": [1, 3, 5],
        "max_features": [1.0, 0.6],
    },
    "Gradient Boosting": {
        "n_estimators": [150, 300],
        "learning_rate": [0.01, 0.03, 0.1],
        "max_depth": [2, 3, 4],
    },
}


def prepare_features(df):
    """Year over year growth features. The target is next year's growth, so forecasts can compound past the training range."""
    df = df.sort_values(['PROVINCE', 'YEAR']).copy()
    le = LabelEncoder()
    df['PROVINCE_CODE'] = le.fit_transform(df['PROVINCE'])
    by_province = df.groupby('PROVINCE')
    df['VALUE_LAG1'] = by_province['TARGET_VALUE'].shift(1)
    df['GROWTH'] = df['TARGET_VALUE'] / df['VALUE_LAG1'] - 1
    growth = df.groupby('PROVINCE')['GROWTH']
    df['GROWTH_LAG1'] = growth.shift(1)
    df['GROWTH_LAG2'] = growth.shift(2)
    df['GROWTH_AVG3'] = growth.transform(lambda g: g.shift(1).rolling(3).mean())
    # Lags use earlier years, but training rows start in FIRST_YEAR
    return df[df['YEAR'] >= FIRST_YEAR], le


def features_for_year(prices, year, province_code):
    """Feature row for `year` from a list of yearly prices that ends at year - 1."""
    growth = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
    return {
        'YEAR': year,
        'PROVINCE_CODE': province_code,
        'GROWTH_LAG1': growth[-1],
        'GROWTH_LAG2': growth[-2],
        'GROWTH_AVG3': float(np.mean(growth[-3:])),
    }


def forecast(model, history, province_code, end_year, adjust=lambda g: g):
    """Compound predicted yearly growth from the last known price up to end_year.

    history: {year: price}. adjust lets the app shift each year's growth for a scenario.
    Returns a list of (year, price, growth).
    """
    years = sorted(history)
    prices = [history[y] for y in years]
    rows = []
    for year in range(years[-1] + 1, end_year + 1):
        row = pd.DataFrame([features_for_year(prices, year, province_code)])[FEATURES]
        g = adjust(float(model.predict(row)[0]))
        prices.append(prices[-1] * (1 + g))
        rows.append((year, prices[-1], g))
    return rows


def _price_metrics(actual, predicted):
    return {
        'r2': r2_score(actual, predicted),
        'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
        'mae': mean_absolute_error(actual, predicted),
    }


def year_folds(years, n_splits=4):
    """Expanding window folds that never split a year: train on earlier years, validate on the next block."""
    unique = np.unique(years)
    return [(np.flatnonzero(np.isin(years, unique[tr])), np.flatnonzero(np.isin(years, unique[va])))
            for tr, va in TimeSeriesSplit(n_splits).split(unique)]


def tune(name, train):
    """Grid search the model's settings on the training years. Returns the best settings and their validation growth MAE."""
    search = GridSearchCV(
        BASELINES[name](), PARAM_GRIDS[name], cv=year_folds(train['YEAR'].to_numpy()),
        scoring='neg_mean_absolute_error', n_jobs=-1,
    ).fit(train[FEATURES], train['GROWTH'])
    return search.best_params_, -search.best_score_ * 100


def ab_test(actual, pred_a, pred_b):
    """One sided paired Wilcoxon signed-rank test on each test row's absolute price error.

    B (tuned) wins only if its errors are significantly smaller than A's (current), otherwise A stays.
    """
    err_a, err_b = np.abs(actual - pred_a), np.abs(actual - pred_b)
    # Identical predictions (the grid picked A's settings) leave nothing to test
    p_value = 1.0 if np.allclose(err_a, err_b) else float(wilcoxon(err_a, err_b, alternative='greater').pvalue)
    return {'mae_a': float(err_a.mean()), 'mae_b': float(err_b.mean()), 'p_value': p_value,
            'rows': len(actual), 'winner': 'B' if p_value < AB_ALPHA else 'A'}


def _evaluate(model, test):
    """One year ahead price metrics on the test years, plus the price predictions."""
    growth_pred = model.predict(test[FEATURES])
    price_pred = test['VALUE_LAG1'].to_numpy() * (1 + growth_pred)
    metrics = _price_metrics(test['TARGET_VALUE'], price_pred)
    metrics['growth_mae_pts'] = mean_absolute_error(test['GROWTH'], growth_pred) * 100
    return metrics, price_pred


def train_final_model(df, data_type, label_encoder, chosen_option, full_history):
    """Tune both models, A/B test tuned against current settings, compare the shipped variants, then refit the winner on everything."""
    print("\n" + "="*60)
    print(f"TRAINING FINAL MODEL ({data_type.upper()})")
    print("="*60)
    if df is None or df.empty:
        print("No data available for training!")
        return None
    print(f"Rows: {len(df)}  Years: {df['YEAR'].min()} to {df['YEAR'].max()}  Regions: {df['PROVINCE'].nunique()}")

    train = df[df['YEAR'] < TEST_FROM]
    test = df[df['YEAR'] >= TEST_FROM]
    print(f"Train: {train['YEAR'].min()} to {train['YEAR'].max()} ({len(train)} rows)  "
          f"Test: {test['YEAR'].min()} to {test['YEAR'].max()} ({len(test)} rows)")

    comparison, test_predictions, ab_tests, chosen_params = {}, {}, {}, {}
    actual = test['TARGET_VALUE'].to_numpy()
    for name, make in BASELINES.items():
        params, cv_mae = tune(name, train)
        print(f"\nTUNING {name}: best of {len(list(ParameterGrid(PARAM_GRIDS[name])))} settings {params} "
              f"(validation growth error {cv_mae:.2f} pts)")
        results = {variant: _evaluate(make().set_params(**p).fit(train[FEATURES], train['GROWTH']), test)
                   for variant, p in (('A', {}), ('B', params))}
        ab = ab_test(actual, results['A'][1], results['B'][1])
        ab['params_b'] = params
        ab_tests[name] = ab
        print(f"A/B TEST {name}: A (current) MAE ${ab['mae_a']:,.0f}  B (tuned) MAE ${ab['mae_b']:,.0f}  "
              f"p = {ab['p_value']:.4f}  -> {'B ships' if ab['winner'] == 'B' else 'A stays'}")
        chosen_params[name] = params if ab['winner'] == 'B' else {}
        comparison[name], test_predictions[name] = results[ab['winner']]

    print("\nMODEL COMPARISON (one year ahead prices, tested on 2018 onward):")
    for name, m in comparison.items():
        print(f"  {name:18} R² {m['r2']:.4f}  MAE ${m['mae']:,.0f}  RMSE ${m['rmse']:,.0f}  "
              f"growth error {m['growth_mae_pts']:.2f} pts")
    best_name = max(comparison, key=lambda name: comparison[name]['r2'])
    print(f"\nSelected model: {best_name} (refit on {df['YEAR'].min()} to {df['YEAR'].max()})")

    # Refit the winner on all years so forecasts start from the latest data
    model = BASELINES[best_name]().set_params(**chosen_params[best_name]).fit(df[FEATURES], df['GROWTH'])
    print("\nFEATURE IMPORTANCE:")
    for feature, value in sorted(zip(FEATURES, model.feature_importances_), key=lambda p: -p[1]):
        print(f"  {feature}: {value:.4f}")

    history = {
        province: dict(zip(group['YEAR'].astype(int), group['TARGET_VALUE'].astype(float)))
        for province, group in full_history.groupby('PROVINCE')
    }
    codes = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    forecasts = []
    for province, past in history.items():
        for year, price, _ in forecast(model, past, codes[province], FORECAST_CHART_END):
            forecasts.append({'PROVINCE': province, 'YEAR': year, 'TARGET_VALUE': price})
    forecast_df = pd.DataFrame(forecasts)
    last_year = max(max(h) for h in history.values())
    canada = forecast_df[forecast_df['PROVINCE'] == 'Canada'].set_index('YEAR')['TARGET_VALUE']
    canada_2100 = forecast(model, history['Canada'], codes['Canada'], 2100)[-1][1]
    print(f"\nCanada forecast: {last_year + 1} ${canada.iloc[0]:,.0f}  "
          f"{last_year + 10} ${canada.loc[last_year + 10]:,.0f}  2100 ${canada_2100:,.0f}")

    # Charts with Matplotlib and Seaborn
    os.makedirs("figures", exist_ok=True)
    history_df = full_history[full_history['YEAR'] >= FIRST_YEAR]
    figures = {
        "price_trends.png": charts.price_trends(history_df),
        "model_comparison.png": charts.model_comparison(comparison),
        "actual_vs_predicted.png": charts.actual_vs_predicted(
            test['TARGET_VALUE'].to_numpy(), test_predictions[best_name], best_name),
        "feature_importance.png": charts.feature_importance(FEATURES, model.feature_importances_, best_name),
        "forecast.png": charts.forecast_paths(history_df, forecast_df),
    }
    for filename, fig in figures.items():
        fig.savefig(os.path.join("figures", filename), dpi=150)
    print(f"Saved {len(figures)} charts to figures/")

    best = comparison[best_name]
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': FEATURES,
        'metrics': {'r2': best['r2'], 'rmse': best['rmse'], 'mae': best['mae'],
                    'mean_value': float(test['TARGET_VALUE'].mean())},
        'model_name': best_name,
        'comparison': comparison,
        'ab_tests': ab_tests,
        'test_actual': test['TARGET_VALUE'].to_numpy(),
        'test_predicted': test_predictions[best_name],
        'history': {p: {y: v for y, v in h.items() if y >= FIRST_YEAR - 4} for p, h in history.items()},
        'data_type': data_type,
        'chosen_option': chosen_option,
        'target_unit': 'CAD',
        'years_range': (int(df['YEAR'].min()), int(df['YEAR'].max())),
        'provinces': sorted(history),
    }
    filename = f"housing_model_{data_type}_{chosen_option}.joblib"
    joblib.dump(model_data, filename)
    print(f"Model saved as '{filename}'")
    return model_data


def train_model():
    print("="*60)
    print("CANADIAN PROPERTY VALUE PREDICTOR")
    print("="*60)
    # Analyze available data
    data_sources = analyze_data_sources()
    # Load House prices
    df_option1, type1 = load_housing_index_with_conversion()
    print(f"\n Using data: {type1}")
    # Prepare features
    df_features, le = prepare_features(df_option1)
    df_features = df_features.dropna(subset=FEATURES + ['GROWTH'])
    if len(df_features) < 10:
        print("Not enough data after feature preparation!")
        return
    # Train model
    model_data = train_final_model(df_features, type1, le, "option1", df_option1)
    if model_data:
        print("\n" + "="*60)
        print("🎉 MODEL TRAINING COMPLETE!")
        print(f"Data type: {model_data['data_type']}")
        print(f"Chosen option: {model_data['chosen_option']}")
        print(f"Target unit: {model_data['target_unit']}")
        print("="*60)

if __name__ == "__main__":
    train_model()