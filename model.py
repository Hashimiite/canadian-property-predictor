import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
def prepare_features(df):
    """Prepare features based on data type"""
    df = df.copy()
    df['TIME_TREND'] = df['YEAR'] - df['YEAR'].min()
    le = LabelEncoder()# Province encoding
    df['PROVINCE_CODE'] = le.fit_transform(df['PROVINCE'])
    df = df.sort_values(['PROVINCE', 'YEAR'])
    df['VALUE_LAG1'] = df.groupby('PROVINCE')['TARGET_VALUE'].shift(1)
    df['GROWTH_RATE'] = (df['TARGET_VALUE'] - df['VALUE_LAG1']) / df['VALUE_LAG1']     # Growth rate
    df['GROWTH_RATE'] = df['GROWTH_RATE'].fillna(0)
    
    # Additional economic context for better model
    if 'YEAR' in df.columns:
        df['DECADE'] = (df['YEAR'] // 10) * 10  #Adding decade feature
        df['ECONOMIC_BOOM'] = df['YEAR'].apply( # Economic boom periods (approximate)
            lambda x: 1 if x in range(1970, 1974) or x in range(2002, 2008) or x in range(2010, 2014) else 0
        )
    return df, le

def train_final_model(df, data_type, label_encoder, chosen_option):
    """Train final model with chosen data"""
    print("\n" + "="*60)
    print(f"TRAINING FINAL MODEL ({data_type.upper()})")
    print("="*60)
    if df is None or df.empty:
        print("No data available for training!")
        return None
    print(f"Data shape: {df.shape}")
    print(f"Years: {df['YEAR'].min()} to {df['YEAR'].max()}")
    print(f"Provinces: {df['PROVINCE'].nunique()}")
    # Features and target
    feature_cols = ['YEAR', 'TIME_TREND', 'PROVINCE_CODE', 'VALUE_LAG1', 'GROWTH_RATE']
    if 'DECADE' in df.columns:
        feature_cols.append('DECADE')
    if 'ECONOMIC_BOOM' in df.columns:
        feature_cols.append('ECONOMIC_BOOM')
    X = df[[col for col in feature_cols if col in df.columns]]
    y = df['TARGET_VALUE']
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Target range: ${y.min():,.0f} to ${y.max():,.0f}")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train both candidates on the same split and keep the better one
    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
    }
    comparison = {}
    fitted = {}
    for name, candidate in candidates.items():
        print(f"\nTraining {name}...")
        candidate.fit(X_train_scaled, y_train)
        y_pred = candidate.predict(X_test_scaled)
        cv_r2 = cross_val_score(candidate, X_train_scaled, y_train, cv=5, scoring="r2")
        comparison[name] = {
            "r2": r2_score(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": mean_absolute_error(y_test, y_pred),
            "cv_r2_mean": cv_r2.mean(),
            "cv_r2_std": cv_r2.std(),
        }
        fitted[name] = (candidate, y_pred)

    print("\n" + "="*60)
    print("MODEL COMPARISON (held out 20%, plus 5 fold CV on the training set):")
    for name, m in comparison.items():
        print(f"  {name:18} R² {m['r2']:.4f}  MAE ${m['mae']:,.0f}  RMSE ${m['rmse']:,.0f}  "
              f"CV R² {m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")
    best_name = max(comparison, key=lambda name: comparison[name]["r2"])
    model, y_pred = fitted[best_name]
    r2, rmse, mae = (comparison[best_name][k] for k in ("r2", "rmse", "mae"))
    print(f"\nSelected model: {best_name}")
    print(f"MAE as % of mean: {(mae / y.mean() * 100):.1f}%")
    print("="*60)
    print("\nFEATURE IMPORTANCE:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Charts with Matplotlib and Seaborn
    os.makedirs("figures", exist_ok=True)
    figures = {
        "price_trends.png": charts.price_trends(df),
        "model_comparison.png": charts.model_comparison(comparison),
        "actual_vs_predicted.png": charts.actual_vs_predicted(y_test.to_numpy(), y_pred, best_name),
        "feature_importance.png": charts.feature_importance(X.columns.tolist(), model.feature_importances_, best_name),
    }
    for filename, fig in figures.items():
        fig.savefig(os.path.join("figures", filename), dpi=150)
    print(f"\nSaved {len(figures)} charts to figures/")

    # Saving model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist(),
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 'mean_value': y.mean()},
        'model_name': best_name,
        'comparison': comparison,
        'test_actual': y_test.to_numpy(),
        'test_predicted': y_pred,
        'data_type': data_type,
        'chosen_option': chosen_option,
        'target_unit': 'CAD',
        'years_range': (df['YEAR'].min(), df['YEAR'].max()),
        'provinces': df['PROVINCE'].unique().tolist()
    }
    filename = f"housing_model_{data_type}_{chosen_option}.joblib"
    joblib.dump(model_data, filename)
    print(f"\n Model saved as '{filename}'")
    # Sample prediction
    print("\n" + "="*60)
    print("SAMPLE PREDICTION:")
    
    sample_idx = np.random.default_rng(42).integers(0, len(df))
    sample = X.iloc[sample_idx:sample_idx+1].copy()
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    actual = y.iloc[sample_idx]
    print(f"Province: {df.iloc[sample_idx]['PROVINCE']}")
    print(f"Year: {df.iloc[sample_idx]['YEAR']}")
    print(f"Actual: ${actual:,.0f}")
    print(f"Predicted: ${prediction:,.0f}")
    print(f"Error: ${abs(prediction - actual):,.0f} ({abs(prediction - actual)/actual*100:.1f}%)")
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
    df_features = df_features.dropna()
    if len(df_features) < 10:
        print("Not enough data after feature preparation!")
        return
    # Train model
    model_data = train_final_model(df_features, type1, le, "option1")
    if model_data:
        print("\n" + "="*60)
        print("🎉 MODEL TRAINING COMPLETE!")
        print(f"Data type: {model_data['data_type']}")
        print(f"Chosen option: {model_data['chosen_option']}")
        print(f"Target unit: {model_data['target_unit']}")
        print("="*60)

if __name__ == "__main__":
    train_model()