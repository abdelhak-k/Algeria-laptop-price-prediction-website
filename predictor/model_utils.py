import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'

FEAT_NUMERIC = ['RAM_SIZE_GB', 'SSD_SIZE_GB']
FEAT_BINARY = ['is_pro']
FEAT_CATEGORICAL = [
    'series', 'cpu_family', 'gpu_tier', 'condition', 'gpu_suffix',
    'cpu_suffix', 'cpu_gen_brand', 'listing_year'
]


def load_model():
    model_path = MODEL_DIR / 'model.pkl'
    preprocessor_path = MODEL_DIR / 'preprocessor.pkl'
    
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Model files not found. Please run the training notebook first to save the model. "
            f"Expected paths: {model_path} and {preprocessor_path}"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    return model, preprocessor


def prepare_input_data(form_data):
    cpu_family = form_data['cpu_family']
    
    if cpu_family in ['i3', 'i5', 'i7', 'i9', 'ultra5', 'ultra7', 'ultra9', 'celeron', 'pentium', 'xeon']:
        cpu_brand = 'INTEL'
    elif cpu_family in ['r3', 'r5', 'r7', 'r9']:
        cpu_brand = 'AMD'
    elif cpu_family in ['m1', 'm2', 'm3', 'm4']:
        cpu_brand = 'APPLE'
    elif cpu_family == 'SNAPDRAGON':
        cpu_brand = 'QUALCOMM'
    elif cpu_family == 'UNKNOWN':
        cpu_brand = 'UNKNOWN'
    else:
        cpu_brand = form_data.get('cpu_brand', 'UNKNOWN')
    
    cpu_generation = form_data.get('cpu_generation', '0')
    if cpu_generation == '' or cpu_generation is None:
        cpu_generation = '0'
    
    if cpu_family in ['m1', 'm2', 'm3', 'm4', 'SNAPDRAGON', 'UNKNOWN']:
        cpu_generation = '0'
    
    cpu_gen_brand = f"{cpu_brand}_{int(cpu_generation)}"
    
    ram_log = np.log1p(form_data['ram_size_gb'])
    ssd_size = form_data.get('ssd_size_gb', 0) or 0
    hdd_size = form_data.get('hdd_size_gb', 0) or 0
    ssd_log = np.log1p(ssd_size)
    
    gpu_tier = form_data.get('gpu_tier', 'NONE') or 'NONE'
    gpu_suffix = form_data.get('gpu_suffix', '') or ''
    cpu_suffix = form_data.get('cpu_suffix', '') or ''
    
    data = {
        'RAM_SIZE_GB': ram_log,
        'SSD_SIZE_GB': ssd_log,
        'is_pro': 1 if form_data.get('is_pro', False) else 0,
        'series': form_data['series'],
        'cpu_family': cpu_family,
        'gpu_tier': gpu_tier,
        'condition': form_data['condition'],
        'gpu_suffix': gpu_suffix,
        'cpu_suffix': cpu_suffix,
        'cpu_gen_brand': cpu_gen_brand,
        'listing_year': str(form_data['listing_year']),
    }
    
    df = pd.DataFrame([data])
    all_features = FEAT_NUMERIC + FEAT_BINARY + FEAT_CATEGORICAL
    df = df[all_features]
    
    return df


def predict_price(form_data):
    try:
        model, preprocessor = load_model()
        input_df = prepare_input_data(form_data)
        X = preprocessor.transform(input_df)
        y_log = model.predict(X)
        predicted_price = np.expm1(y_log[0])
        predicted_price = max(predicted_price, 10000)
        
        model_rmse = 34390
        min_price = max(predicted_price - model_rmse, 5000)  # Minimum floor of 5,000 DZD
        max_price = predicted_price + model_rmse
        
        return {
            'success': True,
            'predicted_price': round(predicted_price, 0),
            'predicted_price_formatted': f"{predicted_price:,.0f} DZD",
            'min_price': round(min_price, 0),
            'max_price': round(max_price, 0),
            'min_price_formatted': f"{min_price:,.0f} DZD",
            'max_price_formatted': f"{max_price:,.0f} DZD"
        }
        
    except FileNotFoundError as e:
        return {
            'success': False,
            'error': str(e)
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Prediction error: {str(e)}"
        }
