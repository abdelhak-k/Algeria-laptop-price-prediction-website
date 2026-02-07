import os
import pickle
import csv
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'

# ============================================================================
# FEATURE DEFINITIONS (must match 03_Regression.ipynb exactly)
# ============================================================================
FEAT_NUMERIC = [
    'RAM_SIZE_GB', 'SSD_SIZE_GB', 'cpu_mark', 'cpu_single_thread',
    'gpu_score', 'build_quality_tier', 'condition_value_retention',
    'cpu_generation', 'storage_perf_score', 'cpu_cores',
    'inferred_ddr_ordinal', 'SCREEN_SIZE_NUM', 'cpu_threads',
]

FEAT_BINARY = ['is_pro', 'is_gaming_series', 'has_gpu']

FEAT_CATEGORICAL = [
    'series', 'cpu_family', 'gpu_tier', 'condition', 'gpu_suffix',
    'cpu_suffix', 'cpu_gen_brand', 'listing_year', 'brand',
    'ram_type_class', 'resolution_class',
]

# ============================================================================
# GAMING SERIES LIST (from preprocessing)
# ============================================================================
GAMING_SERIES = [
    'ROG', 'TUF', 'STRIX', 'LEGION', 'OMEN', 'VICTUS', 'PREDATOR', 'NITRO',
    'ALIENWARE', 'RAIDER', 'STEALTH', 'KATANA', 'SWORD', 'VECTOR', 'TITAN',
    'CYBORG', 'PULSE', 'CROSSHAIR', 'THIN GF', 'ALPHA', 'BRAVO', 'DELTA',
    'LOQ', 'IDEAPAD GAMING', 'PAVILION GAMING', 'DELL G3', 'DELL G5',
    'DELL G7', 'DELL G15', 'DELL G16', 'BLADE', 'AORUS', 'ODYSSEY',
]

# ============================================================================
# BUILD QUALITY TIER MAP (from preprocessing)
# ============================================================================
SERIES_TIER_MAP = {
    # Tier 1 — Budget
    'STREAM': 1,
    # Tier 2 — Consumer (default)
    'PAVILION': 2, 'INSPIRON': 2, 'IDEAPAD': 2, 'VIVOBOOK': 2,
    'ASPIRE': 2, 'MODERN': 2, 'SATELLITE': 2,
    # Tier 3 — Business
    'PROBOOK': 3, 'VOSTRO': 3, 'THINKBOOK': 3, 'EXPERTBOOK': 3,
    'LOQ': 3, 'SPIN': 3, 'TRAVELMATE': 3, 'CYBORG': 3, 'TECRA': 3,
    # Tier 4 — Premium
    'ELITEBOOK': 4, 'ENVY': 4, 'LATITUDE': 4, 'THINKPAD': 4, 'YOGA': 4,
    'ZENBOOK': 4, 'TUF': 4, 'NITRO': 4, 'KATANA': 4, 'SWIFT': 4,
    'PRESTIGE': 4, 'VICTUS': 4, 'G SERIES': 4, 'MACBOOK AIR': 4,
    'MACBOOK': 4, 'PORTEGE': 4, 'GALAXY BOOK': 4, 'MATEBOOK': 4,
    # Tier 5 — Workstation/Gaming Flagship
    'SPECTRE': 5, 'ZBOOK': 5, 'OMEN': 5, 'XPS': 5, 'PRECISION': 5,
    'ALIENWARE': 5, 'LEGION': 5, 'ROG': 5, 'STRIX': 5, 'PROART': 5,
    'PREDATOR': 5, 'CONCEPTD': 5, 'STEALTH': 5, 'RAIDER': 5, 'TITAN': 5,
    'CREATOR': 5, 'MACBOOK PRO': 5, 'AERO': 5, 'AORUS': 5, 'G5': 5,
    'BLADE': 5,
}

# ============================================================================
# CONDITION VALUE RETENTION MAP (from preprocessing)
# ============================================================================
CONDITION_VALUE_RETENTION = {
    'JAMAIS UTILISÉ': 1.00,
    'BON ÉTAT': 0.75,
    'MOYEN': 0.60,
    'UNKNOWN': 0.70,
}

# ============================================================================
# DDR ORDINAL MAP (from preprocessing)
# ============================================================================
DDR_ORDINAL_MAP = {
    'Unknown': 0,
    'DDR3': 3,
    'LPDDR4x': 4,
    'DDR4': 4,
    'LPDDR5': 5,
    'DDR5': 5,
    'LPDDR5X': 5,
}

# ============================================================================
# PASSMARK LOOKUP TABLES
# ============================================================================
_cpu_passmark_cache = None
_gpu_passmark_cache = None


def _load_cpu_passmark():
    """Load CPU PassMark data and build a lookup dict."""
    global _cpu_passmark_cache
    if _cpu_passmark_cache is not None:
        return _cpu_passmark_cache

    csv_path = MODEL_DIR / 'cpu_passmark.csv'
    _cpu_passmark_cache = {}
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip().upper()
                try:
                    _cpu_passmark_cache[name] = {
                        'cpumark': float(row['cpumark']) if row['cpumark'] else None,
                        'single_thread': float(row['single_thread']) if row['single_thread'] else None,
                        'cores': float(row['cores']) if row['cores'] else None,
                        'threads': float(row['threads']) if row['threads'] else None,
                    }
                except (ValueError, KeyError):
                    continue
    return _cpu_passmark_cache


def _load_gpu_passmark():
    """Load GPU PassMark data and build a lookup dict."""
    global _gpu_passmark_cache
    if _gpu_passmark_cache is not None:
        return _gpu_passmark_cache

    csv_path = MODEL_DIR / 'gpu_passmark.csv'
    _gpu_passmark_cache = {}
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip().upper()
                try:
                    _gpu_passmark_cache[name] = {
                        'g3d': float(row['g3d']) if row['g3d'] else None,
                    }
                except (ValueError, KeyError):
                    continue
    return _gpu_passmark_cache


def _lookup_cpu_scores(cpu_family, cpu_generation, cpu_suffix, cpu_brand):
    """
    Attempt to look up CPU PassMark scores by building likely CPU names.
    Returns (cpu_mark, cpu_single_thread, cpu_cores, cpu_threads) or Nones.
    """
    passmark = _load_cpu_passmark()
    if not passmark:
        return None, None, None, None

    gen = int(cpu_generation) if cpu_generation else 0
    suffix_str = cpu_suffix.upper() if cpu_suffix else ''

    search_names = []

    if cpu_family in ('i3', 'i5', 'i7', 'i9'):
        family_upper = cpu_family.upper()
        base_numbers = {
            'i3': f'{gen}100', 'i5': f'{gen}500',
            'i7': f'{gen}700', 'i9': f'{gen}900',
        }
        num = base_numbers.get(cpu_family, f'{gen}00')
        search_names.append(f'INTEL CORE {family_upper}-{num}{suffix_str}')
        search_names.append(f'INTEL CORE {family_upper}-{num}')

    elif cpu_family in ('r3', 'r5', 'r7', 'r9'):
        ryzen_map = {'r3': 'RYZEN 3', 'r5': 'RYZEN 5', 'r7': 'RYZEN 7', 'r9': 'RYZEN 9'}
        ryzen_name = ryzen_map[cpu_family]
        if gen > 0:
            search_names.append(f'AMD {ryzen_name} {gen}600{suffix_str}')
            search_names.append(f'AMD {ryzen_name} {gen}600')
        search_names.append(f'AMD {ryzen_name}')

    elif cpu_family in ('m1', 'm2', 'm3', 'm4'):
        apple_name = cpu_family.upper()
        if suffix_str:
            search_names.append(f'APPLE {apple_name} {suffix_str}')
        search_names.append(f'APPLE {apple_name}')

    elif cpu_family in ('ultra5', 'ultra7', 'ultra9'):
        ultra_num = cpu_family.replace('ultra', '')
        search_names.append(f'INTEL CORE ULTRA {ultra_num}')

    # Try exact matches first, then partial matches
    for name in search_names:
        if name in passmark:
            d = passmark[name]
            return d['cpumark'], d['single_thread'], d['cores'], d['threads']

    # Try partial matching
    for name in search_names:
        for key, d in passmark.items():
            if name in key:
                return d['cpumark'], d['single_thread'], d['cores'], d['threads']

    return None, None, None, None


def _lookup_gpu_score(gpu_tier, gpu_suffix):
    """
    Attempt to look up GPU PassMark G3D score.
    Returns gpu_score or None.
    """
    if gpu_tier == 'NONE':
        return 0.0

    passmark = _load_gpu_passmark()
    if not passmark:
        return None

    suffix_str = f' {gpu_suffix.upper()}' if gpu_suffix else ''

    search_names = []

    if gpu_tier.startswith('RTX') or gpu_tier.startswith('GTX'):
        search_names.append(f'NVIDIA GEFORCE {gpu_tier.upper()}{suffix_str}')
        search_names.append(f'GEFORCE {gpu_tier.upper()}{suffix_str}')
        search_names.append(f'NVIDIA GEFORCE {gpu_tier.upper()}')
        search_names.append(f'GEFORCE {gpu_tier.upper()}')
    elif gpu_tier.startswith('MX'):
        search_names.append(f'NVIDIA GEFORCE {gpu_tier.upper()}')
        search_names.append(f'GEFORCE {gpu_tier.upper()}')
    elif gpu_tier.startswith('Quadro') or gpu_tier.startswith('RTX Workstation'):
        search_names.append(f'NVIDIA {gpu_tier.upper()}')
        search_names.append(gpu_tier.upper())
    elif gpu_tier.startswith('RX'):
        search_names.append(f'AMD RADEON {gpu_tier.upper()}')
        search_names.append(f'RADEON {gpu_tier.upper()}')
    elif gpu_tier == 'ARC':
        search_names.append('INTEL ARC')
    else:
        search_names.append(gpu_tier.upper())

    # Try exact matches
    for name in search_names:
        if name in passmark:
            return passmark[name]['g3d']

    # Try partial matching
    for name in search_names:
        for key, d in passmark.items():
            if name in key:
                return d['g3d']

    return None


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
    """
    Transform user form data into the DataFrame expected by the model preprocessor.
    Computes all derived features (build_quality_tier, condition_value_retention, etc.)
    """
    cpu_family = form_data['cpu_family']
    brand = form_data['brand']
    series = form_data['series']
    condition = form_data['condition']

    # --- CPU brand inference ---
    if cpu_family in ['i3', 'i5', 'i7', 'i9', 'ultra5', 'ultra7', 'ultra9', 'celeron', 'pentium', 'xeon']:
        cpu_brand = 'INTEL'
    elif cpu_family in ['r3', 'r5', 'r7', 'r9']:
        cpu_brand = 'AMD'
    elif cpu_family in ['m1', 'm2', 'm3', 'm4']:
        cpu_brand = 'APPLE'
    elif cpu_family == 'SNAPDRAGON':
        cpu_brand = 'QUALCOMM'
    else:
        cpu_brand = form_data.get('cpu_brand', 'UNKNOWN')

    # --- CPU generation ---
    cpu_generation = form_data.get('cpu_generation', '0')
    if cpu_generation == '' or cpu_generation is None:
        cpu_generation = '0'
    if cpu_family in ['m1', 'm2', 'm3', 'm4', 'SNAPDRAGON', 'UNKNOWN']:
        cpu_generation = '0'

    cpu_gen_brand = f"{cpu_brand}_{int(cpu_generation)}"

    # --- Log transform RAM and SSD (matching notebook) ---
    ram_log = np.log1p(form_data['ram_size_gb'])
    ssd_size = form_data.get('ssd_size_gb', 0) or 0
    hdd_size = form_data.get('hdd_size_gb', 0) or 0
    ssd_log = np.log1p(ssd_size)

    # --- GPU fields ---
    gpu_tier = form_data.get('gpu_tier', 'NONE') or 'NONE'
    gpu_suffix = form_data.get('gpu_suffix', '') or ''
    cpu_suffix = form_data.get('cpu_suffix', '') or ''

    # --- Derived: has_gpu ---
    has_gpu = 1 if gpu_tier != 'NONE' else 0

    # --- Derived: is_gaming_series ---
    series_upper = series.upper() if series else ''
    is_gaming_series = 1 if any(g in series_upper for g in GAMING_SERIES) else 0

    # --- Derived: build_quality_tier ---
    build_quality_tier = SERIES_TIER_MAP.get(series, 2)
    if brand == 'APPLE' and build_quality_tier < 4:
        build_quality_tier = 4

    # --- Derived: condition_value_retention ---
    condition_value_retention = CONDITION_VALUE_RETENTION.get(condition, 0.70)

    # --- Derived: storage_perf_score ---
    if ssd_size > 0:
        storage_perf_score = 2
    elif hdd_size > 0:
        storage_perf_score = 1
    else:
        storage_perf_score = 0

    # --- Derived: inferred_ddr_ordinal ---
    ram_type = form_data.get('ram_type_class', 'Unknown') or 'Unknown'
    inferred_ddr_ordinal = DDR_ORDINAL_MAP.get(ram_type, 0)

    # --- Screen size ---
    screen_size = form_data.get('screen_size', 15.6)
    if screen_size is None:
        screen_size = 15.6

    # --- PassMark lookups ---
    cpu_mark, cpu_single_thread, cpu_cores, cpu_threads = _lookup_cpu_scores(
        cpu_family, cpu_generation, cpu_suffix, cpu_brand
    )
    gpu_score = _lookup_gpu_score(gpu_tier, gpu_suffix)

    # Use NaN for missing values — the preprocessor's SimpleImputer will handle them
    if cpu_mark is None:
        cpu_mark = np.nan
    if cpu_single_thread is None:
        cpu_single_thread = np.nan
    if cpu_cores is None:
        cpu_cores = np.nan
    if cpu_threads is None:
        cpu_threads = np.nan
    if gpu_score is None:
        gpu_score = np.nan

    # --- Resolution class ---
    resolution_class = form_data.get('resolution_class', 'Unknown') or 'Unknown'

    # --- Build the feature dict ---
    data = {
        # Numeric features
        'RAM_SIZE_GB': ram_log,
        'SSD_SIZE_GB': ssd_log,
        'cpu_mark': cpu_mark,
        'cpu_single_thread': cpu_single_thread,
        'gpu_score': gpu_score,
        'build_quality_tier': build_quality_tier,
        'condition_value_retention': condition_value_retention,
        'cpu_generation': int(cpu_generation),
        'storage_perf_score': storage_perf_score,
        'cpu_cores': cpu_cores,
        'inferred_ddr_ordinal': inferred_ddr_ordinal,
        'SCREEN_SIZE_NUM': float(screen_size),
        'cpu_threads': cpu_threads,
        # Binary features
        'is_pro': 1 if form_data.get('is_pro', False) else 0,
        'is_gaming_series': is_gaming_series,
        'has_gpu': has_gpu,
        # Categorical features
        'series': series,
        'cpu_family': cpu_family,
        'gpu_tier': gpu_tier,
        'condition': condition,
        'gpu_suffix': gpu_suffix,
        'cpu_suffix': cpu_suffix,
        'cpu_gen_brand': cpu_gen_brand,
        'listing_year': str(form_data['listing_year']),
        'brand': brand,
        'ram_type_class': ram_type,
        'resolution_class': resolution_class,
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

        model_rmse = 28000  # Approximate RMSE for the Gradient Boosting model
        min_price = max(predicted_price - model_rmse, 5000)
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
