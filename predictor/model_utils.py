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


def _lookup_cpu_scores(cpu_family, cpu_generation, cpu_suffix, cpu_brand, is_pro=False):
    """
    Attempt to look up CPU PassMark scores by building likely CPU names.
    Returns (cpu_mark, cpu_single_thread, cpu_cores, cpu_threads, match_info) or Nones.
    """
    passmark = _load_cpu_passmark()
    if not passmark:
        return None, None, None, None, None

    gen = int(cpu_generation) if cpu_generation else 0
    suffix_str = cpu_suffix.upper() if cpu_suffix else ''

    search_names = []

    # ===== Intel Core i3, i5, i7, i9 =====
    if cpu_family in ('i3', 'i5', 'i7', 'i9'):
        family_upper = cpu_family.upper()
        variant = "XXX"
        model_num = f'{gen}{variant}'
        search_names.append(f'INTEL CORE {family_upper}-{model_num}{suffix_str}')

    # ===== AMD Ryzen 3, 5, 7, 9 =====
    elif cpu_family in ('r3', 'r5', 'r7', 'r9'):
        ryzen_map = {'r3': 'RYZEN 3', 'r5': 'RYZEN 5', 'r7': 'RYZEN 7', 'r9': 'RYZEN 9'}
        ryzen_name = ryzen_map[cpu_family]
        
        if gen > 0:
            variant = 'XXX'
            model_num = f'{gen}{variant}'
            if is_pro:
                search_names.append(f'AMD {ryzen_name} PRO {model_num}{suffix_str}')
            else:
                search_names.append(f'AMD {ryzen_name} {model_num}{suffix_str}')

    # ===== Apple Silicon (M1, M2, M3, M4) =====
    elif cpu_family in ('m1', 'm2', 'm3', 'm4'):
        apple_name = cpu_family.upper()
        if suffix_str in ('PRO', 'MAX', 'ULTRA'):
            search_names.append(f'APPLE {apple_name} {suffix_str}')
        else:
            search_names.append(f'APPLE {apple_name}')

    # ===== Intel Core Ultra (Ultra 5, 7, 9) =====
    elif cpu_family in ('ultra5', 'ultra7', 'ultra9'):
        ultra_num = cpu_family.replace('ultra', '')
        if gen in [1, 2, 3]:
            variant = 'XX'
            search_names.append(f'INTEL CORE ULTRA {ultra_num} {gen}{variant}{suffix_str}')
            
    # ===== Other processors =====
    elif cpu_family in ('celeron', 'pentium', 'xeon'):
        family_name = cpu_family.upper()
        search_names.append(f'INTEL {family_name}')
        
    elif cpu_family == 'SNAPDRAGON':
        search_names.append('QUALCOMM SNAPDRAGON')
        search_names.append('SNAPDRAGON')

    # Try fuzzy matching based on POSITIONAL character similarity
    # First, try to find matches with the SAME generation
    best_match = None
    best_match_data = None
    max_similarity = 0
    search_name_used = None
    
    for name in search_names:
        # Clean the search name once
        name_clean = name.upper().replace('-', '').replace(' ', '')
        
        for key, d in passmark.items():
            # Clean the database key
            key_clean = key.upper().replace('-', '').replace(' ', '')
            
            # CRITICAL: For Intel/AMD CPUs with generations, prioritize same-generation matches
            if gen > 0 and cpu_family in ('i3', 'i5', 'i7', 'i9', 'r3', 'r5', 'r7', 'r9'):
                # Check if the generation appears in the key
                gen_str = str(gen)
                
                # For Intel 10th gen and above, check for the generation number
                if gen >= 10:
                    # Must start with the generation (e.g., "12" in "I5-12450")
                    # Find the position after "I5-" or "RYZEN5"
                    if f'I{cpu_family[-1]}{gen}' not in key_clean and f'RYZEN{cpu_family[-1]}{gen}' not in key_clean:
                        continue  # Skip CPUs from different generations
                else:
                    # For older gens (6-9), look for single digit after family
                    if f'I{cpu_family[-1]}{gen}' not in key_clean and f'RYZEN{cpu_family[-1]}{gen}' not in key_clean:
                        continue
            
            # Count characters that match in the SAME POSITION
            matching_positions = 0
            min_len = min(len(name_clean), len(key_clean))
            
            for i in range(min_len):
                if name_clean[i] == key_clean[i]:
                    matching_positions += 1
            
            # Calculate similarity as ratio of matching positions to search name length
            similarity = matching_positions / len(name_clean) if len(name_clean) > 0 else 0
            
            # Update best match
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = key
                best_match_data = d
                search_name_used = name
    
    # If no match found with generation filter, try again WITHOUT the generation filter
    if not best_match:
        for name in search_names:
            name_clean = name.upper().replace('-', '').replace(' ', '')
            
            for key, d in passmark.items():
                key_clean = key.upper().replace('-', '').replace(' ', '')
                
                matching_positions = 0
                min_len = min(len(name_clean), len(key_clean))
                
                for i in range(min_len):
                    if name_clean[i] == key_clean[i]:
                        matching_positions += 1
                
                similarity = matching_positions / len(name_clean) if len(name_clean) > 0 else 0
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = key
                    best_match_data = d
                    search_name_used = name
    
    # Prepare match info dictionary
    match_info = None
    if best_match:
        match_info = {
            'matched_cpu': best_match,
            'search_pattern': search_name_used,
            'similarity': round(max_similarity * 100, 2),
        }
    
    if best_match_data:
        return (
            best_match_data['cpumark'], 
            best_match_data['single_thread'], 
            best_match_data['cores'], 
            best_match_data['threads'],
            match_info
        )

    return None, None, None, None, None

def _lookup_gpu_score(gpu_tier, gpu_suffix):
    """
    Attempt to look up GPU PassMark G3D score.
    Returns (gpu_score, match_info) or (None, None).
    """
    if gpu_tier == 'NONE':
        return 0.0, None

    passmark = _load_gpu_passmark()
    if not passmark:
        return None, None

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

    # Try exact matches first
    for name in search_names:
        if name in passmark:
            match_info = {
                'matched_gpu': name,
                'search_pattern': name,
                'similarity': 100.0,
                'match_type': 'exact'
            }
            return passmark[name]['g3d'], match_info

    # Try fuzzy matching based on positional character similarity
    best_match = None
    best_match_data = None
    max_similarity = 0
    search_name_used = None
    
    for name in search_names:
        # Clean the search name
        name_clean = name.upper().replace('-', '').replace(' ', '')
        
        for key, d in passmark.items():
            # Clean the database key
            key_clean = key.upper().replace('-', '').replace(' ', '')
            
            # Count characters that match in the SAME POSITION
            matching_positions = 0
            min_len = min(len(name_clean), len(key_clean))
            
            for i in range(min_len):
                if name_clean[i] == key_clean[i]:
                    matching_positions += 1
            
            # Calculate similarity
            similarity = matching_positions / len(name_clean) if len(name_clean) > 0 else 0
            
            # Update best match
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = key
                best_match_data = d
                search_name_used = name

    # Prepare match info
    match_info = None
    if best_match:
        match_info = {
            'matched_gpu': best_match,
            'search_pattern': search_name_used,
            'similarity': round(max_similarity * 100, 2),
            'match_type': 'fuzzy'
        }
        return best_match_data['g3d'], match_info

    return None, None


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
    # --- Define all basic variables first ---
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

    # --- CPU suffix ---
    cpu_suffix = form_data.get('cpu_suffix', '') or ''

    # --- CPU gen_brand ---
    cpu_gen_brand = f"{cpu_brand}_{int(cpu_generation)}"

    # --- Log transform RAM and SSD (matching notebook) ---
    ram_log = np.log1p(form_data['ram_size_gb'])
    ssd_size = form_data.get('ssd_size_gb', 0) or 0
    hdd_size = form_data.get('hdd_size_gb', 0) or 0
    ssd_log = np.log1p(ssd_size)

    # --- GPU fields ---
    gpu_tier = form_data.get('gpu_tier', 'NONE') or 'NONE'
    gpu_suffix = form_data.get('gpu_suffix', '') or ''

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

    # --- PassMark lookups (NOW all variables are defined) ---
    cpu_mark, cpu_single_thread, cpu_cores, cpu_threads, cpu_match_info = _lookup_cpu_scores(
        cpu_family, cpu_generation, cpu_suffix, cpu_brand, 
        is_pro=form_data.get('is_pro', False)
    )
    gpu_score, gpu_match_info = _lookup_gpu_score(gpu_tier, gpu_suffix)

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

    return df, cpu_match_info, gpu_match_info
def get_feature_importance(model, preprocessor, top_n=10):
    """
    Get the top N most important features from the model.
    Returns a list of dicts with feature names and their importance scores.
    """
    try:
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            return []
        
        importances = model.feature_importances_
        
        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()
        
        # Create list of (feature_name, importance) tuples
        feature_imp = list(zip(feature_names, importances))
        
        # Sort by importance (descending)
        feature_imp_sorted = sorted(feature_imp, key=lambda x: x[1], reverse=True)
        
        # Return top N features
        top_features = feature_imp_sorted[:top_n]
        
        # Format as a more readable list of dicts
        result = [
            {
                'feature': _clean_feature_name(feat),
                'importance': round(imp * 100, 2),  # Convert to percentage
                'importance_formatted': f"{imp * 100:.2f}%"
            }
            for feat, imp in top_features
        ]
        
        return result
        
    except Exception as e:
        # If anything goes wrong, return empty list
        return []


def _clean_feature_name(feature_name):
    """
    Clean up feature names from preprocessor output for better readability.
    Example: 'cat__brand_ASUS' -> 'Brand: ASUS'
    """
    # Remove pipeline step prefixes (e.g., 'num__', 'cat__')
    if '__' in feature_name:
        feature_name = feature_name.split('__', 1)[1]
    
    # Replace underscores with spaces and title case
    feature_name = feature_name.replace('_', ' ').title()
    
    return feature_name


def _get_feature_values(input_df):
    """
    Extract and format the actual feature values used for prediction.
    Returns a dict with categorized features.
    """
    row = input_df.iloc[0]
    
    # Numeric features with descriptions
    numeric_features = [
        {'name': 'CPU Mark (PassMark)', 'value': row['cpu_mark'], 'format': '.0f'},
        {'name': 'CPU Single Thread Score', 'value': row['cpu_single_thread'], 'format': '.0f'},
        {'name': 'CPU Cores', 'value': row['cpu_cores'], 'format': '.0f'},
        {'name': 'CPU Threads', 'value': row['cpu_threads'], 'format': '.0f'},
        {'name': 'GPU Score (PassMark G3D)', 'value': row['gpu_score'], 'format': '.0f'},
        {'name': 'RAM Size (log-transformed)', 'value': row['RAM_SIZE_GB'], 'format': '.2f'},
        {'name': 'SSD Size (log-transformed)', 'value': row['SSD_SIZE_GB'], 'format': '.2f'},
        {'name': 'Screen Size', 'value': row['SCREEN_SIZE_NUM'], 'format': '.1f'},
        {'name': 'CPU Generation', 'value': row['cpu_generation'], 'format': '.0f'},
    ]
    
    # Derived features
    derived_features = [
        {'name': 'Build Quality Tier', 'value': row['build_quality_tier'], 'format': '.0f', 'description': '(1=Budget to 5=Flagship)'},
        {'name': 'Condition Value Retention', 'value': row['condition_value_retention'], 'format': '.2f', 'description': '(0.0 to 1.0)'},
        {'name': 'Storage Performance Score', 'value': row['storage_perf_score'], 'format': '.0f', 'description': '(0=None, 1=HDD, 2=SSD)'},
        {'name': 'DDR Generation (Ordinal)', 'value': row['inferred_ddr_ordinal'], 'format': '.0f', 'description': '(3=DDR3, 4=DDR4, 5=DDR5)'},
    ]
    
    # Binary features
    binary_features = [
        {'name': 'Has Dedicated GPU', 'value': 'Yes' if row['has_gpu'] == 1 else 'No'},
        {'name': 'Is Gaming Series', 'value': 'Yes' if row['is_gaming_series'] == 1 else 'No'},
        {'name': 'Is Professional', 'value': 'Yes' if row['is_pro'] == 1 else 'No'},
    ]
    
    # Categorical features
    categorical_features = [
        {'name': 'Brand', 'value': row['brand']},
        {'name': 'Series', 'value': row['series']},
        {'name': 'CPU Family', 'value': row['cpu_family']},
        {'name': 'CPU Gen + Brand', 'value': row['cpu_gen_brand']},
        {'name': 'GPU Tier', 'value': row['gpu_tier']},
        {'name': 'Condition', 'value': row['condition']},
        {'name': 'RAM Type', 'value': row['ram_type_class']},
        {'name': 'Resolution Class', 'value': row['resolution_class']},
        {'name': 'Listing Year', 'value': row['listing_year']},
    ]
    
    # Format numeric values
    for feat in numeric_features:
        val = feat['value']
        if pd.isna(val):
            feat['value_formatted'] = 'N/A (imputed by model)'
        else:
            feat['value_formatted'] = f"{val:{feat['format']}}"
    
    for feat in derived_features:
        val = feat['value']
        feat['value_formatted'] = f"{val:{feat['format']}}"
    
    return {
        'numeric': numeric_features,
        'derived': derived_features,
        'binary': binary_features,
        'categorical': categorical_features,
    }


def predict_price(form_data):
    try:
        model, preprocessor = load_model()
        input_df, cpu_match_info, gpu_match_info = prepare_input_data(form_data)  # Unpack all three
        
        X = preprocessor.transform(input_df)
        
        y_log = model.predict(X)
        predicted_price = np.expm1(y_log[0])
        predicted_price = max(predicted_price, 10000)

        model_rmse = 28000
        min_price = max(predicted_price - model_rmse, 5000)
        max_price = predicted_price + model_rmse

        # Get feature importance
        feature_importance = get_feature_importance(model, preprocessor)
        
        # Get feature values used for prediction
        feature_values = _get_feature_values(input_df)

        return {
            'success': True,
            'predicted_price': round(predicted_price, 0),
            'predicted_price_formatted': f"{predicted_price:,.0f} DZD",
            'min_price': round(min_price, 0),
            'max_price': round(max_price, 0),
            'min_price_formatted': f"{min_price:,.0f} DZD",
            'max_price_formatted': f"{max_price:,.0f} DZD",
            'feature_importance': feature_importance,
            'feature_values': feature_values,
            'cpu_match_info': cpu_match_info,
            'gpu_match_info': gpu_match_info  # Add GPU match info
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