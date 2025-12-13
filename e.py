"""
Generate missing preprocessing artifacts (feature_names.pkl and preprocessing_metadata.json)
Run this once: python generate_missing_artifacts.py
"""

import pickle
import json
from pathlib import Path
import pandas as pd

# Paths
ARTIFACTS_DIR = Path("./data/processed/preprocessing_artifacts")

# The exact feature order from your training data
FEATURE_ORDER = [
    'Year', 'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg', 
    'prcp_growing_total', 'rh_growing_avg', 'gdd_total', 'heat_stress_days', 
    'temp_range_avg', 'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct', 
    'ph', 'bulk_density', 'cec', 'awc', 'Latitude', 'Longitude', 
    'Year_Since_Start', 'Year_Squared', 'temp_prcp_interaction', 'gdd_per_day', 
    'moisture_stress', 'soil_quality_score', 'loam_index', 'temp_variability', 
    'extreme_heat_flag', 'lat_squared', 'long_squared', 'lat_long_interaction', 
    'Crop_Encoded', 'State_Encoded', 'County_Encoded'
]

def generate_artifacts():
    """Generate the missing artifacts"""
    
    print("="*70)
    print("GENERATING MISSING PREPROCESSING ARTIFACTS")
    print("="*70)
    
    # 1. Save feature_names.pkl
    print("\n1. Saving feature_names.pkl...")
    with open(ARTIFACTS_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(FEATURE_ORDER, f)
    print(f"   ✅ Saved {len(FEATURE_ORDER)} features to feature_names.pkl")
    
    # 2. Calculate extreme_heat_threshold from training data
    print("\n2. Calculating extreme_heat_threshold from SCALED training data...")
    try:
        with open(ARTIFACTS_DIR / 'X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        
        # Check if data is scaled
        tmax_mean = X_train['tmax_growing_avg'].mean()
        
        if abs(tmax_mean) < 0.1:
            # Data is scaled - use 95th percentile of scaled values
            extreme_heat_threshold_scaled = X_train['tmax_growing_avg'].quantile(0.95)
            
            # Convert back to unscaled temperature for display
            with open(ARTIFACTS_DIR / 'weather_scaler.pkl', 'rb') as f:
                weather_scaler = pickle.load(f)
            
            # tmax is at index 2 in weather features
            tmax_scaler_mean = weather_scaler.mean_[2]
            tmax_scaler_scale = weather_scaler.scale_[2]
            
            extreme_heat_threshold = (extreme_heat_threshold_scaled * tmax_scaler_scale) + tmax_scaler_mean
            
            print(f"   Scaled threshold (95th percentile): {extreme_heat_threshold_scaled:.4f}")
            print(f"   Unscaled threshold: {extreme_heat_threshold:.1f}°C")
        else:
            # Data is not scaled - use 95th percentile directly
            extreme_heat_threshold = X_train['tmax_growing_avg'].quantile(0.95)
            print(f"   Using threshold: {extreme_heat_threshold:.1f}°C (from unscaled data)")
        
    except Exception as e:
        print(f"   ⚠️  Could not calculate from data: {e}")
        extreme_heat_threshold = 32.0
        print(f"   Using default: {extreme_heat_threshold}°C")
    
    # 3. Create preprocessing_metadata.json
    print("\n3. Saving preprocessing_metadata.json...")
    metadata = {
        'feature_names': FEATURE_ORDER,
        'n_features': len(FEATURE_ORDER),
        'extreme_heat_threshold': extreme_heat_threshold,
        'feature_groups': {
            'weather': [
                'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg',
                'prcp_growing_total', 'rh_growing_avg', 'gdd_total',
                'heat_stress_days', 'temp_range_avg',
                'temp_prcp_interaction', 'gdd_per_day', 'moisture_stress',
                'temp_variability'
            ],
            'soil': [
                'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct',
                'ph', 'bulk_density', 'cec', 'awc',
                'soil_quality_score', 'loam_index'
            ],
            'geo': [
                'Latitude', 'Longitude', 'lat_squared', 
                'long_squared', 'lat_long_interaction'
            ],
            'time': [
                'Year', 'Year_Since_Start', 'Year_Squared'
            ],
            'categorical': [
                'Crop_Encoded', 'State_Encoded', 'County_Encoded'
            ],
            'flags': [
                'extreme_heat_flag'
            ]
        },
        'version': '1.0',
        'description': 'Preprocessing metadata for crop yield prediction model'
    }
    
    with open(ARTIFACTS_DIR / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ Saved preprocessing_metadata.json")
    
    # 4. Verify all artifacts exist
    print("\n4. Verifying all required artifacts...")
    required_files = [
        'feature_names.pkl',
        'preprocessing_metadata.json',
        'crop_encoder.pkl',
        'state_encoder.pkl',
        'county_encoder.pkl',
        'numerical_scaler.pkl',
        'weather_scaler.pkl',
        'soil_scaler.pkl',
        'geo_scaler.pkl',
        'X_train.pkl',
        'y_train.pkl',
        'X_val.pkl',
        'y_val.pkl',
        'X_test.pkl',
        'y_test.pkl'
    ]
    
    missing = []
    for filename in required_files:
        filepath = ARTIFACTS_DIR / filename
        if filepath.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (MISSING)")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} files are missing!")
        print("You may need to run preprocessing again.")
    else:
        print("\n✅ All artifacts verified!")
    
    print("\n" + "="*70)
    print("DONE! Now copy artifacts to your Flask app:")
    print("="*70)
    print("\n# Copy all preprocessing artifacts")
    print("cp data/processed/preprocessing_artifacts/*.pkl api/artifacts/")
    print("cp data/processed/preprocessing_artifacts/*.json api/artifacts/")
    print("\n# Copy your trained model")
    print("cp models/xgboost.pkl api/artifacts/")
    print("\n" + "="*70)

if __name__ == "__main__":
    # Create artifacts directory if it doesn't exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_artifacts()