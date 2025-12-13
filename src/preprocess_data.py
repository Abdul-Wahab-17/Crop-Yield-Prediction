"""
Comprehensive Preprocessing Pipeline for Crop Yield Prediction
Location: src/preprocess_data.py
Run from root directory: python src/preprocess_data.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import pickle
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

class CropYieldPreprocessor:
    def __init__(self):
        # Set up paths relative to project root
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.artifacts_dir = self.processed_dir / 'preprocessing_artifacts'
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize all transformers
        self.crop_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        self.county_encoder = LabelEncoder()
        
        self.numerical_scaler = RobustScaler()
        self.weather_scaler = StandardScaler()
        self.soil_scaler = StandardScaler()
        self.geo_scaler = StandardScaler()
        
        # Storage for feature info
        self.feature_columns = {}
        self.crop_stats = {}
        self.metadata = {}
        
    def load_and_validate(self):
        """Load data from processed directory"""
        print("Loading data...")
        filepath = self.processed_dir / 'master_yield_dataset.csv'
        
        if not filepath.exists():
            raise FileNotFoundError(f"Master dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        
        print(f"Initial shape: {df.shape}")
        print(f"\nColumns in dataset:")
        print(df.columns.tolist())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        # Basic validation
        assert 'Yield' in df.columns, "Target variable 'Yield' not found"
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values strategically"""
        print("\nHandling missing values...")
        
        # Drop AREA_HARVESTED and PRODUCTION if they exist (outcome variables)
        cols_to_drop = ['AREA_HARVESTED', 'PRODUCTION']
        existing_cols = [col for col in cols_to_drop if col in df.columns]
        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"Dropped outcome columns: {existing_cols}")
        
        # Remove rows where Yield is missing
        initial_len = len(df)
        df = df[df['Yield'].notna()]
        if len(df) < initial_len:
            print(f"Removed {initial_len - len(df)} rows with missing Yield")
        
        # Check for other missing values
        remaining_missing = df.isnull().sum()
        if remaining_missing.any() and remaining_missing.sum() > 0:
            print(f"Warning: Remaining missing values:\n{remaining_missing[remaining_missing > 0]}")
            df = df.dropna()
            print(f"Dropped rows with missing values. New shape: {df.shape}")
        
        return df
    
    def create_features(self, df):
        """Engineer additional features"""
        print("\nEngineering features...")
        
        # Time-based features
        df['Year_Since_Start'] = df['Year'] - df['Year'].min()
        df['Year_Squared'] = df['Year'] ** 2
        
        # Climate interaction features
        df['temp_prcp_interaction'] = df['tavg_growing_avg'] * df['prcp_growing_total']
        df['gdd_per_day'] = df['gdd_total'] / 180
        df['moisture_stress'] = df['prcp_growing_total'] / (df['tavg_growing_avg'] + 1)
        
        # Soil quality composite
        df['soil_quality_score'] = (
            df['organic_matter_pct'] * 0.3 +
            (7 - abs(df['ph'] - 6.5)) * 0.2 +
            df['awc'] * 100 * 0.3 +
            df['cec'] * 0.2
        )
        
        # Texture-based features
        df['loam_index'] = 100 - (abs(df['clay_pct'] - 20) + abs(df['sand_pct'] - 40) + abs(df['silt_pct'] - 40))
        
        # Weather extremes
        df['temp_variability'] = df['temp_range_avg'] * df['heat_stress_days']
        df['extreme_heat_flag'] = (df['heat_stress_days'] > df.groupby('Crop')['heat_stress_days'].transform('median')).astype(int)
        
        # Geographic features
        df['lat_squared'] = df['Latitude'] ** 2
        df['long_squared'] = df['Longitude'] ** 2
        df['lat_long_interaction'] = df['Latitude'] * df['Longitude']
        
        print(f"Created 13 new engineered features")
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        # Encode Crop
        if fit:
            df['Crop_Encoded'] = self.crop_encoder.fit_transform(df['Crop'])
            self.metadata['crop_classes'] = self.crop_encoder.classes_.tolist()
        else:
            df['Crop_Encoded'] = self.crop_encoder.transform(df['Crop'])
        
        # Encode State
        if fit:
            df['State_Encoded'] = self.state_encoder.fit_transform(df['State'])
            self.metadata['state_classes'] = self.state_encoder.classes_.tolist()
        else:
            df['State_Encoded'] = self.state_encoder.transform(df['State'])
        
        # County encoding (within state context)
        df['County_State'] = df['County'] + '_' + df['State']
        if fit:
            df['County_Encoded'] = self.county_encoder.fit_transform(df['County_State'])
            self.metadata['county_classes'] = self.county_encoder.classes_.tolist()
        else:
            known_counties = set(self.county_encoder.classes_)
            df['County_Encoded'] = df['County_State'].apply(
                lambda x: self.county_encoder.transform([x])[0] if x in known_counties else -1
            )
        
        df = df.drop(columns=['County_State'])
        
        print(f"Encoded {len(self.metadata.get('crop_classes', []))} crops, "
              f"{len(self.metadata.get('state_classes', []))} states, "
              f"{len(self.metadata.get('county_classes', []))} counties")
        
        return df
    
    def compute_crop_statistics(self, df):
        """Compute crop-specific statistics"""
        print("\nComputing crop statistics...")
        
        for crop in df['Crop'].unique():
            crop_data = df[df['Crop'] == crop]
            self.crop_stats[crop] = {
                'mean_yield': float(crop_data['Yield'].mean()),
                'std_yield': float(crop_data['Yield'].std()),
                'median_yield': float(crop_data['Yield'].median()),
                'min_yield': float(crop_data['Yield'].min()),
                'max_yield': float(crop_data['Yield'].max()),
                'count': int(len(crop_data))
            }
        
        print(f"Computed statistics for {len(self.crop_stats)} crops")
    
    def scale_features(self, df, fit=True):
        """Scale different feature groups"""
        print("\nScaling features...")
        
        # Define feature groups
        weather_features = [
            'tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg',
            'prcp_growing_total', 'rh_growing_avg', 'gdd_total',
            'heat_stress_days', 'temp_range_avg',
            'temp_prcp_interaction', 'gdd_per_day', 'moisture_stress',
            'temp_variability'
        ]
        
        soil_features = [
            'clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct',
            'ph', 'bulk_density', 'cec', 'awc',
            'soil_quality_score', 'loam_index'
        ]
        
        geo_features = [
            'Latitude', 'Longitude', 'lat_squared', 'long_squared',
            'lat_long_interaction'
        ]
        
        time_features = ['Year', 'Year_Since_Start', 'Year_Squared']
        
        # Store feature groups
        self.feature_columns['weather'] = weather_features
        self.feature_columns['soil'] = soil_features
        self.feature_columns['geo'] = geo_features
        self.feature_columns['time'] = time_features
        
        # Scale each group
        if fit:
            df[weather_features] = self.weather_scaler.fit_transform(df[weather_features])
            df[soil_features] = self.soil_scaler.fit_transform(df[soil_features])
            df[geo_features] = self.geo_scaler.fit_transform(df[geo_features])
            df[time_features] = self.numerical_scaler.fit_transform(df[time_features])
        else:
            df[weather_features] = self.weather_scaler.transform(df[weather_features])
            df[soil_features] = self.soil_scaler.transform(df[soil_features])
            df[geo_features] = self.geo_scaler.transform(df[geo_features])
            df[time_features] = self.numerical_scaler.transform(df[time_features])
        
        print(f"Scaled features: {len(weather_features)} weather, {len(soil_features)} soil, "
              f"{len(geo_features)} geo, {len(time_features)} time")
        
        return df
    
    def prepare_model_data(self, df):
        """Prepare final feature matrix and target"""
        print("\nPreparing model data...")
        
        # Columns to exclude from features
        exclude_cols = [
            'Yield',  # Target
            'Crop', 'State', 'County',  # Original categorical (encoded versions kept)
        ]
        
        # Add optional columns if they exist
        optional_exclude = ['State_Name', 'STATE_FIPS_CODE', 'COUNTY_CODE']
        for col in optional_exclude:
            if col in df.columns:
                exclude_cols.append(col)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['Yield']
        
        self.feature_columns['final_features'] = feature_cols
        self.metadata['n_features'] = len(feature_cols)
        
        print(f"Final feature count: {len(feature_cols)}")
        print(f"Features: {feature_cols}")
        
        return X, y, df[['Crop', 'State', 'County', 'Year']]
    
    def temporal_split(self, X, y, metadata, test_years=3, val_years=2):
        """Split data by time to prevent data leakage"""
        print(f"\nPerforming temporal split...")
        
        unique_years = sorted(metadata['Year'].unique())
        
        train_years = unique_years[:-test_years-val_years]
        val_years_list = unique_years[-test_years-val_years:-test_years]
        test_years_list = unique_years[-test_years:]
        
        train_idx = metadata['Year'].isin(train_years)
        val_idx = metadata['Year'].isin(val_years_list)
        test_idx = metadata['Year'].isin(test_years_list)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"Train years: {train_years[0]}-{train_years[-1]} ({len(train_years)} years, {len(X_train)} samples)")
        print(f"Val years: {val_years_list[0]}-{val_years_list[-1]} ({len(val_years_list)} years, {len(X_val)} samples)")
        print(f"Test years: {test_years_list[0]}-{test_years_list[-1]} ({len(test_years_list)} years, {len(X_test)} samples)")
        
        self.metadata['split_info'] = {
            'train_years': [int(y) for y in train_years],
            'val_years': [int(y) for y in val_years_list],
            'test_years': [int(y) for y in test_years_list]
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_artifacts(self):
        """Save all preprocessing artifacts"""
        print("\nSaving preprocessing artifacts...")
        
        artifacts = {
            'crop_encoder': self.crop_encoder,
            'state_encoder': self.state_encoder,
            'county_encoder': self.county_encoder,
            'weather_scaler': self.weather_scaler,
            'soil_scaler': self.soil_scaler,
            'geo_scaler': self.geo_scaler,
            'numerical_scaler': self.numerical_scaler,
        }
        
        for name, artifact in artifacts.items():
            filepath = self.artifacts_dir / f'{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(artifact, f)
            print(f"  Saved {name}")
        
        # Save metadata and feature info
        with open(self.artifacts_dir / 'feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        with open(self.artifacts_dir / 'crop_stats.json', 'w') as f:
            json.dump(self.crop_stats, f, indent=2)
        
        with open(self.artifacts_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"\nAll artifacts saved to {self.artifacts_dir}")
    
    def save_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed datasets"""
        print("\nSaving processed datasets...")
        
        datasets = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        for name, data in datasets.items():
            filepath = self.artifacts_dir / f'{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            shape = data.shape if hasattr(data, 'shape') else len(data)
            print(f"  Saved {name}: {shape}")
    
    def run_full_pipeline(self, test_years=3, val_years=2):
        """Run complete preprocessing pipeline"""
        print("="*70)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*70)
        
        # Load and clean
        df = self.load_and_validate()
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.create_features(df)
        
        # Encoding
        df = self.encode_categorical(df, fit=True)
        
        # Statistics
        self.compute_crop_statistics(df)
        
        # Scaling
        df = self.scale_features(df, fit=True)
        
        # Prepare data
        X, y, metadata = self.prepare_model_data(df)
        
        # Temporal split
        X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_split(
            X, y, metadata, test_years, val_years
        )
        
        # Save everything
        self.save_artifacts()
        self.save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"\nArtifacts saved to: {self.artifacts_dir}")
        print(f"\nDataset sizes:")
        print(f"  Training:   {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test:       {X_test.shape}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }


if __name__ == "__main__":
    preprocessor = CropYieldPreprocessor()
    datasets = preprocessor.run_full_pipeline(test_years=3, val_years=2)
    
    print("\n✅ Preprocessing completed successfully!")
    print("Next step: Run training script")
    print("Command: python src/model/train_models.py")