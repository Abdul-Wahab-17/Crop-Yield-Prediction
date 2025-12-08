"""
Complete ML Pipeline for Crop Yield Prediction
Location: src/model/train_models.py
Run from root directory: python src/model/train_models.py

Includes: Baselines, Classical ML, Ensemble Models, Evaluation & Comparison
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                            mean_absolute_percentage_error, median_absolute_error)

# Advanced models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class CropYieldModelPipeline:
    def __init__(self):
        # Set up paths relative to project root
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data' / 'processed'
        self.artifacts_dir = self.data_dir / 'preprocessing_artifacts'
        self.models_dir = self.project_root / 'models'
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        self.results = {}
        self.trained_models = {}
        self.baseline_results = {}
        
        print(f"Project root: {self.project_root}")
        print(f"Artifacts dir: {self.artifacts_dir}")
        print(f"Models dir: {self.models_dir}")
        
    def load_data(self):
        """Load preprocessed data"""
        print("\n" + "="*70)
        print("LOADING PREPROCESSED DATA")
        print("="*70)
        
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Preprocessing artifacts not found at {self.artifacts_dir}\n"
                "Please run preprocessing first: python src/preprocess_data.py"
            )
        
        data_files = {
            'X_train': 'X_train.pkl',
            'y_train': 'y_train.pkl',
            'X_val': 'X_val.pkl',
            'y_val': 'y_val.pkl',
            'X_test': 'X_test.pkl',
            'y_test': 'y_test.pkl'
        }
        
        data = {}
        for name, filename in data_files.items():
            filepath = self.artifacts_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Required file not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                data[name] = pickle.load(f)
            shape = data[name].shape if hasattr(data[name], 'shape') else len(data[name])
            print(f"  ✓ Loaded {name}: {shape}")
        
        return data
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Comprehensive evaluation metrics"""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        
        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # Custom metrics
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = np.abs((y_true - y_pred) / y_true)
            within_5pct = np.mean(pct_errors <= 0.05) * 100
            within_10pct = np.mean(pct_errors <= 0.10) * 100
            within_20pct = np.mean(pct_errors <= 0.20) * 100
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MedAE': medae,
            'R²': r2,
            'MAPE (%)': mape,
            'Bias': bias,
            'Within 5%': within_5pct,
            'Within 10%': within_10pct,
            'Within 20%': within_20pct
        }
    
    def train_baselines(self, data):
        """Train baseline models"""
        print("\n" + "="*70)
        print("TRAINING BASELINE MODELS")
        print("="*70)
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        baselines = {}
        
        # 1. Mean baseline
        print("\n1. Mean Baseline")
        mean_pred = np.full(len(y_val), y_train.mean())
        baselines['Mean'] = {
            'predictions': mean_pred,
            'metrics': self.evaluate_model(y_val, mean_pred, 'Mean'),
            'description': 'Predicts global mean yield',
            'training_time': 0.001
        }
        print(f"   R² = {baselines['Mean']['metrics']['R²']:.4f}, "
              f"RMSE = {baselines['Mean']['metrics']['RMSE']:.2f}")
        
        # 2. Median baseline
        print("\n2. Median Baseline")
        median_pred = np.full(len(y_val), y_train.median())
        baselines['Median'] = {
            'predictions': median_pred,
            'metrics': self.evaluate_model(y_val, median_pred, 'Median'),
            'description': 'Predicts global median yield',
            'training_time': 0.001
        }
        print(f"   R² = {baselines['Median']['metrics']['R²']:.4f}, "
              f"RMSE = {baselines['Median']['metrics']['RMSE']:.2f}")
        
        # 3. Last-value baseline
        print("\n3. Last Value Baseline")
        last_val_pred = np.full(len(y_val), y_train.iloc[-1000:].mean())
        baselines['Last Value'] = {
            'predictions': last_val_pred,
            'metrics': self.evaluate_model(y_val, last_val_pred, 'Last Value'),
            'description': 'Uses recent historical average',
            'training_time': 0.001
        }
        print(f"   R² = {baselines['Last Value']['metrics']['R²']:.4f}, "
              f"RMSE = {baselines['Last Value']['metrics']['RMSE']:.2f}")
        
        # 4. Moving average baseline
        print("\n4. Moving Average Baseline")
        window_size = min(1000, len(y_train) // 10)
        ma_value = y_train.rolling(window=window_size, min_periods=1).mean().iloc[-1]
        ma_pred = np.full(len(y_val), ma_value)
        baselines['Moving Average'] = {
            'predictions': ma_pred,
            'metrics': self.evaluate_model(y_val, ma_pred, 'Moving Average'),
            'description': f'Moving average (window={window_size})',
            'training_time': 0.001
        }
        print(f"   R² = {baselines['Moving Average']['metrics']['R²']:.4f}, "
              f"RMSE = {baselines['Moving Average']['metrics']['RMSE']:.2f}")
        
        self.baseline_results = baselines
        return baselines
    
    def train_classical_models(self, data):
        """Train classical ML models"""
        print("\n" + "="*70)
        print("TRAINING CLASSICAL ML MODELS")
        print("="*70)
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        models = {}
        
        # 1. Linear Regression
        print("\n1. Linear Regression")
        start_time = time.time()
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = lr.predict(X_val)
        models['Linear Regression'] = {
            'model': lr,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Linear Regression'),
            'training_time': train_time,
            'complexity': 'Low',
            'interpretability': 'High',
            'description': 'Simple linear model, highly interpretable'
        }
        print(f"   R² = {models['Linear Regression']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Linear Regression']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 2. Ridge Regression
        print("\n2. Ridge Regression")
        start_time = time.time()
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = ridge.predict(X_val)
        models['Ridge Regression'] = {
            'model': ridge,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Ridge Regression'),
            'training_time': train_time,
            'complexity': 'Low',
            'interpretability': 'High',
            'description': 'Regularized linear model'
        }
        print(f"   R² = {models['Ridge Regression']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Ridge Regression']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 3. Decision Tree
        print("\n3. Decision Tree")
        start_time = time.time()
        dt = DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)
        dt.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = dt.predict(X_val)
        models['Decision Tree'] = {
            'model': dt,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Decision Tree'),
            'training_time': train_time,
            'complexity': 'Medium',
            'interpretability': 'Medium',
            'description': 'Tree-based, captures non-linearity'
        }
        print(f"   R² = {models['Decision Tree']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Decision Tree']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 4. K-Nearest Neighbors
        print("\n4. K-Nearest Neighbors")
        start_time = time.time()
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
        knn.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = knn.predict(X_val)
        models['K-NN'] = {
            'model': knn,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'K-NN'),
            'training_time': train_time,
            'complexity': 'Medium',
            'interpretability': 'Low',
            'description': 'Instance-based learning'
        }
        print(f"   R² = {models['K-NN']['metrics']['R²']:.4f}, "
              f"RMSE = {models['K-NN']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 5. Support Vector Regression (subset)
        print("\n5. Support Vector Regression (training on subset)")
        subset_size = min(20000, len(X_train))
        X_train_subset = X_train.iloc[:subset_size] if hasattr(X_train, 'iloc') else X_train[:subset_size]
        y_train_subset = y_train.iloc[:subset_size] if hasattr(y_train, 'iloc') else y_train[:subset_size]
        
        start_time = time.time()
        svr = SVR(kernel='rbf', C=10, gamma='scale', cache_size=1000)
        svr.fit(X_train_subset, y_train_subset)
        train_time = time.time() - start_time
        
        y_pred = svr.predict(X_val)
        models['SVR'] = {
            'model': svr,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'SVR'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Low',
            'description': f'SVM regression (trained on {subset_size} samples)'
        }
        print(f"   R² = {models['SVR']['metrics']['R²']:.4f}, "
              f"RMSE = {models['SVR']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        self.trained_models.update(models)
        return models
    
    def train_ensemble_models(self, data):
        """Train ensemble and advanced models"""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE & ADVANCED MODELS")
        print("="*70)
        
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        
        models = {}
        
        # 1. Random Forest
        print("\n1. Random Forest")
        start_time = time.time()
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = rf.predict(X_val)
        models['Random Forest'] = {
            'model': rf,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Random Forest'),
            'training_time': train_time,
            'complexity': 'Medium',
            'interpretability': 'Medium',
            'description': 'Bagging ensemble of trees'
        }
        print(f"   R² = {models['Random Forest']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Random Forest']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 2. Gradient Boosting
        print("\n2. Gradient Boosting")
        start_time = time.time()
        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = gb.predict(X_val)
        models['Gradient Boosting'] = {
            'model': gb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Gradient Boosting'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': 'Sequential boosting'
        }
        print(f"   R² = {models['Gradient Boosting']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Gradient Boosting']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 3. XGBoost
        print("\n3. XGBoost")
        start_time = time.time()
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = xgb.predict(X_val)
        models['XGBoost'] = {
            'model': xgb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'XGBoost'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': 'Optimized gradient boosting'
        }
        print(f"   R² = {models['XGBoost']['metrics']['R²']:.4f}, "
              f"RMSE = {models['XGBoost']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 4. LightGBM
        print("\n4. LightGBM")
        start_time = time.time()
        lgb = LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = lgb.predict(X_val)
        models['LightGBM'] = {
            'model': lgb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'LightGBM'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': 'Fast gradient boosting'
        }
        print(f"   R² = {models['LightGBM']['metrics']['R²']:.4f}, "
              f"RMSE = {models['LightGBM']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 5. CatBoost
        print("\n5. CatBoost")
        start_time = time.time()
        cb = CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
        cb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = cb.predict(X_val)
        models['CatBoost'] = {
            'model': cb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'CatBoost'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': 'Robust boosting'
        }
        print(f"   R² = {models['CatBoost']['metrics']['R²']:.4f}, "
              f"RMSE = {models['CatBoost']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 6. Voting Ensemble
        print("\n6. Voting Ensemble (XGB + LGB + RF)")
        start_time = time.time()
        voting = VotingRegressor(
            estimators=[
                ('xgb', models['XGBoost']['model']),
                ('lgb', models['LightGBM']['model']),
                ('rf', models['Random Forest']['model'])
            ],
            n_jobs=-1
        )
        voting.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = voting.predict(X_val)
        models['Voting Ensemble'] = {
            'model': voting,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Voting Ensemble'),
            'training_time': train_time,
            'complexity': 'Very High',
            'interpretability': 'Low',
            'description': 'Average of top 3 models'
        }
        print(f"   R² = {models['Voting Ensemble']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Voting Ensemble']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        # 7. Stacking Ensemble
        print("\n7. Stacking Ensemble (XGB + LGB + RF → Ridge)")
        start_time = time.time()
        stacking = StackingRegressor(
            estimators=[
                ('xgb', XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
                ('lgb', LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
            ],
            final_estimator=Ridge(alpha=1.0),
            n_jobs=-1
        )
        stacking.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = stacking.predict(X_val)
        models['Stacking Ensemble'] = {
            'model': stacking,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Stacking Ensemble'),
            'training_time': train_time,
            'complexity': 'Very High',
            'interpretability': 'Low',
            'description': 'Meta-learner on base predictions'
        }
        print(f"   R² = {models['Stacking Ensemble']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Stacking Ensemble']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        
        self.trained_models.update(models)
        return models
    
    def create_comparison_table(self):
        """Create comprehensive comparison table"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        comparison_data = []
        
        # Add baselines
        for name, info in self.baseline_results.items():
            comparison_data.append({
                'Model': f"[BASELINE] {name}",
                'R²': info['metrics']['R²'],
                'RMSE': info['metrics']['RMSE'],
                'MAE': info['metrics']['MAE'],
                'MAPE (%)': info['metrics']['MAPE (%)'],
                'Within 10%': info['metrics']['Within 10%'],
                'Training Time (s)': info['training_time'],
                'Complexity': 'Minimal',
                'Interpretability': 'High',
                'Type': 'Baseline'
            })
        
        # Add trained models
        for name, info in self.trained_models.items():
            model_type = 'Classical ML' if name in ['Linear Regression', 'Ridge Regression', 
                                                      'Decision Tree', 'K-NN', 'SVR'] else 'Ensemble/Advanced'
            comparison_data.append({
                'Model': name,
                'R²': info['metrics']['R²'],
                'RMSE': info['metrics']['RMSE'],
                'MAE': info['metrics']['MAE'],
                'MAPE (%)': info['metrics']['MAPE (%)'],
                'Within 10%': info['metrics']['Within 10%'],
                'Training Time (s)': info['training_time'],
                'Complexity': info['complexity'],
                'Interpretability': info['interpretability'],
                'Type': model_type
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('R²', ascending=False)
        
        return df_comparison
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        for name, info in self.trained_models.items():
            if 'model' in info:
                filename = name.replace(' ', '_').lower() + '.pkl'
                filepath = self.models_dir / filename
                with open(filepath, 'wb') as f:
                    pickle.dump(info['model'], f)
                print(f"  ✓ Saved {name}")
    
    def save_results(self, comparison_df):
        """Save comparison results"""
        filepath = self.models_dir / 'model_comparison.csv'
        comparison_df.to_csv(filepath, index=False)
        print(f"\n✓ Comparison table saved to {filepath}")
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        # Load data
        data = self.load_data()
        
        # Train all models
        self.train_baselines(data)
        self.train_classical_models(data)
        self.train_ensemble_models(data)
        
        # Create comparison
        comparison_df = self.create_comparison_table()
        
        print("\n" + "="*70)
        print("FINAL COMPARISON TABLE")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        # Save everything
        self.save_models()
        self.save_results(comparison_df)
        
        return comparison_df


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("CROP YIELD PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    pipeline = CropYieldModelPipeline()
    comparison_df = pipeline.run_full_pipeline()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nTop 5 Models by R²:")
    top_5 = comparison_df.nlargest(5, 'R²')[['Model', 'R²', 'RMSE', 'MAE', 'Training Time (s)']]
    print(top_5.to_string(index=False))
    
    print("\n✅ All models saved successfully!")
    print(f"📁 Models directory: {pipeline.models_dir}")