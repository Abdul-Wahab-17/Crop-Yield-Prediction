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
from sklearn.model_selection import GridSearchCV

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
        self.feature_importances = {}
        
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
        
        # Try to load feature names if available
        feature_names_file = self.artifacts_dir / 'feature_names.json'
        if feature_names_file.exists():
            with open(feature_names_file, 'r') as f:
                self.feature_names = json.load(f)
            print(f"  ✓ Loaded feature names: {len(self.feature_names)} features")
        else:
            self.feature_names = None
            print("  ℹ️ Feature names not found, using default column names")
        
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
        """Train classical ML models with hyperparameter tuning + CV"""
        print("\n" + "="*70)
        print("TRAINING CLASSICAL ML MODELS (WITH GRID SEARCH CV)")
        print("="*70)

        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']

        models = {}

        # ----------------------------
        # Linear Regression (no tuning needed)
        # ----------------------------
        print("\n1. Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)
        models['Linear Regression'] = {
            'model': lr,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Linear Regression'),
            'training_time': 0,  # negligible
            'complexity': 'Low',
            'interpretability': 'High',
            'description': 'Simple linear model, highly interpretable'
        }
        print(f"   R² = {models['Linear Regression']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Linear Regression']['metrics']['RMSE']:.2f}")

        # ----------------------------
        # Ridge Regression
        # ----------------------------
        print("\n2. Ridge Regression")
        start_time = time.time()
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 50.0]}
        ridge = Ridge(random_state=42)
        grid_ridge = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_ridge.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_ridge = grid_ridge.best_estimator_
        y_pred = best_ridge.predict(X_val)
        models['Ridge Regression'] = {
            'model': best_ridge,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Ridge Regression'),
            'training_time': train_time,
            'complexity': 'Low',
            'interpretability': 'High',
            'description': f'Regularized linear model with CV-tuned alpha (best: {grid_ridge.best_params_["alpha"]})'
        }
        print(f"   R² = {models['Ridge Regression']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Ridge Regression']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_ridge.best_params_}")

        # ----------------------------
        # Decision Tree
        # ----------------------------
        print("\n3. Decision Tree")
        start_time = time.time()
        dt_params = {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        dt = DecisionTreeRegressor(random_state=42)
        grid_dt = GridSearchCV(dt, dt_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_dt.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_dt = grid_dt.best_estimator_
        y_pred = best_dt.predict(X_val)
        models['Decision Tree'] = {
            'model': best_dt,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Decision Tree'),
            'training_time': train_time,
            'complexity': 'Medium',
            'interpretability': 'Medium',
            'description': f'Tree-based model with CV-tuned hyperparameters (best depth: {grid_dt.best_params_["max_depth"]})'
        }
        print(f"   R² = {models['Decision Tree']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Decision Tree']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_dt.best_params_}")

        # ----------------------------
        # K-Nearest Neighbors
        # ----------------------------
        print("\n4. K-Nearest Neighbors")
        start_time = time.time()
        knn_params = {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance']}
        knn = KNeighborsRegressor(n_jobs=-1)
        grid_knn = GridSearchCV(knn, knn_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_knn.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_knn = grid_knn.best_estimator_
        y_pred = best_knn.predict(X_val)
        models['K-NN'] = {
            'model': best_knn,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'K-NN'),
            'training_time': train_time,
            'complexity': 'Medium',
            'interpretability': 'Low',
            'description': f'Instance-based learning with CV-tuned k (best: {grid_knn.best_params_["n_neighbors"]})'
        }
        print(f"   R² = {models['K-NN']['metrics']['R²']:.4f}, "
              f"RMSE = {models['K-NN']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_knn.best_params_}")

        # ----------------------------
        # SVR
        # ----------------------------
        print("\n5. Support Vector Regression (subset)")
        subset_size = min(20000, len(X_train))
        X_train_subset = X_train.iloc[:subset_size] if hasattr(X_train, 'iloc') else X_train[:subset_size]
        y_train_subset = y_train.iloc[:subset_size] if hasattr(y_train, 'iloc') else y_train[:subset_size]

        start_time = time.time()
        svr_params = {'C': [1, 10, 50], 'gamma': ['scale', 'auto']}
        svr = SVR(kernel='rbf', cache_size=1000)
        grid_svr = GridSearchCV(svr, svr_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_svr.fit(X_train_subset, y_train_subset)
        train_time = time.time() - start_time
        
        best_svr = grid_svr.best_estimator_
        y_pred = best_svr.predict(X_val)
        models['SVR'] = {
            'model': best_svr,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'SVR'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Low',
            'description': f'SVM regression with CV-tuned hyperparameters (trained on {subset_size} samples)'
        }
        print(f"   R² = {models['SVR']['metrics']['R²']:.4f}, "
              f"RMSE = {models['SVR']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_svr.best_params_}")

        self.trained_models.update(models)
        return models
    
    def train_ensemble_models(self, data):
        """Train ensemble/advanced models with hyperparameter tuning + feature importance + CV"""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE & ADVANCED MODELS (WITH CV + FEATURE IMPORTANCE)")
        print("="*70)

        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']

        models = {}
        feature_importances = {}

        # ----------------------------
        # Random Forest
        # ----------------------------
        print("\n1. Random Forest")
        start_time = time.time()
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_rf = grid_rf.best_estimator_
        y_pred = best_rf.predict(X_val)

        models['Random Forest'] = {
            'model': best_rf,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Random Forest'),
            'training_time': train_time,
            'complexity': 'Medium',
            'interpretability': 'Medium',
            'description': f'Bagging ensemble of trees (CV-tuned) (best n_estimators: {grid_rf.best_params_["n_estimators"]})'
        }
        print(f"   R² = {models['Random Forest']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Random Forest']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_rf.best_params_}")

        feature_importances['Random Forest'] = best_rf.feature_importances_

        # ----------------------------
        # Gradient Boosting
        # ----------------------------
        print("\n2. Gradient Boosting")
        start_time = time.time()
        gb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}
        gb = GradientBoostingRegressor(random_state=42)
        grid_gb = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_gb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_gb = grid_gb.best_estimator_
        y_pred = best_gb.predict(X_val)

        models['Gradient Boosting'] = {
            'model': best_gb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'Gradient Boosting'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': f'Sequential boosting (CV-tuned) (best learning_rate: {grid_gb.best_params_["learning_rate"]})'
        }
        print(f"   R² = {models['Gradient Boosting']['metrics']['R²']:.4f}, "
              f"RMSE = {models['Gradient Boosting']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_gb.best_params_}")

        feature_importances['Gradient Boosting'] = best_gb.feature_importances_

        # ----------------------------
        # XGBoost
        # ----------------------------
        print("\n3. XGBoost")
        start_time = time.time()
        xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6, 8], 'learning_rate': [0.05, 0.1]}
        xgb = XGBRegressor(random_state=42, n_jobs=-1)
        grid_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_xgb = grid_xgb.best_estimator_
        y_pred = best_xgb.predict(X_val)

        models['XGBoost'] = {
            'model': best_xgb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'XGBoost'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': f'Optimized gradient boosting (CV-tuned) (best max_depth: {grid_xgb.best_params_["max_depth"]})'
        }
        print(f"   R² = {models['XGBoost']['metrics']['R²']:.4f}, "
              f"RMSE = {models['XGBoost']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_xgb.best_params_}")

        feature_importances['XGBoost'] = best_xgb.feature_importances_

        # ----------------------------
        # LightGBM
        # ----------------------------
        print("\n4. LightGBM")
        start_time = time.time()
        lgb_params = {'n_estimators': [100, 200], 'max_depth': [5, 8], 'learning_rate': [0.05, 0.1]}
        lgb = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        grid_lgb = GridSearchCV(lgb, lgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_lgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_lgb = grid_lgb.best_estimator_
        y_pred = best_lgb.predict(X_val)

        models['LightGBM'] = {
            'model': best_lgb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'LightGBM'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': f'Fast gradient boosting (CV-tuned) (best learning_rate: {grid_lgb.best_params_["learning_rate"]})'
        }
        print(f"   R² = {models['LightGBM']['metrics']['R²']:.4f}, "
              f"RMSE = {models['LightGBM']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_lgb.best_params_}")

        feature_importances['LightGBM'] = best_lgb.feature_importances_

        # ----------------------------
        # CatBoost
        # ----------------------------
        print("\n5. CatBoost")
        start_time = time.time()
        cb_params = {'depth': [6, 8], 'iterations': [100, 200], 'learning_rate': [0.05, 0.1]}
        cb = CatBoostRegressor(random_state=42, verbose=False)
        grid_cb = GridSearchCV(cb, cb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_cb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        best_cb = grid_cb.best_estimator_
        y_pred = best_cb.predict(X_val)

        models['CatBoost'] = {
            'model': best_cb,
            'predictions': y_pred,
            'metrics': self.evaluate_model(y_val, y_pred, 'CatBoost'),
            'training_time': train_time,
            'complexity': 'High',
            'interpretability': 'Medium',
            'description': f'Robust boosting (CV-tuned) (best iterations: {grid_cb.best_params_["iterations"]})'
        }
        print(f"   R² = {models['CatBoost']['metrics']['R²']:.4f}, "
              f"RMSE = {models['CatBoost']['metrics']['RMSE']:.2f}, "
              f"Time = {train_time:.2f}s")
        print(f"   Best params: {grid_cb.best_params_}")

        feature_importances['CatBoost'] = best_cb.feature_importances_

        # ----------------------------
        # Voting Ensemble (base models already tuned)
        # ----------------------------
        print("\n6. Voting Ensemble (XGB + LGB + RF)")
        start_time = time.time()
        voting = VotingRegressor(
            estimators=[
                ('xgb', best_xgb),
                ('lgb', best_lgb),
                ('rf', best_rf)
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

        # ----------------------------
        # Stacking Ensemble (base models already tuned)
        # ----------------------------
        print("\n7. Stacking Ensemble (XGB + LGB + RF → Ridge)")
        start_time = time.time()
        stacking = StackingRegressor(
            estimators=[
                ('xgb', best_xgb),
                ('lgb', best_lgb),
                ('rf', best_rf)
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

        # Save feature importances
        self.feature_importances = feature_importances

        self.trained_models.update(models)
        return models
    
    def save_feature_importances(self, feature_names=None):
        """Save feature importances for all tree-based models"""
        if not hasattr(self, 'feature_importances') or not self.feature_importances:
            print("\nNo feature importance available - skipping feature importance export")
            return

        # Use provided feature_names or try to get from data
        if feature_names is None:
            if hasattr(self, 'feature_names') and self.feature_names:
                feature_names = self.feature_names
            elif hasattr(self, 'trained_models') and len(self.trained_models) > 0:
                # Get feature names from the first model
                first_model = list(self.trained_models.values())[0]['model']
                if hasattr(first_model, 'feature_names_in_'):
                    feature_names = first_model.feature_names_in_
                elif hasattr(self, 'X_train'):
                    if isinstance(self.X_train, pd.DataFrame):
                        feature_names = self.X_train.columns.tolist()
                    else:
                        feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
                else:
                    print("Warning: Could not determine feature names")
                    return
        
        if feature_names is None:
            print("Warning: Feature names not available")
            return

        importance_dir = self.models_dir / "feature_importance"
        importance_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nSaving feature importances to {importance_dir}/")
        
        for model_name, importances in self.feature_importances.items():
            if len(importances) != len(feature_names):
                print(f"  Skipping {model_name}: Importance array length doesn't match feature names")
                continue
            
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            filepath = importance_dir / f"{model_name.replace(' ', '_').lower()}_importance.csv"
            df.to_csv(filepath, index=False)
            print(f"  ✓ Saved feature importance: {model_name}")
            
            # Print top 10 features
            print(f"    Top 5 features: {df['Feature'].head(5).tolist()}")
    
    def error_analysis(self, model_name, y_true, y_pred):
        """Print top 10 largest errors and percent errors"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate errors
        errors = y_pred - y_true
        percent_errors = np.abs(errors / (y_true + 1e-6)) * 100  # avoid div by 0
        
        # Find indices of top errors
        top_idx = np.argsort(-percent_errors)[:10]
        
        print(f"\n{'='*50}")
        print(f"ERROR ANALYSIS: {model_name}")
        print(f"{'='*50}")
        
        # Overall error statistics
        print(f"Overall Error Statistics:")
        print(f"  Mean Absolute Error: {np.mean(np.abs(errors)):.2f}")
        print(f"  Mean Absolute % Error: {np.mean(percent_errors):.2f}%")
        print(f"  Max Absolute % Error: {np.max(percent_errors):.2f}%")
        print(f"  Std of % Errors: {np.std(percent_errors):.2f}%")
        
        # Print top 10 errors
        print(f"\nTop 10 Largest Percentage Errors:")
        for i, idx in enumerate(top_idx, 1):
            print(f"  {i:2d}. True: {y_true[idx]:8.2f}, Pred: {y_pred[idx]:8.2f}, "
                  f"Error: {errors[idx]:8.2f}, %Error: {percent_errors[idx]:6.2f}%")
        
        # Analyze error distribution
        print(f"\nError Distribution by Percent Error Range:")
        ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, np.inf)]
        for low, high in ranges:
            if high == np.inf:
                count = np.sum(percent_errors >= low)
                print(f"  {low}+%: {count} samples ({count/len(y_true)*100:.1f}%)")
            else:
                count = np.sum((percent_errors >= low) & (percent_errors < high))
                print(f"  {low}-{high}%: {count} samples ({count/len(y_true)*100:.1f}%)")
    
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
        # Save CSV comparison
        filepath = self.models_dir / 'model_comparison.csv'
        comparison_df.to_csv(filepath, index=False)
        print(f"\n✓ Comparison table saved to {filepath}")
        
        # Also save as JSON for easy reading
        json_filepath = self.models_dir / 'model_comparison.json'
        comparison_df.to_json(json_filepath, orient='records', indent=2)
        print(f"✓ Comparison table saved to {json_filepath}")
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        # Load data
        data = self.load_data()
        
        # Store X_train for feature importance
        self.X_train = data['X_train']
        
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
        
        # Save feature importances
        self.save_feature_importances()
        
        # Perform error analysis for all trained models
        print("\n" + "="*70)
        print("ERROR ANALYSIS FOR ALL MODELS")
        print("="*70)
        
        for name, info in self.trained_models.items():
            self.error_analysis(name, data['y_val'], info['predictions'])
        
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