"""
Evaluate all trained models on the test set
Location: src/model/evaluate_test.py
Run from root directory: python src/model/evaluate_test.py

This version:
- Assumes same folder structure as train_models.py
- Uses the same models/ directory
- Overwrites / extends training outputs
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error


class TestSetEvaluator:
    def __init__(self):
        # SAME ROOT LOGIC AS train_models.py
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / 'models'
        self.artifacts_dir = self.project_root / 'data' / 'processed' / 'preprocessing_artifacts'

        # Reuse same error_analysis folder
        self.error_analysis_dir = self.models_dir / 'error_analysis'
        self.error_analysis_dir.mkdir(exist_ok=True, parents=True)

        print(f"Project root: {self.project_root}")
        print(f"Models directory: {self.models_dir}")
        print(f"Artifacts directory: {self.artifacts_dir}")

    # ------------------------------------------------------------------
    def load_test_data(self):
        print("\n" + "=" * 70)
        print("LOADING TEST DATA")
        print("=" * 70)

        with open(self.artifacts_dir / 'X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(self.artifacts_dir / 'y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)

        print(f"  ✓ X_test shape: {X_test.shape}")
        print(f"  ✓ y_test shape: {y_test.shape}")

        return X_test, y_test

    # ------------------------------------------------------------------
    def load_models(self):
        print("\n" + "=" * 70)
        print("LOADING TRAINED MODELS")
        print("=" * 70)

        model_files = list(self.models_dir.glob('*.pkl'))
        models = {}

        if not model_files:
            raise FileNotFoundError("No trained models found. Run train_models.py first.")

        for model_file in model_files:
            model_name = model_file.stem.replace('_', ' ').title()
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                models[model_name] = model
                print(f"  ✓ Loaded: {model_name}")
            except ModuleNotFoundError as e:
                print(f"  ⚠ Skipped {model_name} (missing dependency: {e.name})")
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")

        print(f"\nTotal models loaded: {len(models)}")
        return models


    # ------------------------------------------------------------------
    def evaluate_model(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)

        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

        pct_errors = np.abs((y_true - y_pred) / (y_true + 1e-6))
        within_10 = np.mean(pct_errors <= 0.10) * 100

        bias = np.mean(y_pred - y_true)

        return {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MedAE': medae,
            'MAPE (%)': mape,
            'Bias': bias,
            'Within 10%': within_10
        }

    # ------------------------------------------------------------------
    def error_analysis(self, model_name, y_true, y_pred):
        """Standardized error analysis matching train_models.py"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate errors
        errors = y_pred - y_true
        # Avoid division by zero
        percent_errors = np.abs(errors / (y_true + 1e-6)) * 100
        
        # Find indices of top errors
        top_idx = np.argsort(-percent_errors)[:10]

        print(f"\nERROR ANALYSIS (TEST): {model_name}")

        # Overall error statistics
        overall_stats = {
            'mean_absolute_error': float(np.mean(np.abs(errors))),
            'mean_absolute_percent_error': float(np.mean(percent_errors)),
            'max_absolute_percent_error': float(np.max(percent_errors)),
            'std_percent_errors': float(np.std(percent_errors))
        }

        # Top 10 errors
        top_errors = []
        for i, idx in enumerate(top_idx, 1):
            error_data = {
                'rank': i,
                'true_value': float(y_true[idx]),
                'predicted_value': float(y_pred[idx]),
                'error': float(errors[idx]),
                'percent_error': float(percent_errors[idx])
            }
            top_errors.append(error_data)

        # Error distribution
        error_distribution = []
        
        # Create (0, 10), (10, 20) ... (90, 100) ranges
        ranges = [(i, i+10) for i in range(0, 100, 10)]
        # Add the final range
        ranges.append((100, np.inf))

        for low, high in ranges:
            if high == np.inf:
                count = np.sum(percent_errors >= low)
                range_label = f'{low}+%'
            else:
                count = np.sum((percent_errors >= low) & (percent_errors < high))
                range_label = f'{low}-{high}%'
            
            percentage = count / len(y_true) * 100
            dist_data = {
                'range': range_label,
                'count': int(count),
                'percentage': float(percentage)
            }
            error_distribution.append(dist_data)

        # Output structure
        error_analysis_data = {
            'model_name': model_name,
            'overall_statistics': overall_stats,
            'top_10_errors': top_errors,
            'error_distribution': error_distribution,
            'total_samples': int(len(y_true))
        }

        # Save to correct filename
        model_file_name = model_name.replace(' ', '_').replace('-', '_').lower()
        json_filepath = self.error_analysis_dir / f'{model_file_name}_error_analysis.json'

        with open(json_filepath, 'w') as f:
            json.dump(error_analysis_data, f, indent=2)

        print(f"  ✓ Saved test error analysis → {json_filepath}")
        return error_analysis_data

    # ------------------------------------------------------------------
    def load_training_data(self):
        """Need y_train for baselines"""
        with open(self.artifacts_dir / 'y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        return y_train

    # ------------------------------------------------------------------
    def evaluate_baselines(self, y_train, y_test):
        print("\n" + "=" * 70)
        print("EVALUATING BASELINES")
        print("=" * 70)
        
        baselines = {}
        results = []
        
        # 1. Mean
        mean_pred = np.full(len(y_test), y_train.mean())
        baselines['[BASELINE] Mean'] = mean_pred
        
        # 2. Median
        median_pred = np.full(len(y_test), y_train.median())
        baselines['[BASELINE] Median'] = median_pred
        
        # 3. Last Value
        last_val_pred = np.full(len(y_test), y_train.iloc[-1000:].mean())
        baselines['[BASELINE] Last Value'] = last_val_pred
        
        # 4. Moving Average
        window_size = min(1000, len(y_train) // 10)
        ma_value = y_train.rolling(window=window_size, min_periods=1).mean().iloc[-1]
        ma_pred = np.full(len(y_test), ma_value)
        baselines['[BASELINE] Moving Average'] = ma_pred

        for name, pred in baselines.items():
            print(f"\nEvaluating {name}")
            metrics = self.evaluate_model(y_test, pred)
            
            row = {
                'Model': name,
                'R²': metrics['R²'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE (%)': metrics['MAPE (%)'],
                'Within 10%': metrics['Within 10%'],
                'Training Time (s)': 0.001,
                'Complexity': 'Minimal',
                'Interpretability': 'High',
                'Type': 'Baseline'
            }
            results.append(row)
            print(f"  R²={metrics['R²']:.4f}, RMSE={metrics['RMSE']:.2f}")
            
            self.error_analysis(name, y_test, pred)
            
        return results

    # ------------------------------------------------------------------
    def run(self):
        X_test, y_test = self.load_test_data()
        y_train = self.load_training_data()
        models = self.load_models()

        results = []

        print("\n" + "=" * 70)
        print("TEST SET EVALUATION")
        print("=" * 70)

        # 1. Evaluate Trained Models
        for name, model in models.items():
            print(f"\nEvaluating {name}")
            y_pred = model.predict(X_test)

            metrics = self.evaluate_model(y_test, y_pred)
            
            # Add metadata matching schema
            model_type = 'Classical ML' if name in ['Linear Regression', 'Ridge Regression', 
                                                    'Decision Tree', 'K-NN', 'SVR'] else 'Ensemble/Advanced'
            
            complexity = 'Medium'
            interpretability = 'Medium'
            if 'Linear' in name or 'Ridge' in name:
                complexity = 'Low'
                interpretability = 'High'
            elif 'Stacking' in name or 'Voting' in name:
                complexity = 'Very High'
                interpretability = 'Low'
            
            row = {
                'Model': name,
                'R²': metrics['R²'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE (%)': metrics['MAPE (%)'],
                'Within 10%': metrics['Within 10%'],
                'Training Time (s)': 0,
                'Complexity': complexity,
                'Interpretability': interpretability,
                'Type': model_type
            }
            results.append(row)
            print(f"  R²={metrics['R²']:.4f}, RMSE={metrics['RMSE']:.2f}")

            self.error_analysis(name, y_test, y_pred)
            
        # 2. Evaluate Baselines
        baseline_results = self.evaluate_baselines(y_train, y_test)
        results.extend(baseline_results)

        df = pd.DataFrame(results).sort_values('R²', ascending=False)

        # OVERWRITE TRAIN OUTPUTS
        csv_path = self.models_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 70)
        print("FINAL TEST RESULTS (Overwriting comparison table)")
        print("=" * 70)
        print(df.to_string(index=False))

        print(f"\n✓ Saved test results to: {csv_path}")

        return df


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("TEST SET EVALUATION (ALIGNED WITH TRAINING PIPELINE)")
    print("=" * 70)

    evaluator = TestSetEvaluator()
    evaluator.run()

    print("\n✅ Test evaluation complete!")
