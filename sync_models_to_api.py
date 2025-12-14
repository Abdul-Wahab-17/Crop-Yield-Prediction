"""
Script to sync trained models from models/ to api/artifacts/
Run this after training models to update the API with the latest models.
"""

import shutil
from pathlib import Path

def sync_models():
    """Copy trained models and comparison CSV to API artifacts directory"""
    
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    api_artifacts_dir = project_root / 'api' / 'artifacts'
    
    print("="*70)
    print("SYNCING MODELS TO API")
    print("="*70)
    
    # Models to copy (excluding catboost if it doesn't exist)
    model_files = [
        'xgboost.pkl',
        'lightgbm.pkl', 
        'random_forest.pkl',
        'gradient_boosting.pkl',
        'voting_ensemble.pkl',
        'stacking_ensemble.pkl',
        'decision_tree.pkl',
        'k-nn.pkl',
        'linear_regression.pkl',
        'ridge_regression.pkl',
        'svr.pkl',
        'catboost.pkl'  # Will skip if doesn't exist
    ]
    
    # Copy models
    copied = 0
    skipped = 0
    for model_file in model_files:
        source = models_dir / model_file
        dest = api_artifacts_dir / model_file
        
        if source.exists():
            shutil.copy2(source, dest)
            print(f"  ✓ Copied {model_file}")
            copied += 1
        else:
            print(f"  ⊘ Skipped {model_file} (not found)")
            skipped += 1
    
    # Copy model comparison CSV
    comparison_csv = models_dir / 'model_comparison.csv'
    if comparison_csv.exists():
        shutil.copy2(comparison_csv, api_artifacts_dir / 'model_comparison.csv')
        print(f"  ✓ Copied model_comparison.csv")
    else:
        print(f"  ⚠ model_comparison.csv not found")
    
    # Copy error analysis files
    error_analysis_dir = models_dir / 'error_analysis'
    if error_analysis_dir.exists():
        api_error_dir = api_artifacts_dir / 'error_analysis'
        api_error_dir.mkdir(exist_ok=True, parents=True)
        error_files = list(error_analysis_dir.glob('*_error_analysis.json'))
        for error_file in error_files:
            shutil.copy2(error_file, api_error_dir / error_file.name)
        print(f"  ✓ Copied {len(error_files)} error analysis files")
    else:
        print(f"  ⚠ error_analysis directory not found (run training to generate)")
    
    print("\n" + "="*70)
    print(f"✅ Sync complete: {copied} models copied, {skipped} skipped")
    print("="*70)
    print(f"\n📁 Models are now available in: {api_artifacts_dir}")
    print("\n💡 You can now restart your Flask API to load the updated models.")

if __name__ == "__main__":
    sync_models()


