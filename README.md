# Crop Yield Prediction Using Machine Learning

A comprehensive machine learning system for predicting crop yields across 11 crop types in 50 US states using climate, soil, and geospatial data.

## 🎯 Project Overview

This project implements a production-ready crop yield prediction system that:
- Trains and compares 17+ machine learning models
- Provides a REST API for predictions
- Features an interactive React web dashboard
- Achieves R² > 0.98 on test data

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Preprocessed data and artifacts
├── src/
│   ├── preprocess_data.py      # Data preprocessing pipeline
│   └── model/
│       └── train_models.py     # Model training pipeline
├── api/
│   ├── app.py                  # Flask REST API
│   └── artifacts/              # Models and preprocessors
├── eeee/                       # React frontend (main branch)
│   └── src/
│       └── pages/              # Frontend pages
├── models/                      # Trained model files
├── notebooks/                   # Jupyter notebooks (EDA)
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd eee
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd eeee
   npm install
   ```

### Running the Project

#### Step 1: Preprocess Data (if needed)
```bash
python src/preprocess_data.py
```

#### Step 2: Train Models (if needed)
```bash
python src/model/train_models.py
```

#### Step 3: Sync Models to API
```bash
python sync_models_to_api.py
```

#### Step 4: Start the API Server
```bash
cd api
python app.py
```
API will run on `http://localhost:5000`

#### Step 5: Start the Frontend
```bash
cd eeee
npm start
```
Frontend will run on `http://localhost:3000`

## 📡 API Endpoints

### Core Endpoints

- `GET /health` - API health check
- `GET /metadata` - Available crops, states, and models
- `GET /counties/<state>` - Counties for a given state
- `GET /valid-states/<crop>` - Valid states for a crop
- `GET /models` - List all available models
- `POST /models/<model_name>` - Switch active model
- `POST /predict` - Single crop yield prediction
- `POST /predict-regional` - Predict all crops for a region
- `GET /metrics` - Model comparison metrics

### Bonus Endpoints

- `GET /feature-engineering` - Feature engineering documentation
- `GET /feature-importance` - List models with feature importance
- `GET /feature-importance/<model>` - Feature importance for a model

## 🎨 Frontend Features

- **Home Page**: Project overview and statistics
- **Predict Page**: Single crop yield prediction with comprehensive form
- **Regional Page**: Batch predictions for all crops in a location
- **Metrics Page**: Interactive charts comparing model performance
  - R² Score comparison
  - Training Time comparison
  - MAE comparison
  - RMSE comparison
  - Feature Engineering documentation
  - Feature Importance visualization

## 🤖 Models

The system includes 17+ models across three categories:

### Baselines
- Mean Baseline
- Median Baseline
- Last Value Baseline
- Moving Average Baseline

### Classical ML
- Linear Regression
- Ridge Regression
- Decision Tree
- K-Nearest Neighbors
- Support Vector Regression

### Ensemble/Advanced
- Random Forest (Best: R² = 0.9876)
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost (optional)
- Voting Ensemble
- Stacking Ensemble

## 📊 Performance

Top 5 Models (by R²):
1. Random Forest - R²: 0.9876, RMSE: 121.50
2. Voting Ensemble - R²: 0.9854, RMSE: 131.91
3. XGBoost - R²: 0.9846, RMSE: 135.42
4. Gradient Boosting - R²: 0.9835, RMSE: 140.18
5. Decision Tree - R²: 0.9818, RMSE: 147.29

## 🔧 Feature Engineering

The system creates 13 engineered features:
- **Time Features (2)**: Year_Since_Start, Year_Squared
- **Climate Interactions (4)**: temp_prcp_interaction, gdd_per_day, moisture_stress, temp_variability
- **Soil Features (2)**: soil_quality_score, loam_index
- **Geographic Features (3)**: lat_squared, long_squared, lat_long_interaction

## 📝 Usage Examples

### API: Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Crop": "CORN",
    "State": "IOWA",
    "County": "POLK",
    "Year": 2024
  }'
```

### API: Regional Prediction
```bash
curl -X POST http://localhost:5000/predict-regional \
  -H "Content-Type: application/json" \
  -d '{
    "State": "IOWA",
    "County": "POLK",
    "Year": 2024
  }'
```

## 🛠️ Development

### Project Branches
- `de` - Development branch (frontend in `eeee/`)
- `dev` - Development branch
- `eee` - Alternative branch

### Key Scripts
- `src/preprocess_data.py` - Data preprocessing
- `src/model/train_models.py` - Model training
- `sync_models_to_api.py` - Sync models to API artifacts
- `api/app.py` - Flask API server

## 📚 Documentation

- `PROJECT_COMPLETENESS_CHECK.md` - Project completeness analysis
- `TRAINING_OUTPUT_ANALYSIS.md` - Training results analysis
- `eee/project_report.tex` - Full project report (LaTeX)

## ⚠️ Troubleshooting

### Models not loading
- Ensure models are in `api/artifacts/` directory
- Run `python sync_models_to_api.py` to sync models
- Check that all required artifacts exist

### API errors
- Verify Flask is installed: `pip install flask flask-cors`
- Check that artifacts directory exists
- Review console output for specific error messages

### Frontend not connecting
- Ensure API is running on port 5000
- Check CORS settings in `api/app.py`
- Verify API endpoint URLs in frontend code

## 📄 License

[Add your license here]

## 👥 Authors

[Add author names here]

## 🙏 Acknowledgments

- USDA NASS for agricultural data
- Climate data sources
- Open-source ML libraries


