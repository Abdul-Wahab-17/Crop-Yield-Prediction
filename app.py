"""
Streamlit Dashboard for Crop Yield Prediction
Location: app.py (in root directory)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


class CropYieldPredictor:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.artifacts_dir = self.project_root / 'data' / 'processed' / 'preprocessing_artifacts'
        self.models_dir = self.project_root / 'models'
        
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = {}
        self.metadata = {}
        self.crop_stats = {}
        self.available_models = []
        
        # These will be populated from encoders
        self.crop_classes = []
        self.state_classes = []
        self.county_classes = []
        
    @st.cache_resource
    def load_artifacts(_self):
        """Load all preprocessing artifacts"""
        try:
            # Load encoders
            with open(_self.artifacts_dir / 'crop_encoder.pkl', 'rb') as f:
                _self.encoders['crop'] = pickle.load(f)
                _self.crop_classes = _self.encoders['crop'].classes_.tolist()
                
            with open(_self.artifacts_dir / 'state_encoder.pkl', 'rb') as f:
                _self.encoders['state'] = pickle.load(f)
                _self.state_classes = _self.encoders['state'].classes_.tolist()
                
            with open(_self.artifacts_dir / 'county_encoder.pkl', 'rb') as f:
                _self.encoders['county'] = pickle.load(f)
                _self.county_classes = _self.encoders['county'].classes_.tolist()
            
            # Load scalers
            with open(_self.artifacts_dir / 'weather_scaler.pkl', 'rb') as f:
                _self.scalers['weather'] = pickle.load(f)
            with open(_self.artifacts_dir / 'soil_scaler.pkl', 'rb') as f:
                _self.scalers['soil'] = pickle.load(f)
            with open(_self.artifacts_dir / 'geo_scaler.pkl', 'rb') as f:
                _self.scalers['geo'] = pickle.load(f)
            with open(_self.artifacts_dir / 'numerical_scaler.pkl', 'rb') as f:
                _self.scalers['numerical'] = pickle.load(f)
            
            # Load metadata
            with open(_self.artifacts_dir / 'feature_columns.json', 'r') as f:
                _self.feature_columns = json.load(f)
            with open(_self.artifacts_dir / 'metadata.json', 'r') as f:
                _self.metadata = json.load(f)
            with open(_self.artifacts_dir / 'crop_stats.json', 'r') as f:
                _self.crop_stats = json.load(f)
            
            st.success(f"✅ Loaded {len(_self.crop_classes)} crops, {len(_self.state_classes)} states")
            return True
        except Exception as e:
            st.error(f"Error loading artifacts: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    @st.cache_resource
    def load_available_models(_self):
        """Load list of available models"""
        models = []
        if _self.models_dir.exists():
            for model_file in _self.models_dir.glob('*.pkl'):
                model_name = model_file.stem.replace('_', ' ').title()
                models.append((model_name, model_file))
        return sorted(models)
    
    def load_model(self, model_path):
        """Load a specific model"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def engineer_features(self, input_data):
        """Create engineered features from input"""
        df = pd.DataFrame([input_data])
        
        # Time-based features
        min_year = self.metadata.get('split_info', {}).get('train_years', [1981])[0]
        df['Year_Since_Start'] = df['Year'] - min_year
        df['Year_Squared'] = df['Year'] ** 2
        
        # Climate interactions
        df['temp_prcp_interaction'] = df['tavg_growing_avg'] * df['prcp_growing_total']
        df['gdd_per_day'] = df['gdd_total'] / 180
        df['moisture_stress'] = df['prcp_growing_total'] / (df['tavg_growing_avg'] + 1)
        
        # Soil quality
        df['soil_quality_score'] = (
            df['organic_matter_pct'] * 0.3 +
            (7 - abs(df['ph'] - 6.5)) * 0.2 +
            df['awc'] * 100 * 0.3 +
            df['cec'] * 0.2
        )
        
        df['loam_index'] = 100 - (abs(df['clay_pct'] - 20) + abs(df['sand_pct'] - 40) + abs(df['silt_pct'] - 40))
        
        # Weather extremes
        df['temp_variability'] = df['temp_range_avg'] * df['heat_stress_days']
        
        # Get median heat stress for crop (default to 10 if not available)
        median_heat = 10
        df['extreme_heat_flag'] = int(df['heat_stress_days'].values[0] > median_heat)
        
        # Geographic features
        df['lat_squared'] = df['Latitude'] ** 2
        df['long_squared'] = df['Longitude'] ** 2
        df['lat_long_interaction'] = df['Latitude'] * df['Longitude']
        
        return df
    
    def encode_and_scale(self, df):
        """Encode categorical and scale numerical features"""
        # Encode crop
        df['Crop_Encoded'] = self.encoders['crop'].transform([df['Crop'].values[0]])[0]
        
        # Encode state
        df['State_Encoded'] = self.encoders['state'].transform([df['State'].values[0]])[0]
        
        # Encode county
        county_state = df['County'].values[0] + '_' + df['State'].values[0]
        known_counties = set(self.encoders['county'].classes_)
        if county_state in known_counties:
            df['County_Encoded'] = self.encoders['county'].transform([county_state])[0]
        else:
            df['County_Encoded'] = -1
        
        # Scale features
        weather_features = self.feature_columns['weather']
        soil_features = self.feature_columns['soil']
        geo_features = self.feature_columns['geo']
        time_features = self.feature_columns['time']
        
        df[weather_features] = self.scalers['weather'].transform(df[weather_features])
        df[soil_features] = self.scalers['soil'].transform(df[soil_features])
        df[geo_features] = self.scalers['geo'].transform(df[geo_features])
        df[time_features] = self.scalers['numerical'].transform(df[time_features])
        
        return df
    
    def prepare_input(self, input_data):
        """Prepare input for model prediction"""
        # Engineer features
        df = self.engineer_features(input_data)
        
        # Encode and scale
        df = self.encode_and_scale(df)
        
        # Select final features in correct order
        final_features = self.feature_columns['final_features']
        X = df[final_features]
        
        return X
    
    def predict(self, model, input_data):
        """Make prediction"""
        X = self.prepare_input(input_data)
        prediction = model.predict(X)[0]
        return prediction


def main():
    st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = CropYieldPredictor()
    
    # Load artifacts
    with st.spinner('Loading preprocessing artifacts...'):
        if not predictor.load_artifacts():
            st.error("❌ Failed to load preprocessing artifacts. Please run preprocessing first.")
            st.stop()
    
    # Load available models
    available_models = predictor.load_available_models()
    if not available_models:
        st.error("❌ No trained models found. Please train models first.")
        st.stop()
    
    st.success(f"✅ Loaded {len(available_models)} models successfully!")
    
    # Sidebar - Model Selection
    st.sidebar.markdown("## 🎯 Model Selection")
    model_names = [name for name, _ in available_models]
    
    # Try to default to XGBoost if available
    default_index = 0
    for i, name in enumerate(model_names):
        if 'xgboost' in name.lower():
            default_index = i
            break
    
    selected_model_name = st.sidebar.selectbox(
        "Choose a model:",
        model_names,
        index=default_index
    )
    
    # Get selected model path
    selected_model_path = [path for name, path in available_models if name == selected_model_name][0]
    
    # Load comparison table if available
    comparison_path = predictor.models_dir / 'model_comparison.csv'
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        model_row = comparison_df[comparison_df['Model'] == selected_model_name]
        
        if not model_row.empty:
            st.sidebar.markdown("### 📊 Model Performance")
            st.sidebar.metric("R² Score", f"{model_row['R²'].values[0]:.4f}")
            st.sidebar.metric("RMSE", f"{model_row['RMSE'].values[0]:.2f}")
            st.sidebar.metric("MAE", f"{model_row['MAE'].values[0]:.2f}")
            st.sidebar.metric("Within 10%", f"{model_row['Within 10%'].values[0]:.1f}%")
    
    # Main content - Input Features
    st.markdown('<h2 class="sub-header">📝 Input Features</h2>', unsafe_allow_html=True)
    
    # Create tabs for different input categories
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Basic Info", "🌤️ Weather", "🌍 Soil", "📍 Location"])
    
    input_data = {}
    
    # Tab 1: Basic Info
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Crop selection - FIXED: Use predictor.crop_classes instead of metadata
            crops = predictor.crop_classes
            input_data['Crop'] = st.selectbox("Select Crop", crops, index=0)
            
            # Show crop statistics
            if input_data['Crop'] in predictor.crop_stats:
                stats = predictor.crop_stats[input_data['Crop']]
                st.info(f"""
                **{input_data['Crop']} Statistics:**
                - Mean Yield: {stats['mean_yield']:.2f}
                - Median Yield: {stats['median_yield']:.2f}
                - Sample Count: {stats['count']:,}
                """)
        
        with col2:
            # State selection - FIXED: Use predictor.state_classes instead of metadata
            states = predictor.state_classes
            input_data['State'] = st.selectbox("Select State", states, index=0)
            
            # Year
            current_year = 2024
            input_data['Year'] = st.number_input(
                "Year",
                min_value=1980,
                max_value=2030,
                value=current_year,
                step=1
            )
        
        # County (full width) - FIXED: Use predictor.county_classes
        counties = [c.split('_')[0] for c in predictor.county_classes if c.endswith(f"_{input_data['State']}")]
        if counties:
            input_data['County'] = st.selectbox("Select County", sorted(set(counties)), index=0)
        else:
            input_data['County'] = st.text_input("County", "STORY")
            st.warning(f"⚠️ No counties found for {input_data['State']}. Using custom input.")
    
    # Tab 2: Weather Features
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data['tavg_growing_avg'] = st.number_input(
                "Avg Temperature (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=20.0,
                step=0.5,
                help="Average growing season temperature"
            )
            
            input_data['tmin_growing_avg'] = st.number_input(
                "Min Temperature (°C)",
                min_value=-20.0,
                max_value=30.0,
                value=15.0,
                step=0.5
            )
            
            input_data['tmax_growing_avg'] = st.number_input(
                "Max Temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=25.0,
                step=0.5
            )
        
        with col2:
            input_data['prcp_growing_total'] = st.number_input(
                "Total Precipitation (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=650.0,
                step=10.0,
                help="Total growing season precipitation"
            )
            
            input_data['rh_growing_avg'] = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )
            
            input_data['temp_range_avg'] = st.number_input(
                "Temperature Range (°C)",
                min_value=0.0,
                max_value=30.0,
                value=10.0,
                step=0.5,
                help="Average daily temperature range"
            )
        
        with col3:
            input_data['gdd_total'] = st.number_input(
                "Growing Degree Days",
                min_value=0.0,
                max_value=5000.0,
                value=2800.0,
                step=50.0,
                help="Total accumulated growing degree days"
            )
            
            input_data['heat_stress_days'] = st.number_input(
                "Heat Stress Days",
                min_value=0,
                max_value=100,
                value=5,
                step=1,
                help="Number of days with extreme heat"
            )
    
    # Tab 3: Soil Features
    with tab3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_data['clay_pct'] = st.slider(
                "Clay (%)",
                min_value=0,
                max_value=100,
                value=25,
                help="Percentage of clay in soil"
            )
            
            input_data['sand_pct'] = st.slider(
                "Sand (%)",
                min_value=0,
                max_value=100,
                value=35
            )
            
            input_data['silt_pct'] = st.slider(
                "Silt (%)",
                min_value=0,
                max_value=100,
                value=40
            )
            
            # Show total
            total = input_data['clay_pct'] + input_data['sand_pct'] + input_data['silt_pct']
            if total != 100:
                st.warning(f"⚠️ Soil composition total: {total}% (should be 100%)")
        
        with col2:
            input_data['organic_matter_pct'] = st.number_input(
                "Organic Matter (%)",
                min_value=0.0,
                max_value=20.0,
                value=4.5,
                step=0.1
            )
            
            input_data['ph'] = st.number_input(
                "Soil pH",
                min_value=3.0,
                max_value=10.0,
                value=6.5,
                step=0.1,
                help="Soil acidity/alkalinity"
            )
            
            input_data['bulk_density'] = st.number_input(
                "Bulk Density (g/cm³)",
                min_value=0.5,
                max_value=2.0,
                value=1.3,
                step=0.05
            )
        
        with col3:
            input_data['cec'] = st.number_input(
                "CEC (meq/100g)",
                min_value=0,
                max_value=50,
                value=18,
                step=1,
                help="Cation Exchange Capacity"
            )
            
            input_data['awc'] = st.number_input(
                "Available Water Capacity",
                min_value=0.0,
                max_value=0.5,
                value=0.18,
                step=0.01,
                help="Water holding capacity"
            )
    
    # Tab 4: Location
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['Latitude'] = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=42.0,
                step=0.1
            )
        
        with col2:
            input_data['Longitude'] = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-93.5,
                step=0.1
            )
        
        # Optional: Show map
        try:
            st.map(pd.DataFrame({'lat': [input_data['Latitude']], 'lon': [input_data['Longitude']]}))
        except:
            pass
    
    # Prediction Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 Predict Yield", use_container_width=True, type="primary")
    
    # Make Prediction
    if predict_button:
        with st.spinner('Making prediction...'):
            # Load model
            model = predictor.load_model(selected_model_path)
            
            if model is not None:
                try:
                    # Make prediction
                    prediction = predictor.predict(model, input_data)
                    
                    # Display result
                    st.markdown('<h2 class="sub-header">🎯 Prediction Result</h2>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">Predicted Yield</p>
                        <p class="prediction-value">{prediction:.2f}</p>
                        <p style="font-size: 1rem; color: #666;">bushels per acre (or appropriate unit)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show comparison with crop statistics
                    if input_data['Crop'] in predictor.crop_stats:
                        stats = predictor.crop_stats[input_data['Crop']]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Predicted", f"{prediction:.2f}")
                        with col2:
                            diff_mean = prediction - stats['mean_yield']
                            st.metric("vs Mean", f"{stats['mean_yield']:.2f}", f"{diff_mean:+.2f}")
                        with col3:
                            diff_median = prediction - stats['median_yield']
                            st.metric("vs Median", f"{stats['median_yield']:.2f}", f"{diff_median:+.2f}")
                        with col4:
                            pct_of_max = (prediction / stats['max_yield']) * 100
                            st.metric("% of Max", f"{pct_of_max:.1f}%")
                        
                        # Visualization
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=['Min', 'Mean', 'Predicted', 'Median', 'Max'],
                            y=[stats['min_yield'], stats['mean_yield'], prediction, 
                               stats['median_yield'], stats['max_yield']],
                            marker_color=['#B0BEC5', '#90CAF9', '#4CAF50', '#90CAF9', '#B0BEC5'],
                            text=[f"{stats['min_yield']:.1f}", f"{stats['mean_yield']:.1f}", 
                                  f"{prediction:.1f}", f"{stats['median_yield']:.1f}", 
                                  f"{stats['max_yield']:.1f}"],
                            textposition='auto',
                        ))
                        
                        fig.update_layout(
                            title=f"{input_data['Crop']} Yield Comparison",
                            yaxis_title="Yield",
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Export option
                    st.markdown("### 📥 Export Results")
                    result_data = input_data.copy()
                    result_data['Predicted_Yield'] = prediction
                    result_data['Model_Used'] = selected_model_name
                    result_df = pd.DataFrame([result_data])
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Prediction as CSV",
                        data=csv,
                        file_name=f"yield_prediction_{input_data['Crop']}_{input_data['Year']}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🌾 Crop Yield Prediction System | Built with Streamlit</p>
        <p style="font-size: 0.9rem;">Trained on 385k+ samples across 12 crops, 38 states, and 43 years (1981-2023)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()