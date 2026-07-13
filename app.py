# =============================================================================
# HMS GEOMETALLURGY NSR PREDICTION DASHBOARD
# Interactive Streamlit Application
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="HMS Geometallurgy NSR Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL AND FEATURES
# =============================================================================

@st.cache_resource
def load_production_model():
    """Load the production Random Forest model"""
    model_path = Path("production_model_top10_features.pkl")
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

@st.cache_resource
def load_feature_medians():
    """Load feature medians for imputation"""
    median_path = Path("feature_medians_top10.csv")
    if median_path.exists():
        try:
            return pd.read_csv(median_path)
        except:
            return None
    return None

@st.cache_data
def load_model_comparison():
    """Load model comparison results"""
    csv_path = Path("model_comparison_table.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

# Try to load the production model
production_model = load_production_model()
feature_medians = load_feature_medians()
model_comparison = load_model_comparison()

# =============================================================================
# TOP 10 FEATURES (from your production model)
# =============================================================================

TOP_10_FEATURES = [
    'Ilmenite_Percent',
    'THM_Percent',
    'TiO2_Percent',
    'Zircon_Percent',
    'Rutile_Percent',
    'Total_Valuables_Recovery',
    'Leucoxene_Percent',
    'ZrO2_Percent',
    'Slimes_Percent',
    'Desliming_Efficiency_Percent'
]

# Feature ranges for sliders (adjust based on your data)
FEATURE_RANGES = {
    'Ilmenite_Percent': (0.0, 100.0, 45.0),
    'THM_Percent': (0.0, 100.0, 15.0),
    'TiO2_Percent': (0.0, 100.0, 55.0),
    'Zircon_Percent': (0.0, 100.0, 5.0),
    'Rutile_Percent': (0.0, 100.0, 3.0),
    'Total_Valuables_Recovery': (0.0, 100.0, 75.0),
    'Leucoxene_Percent': (0.0, 100.0, 2.0),
    'ZrO2_Percent': (0.0, 100.0, 3.0),
    'Slimes_Percent': (0.0, 100.0, 15.0),
    'Desliming_Efficiency_Percent': (0.0, 100.0, 85.0)
}

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.title("🎯 Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "📊 NSR Predictor",
        "📈 Model Comparison",
        "📁 Batch Prediction",
        "🔬 Feature Analysis",
        "📝 About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Stats")
if production_model:
    st.sidebar.success("✅ Production Model Loaded")
    st.sidebar.info(f"📦 Features: {len(TOP_10_FEATURES)}")
    st.sidebar.info(f"📊 Model R²: 0.9847")
    st.sidebar.info(f"📉 MAE: $17.00/t")
else:
    st.sidebar.warning("⚠️ Model not found")

# =============================================================================
# PAGE 1: NSR PREDICTOR
# =============================================================================

if page == "📊 NSR Predictor":
    st.markdown('<p class="main-header">💎 NSR Value Predictor</p>', unsafe_allow_html=True)
    st.markdown("### Enter your sample characteristics below to predict Net Smelter Return")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🧪 Sample Input Parameters")
        
        input_data = {}
        for feature in TOP_10_FEATURES:
            min_val, max_val, default_val = FEATURE_RANGES[feature]
            input_data[feature] = st.slider(
                feature.replace('_', ' '),
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=0.1
            )
        
        input_df = pd.DataFrame([input_data])
    
    with col2:
        st.markdown("#### 📊 Prediction Result")
        
        if st.button("🔮 Predict NSR", type="primary", use_container_width=True):
            if production_model:
                predicted_nsrs = production_model.predict(input_df)
                predicted_value = predicted_nsrs[0]
                
                st.metric(
                    label="Predicted NSR",
                    value=f"${predicted_value:,.2f}/t",
                    delta=None
                )
                
                if predicted_value >= 1100:
                    st.success("💰 **High Grade** - Economically attractive")
                elif predicted_value >= 950:
                    st.info("✅ **Medium Grade** - Marginal economics")
                else:
                    st.warning("⚠️ **Low Grade** - Below economic cutoff")
                
                st.session_state['last_prediction'] = {
                    'input': input_data,
                    'nsr': predicted_value
                }
            else:
                st.error("❌ Production model not found.")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state['last_prediction']
            download_df = pd.DataFrame([pred['input']])
            download_df['Predicted_NSR_USD_per_t'] = pred['nsr']
            
            csv = download_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Prediction",
                csv,
                "nsr_prediction.csv",
                "text/csv",
                key='download-prediction'
            )
    
    if production_model:
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### 🎯 Feature Importance (Production Model)")
            importances = production_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': TOP_10_FEATURES,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#667eea')
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col4:
            st.markdown("#### 📋 Input Summary")
            summary_df = pd.DataFrame({
                'Feature': TOP_10_FEATURES,
                'Value': [f"{input_data[f]:.1f}%" for f in TOP_10_FEATURES]
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

# =============================================================================
# PAGE 2: MODEL COMPARISON
# =============================================================================

elif page == "📈 Model Comparison":
    st.markdown('<p class="main-header">📈 Model Performance Comparison</p>', unsafe_allow_html=True)
    st.markdown("### Comparing 6 ML models for NSR prediction")
    
    if model_comparison is not None:
        st.markdown("#### 📊 Performance Metrics")
        st.dataframe(
            model_comparison.round(4),
            hide_index=True,
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Test R² Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(model_comparison)))
            ax.bar(model_comparison['Model'], model_comparison['Test R²'], color=colors)
            ax.set_ylabel('R² Score')
            ax.set_title('Model Performance (Test Set)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📉 Test RMSE Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(model_comparison)))
            ax.bar(model_comparison['Model'], model_comparison['Test RMSE'], color=colors)
            ax.set_ylabel('RMSE ($/t)')
            ax.set_title('Model Error (Test Set)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        st.markdown("#### 📁 Detailed Visualizations")
        
        viz_cols = st.columns(2)
        
        with viz_cols[0]:
            perf_path = Path("dashboard/performance_comparison.png")
            if perf_path.exists():
                st.image(perf_path, caption="Performance Comparison", use_container_width=True)
        
        with viz_cols[1]:
            pred_path = Path("dashboard/predictions_vs_actual.png")
            if pred_path.exists():
                st.image(pred_path, caption="Predictions vs Actual", use_container_width=True)
        
        csv = model_comparison.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Model Comparison Table",
            csv,
            "model_comparison.csv",
            "text/csv",
            key='download-comparison'
        )
    else:
        st.warning("⚠️ Model comparison data not found. Please run compare_models.py first.")

# =============================================================================
# PAGE 3: BATCH PREDICTION
# =============================================================================

elif page == "📁 Batch Prediction":
    st.markdown('<p class="main-header">📁 Batch Prediction from CSV</p>', unsafe_allow_html=True)
    st.markdown("### Upload a CSV file with the 10 required features")
    
    st.markdown("#### 📋 Required Columns:")
    st.write(TOP_10_FEATURES)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} samples")
            
            missing_cols = [col for col in TOP_10_FEATURES if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                if production_model:
                    predictions = production_model.predict(df[TOP_10_FEATURES])
                    
                    df['Predicted_NSR_USD_per_t'] = predictions
                    
                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean NSR", f"${df['Predicted_NSR_USD_per_t'].mean():,.2f}/t")
                    with col2:
                        st.metric("Min NSR", f"${df['Predicted_NSR_USD_per_t'].min():,.2f}/t")
                    with col3:
                        st.metric("Max NSR", f"${df['Predicted_NSR_USD_per_t'].max():,.2f}/t")
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        "batch_predictions.csv",
                        "text/csv",
                        key='download-batch'
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df['Predicted_NSR_USD_per_t'], bins=30, color='#667eea', edgecolor='black')
                    ax.set_xlabel('NSR ($/t)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Predicted NSR Values')
                    st.pyplot(fig)
                else:
                    st.error("❌ Production model not found.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# =============================================================================
# PAGE 4: FEATURE ANALYSIS
# =============================================================================

elif page == "🔬 Feature Analysis":
    st.markdown('<p class="main-header">🔬 Feature Analysis</p>', unsafe_allow_html=True)
    st.markdown("### Understanding the Top 10 Features")
    
    st.markdown("#### 📊 Feature Reduction Results")
    
    reduction_data = {
        'Features': [1, 2, 3, 5, 10, 20, 81],
        'R²': [0.5806, 0.7306, 0.8052, 0.9779, 0.9840, 0.9820, 0.9784],
        'MAE': [100.90, 80.58, 69.02, 22.04, 17.25, 18.67, 20.70],
        'RMSE': [146.17, 117.16, 99.62, 33.58, 28.53, 30.26, 33.16]
    }
    
    reduction_df = pd.DataFrame(reduction_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(reduction_df['Features'], reduction_df['R²'], marker='o', linewidth=2, markersize=8, color='#667eea')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('R² Score')
        ax.set_title('R² vs Number of Features')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(reduction_df['Features'], reduction_df['MAE'], marker='s', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('MAE ($/t)')
        ax.set_title('MAE vs Number of Features')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("#### 💡 Key Insight")
    st.info("""
    **Top 10 features achieve 98.4% R²** - only 0.61% drop from full model (81 features),
    but requires collecting **71 fewer features**! This dramatically reduces data collection
    costs and complexity for deployment.
    """)
    
    feat_imp_path = Path("dashboard/feature_importance.png")
    if feat_imp_path.exists():
        st.image(feat_imp_path, caption="Feature Importance Analysis", use_container_width=True)

# =============================================================================
# PAGE 5: ABOUT
# =============================================================================

elif page == "📝 About":
    st.markdown('<p class="main-header">📝 About This Dashboard</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This dashboard provides an interactive interface for predicting Net Smelter Return (NSR) 
    values for HMS (Heavy Mineral Sands) geometallurgical samples using machine learning.
    
    ### 🏗️ Model Architecture
    - **Production Model**: Random Forest Regressor
    - **Features**: Top 10 most important features (from 81 original features)
    - **Performance**: R² = 0.9847, MAE = $17.00/t, RMSE = $27.87/t
    - **Training Data**: 4,800 samples (synthetic)
    - **Test Data**: 1,200 samples (synthetic)
    
    ### 📦 Top 10 Features
    1. Ilmenite_Percent
    2. THM_Percent
    3. TiO2_Percent
    4. Zircon_Percent
    5. Rutile_Percent
    6. Total_Valuables_Recovery
    7. Leucoxene_Percent
    8. ZrO2_Percent
    9. Slimes_Percent
    10. Desliming_Efficiency_Percent
    
    ### 🚀 How to Use
    1. **NSR Predictor**: Enter sample parameters using sliders for instant predictions
    2. **Model Comparison**: Review performance metrics across 6 ML models
    3. **Batch Prediction**: Upload CSV files for bulk predictions
    4. **Feature Analysis**: Understand feature importance and reduction results
    
    ### 📁 Output Files
    All visualizations and metrics are saved to the `dashboard/` folder:
    - performance_comparison.png
    - predictions_vs_actual.png
    - residuals_analysis.png
    - feature_importance.png
    - model_comparison_table.csv
    - dashboard_summary.txt
    
    ---
    **Built with Streamlit** | HMS Geometallurgy Project
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "HMS Geometallurgical Model Project | © 2026"
    "</div>",
    unsafe_allow_html=True
)
