import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sea_level_predictor import SeaLevelPredictor
from datetime import datetime
import requests

st.set_page_config(
    page_title="Sea Level Rise Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the sea level data"""
    try:
        predictor = SeaLevelPredictor()
        data = predictor.preprocess_data()
        return predictor, data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def create_interactive_plot(predictor, data, end_year=2100):
    """Create interactive Plotly visualization"""
    # Fit models
    predictor.fit_linear_model()
    predictor.fit_linear_model(2000)
    predictor.fit_polynomial_model(2)
    predictor.predict_future(end_year)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Historical Data & Projections', 'Rate of Change', 
                       'Decadal Averages', 'Model Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Historical data scatter
    fig.add_trace(
        go.Scatter(
            x=data['Year'], 
            y=data['CSIRO Adjusted Sea Level'],
            mode='markers',
            name='Historical Data',
            marker=dict(size=4, opacity=0.7),
            hovertemplate='Year: %{x}<br>Sea Level: %{y:.2f} inches<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add trend lines
    years_extended = np.arange(data['Year'].min(), end_year + 1)
    for key, model_info in predictor.models.items():
        if 'linear' in key:
            X_pred = years_extended.reshape(-1, 1)
            if '2000' in key:
                X_pred = X_pred[X_pred.flatten() >= 2000]
            y_pred = model_info['model'].predict(X_pred)
            
            fig.add_trace(
                go.Scatter(
                    x=X_pred.flatten(),
                    y=y_pred,
                    mode='lines',
                    name=f"{key.replace('_', ' ').title()} (R²={model_info['r2']:.3f})",
                    line=dict(width=2),
                    hovertemplate='Year: %{x}<br>Predicted: %{y:.2f} inches<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Rate of change
    data['Rate_Change'] = data['CSIRO Adjusted Sea Level'].diff()
    rolling_rate = data['Rate_Change'].rolling(window=10, center=True).mean()
    
    fig.add_trace(
        go.Scatter(
            x=data['Year'],
            y=rolling_rate,
            mode='lines',
            name='Rate of Change',
            line=dict(color='red', width=2),
            hovertemplate='Year: %{x}<br>Rate: %{y:.3f} inches/year<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Decadal averages
    decade_avg = data.groupby('Decade')['CSIRO Adjusted Sea Level'].mean()
    fig.add_trace(
        go.Bar(
            x=decade_avg.index,
            y=decade_avg.values,
            name='Decadal Average',
            marker_color='green',
            opacity=0.7,
            hovertemplate='Decade: %{x}s<br>Average: %{y:.2f} inches<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Future projections comparison
    future_years = np.arange(2020, end_year + 1)
    for key, pred_info in predictor.predictions.items():
        if 'linear' in key:
            mask = pred_info['years'] >= 2020
            fig.add_trace(
                go.Scatter(
                    x=pred_info['years'][mask],
                    y=pred_info['predictions'][mask],
                    mode='lines',
                    name=f"Future: {key.replace('_', ' ').title()}",
                    line=dict(width=2, dash='dash'),
                    hovertemplate='Year: %{x}<br>Prediction: %{y:.2f} inches<extra></extra>'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        title_text="Advanced Sea Level Rise Analysis",
        title_x=0.5,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Decade", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    
    fig.update_yaxes(title_text="Sea Level (inches)", row=1, col=1)
    fig.update_yaxes(title_text="Rate (inches/year)", row=1, col=2)
    fig.update_yaxes(title_text="Sea Level (inches)", row=2, col=1)
    fig.update_yaxes(title_text="Sea Level (inches)", row=2, col=2)
    
    return fig

def main():
    st.title("🌊 Advanced Sea Level Rise Predictor")
    st.markdown("### Comprehensive Analysis of Global Sea Level Trends")
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    end_year = st.sidebar.slider("Prediction End Year", 2030, 2150, 2100, 5)
    show_uncertainty = st.sidebar.checkbox("Show Uncertainty Bands", False)
    
    # Load data
    predictor, data = load_data()
    
    if predictor is None or data is None:
        st.error("Failed to load data. Please check the data file.")
        return
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_rise = data['CSIRO Adjusted Sea Level'].iloc[-1] - data['CSIRO Adjusted Sea Level'].iloc[0]
        st.metric("Total Rise Since 1880", f"{total_rise:.2f} inches", f"{total_rise*2.54:.1f} cm")
    
    with col2:
        avg_rise = data['CSIRO Adjusted Sea Level'].diff().mean()
        st.metric("Average Annual Rise", f"{avg_rise:.3f} inches/year", f"{avg_rise*2.54:.2f} cm/year")
    
    with col3:
        recent_data = data[data['Year'] >= 2000]
        if len(recent_data) > 1:
            recent_rise = recent_data['CSIRO Adjusted Sea Level'].diff().mean()
            st.metric("Recent Annual Rise (2000+)", f"{recent_rise:.3f} inches/year", f"{recent_rise*2.54:.2f} cm/year")
    
    # Interactive plot
    st.subheader("Interactive Analysis Dashboard")
    fig = create_interactive_plot(predictor, data, end_year)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    st.subheader("Model Performance Comparison")
    
    # Fit models for comparison
    predictor.fit_linear_model()
    predictor.fit_linear_model(2000)
    predictor.fit_polynomial_model(2)
    predictor.fit_polynomial_model(3, 2000)
    
    model_data = []
    for key, model_info in predictor.models.items():
        model_data.append({
            'Model': key.replace('_', ' ').title(),
            'R² Score': f"{model_info['r2']:.4f}",
            'RMSE': f"{model_info['rmse']:.3f}",
            'Start Year': model_info['start_year']
        })
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True)
    
    # Future predictions table
    st.subheader("Future Predictions")
    predictor.predict_future(end_year)
    
    prediction_years = [2030, 2050, 2070, 2100]
    prediction_data = []
    
    for year in prediction_years:
        if year <= end_year:
            row = {'Year': year}
            for key, pred_info in predictor.predictions.items():
                if 'linear' in key:
                    year_idx = np.where(pred_info['years'] == year)[0]
                    if len(year_idx) > 0:
                        prediction = pred_info['predictions'][year_idx[0]]
                        row[key.replace('_', ' ').title()] = f"{prediction:.2f} inches"
            prediction_data.append(row)
    
    if prediction_data:
        pred_df = pd.DataFrame(prediction_data)
        st.dataframe(pred_df, use_container_width=True)
    
    # Data insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Acceleration in Sea Level Rise:**
        - The rate of sea level rise has accelerated significantly since 2000
        - Modern satellite measurements show higher precision in recent decades
        - Climate change impacts are becoming more pronounced
        """)
    
    with col2:
        st.warning("""
        **Future Projections:**
        - Linear models may underestimate future rise due to acceleration
        - Polynomial models capture non-linear trends better
        - Uncertainty increases significantly for distant future predictions
        """)
    
    # Download section
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Report"):
            # Create a summary report
            stats = predictor.calculate_statistics()
            report = f"""
# Sea Level Rise Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total rise since 1880: {stats['total_rise']:.2f} inches ({stats['total_rise']*2.54:.1f} cm)
- Average annual rise: {stats['avg_annual_rise']:.3f} inches/year
- Recent trend acceleration: {stats.get('acceleration', 'N/A'):.4f} inches/year²

## Model Performance
"""
            for key, model_info in predictor.models.items():
                report += f"- {key.replace('_', ' ').title()}: R² = {model_info['r2']:.4f}, RMSE = {model_info['rmse']:.3f}\n"
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"sea_level_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    with col2:
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="Download Data (CSV)",
            data=csv_data,
            file_name="sea_level_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()