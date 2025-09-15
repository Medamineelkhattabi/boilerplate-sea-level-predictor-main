import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SeaLevelPredictor:
    def __init__(self, data_file='epa-sea-level.csv'):
        self.data = pd.read_csv(data_file)
        self.models = {}
        self.predictions = {}
        
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        # Remove rows with missing CSIRO data
        self.data = self.data.dropna(subset=['CSIRO Adjusted Sea Level'])
        
        # Create additional features
        self.data['Decade'] = (self.data['Year'] // 10) * 10
        self.data['Years_Since_1880'] = self.data['Year'] - 1880
        
        return self.data
    
    def fit_linear_model(self, start_year=None):
        """Fit linear regression model"""
        data = self.data if start_year is None else self.data[self.data['Year'] >= start_year]
        
        X = data['Year'].values.reshape(-1, 1)
        y = data['CSIRO Adjusted Sea Level'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        key = f'linear_{start_year or "all"}'
        self.models[key] = {
            'model': model,
            'r2': r2_score(y, model.predict(X)),
            'rmse': np.sqrt(mean_squared_error(y, model.predict(X))),
            'start_year': start_year or self.data['Year'].min()
        }
        
        return model
    
    def fit_polynomial_model(self, degree=2, start_year=None):
        """Fit polynomial regression model"""
        data = self.data if start_year is None else self.data[self.data['Year'] >= start_year]
        
        X = data['Year'].values.reshape(-1, 1)
        y = data['CSIRO Adjusted Sea Level'].values
        
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        key = f'poly_{degree}_{start_year or "all"}'
        self.models[key] = {
            'model': model,
            'poly_features': poly_features,
            'r2': r2_score(y, model.predict(X_poly)),
            'rmse': np.sqrt(mean_squared_error(y, model.predict(X_poly))),
            'degree': degree,
            'start_year': start_year or self.data['Year'].min()
        }
        
        return model
    
    def predict_future(self, end_year=2050):
        """Generate predictions for future years"""
        future_years = np.arange(self.data['Year'].max() + 1, end_year + 1)
        
        for key, model_info in self.models.items():
            X_future = future_years.reshape(-1, 1)
            
            if 'poly' in key:
                X_future = model_info['poly_features'].transform(X_future)
            
            predictions = model_info['model'].predict(X_future)
            
            self.predictions[key] = {
                'years': future_years,
                'predictions': predictions
            }
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        stats = {
            'total_rise': self.data['CSIRO Adjusted Sea Level'].iloc[-1] - self.data['CSIRO Adjusted Sea Level'].iloc[0],
            'avg_annual_rise': self.data['CSIRO Adjusted Sea Level'].diff().mean(),
            'acceleration': None,
            'recent_trend': None
        }
        
        # Calculate acceleration (change in rate)
        recent_data = self.data[self.data['Year'] >= 2000]
        if len(recent_data) > 1:
            recent_slope = linregress(recent_data['Year'], recent_data['CSIRO Adjusted Sea Level']).slope
            early_data = self.data[self.data['Year'] <= 1950]
            early_slope = linregress(early_data['Year'], early_data['CSIRO Adjusted Sea Level']).slope
            stats['acceleration'] = recent_slope - early_slope
            stats['recent_trend'] = recent_slope
        
        return stats

def draw_plot():
    """Original function for compatibility with tests"""
    predictor = SeaLevelPredictor()
    predictor.preprocess_data()
    
    # Read data from file
    df = pd.read_csv('epa-sea-level.csv')
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], alpha=0.7, s=30)
    
    # Create first line of best fit (all data)
    slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
    years_extended = np.arange(df['Year'].min(), 2051)
    line1 = slope * years_extended + intercept
    plt.plot(years_extended, line1, 'r-', label=f'Linear fit (all data): R²={r_value**2:.3f}')
    
    # Create second line of best fit (from 2000)
    df_2000 = df[df['Year'] >= 2000]
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(df_2000['Year'], df_2000['CSIRO Adjusted Sea Level'])
    years_2000 = np.arange(2000, 2051)
    line2 = slope2 * years_2000 + intercept2
    plt.plot(years_2000, line2, 'g-', label=f'Linear fit (2000+): R²={r_value2**2:.3f}')
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.title('Rise in Sea Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()

def create_advanced_analysis():
    """Create comprehensive analysis with multiple visualizations"""
    predictor = SeaLevelPredictor()
    data = predictor.preprocess_data()
    
    # Fit multiple models
    predictor.fit_linear_model()
    predictor.fit_linear_model(2000)
    predictor.fit_polynomial_model(2)
    predictor.fit_polynomial_model(3, 2000)
    
    # Generate predictions
    predictor.predict_future(2100)
    
    # Calculate statistics
    stats = predictor.calculate_statistics()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Sea Level Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Historical data with multiple trend lines
    ax1 = axes[0, 0]
    ax1.scatter(data['Year'], data['CSIRO Adjusted Sea Level'], alpha=0.6, s=20, color='blue')
    
    # Add trend lines
    years_full = np.arange(data['Year'].min(), 2101)
    for key, model_info in predictor.models.items():
        if 'linear' in key:
            X_pred = years_full.reshape(-1, 1)
            if 'linear_2000' in key:
                X_pred = X_pred[X_pred.flatten() >= 2000]
            y_pred = model_info['model'].predict(X_pred)
            label = f"{key.replace('_', ' ').title()}: R²={model_info['r2']:.3f}"
            ax1.plot(X_pred.flatten(), y_pred, label=label, linewidth=2)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Sea Level (inches)')
    ax1.set_title('Historical Trends & Projections')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rate of change over time
    ax2 = axes[0, 1]
    data['Rate_Change'] = data['CSIRO Adjusted Sea Level'].diff() / data['Year'].diff()
    rolling_rate = data['Rate_Change'].rolling(window=10, center=True).mean()
    ax2.plot(data['Year'], rolling_rate, color='red', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Rate of Change (inches/year)')
    ax2.set_title('Rate of Sea Level Rise Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Decadal averages
    ax3 = axes[1, 0]
    decade_avg = data.groupby('Decade')['CSIRO Adjusted Sea Level'].mean()
    ax3.bar(decade_avg.index, decade_avg.values, width=8, alpha=0.7, color='green')
    ax3.set_xlabel('Decade')
    ax3.set_ylabel('Average Sea Level (inches)')
    ax3.set_title('Decadal Average Sea Levels')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model comparison and uncertainty
    ax4 = axes[1, 1]
    future_years = np.arange(2020, 2101)
    
    for key, pred_info in predictor.predictions.items():
        if 'linear' in key:
            mask = pred_info['years'] >= 2020
            ax4.plot(pred_info['years'][mask], pred_info['predictions'][mask], 
                    label=key.replace('_', ' ').title(), linewidth=2)
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Predicted Sea Level (inches)')
    ax4.set_title('Future Projections Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_sea_level_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n=== SEA LEVEL RISE STATISTICS ===")
    print(f"Total rise since 1880: {stats['total_rise']:.2f} inches")
    print(f"Average annual rise: {stats['avg_annual_rise']:.3f} inches/year")
    if stats['acceleration']:
        print(f"Acceleration in rise rate: {stats['acceleration']:.4f} inches/year²")
        print(f"Recent trend (2000+): {stats['recent_trend']:.3f} inches/year")
    
    print("\n=== MODEL PERFORMANCE ===")
    for key, model_info in predictor.models.items():
        print(f"{key.replace('_', ' ').title()}: R² = {model_info['r2']:.4f}, RMSE = {model_info['rmse']:.3f}")
    
    return predictor, stats

if __name__ == "__main__":
    # Run advanced analysis
    predictor, stats = create_advanced_analysis()