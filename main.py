#!/usr/bin/env python3
"""
Advanced Sea Level Predictor - Main Entry Point

This file demonstrates the enhanced capabilities of the sea level predictor.
Run this file to see a comprehensive analysis with multiple models and visualizations.
"""

import sea_level_predictor
from sea_level_predictor import SeaLevelPredictor, create_advanced_analysis
from unittest import main
import sys

def main_demo():
    """Run a comprehensive demonstration of the advanced features"""
    print("🌊 Advanced Sea Level Rise Predictor")
    print("=" * 50)
    
    try:
        # Initialize the predictor
        print("\n1. Initializing predictor and loading data...")
        predictor = SeaLevelPredictor()
        data = predictor.preprocess_data()
        print(f"   ✓ Loaded {len(data)} data points from {data['Year'].min()} to {data['Year'].max()}")
        
        # Fit multiple models
        print("\n2. Training multiple prediction models...")
        predictor.fit_linear_model()
        predictor.fit_linear_model(2000)
        predictor.fit_polynomial_model(2)
        predictor.fit_polynomial_model(3, 2000)
        print(f"   ✓ Trained {len(predictor.models)} models")
        
        # Generate predictions
        print("\n3. Generating future projections...")
        predictor.predict_future(2100)
        print("   ✓ Generated predictions through 2100")
        
        # Calculate statistics
        print("\n4. Calculating comprehensive statistics...")
        stats = predictor.calculate_statistics()
        print(f"   ✓ Total rise since 1880: {stats['total_rise']:.2f} inches")
        print(f"   ✓ Average annual rise: {stats['avg_annual_rise']:.3f} inches/year")
        if stats['acceleration']:
            print(f"   ✓ Acceleration detected: {stats['acceleration']:.4f} inches/year²")
        
        # Model performance
        print("\n5. Model Performance Summary:")
        for key, model_info in predictor.models.items():
            print(f"   • {key.replace('_', ' ').title()}: R² = {model_info['r2']:.4f}, RMSE = {model_info['rmse']:.3f}")
        
        # Key predictions
        print("\n6. Key Future Predictions (inches):")
        key_years = [2030, 2050, 2100]
        for year in key_years:
            print(f"   {year}:")
            for key, pred_info in predictor.predictions.items():
                year_idx = pred_info['years'] == year
                if year_idx.any():
                    prediction = pred_info['predictions'][year_idx][0]
                    print(f"     • {key.replace('_', ' ').title()}: {prediction:.2f}")
        
        print("\n7. Generating advanced visualizations...")
        # Run the original function for compatibility
        sea_level_predictor.draw_plot()
        print("   ✓ Basic plot saved as 'sea_level_plot.png'")
        
        # Run advanced analysis
        predictor_advanced, stats_advanced = create_advanced_analysis()
        print("   ✓ Advanced analysis saved as 'advanced_sea_level_analysis.png'")
        
        print("\n🎉 Analysis complete! Check the generated plots and consider running:")
        print("   • streamlit run streamlit_app.py  (for interactive dashboard)")
        print("   • jupyter notebook analysis_notebook.ipynb  (for detailed analysis)")
        print("   • python cli.py --help  (for command-line options)")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check that all dependencies are installed: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    # Check if user wants to run tests instead
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running unit tests...")
        # Run original tests
        main(module='test_module', exit=False)
        # Run advanced tests
        try:
            import test_advanced
            main(module='test_advanced', exit=False)
        except ImportError:
            print("Advanced tests not available")
    else:
        # Run the main demonstration
        success = main_demo()
        if not success:
            sys.exit(1)