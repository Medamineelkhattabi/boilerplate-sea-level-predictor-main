import unittest
import numpy as np
from sea_level_predictor import SeaLevelPredictor
import pandas as pd
from sklearn.metrics import r2_score

class TestSeaLevelPredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = SeaLevelPredictor()
        self.data = self.predictor.preprocess_data()
    
    def test_data_loading(self):
        """Test data loading and preprocessing"""
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertGreater(len(self.data), 0)
        self.assertIn('Year', self.data.columns)
        self.assertIn('CSIRO Adjusted Sea Level', self.data.columns)
    
    def test_linear_model_fitting(self):
        """Test linear model fitting"""
        model = self.predictor.fit_linear_model()
        self.assertIsNotNone(model)
        self.assertIn('linear_all', self.predictor.models)
        
        # Check model performance
        model_info = self.predictor.models['linear_all']
        self.assertGreater(model_info['r2'], 0.8)  # Should have good fit
        self.assertGreater(model_info['rmse'], 0)
    
    def test_polynomial_model_fitting(self):
        """Test polynomial model fitting"""
        model = self.predictor.fit_polynomial_model(degree=2)
        self.assertIsNotNone(model)
        self.assertIn('poly_2_all', self.predictor.models)
        
        # Polynomial should fit better than linear
        self.predictor.fit_linear_model()
        poly_r2 = self.predictor.models['poly_2_all']['r2']
        linear_r2 = self.predictor.models['linear_all']['r2']
        self.assertGreaterEqual(poly_r2, linear_r2)
    
    def test_future_predictions(self):
        """Test future predictions generation"""
        self.predictor.fit_linear_model()
        self.predictor.predict_future(2050)
        
        self.assertIn('linear_all', self.predictor.predictions)
        pred_info = self.predictor.predictions['linear_all']
        
        self.assertGreater(len(pred_info['years']), 0)
        self.assertGreater(len(pred_info['predictions']), 0)
        self.assertEqual(len(pred_info['years']), len(pred_info['predictions']))
    
    def test_statistics_calculation(self):
        """Test statistics calculation"""
        stats = self.predictor.calculate_statistics()
        
        self.assertIn('total_rise', stats)
        self.assertIn('avg_annual_rise', stats)
        self.assertGreater(stats['total_rise'], 0)  # Sea level should be rising
        self.assertGreater(stats['avg_annual_rise'], 0)
    
    def test_model_comparison(self):
        """Test multiple model comparison"""
        # Fit multiple models
        self.predictor.fit_linear_model()
        self.predictor.fit_linear_model(2000)
        self.predictor.fit_polynomial_model(2)
        
        # Should have multiple models
        self.assertGreaterEqual(len(self.predictor.models), 3)
        
        # All models should have required metrics
        for key, model_info in self.predictor.models.items():
            self.assertIn('r2', model_info)
            self.assertIn('rmse', model_info)
            self.assertBetween(model_info['r2'], 0, 1)
            self.assertGreater(model_info['rmse'], 0)
    
    def test_data_quality(self):
        """Test data quality and consistency"""
        # Check for missing values in key columns
        self.assertEqual(self.data['CSIRO Adjusted Sea Level'].isnull().sum(), 0)
        
        # Check year range
        self.assertGreaterEqual(self.data['Year'].min(), 1880)
        self.assertLessEqual(self.data['Year'].max(), 2020)
        
        # Check for reasonable sea level values
        self.assertGreater(self.data['CSIRO Adjusted Sea Level'].min(), -5)
        self.assertLess(self.data['CSIRO Adjusted Sea Level'].max(), 20)
    
    def test_prediction_reasonableness(self):
        """Test that predictions are reasonable"""
        self.predictor.fit_linear_model()
        self.predictor.predict_future(2100)
        
        pred_info = self.predictor.predictions['linear_all']
        
        # Future predictions should be higher than current levels
        current_level = self.data['CSIRO Adjusted Sea Level'].iloc[-1]
        future_2100 = pred_info['predictions'][-1]
        
        self.assertGreater(future_2100, current_level)
        
        # But not unreasonably high (less than 50 inches rise by 2100)
        self.assertLess(future_2100 - current_level, 50)
    
    def assertBetween(self, value, min_val, max_val):
        """Helper method to check if value is between min and max"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

class TestAdvancedFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for advanced features"""
        self.predictor = SeaLevelPredictor()
        self.data = self.predictor.preprocess_data()
    
    def test_model_performance_metrics(self):
        """Test model performance evaluation"""
        self.predictor.fit_linear_model()
        self.predictor.fit_polynomial_model(2)
        
        # Both models should have performance metrics
        for key, model_info in self.predictor.models.items():
            self.assertIn('r2', model_info)
            self.assertIn('rmse', model_info)
            
            # R² should be reasonable for sea level data
            self.assertGreater(model_info['r2'], 0.7)
    
    def test_acceleration_detection(self):
        """Test detection of acceleration in sea level rise"""
        stats = self.predictor.calculate_statistics()
        
        if stats['acceleration'] is not None:
            # Should detect positive acceleration (increasing rate)
            self.assertGreater(stats['acceleration'], 0)
    
    def test_uncertainty_quantification(self):
        """Test uncertainty in predictions"""
        # Fit models with different start years
        self.predictor.fit_linear_model()
        self.predictor.fit_linear_model(2000)
        self.predictor.predict_future(2050)
        
        # Different models should give different predictions (uncertainty)
        pred1 = self.predictor.predictions['linear_all']['predictions'][-1]
        pred2 = self.predictor.predictions['linear_2000']['predictions'][-1]
        
        # There should be some difference between models
        self.assertNotEqual(pred1, pred2)

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)