#!/usr/bin/env python3
"""
Sea Level Predictor CLI Tool

A command-line interface for the advanced sea level predictor.
Provides various analysis options and output formats.
"""

import argparse
import sys
import json
from datetime import datetime
from sea_level_predictor import SeaLevelPredictor, create_advanced_analysis
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Sea Level Rise Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --analyze --end-year 2100
  %(prog)s --predict 2050 --model linear
  %(prog)s --stats --export json
  %(prog)s --plot --save-plot sea_level_analysis.png
        """
    )
    
    # Analysis options
    parser.add_argument('--analyze', action='store_true',
                       help='Run complete analysis with visualizations')
    parser.add_argument('--predict', type=int, metavar='YEAR',
                       help='Generate predictions up to specified year')
    parser.add_argument('--stats', action='store_true',
                       help='Calculate and display statistics')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    # Model options
    parser.add_argument('--model', choices=['linear', 'polynomial', 'all'],
                       default='all', help='Model type to use')
    parser.add_argument('--start-year', type=int,
                       help='Start year for model training')
    parser.add_argument('--end-year', type=int, default=2100,
                       help='End year for predictions (default: 2100)')
    
    # Output options
    parser.add_argument('--export', choices=['json', 'csv', 'txt'],
                       help='Export results in specified format')
    parser.add_argument('--save-plot', metavar='FILENAME',
                       help='Save plot to file')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = SeaLevelPredictor()
        data = predictor.preprocess_data()
        if args.verbose:
            print(f"Loaded {len(data)} data points from {data['Year'].min()} to {data['Year'].max()}")
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    
    results = {}
    
    # Run analysis
    if args.analyze:
        if args.verbose:
            print("Running complete analysis...")
        try:
            predictor_advanced, stats = create_advanced_analysis()
            results['analysis'] = 'Complete analysis generated'
            results['statistics'] = stats
        except Exception as e:
            print(f"Error in analysis: {e}", file=sys.stderr)
            return 1
    
    # Generate predictions
    if args.predict:
        if args.verbose:
            print(f"Generating predictions up to {args.predict}...")
        
        try:
            # Fit models based on selection
            if args.model in ['linear', 'all']:
                predictor.fit_linear_model(args.start_year)
                if args.start_year != 2000:
                    predictor.fit_linear_model(2000)
            
            if args.model in ['polynomial', 'all']:
                predictor.fit_polynomial_model(2, args.start_year)
                predictor.fit_polynomial_model(3, args.start_year)
            
            predictor.predict_future(args.predict)
            
            # Format predictions
            predictions = {}
            for key, pred_info in predictor.predictions.items():
                predictions[key] = {
                    'years': pred_info['years'].tolist(),
                    'predictions': pred_info['predictions'].tolist()
                }
            
            results['predictions'] = predictions
            
            # Print key predictions
            key_years = [2030, 2050, 2070, 2100]
            print(f"\nKey Predictions (inches above 1880 baseline):")
            print("-" * 50)
            
            for year in key_years:
                if year <= args.predict:
                    print(f"\n{year}:")
                    for key, pred_info in predictor.predictions.items():
                        year_idx = pred_info['years'] == year
                        if year_idx.any():
                            prediction = pred_info['predictions'][year_idx][0]
                            print(f"  {key.replace('_', ' ').title()}: {prediction:.2f} inches")
            
        except Exception as e:
            print(f"Error generating predictions: {e}", file=sys.stderr)
            return 1
    
    # Calculate statistics
    if args.stats:
        if args.verbose:
            print("Calculating statistics...")
        
        try:
            stats = predictor.calculate_statistics()
            results['statistics'] = stats
            
            print("\nSea Level Rise Statistics:")
            print("=" * 40)
            print(f"Total rise since 1880: {stats['total_rise']:.2f} inches ({stats['total_rise']*2.54:.1f} cm)")
            print(f"Average annual rise: {stats['avg_annual_rise']:.3f} inches/year")
            
            if stats['acceleration']:
                print(f"Acceleration: {stats['acceleration']:.4f} inches/year²")
                print(f"Recent trend (2000+): {stats['recent_trend']:.3f} inches/year")
            
            # Model performance if models are fitted
            if predictor.models:
                print(f"\nModel Performance:")
                print("-" * 30)
                for key, model_info in predictor.models.items():
                    print(f"{key.replace('_', ' ').title()}: R² = {model_info['r2']:.4f}, RMSE = {model_info['rmse']:.3f}")
            
        except Exception as e:
            print(f"Error calculating statistics: {e}", file=sys.stderr)
            return 1
    
    # Generate plots
    if args.plot:
        if args.verbose:
            print("Generating plots...")
        
        try:
            # Ensure models are fitted
            if not predictor.models:
                predictor.fit_linear_model()
                predictor.fit_linear_model(2000)
                predictor.fit_polynomial_model(2)
            
            if not predictor.predictions:
                predictor.predict_future(args.end_year)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Historical data
            plt.scatter(data['Year'], data['CSIRO Adjusted Sea Level'], 
                       alpha=0.6, s=20, color='blue', label='Historical Data')
            
            # Predictions
            colors = ['red', 'green', 'orange', 'purple']
            for i, (key, pred_info) in enumerate(predictor.predictions.items()):
                if i < len(colors):
                    plt.plot(pred_info['years'], pred_info['predictions'],
                            color=colors[i], linewidth=2, linestyle='--',
                            label=f"{key.replace('_', ' ').title()}")
            
            plt.axvline(x=data['Year'].max(), color='black', linestyle=':', 
                       alpha=0.7, label='End of Historical Data')
            
            plt.xlabel('Year')
            plt.ylabel('Sea Level (inches)')
            plt.title('Sea Level Rise: Historical Data and Projections')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if args.save_plot:
                plt.savefig(f"{args.output_dir}/{args.save_plot}", dpi=300, bbox_inches='tight')
                print(f"Plot saved to {args.output_dir}/{args.save_plot}")
            else:
                plt.show()
            
        except Exception as e:
            print(f"Error generating plots: {e}", file=sys.stderr)
            return 1
    
    # Export results
    if args.export and results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{args.output_dir}/sea_level_results_{timestamp}.{args.export}"
        
        try:
            if args.export == 'json':
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            elif args.export == 'csv' and 'predictions' in results:
                import pandas as pd
                # Export predictions as CSV
                pred_data = []
                for key, pred_info in results['predictions'].items():
                    for year, pred in zip(pred_info['years'], pred_info['predictions']):
                        pred_data.append({'Model': key, 'Year': year, 'Prediction': pred})
                
                df = pd.DataFrame(pred_data)
                df.to_csv(filename, index=False)
            
            elif args.export == 'txt':
                with open(filename, 'w') as f:
                    f.write(f"Sea Level Predictor Results\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for key, value in results.items():
                        f.write(f"{key.upper()}:\n")
                        f.write(str(value) + "\n\n")
            
            print(f"Results exported to {filename}")
            
        except Exception as e:
            print(f"Error exporting results: {e}", file=sys.stderr)
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())