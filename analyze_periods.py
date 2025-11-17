import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import LiquidityAdjustedVaRAnalyzer

def analyze_different_periods():
    analyzer = LiquidityAdjustedVaRAnalyzer()
    
    # Портфель технологических акций
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    tech_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    # Периоды для анализа
    periods = [
        ('2021-01-01', '2024-01-01', '3 года'),
        ('2022-01-01', '2024-01-01', '2 года'), 
        ('2023-01-01', '2024-01-01', '1 год'),
        ('2023-06-01', '2024-01-01', '6 месяцев')
    ]
    
    for start_date, end_date, period_name in periods:
        print(f"\n{'='*60}")
        print(f"АНАЛИЗ ЗА ПЕРИОД: {period_name} ({start_date} - {end_date})")
        print(f"{'='*60}")
        
        results = analyzer.run_analysis(
            tickers=tech_tickers,
            weights=tech_weights,
            portfolio_value=1000000,
            start_date=start_date,
            end_date=end_date,
            confidence_level=0.95,
            time_horizon=10,
            save_report=f'tech_portfolio_{period_name.replace(" ", "_")}.png'
        )

if __name__ == "__main__":
    analyze_different_periods()