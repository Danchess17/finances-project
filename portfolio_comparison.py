import sys
import os
sys.path.append(os.path.dirname(__file__))

from main import LiquidityAdjustedVaRAnalyzer

def compare_portfolios():
    analyzer = LiquidityAdjustedVaRAnalyzer()
    
    portfolios = {
        'Технологический': {
            'tickers': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
            'weights': [0.3, 0.25, 0.2, 0.15, 0.1]
        },
        'Диверсифицированный': {
            'tickers': ['SPY', 'QQQ', 'GLD', 'TLT', 'IWM'],
            'weights': [0.4, 0.2, 0.15, 0.15, 0.1]
        },
        'Дивидендный': {
            'tickers': ['JNJ', 'PG', 'KO', 'PEP', 'XOM'],
            'weights': [0.25, 0.25, 0.2, 0.15, 0.15]
        }
    }
    
    for portfolio_name, portfolio_config in portfolios.items():
        print(f"\n{'='*60}")
        print(f"АНАЛИЗ: {portfolio_name} портфель")
        print(f"{'='*60}")
        
        results = analyzer.run_analysis(
            tickers=portfolio_config['tickers'],
            weights=portfolio_config['weights'],
            portfolio_value=1000000,
            start_date='2023-01-01',
            end_date='2024-01-01',
            confidence_level=0.95,
            time_horizon=10,
            save_report=f'{portfolio_name}_portfolio.png'
        )

if __name__ == "__main__":
    compare_portfolios()