import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import LiquidityAdjustedVaRAnalyzer

def example_custom_portfolio():
    """Пример создания собственного портфеля"""
    
    analyzer = LiquidityAdjustedVaRAnalyzer()
    
    # Ваш собственный портфель
    my_tickers = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC']
    my_weights = [0.35, 0.25, 0.15, 0.15, 0.1]
    my_portfolio_value = 750000  # 750 тыс долларов
    
    results = analyzer.run_analysis(
        tickers=my_tickers,
        weights=my_weights,
        portfolio_value=my_portfolio_value,
        start_date='2023-06-01',  # Полгода данных
        confidence_level=0.95,
        time_horizon=10,
        save_report='my_portfolio_analysis.png'
    )
    
    return results

if __name__ == "__main__":
    example_custom_portfolio()