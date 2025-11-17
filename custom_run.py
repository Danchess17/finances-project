import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(__file__))
from main import LiquidityAdjustedVaRAnalyzer

def main():
    if len(sys.argv) > 1:
        # Использование аргументов командной строки
        start_date = sys.argv[1]
        end_date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
    else:
        # По умолчанию - последний год
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    analyzer = LiquidityAdjustedVaRAnalyzer()
    
    results = analyzer.run_analysis(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        weights=[0.3, 0.25, 0.2, 0.15, 0.1],
        portfolio_value=1000000,
        start_date=start_date,
        end_date=end_date,
        confidence_level=0.95,
        time_horizon=10,
        save_report=f'analysis_{start_date}_to_{end_date}.png'
    )

if __name__ == "__main__":
    main()