import pandas as pd
import numpy as np
from datetime import datetime

class LiquidityAnalyzer:
    """Анализ ликвидности активов с использованием HL-спредов"""
    
    def __init__(self):
        self.hl_spreads_daily = None
        self.hl_spreads_avg = None
        
    def calculate_hl_spread_daily(self, portfolio_data):
        """
        Расчет ежедневных HL-спредов для каждого актива
        """
        print("Расчет ежедневных HL-спредов...")
        hl_spreads_daily = {}
        hl_spreads_avg = {}
        
        for ticker, data in portfolio_data.items():
            try:
                # Расчет дневного спреда
                daily_spread = 2 * (data['High'] - data['Low']) / (data['High'] + data['Low'])
                hl_spreads_daily[ticker] = daily_spread
                
                # Усреднение за весь период
                avg_spread = float(daily_spread.mean())
                hl_spreads_avg[ticker] = avg_spread
                print(f"  {ticker}: средний спред {avg_spread:.4f}")
                
            except Exception as e:
                print(f"  Ошибка расчета спреда для {ticker}: {e}")
                hl_spreads_avg[ticker] = 0.01
        
        self.hl_spreads_daily = hl_spreads_daily
        self.hl_spreads_avg = hl_spreads_avg
        
        return hl_spreads_daily, hl_spreads_avg
    
    def calculate_portfolio_liquidity(self, weights, portfolio_data):
        """
        Расчет ликвидности портфеля с течением времени
        """
        if self.hl_spreads_daily is None:
            self.calculate_hl_spread_daily(portfolio_data)
        
        # Создаем DataFrame с ежедневными спредами портфеля
        portfolio_spreads = pd.DataFrame()
        
        for i, (ticker, daily_spread) in enumerate(self.hl_spreads_daily.items()):
            portfolio_spreads[ticker] = daily_spread * weights[i]
        
        # Суммируем вклад каждого актива в общий спред портфеля
        portfolio_spreads['portfolio'] = portfolio_spreads.sum(axis=1)
        
        return portfolio_spreads
    
    def get_liquidity_timeseries(self, portfolio_data, weights):
        """
        Возвращает временные ряды ликвидности для анализа
        """
        portfolio_liquidity = self.calculate_portfolio_liquidity(weights, portfolio_data)
        
        return {
            'portfolio': portfolio_liquidity['portfolio'],
            'assets': portfolio_liquidity.drop('portfolio', axis=1),
            'dates': portfolio_liquidity.index
        }