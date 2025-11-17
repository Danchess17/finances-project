import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskCalculator:
    """Калькулятор рыночных рисков"""
    
    def __init__(self):
        self.returns = None
        self.cov_matrix = None
        self.var_results = None
        self.lvar_results = None
        
    def calculate_returns(self, portfolio_data):
        """Расчет логарифмических доходностей"""
        print("Расчет доходностей...")
        
        close_prices = pd.DataFrame()
        for ticker, data in portfolio_data.items():
            close_prices[ticker] = data['Close']
        
        # Обработка пропусков
        close_prices = close_prices.ffill().bfill().dropna()
        
        # Расчет логарифмических доходностей
        self.returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        return self.returns
    
    def calculate_var(self, weights, portfolio_value, confidence_level=0.95, time_horizon=10):
        """Расчет Value at Risk"""
        if self.returns is None:
            raise ValueError("Сначала рассчитайте доходности")
        
        print("Расчет VaR...")
        
        weights = np.array(weights)
        
        # Проверка весов
        if abs(np.sum(weights) - 1.0) > 0.01:
            raise ValueError(f"Сумма весов должна быть 1.0, получено: {np.sum(weights)}")
        
        # Расчет ковариационной матрицы
        self.cov_matrix = self.returns.cov()
        
        # Стандартное отклонение портфеля
        portfolio_std = float(np.sqrt(weights.T @ self.cov_matrix @ weights))
        
        # Квантиль нормального распределения
        z_score = norm.ppf(confidence_level)
        
        # VaR расчет
        var_parametric = float(portfolio_value * z_score * portfolio_std * np.sqrt(time_horizon))
        var_percentage = float((var_parametric / portfolio_value) * 100)
        
        self.var_results = {
            'value': var_parametric,
            'percentage': var_percentage,
            'portfolio_std': portfolio_std,
            'z_score': z_score
        }
        
        print(f"VaR ({confidence_level*100}% доверия, {time_horizon} дней): "
              f"{var_parametric:,.2f} ({var_percentage:.2f}%)")
        
        return self.var_results
    
    def calculate_lvar(self, weights, portfolio_value, hl_spreads_avg, 
                      confidence_level=0.95, time_horizon=10):
        """Расчет Liquidity-Adjusted VaR"""
        if self.var_results is None:
            self.calculate_var(weights, portfolio_value, confidence_level, time_horizon)
        
        print("Расчет LVaR...")
        
        weights = np.array(weights)
        tickers = list(hl_spreads_avg.keys())
        
        # Средневзвешенный спред портфеля
        portfolio_spread = float(sum(weights[i] * hl_spreads_avg[ticker] 
                              for i, ticker in enumerate(tickers)))
        
        # Корректировка на ликвидность
        liquidity_adjustment = float(0.5 * portfolio_value * portfolio_spread)
        
        # LVaR расчет
        lvar_value = float(self.var_results['value'] + liquidity_adjustment)
        lvar_percentage = float((lvar_value / portfolio_value) * 100)
        
        self.lvar_results = {
            'value': lvar_value,
            'percentage': lvar_percentage,
            'portfolio_spread': portfolio_spread,
            'liquidity_adjustment': liquidity_adjustment
        }
        
        print(f"LVaR ({confidence_level*100}% доверия, {time_horizon} дней): "
              f"{lvar_value:,.2f} ({lvar_percentage:.2f}%)")
        print(f"Корректировка на ликвидность: {liquidity_adjustment:,.2f}")
        
        return self.lvar_results