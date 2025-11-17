from datetime import datetime, timedelta

class PortfolioManager:
    """Управление портфелем и параметрами анализа"""
    
    def __init__(self):
        self.tickers = []
        self.weights = []
        self.portfolio_value = 0
        self.start_date = None
        self.end_date = None
        self.confidence_level = 0.95
        self.time_horizon = 10
    
    def set_portfolio(self, tickers, weights, portfolio_value):
        """Установка состава портфеля"""
        if len(tickers) != len(weights):
            raise ValueError("Количество тикеров и весов должно совпадать")
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError(f"Сумма весов должна быть 1.0, получено: {sum(weights)}")
        
        self.tickers = tickers
        self.weights = weights
        self.portfolio_value = portfolio_value
    
    def set_period(self, start_date, end_date=None):
        """Установка периода анализа"""
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    
    def set_risk_parameters(self, confidence_level=0.95, time_horizon=10):
        """Установка параметров риска"""
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
    
    def get_portfolio_info(self):
        """Информация о портфеле"""
        return {
            'tickers': self.tickers,
            'weights': self.weights,
            'portfolio_value': self.portfolio_value,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'confidence_level': self.confidence_level,
            'time_horizon': self.time_horizon
        }