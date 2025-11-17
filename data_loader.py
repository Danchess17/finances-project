import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Класс для загрузки и управления финансовыми данными"""
    
    def __init__(self):
        self.portfolio_data = None
        
    def fetch_data(self, tickers, start_date, end_date, source='yfinance'):
        """
        Загрузка исторических данных по активам
        
        Parameters:
        -----------
        tickers : list
            Список тикеров активов
        start_date : str
            Начальная дата в формате 'YYYY-MM-DD'
        end_date : str
            Конечная дата в формате 'YYYY-MM-DD'
        source : str
            Источник данных ('yfinance', 'moex')
        """
        print("Загрузка данных...")
        data = {}
        
        if source == 'yfinance':
            for ticker in tickers:
                try:
                    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not stock.empty:
                        data[ticker] = stock[['High', 'Low', 'Close', 'Volume']]
                        print(f"✓ {ticker}: {len(stock)} дней данных")
                    else:
                        print(f"✗ {ticker}: нет данных")
                except Exception as e:
                    print(f"✗ {ticker}: {e}")
        
        self.portfolio_data = data
        return data
    
    def get_common_period(self):
        """Возвращает общий период данных для всех активов"""
        if not self.portfolio_data:
            return None
            
        common_start = None
        common_end = None
        
        for ticker, data in self.portfolio_data.items():
            if data.empty:
                continue
                
            start = data.index.min()
            end = data.index.max()
            
            if common_start is None or start > common_start:
                common_start = start
            if common_end is None or end < common_end:
                common_end = end
        
        return common_start, common_end