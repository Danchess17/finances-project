import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

class DataLoader:
    """Класс для загрузки и управления финансовыми данными"""
    
    def __init__(self, data_dir="data"):
        self.portfolio_data = None
        self.data_dir = data_dir
        # Создаем папку для данных, если её нет
        os.makedirs(data_dir, exist_ok=True)
    
    def _generate_filename(self, tickers, start_date, end_date):
        """Генерация имени файла в формате <активы_через_>_дата_начала_дата_окончания_yf.csv"""
        tickers_str = "_".join(tickers)
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return f"{tickers_str}_{start_clean}_{end_clean}_yf.csv"
    
    def _save_to_csv(self, portfolio_data, tickers, start_date, end_date):
        """Сохранение данных в CSV файл"""
        filename = self._generate_filename(tickers, start_date, end_date)
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Создаем пустой DataFrame с правильным индексом
            if not portfolio_data:
                return None
                
            # Берем первый тикер для получения индекса (дат)
            first_ticker = list(portfolio_data.keys())[0]
            index_dates = portfolio_data[first_ticker].index
            
            # Создаем DataFrame с правильной структурой
            all_data = pd.DataFrame(index=index_dates)
            
            for ticker, data in portfolio_data.items():
                for column in ['High', 'Low', 'Close', 'Volume']:
                    if column in data.columns:
                        all_data[f"{ticker}_{column}"] = data[column]
            
            # Сохраняем с индексом (датами)
            all_data.to_csv(filepath, encoding='utf-8')
            print(f"✓ Данные сохранены в: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"✗ Ошибка сохранения в CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_from_csv(self, tickers, start_date, end_date):
        """Загрузка данных из CSV файла"""
        filename = self._generate_filename(tickers, start_date, end_date)
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
            
        try:
            print(f"Загрузка данных из кэша: {filepath}")
            combined_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Восстанавливаем структуру portfolio_data
            portfolio_data = {}
            for ticker in tickers:
                ticker_data = {}
                for column in ['High', 'Low', 'Close', 'Volume']:
                    col_name = f"{ticker}_{column}"
                    if col_name in combined_df.columns:
                        ticker_data[column] = combined_df[col_name]
                
                if ticker_data:
                    # Создаем DataFrame с правильным индексом
                    ticker_df = pd.DataFrame(ticker_data, index=combined_df.index)
                    portfolio_data[ticker] = ticker_df
            
            self.portfolio_data = portfolio_data
            return portfolio_data
            
        except Exception as e:
            print(f"✗ Ошибка загрузки из CSV: {e}")
            return None
    
    def fetch_data(self, tickers, start_date, end_date, source='yfinance', use_cache=True):
        """
        Загрузка исторических данных по активам с кэшированием в CSV
        
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
        use_cache : bool
            Использовать кэшированные данные если доступны
        """
        
        # Пытаемся загрузить из кэша
        if use_cache:
            cached_data = self._load_from_csv(tickers, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        print("Загрузка данных из Yahoo Finance...")
        data = {}
        successful_tickers = []
        
        if source == 'yfinance':
            for ticker in tickers:
                try:
                    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not stock.empty:
                        data[ticker] = stock[['High', 'Low', 'Close', 'Volume']]
                        successful_tickers.append(ticker)
                        print(f"✓ {ticker}: {len(stock)} дней данных")
                    else:
                        print(f"✗ {ticker}: нет данных")
                except Exception as e:
                    print(f"✗ {ticker}: {e}")
        
        if data:
            self.portfolio_data = data
            # Сохраняем в CSV
            self._save_to_csv(data, successful_tickers, start_date, end_date)
        
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

    def list_saved_datasets(self):
        """Показать все сохраненные датасеты"""
        if not os.path.exists(self.data_dir):
            print("Папка с данными не существует")
            return []
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        print(f"\nСохраненные датасеты ({len(csv_files)}):")
        for file in sorted(csv_files):
            file_path = os.path.join(self.data_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  📁 {file} ({file_size} bytes)")
        return csv_files

# Пример использования
if __name__ == "__main__":
    loader = DataLoader()
    
    # Загрузка и автоматическое сохранение
    data = loader.fetch_data(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Показать сохраненные файлы
    loader.list_saved_datasets()