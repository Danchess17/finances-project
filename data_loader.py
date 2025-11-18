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
        # Создаем папки для данных, если их нет
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "portfolio"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "individual"), exist_ok=True)
    
    def _generate_portfolio_filename(self, tickers, start_date, end_date):
        """Генерация имени файла для портфеля"""
        tickers_str = "_".join(tickers)
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return f"portfolio/{tickers_str}_{start_clean}_{end_clean}_yf.csv"
    
    def _generate_individual_filename(self, ticker, start_date, end_date):
        """Генерация имени файла для отдельной акции"""
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        return f"individual/{ticker}_{start_clean}_{end_clean}_yf.csv"
    
    def _save_portfolio_to_csv(self, portfolio_data, tickers, start_date, end_date):
        """Сохранение портфельных данных в CSV"""
        filename = self._generate_portfolio_filename(tickers, start_date, end_date)
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if not portfolio_data:
                return None
                
            # Создаем DataFrame с правильным индексом
            first_ticker = list(portfolio_data.keys())[0]
            index_dates = portfolio_data[first_ticker].index
            all_data = pd.DataFrame(index=index_dates)
            
            for ticker, data in portfolio_data.items():
                for column in ['High', 'Low', 'Close', 'Volume']:
                    if column in data.columns:
                        all_data[f"{ticker}_{column}"] = data[column]
            
            all_data.to_csv(filepath, encoding='utf-8')
            print(f"✓ Портфельные данные сохранены в: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"✗ Ошибка сохранения портфеля в CSV: {e}")
            return None
    
    def _save_individual_to_csv(self, portfolio_data, start_date, end_date):
        """Сохранение данных по каждой акции отдельно"""
        saved_files = []
        
        for ticker, data in portfolio_data.items():
            try:
                filename = self._generate_individual_filename(ticker, start_date, end_date)
                filepath = os.path.join(self.data_dir, filename)
                
                # Сохраняем данные одной акции
                data.to_csv(filepath, encoding='utf-8')
                saved_files.append(filepath)
                print(f"✓ Данные {ticker} сохранены в: {filepath}")
                
            except Exception as e:
                print(f"✗ Ошибка сохранения {ticker} в CSV: {e}")
        
        return saved_files
    
    def _load_portfolio_from_csv(self, tickers, start_date, end_date):
        """Загрузка портфельных данных из CSV"""
        filename = self._generate_portfolio_filename(tickers, start_date, end_date)
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return None
            
        try:
            print(f"Загрузка портфельных данных из кэша: {filepath}")
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
                    ticker_df = pd.DataFrame(ticker_data, index=combined_df.index)
                    portfolio_data[ticker] = ticker_df
            
            self.portfolio_data = portfolio_data
            return portfolio_data
            
        except Exception as e:
            print(f"✗ Ошибка загрузки портфеля из CSV: {e}")
            return None
    
    def _load_individual_from_csv(self, tickers, start_date, end_date):
        """Загрузка данных по каждой акции отдельно"""
        portfolio_data = {}
        loaded_tickers = []
        
        for ticker in tickers:
            filename = self._generate_individual_filename(ticker, start_date, end_date)
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                continue
                
            try:
                print(f"Загрузка данных {ticker} из кэша: {filepath}")
                individual_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                portfolio_data[ticker] = individual_data
                loaded_tickers.append(ticker)
                
            except Exception as e:
                print(f"✗ Ошибка загрузки {ticker} из CSV: {e}")
        
        if portfolio_data:
            self.portfolio_data = portfolio_data
            return portfolio_data
        else:
            return None
    
    def fetch_data(self, tickers, start_date, end_date, source='yfinance', use_cache=True, save_individual=True):
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
        save_individual : bool
            Сохранять данные по каждой акции отдельно
        """
        
        # Пытаемся загрузить из кэша (сначала портфель, потом индивидуальные)
        if use_cache:
            # Сначала пробуем загрузить весь портфель
            cached_data = self._load_portfolio_from_csv(tickers, start_date, end_date)
            if cached_data is not None:
                return cached_data
            
            # Если портфеля нет, пробуем загрузить индивидуальные файлы
            cached_data = self._load_individual_from_csv(tickers, start_date, end_date)
            if cached_data is not None:
                print("✓ Данные загружены из индивидуальных файлов")
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
            
            # Сохраняем портфельные данные
            self._save_portfolio_to_csv(data, successful_tickers, start_date, end_date)
            
            # Сохраняем индивидуальные данные (если включено)
            if save_individual:
                self._save_individual_to_csv(data, start_date, end_date)
        
        return data
    
    def load_individual_ticker(self, ticker, start_date, end_date, use_cache=True):
        """
        Загрузка данных по одному тикеру
        """
        filename = self._generate_individual_filename(ticker, start_date, end_date)
        filepath = os.path.join(self.data_dir, filename)
        
        if use_cache and os.path.exists(filepath):
            try:
                print(f"Загрузка данных {ticker} из кэша: {filepath}")
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                return data
            except Exception as e:
                print(f"✗ Ошибка загрузки {ticker} из CSV: {e}")
        
        # Если кэша нет или ошибка, загружаем из Yahoo Finance
        print(f"Загрузка данных {ticker} из Yahoo Finance...")
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not stock.empty:
                data = stock[['High', 'Low', 'Close', 'Volume']]
                
                # Сохраняем в кэш
                try:
                    data.to_csv(filepath, encoding='utf-8')
                    print(f"✓ Данные {ticker} сохранены в: {filepath}")
                except Exception as e:
                    print(f"✗ Ошибка сохранения {ticker} в CSV: {e}")
                
                return data
            else:
                print(f"✗ {ticker}: нет данных")
                return None
                
        except Exception as e:
            print(f"✗ {ticker}: {e}")
            return None
    
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
            print("Папка с данными не существует!")
            return []
        
        print("\n📁 СОХРАНЕННЫЕ ДАННЫЕ:")
        
        # Портфельные данные
        portfolio_dir = os.path.join(self.data_dir, "portfolio")
        if os.path.exists(portfolio_dir):
            portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('.csv')]
            print(f"\n🎯 ПОРТФЕЛИ ({len(portfolio_files)}):")
            for file in sorted(portfolio_files):
                file_path = os.path.join(portfolio_dir, file)
                file_size = os.path.getsize(file_path) / 1024
                print(f"  📊 {file} ({file_size:.1f} KB)")
        
        # Индивидуальные данные
        individual_dir = os.path.join(self.data_dir, "individual")
        if os.path.exists(individual_dir):
            individual_files = [f for f in os.listdir(individual_dir) if f.endswith('.csv')]
            print(f"\n📈 ИНДИВИДУАЛЬНЫЕ АКЦИИ ({len(individual_files)}):")
            
            # Группируем по тикерам
            ticker_files = {}
            for file in individual_files:
                ticker = file.split('_')[0]
                if ticker not in ticker_files:
                    ticker_files[ticker] = []
                ticker_files[ticker].append(file)
            
            for ticker, files in sorted(ticker_files.items()):
                print(f"  {ticker}: {len(files)} файлов")
                for file in sorted(files):
                    file_path = os.path.join(individual_dir, file)
                    file_size = os.path.getsize(file_path) / 1024
                    print(f"    📄 {file} ({file_size:.1f} KB)")

# Пример использования
if __name__ == "__main__":
    loader = DataLoader()
    
    # Загрузка портфеля с сохранением индивидуальных файлов
    print("=== ЗАГРУЗКА ПОРТФЕЛЯ ===")
    portfolio_data = loader.fetch_data(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        save_individual=True  # Сохранять индивидуальные файлы
    )
    
    # Загрузка отдельной акции
    print("\n=== ЗАГРУЗКА ОТДЕЛЬНОЙ АКЦИИ ===")
    tsla_data = loader.load_individual_ticker(
        ticker='TSLA',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Показать сохраненные файлы
    print("\n=== СПИСОК ФАЙЛОВ ===")
    loader.list_saved_datasets()