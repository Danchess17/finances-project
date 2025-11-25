# moex_data_loader.py
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from moex_parser import MoexParser
from data_manager import DataManager

class MoexDataLoader:
    def __init__(self, base_dir="data", debug=False):
        self.debug = debug
        self.parser = MoexParser(debug=debug)
        self.data_manager = DataManager(output_dir=base_dir)
        self.base_dir = base_dir
        self._create_directories()
    
    def _print(self, *args, **kwargs):
        """Выводит сообщение только если включен debug режим"""
        if self.debug:
            print(*args, **kwargs)
    
    def _create_directories(self):
        """Создаем необходимые директории"""
        directories = [
            self.base_dir,
            os.path.join(self.base_dir, "portfolio"),
            os.path.join(self.base_dir, "individual"),
            os.path.join(self.base_dir, "russian_portfolio")
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self._print(f"Создана директория: {directory}")
    
    def load_portfolio_data(self, symbols, start_date, end_date, portfolio_name=None):
        """
        Загружает данные портфеля российских акций
        
        Parameters:
            symbols (list): Список тикеров (например, ['SBER', 'GAZP', 'LKOH'])
            start_date (str): Дата начала в формате 'YYYY-MM-DD'
            end_date (str): Дата окончания в формате 'YYYY-MM-DD'
            portfolio_name (str): Название портфеля (если None - генерируется автоматически)
        """
        self._print("=== ЗАГРУЗКА РОССИЙСКОГО ПОРТФЕЛЯ ===")
        self._print(f"Загрузка данных с MOEX...")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if portfolio_name is None:
            portfolio_name = "_".join(symbols)
        
        all_data = []
        successful_symbols = []
        
        for symbol in symbols:
            self._print(f"Загружаем {symbol}...", end=" ")
            
            df = self.parser.parse_stock_data(symbol, start_dt, end_dt)
            
            if df is not None and not df.empty:
                # Добавляем дополнительные колонки для совместимости
                df['Adj Close'] = df['Close']  # Для совместимости с Yahoo Finance
                df['Dividends'] = 0  # MOEX не предоставляет дивиденды в этом API
                df['Stock Splits'] = 0
                
                all_data.append(df)
                successful_symbols.append(symbol)
                self._print(f"{len(df)} дней данных")
            else:
                self._print(f"✗ не удалось загрузить")
            
            time.sleep(1)  # Пауза между запросами
        
        if not all_data:
            self._print("Не удалось загрузить данные ни по одной акции")
            return None, []
        
        # Объединяем все данные
        portfolio_df = pd.concat(all_data, ignore_index=True)
        
        # Сохраняем объединенный портфель
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{portfolio_name}_{start_date}_{end_date}_moex.csv"
        filepath = os.path.join(self.base_dir, "russian_portfolio", filename)
        
        portfolio_df.to_csv(filepath, index=False)
        
        self._print(f"Портфельные данные сохранены в: {filepath}")
        self._print(f"Запрошенный период: {start_date} - {end_date}")
        
        # Проверяем реальный период данных
        actual_start = portfolio_df['Date'].min()
        actual_end = portfolio_df['Date'].max()
        self._print(f"Реальный период: {actual_start} - {actual_end}")
        
        if actual_start != start_date or actual_end != end_date:
            self._print(f"ВНИМАНИЕ: Реальный период данных не совпадает с запрошенным!")
        
        # Сохраняем индивидуальные акции
        self._print("\n Сохранение индивидуальных файлов:")
        for i, df in enumerate(all_data):
            symbol = successful_symbols[i]
            self._save_individual_stock(df, symbol, start_date, end_date)
        
        return portfolio_df, successful_symbols
    
    def _save_individual_stock(self, df, symbol, start_date, end_date):
        """Сохраняет данные по отдельной акции"""
        filename = f"{symbol}_{start_date}_{end_date}_moex.csv"
        filepath = os.path.join(self.base_dir, "individual", filename)
        
        df.to_csv(filepath, index=False)
        
        actual_start = df['Date'].min()
        actual_end = df['Date'].max()
        
        self._print(f"Данные {symbol} сохранены в: {filepath}")
        self._print(f"Запрошенный период: {start_date} - {end_date}")
        self._print(f"Реальный период: {actual_start} - {actual_end}")
        
        if actual_start != start_date or actual_end != end_date:
            self._print(f"ВНИМАНИЕ: Реальный период данных не совпадает с запрошенным!")
    
    def load_single_stock(self, symbol, start_date, end_date):
        """Загружает данные по одной российской акции"""
        self._print("\n=== ЗАГРУЗКА ОТДЕЛЬНОЙ АКЦИИ ===")
        self._print(f"Загрузка данных {symbol} с MOEX...")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        df = self.parser.parse_stock_data(symbol, start_dt, end_dt)
        
        if df is not None:
            # Добавляем дополнительные колонки
            df['Adj Close'] = df['Close']
            df['Dividends'] = 0
            df['Stock Splits'] = 0
            
            self._save_individual_stock(df, symbol, start_date, end_date)
            return df
        else:
            self._print(f"Не удалось загрузить данные для {symbol}")
            return None
    
    def list_saved_data(self):
        """Показывает список сохраненных данных"""
        print("\n=== СПИСОК ФАЙЛОВ ===")
        print("\n СОХРАНЕННЫЕ ДАННЫЕ MOEX:")
        
        # Российские портфели
        portfolio_dir = os.path.join(self.base_dir, "russian_portfolio")
        if os.path.exists(portfolio_dir):
            portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('_moex.csv')]
            if portfolio_files:
                print(f"\n РОССИЙСКИЕ ПОРТФЕЛИ ({len(portfolio_files)}):")
                for file in portfolio_files:
                    filepath = os.path.join(portfolio_dir, file)
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"{file} ({size_kb:.1f} KB)")
        
        # Индивидуальные акции
        individual_dir = os.path.join(self.base_dir, "individual")
        if os.path.exists(individual_dir):
            individual_files = [f for f in os.listdir(individual_dir) if f.endswith('_moex.csv')]
            if individual_files:
                # Группируем по тикерам
                symbols = {}
                for file in individual_files:
                    symbol = file.split('_')[0]
                    if symbol not in symbols:
                        symbols[symbol] = []
                    symbols[symbol].append(file)
                
                print(f"\n ИНДИВИДУАЛЬНЫЕ АКЦИИ ({len(individual_files)}):")
                for symbol, files in symbols.items():
                    print(f"{symbol}: {len(files)} файлов")
                    for file in files:
                        filepath = os.path.join(individual_dir, file)
                        size_kb = os.path.getsize(filepath) / 1024
                        print(f"{file} ({size_kb:.1f} KB)")
    
    def get_available_periods(self, symbol):
        """Показывает доступные периоды данных для акции"""
        print(f"\n ДОСТУПНЫЕ ПЕРИОДЫ ДЛЯ {symbol}:")
        
        # Тестируем разные периоды
        test_periods = [
            ("2020-01-01", "2020-12-31", "2020 год"),
            ("2021-01-01", "2021-12-31", "2021 год"),
            ("2022-01-01", "2022-12-31", "2022 год"),
            ("2023-01-01", "2023-12-31", "2023 год"),
            ("2024-01-01", "2024-12-31", "2024 год"),
        ]
        
        for start, end, description in test_periods:
            start_dt = datetime.strptime(start, '%Y-%m-%d')
            end_dt = datetime.strptime(end, '%Y-%m-%d')
            
            df = self.parser.parse_stock_data(symbol, start_dt, end_dt)
            if df is not None and not df.empty:
                actual_start = df['Date'].min()
                actual_end = df['Date'].max()
                print(f"{description}: {actual_start} - {actual_end} ({len(df)} дней)")
            else:
                print(f"{description}: нет данных")
            
            time.sleep(0.5)

def main():
    """Основная функция для демонстрации"""
    loader = MoexDataLoader()
    
    # Пример 1: Загрузка портфеля российских акций
    print("ЗАГРУЗКА ДЕМО-ПОРТФЕЛЯ РОССИЙСКИХ АКЦИЙ")
    
    # Популярные российские акции
    russian_portfolio = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'YNDX']
    
    # Используем реальные даты (например, последние 3 года)
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    portfolio_df, successful_symbols = loader.load_portfolio_data(
        symbols=russian_portfolio,
        start_date=start_date,
        end_date=end_date,
        portfolio_name="RUSSIAN_BLUECHIPS"
    )
    
    # Пример 2: Загрузка отдельной акции
    print("\n" + "="*50)
    loader.load_single_stock('VTBR', '2023-01-01', '2023-12-31')
    
    # Показываем список файлов
    print("\n" + "="*50)
    loader.list_saved_data()
    
    # Показываем доступные периоды для SBER
    print("\n" + "="*50)
    loader.get_available_periods('SBER')

if __name__ == "__main__":
    main()