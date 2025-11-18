# run_moex_loader.py
from moex_data_loader import MoexDataLoader
from datetime import datetime, timedelta
import argparse

def quick_load_portfolio(symbols, years=3, portfolio_name=None):
    """
    Быстрая загрузка портфеля за последние N лет
    
    Parameters:
        symbols (list): Список тикеров
        years (int): Количество лет исторических данных
        portfolio_name (str): Название портфеля
    """
    loader = MoexDataLoader()
    
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    print(f"🚀 Загрузка портфеля за последние {years} лет...")
    print(f"📅 Период: {start_date} - {end_date}")
    print(f"📊 Акции: {', '.join(symbols)}")
    
    portfolio_df, successful_symbols = loader.load_portfolio_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        portfolio_name=portfolio_name
    )
    
    return portfolio_df, successful_symbols

def quick_load_single(symbol, years=3):
    """Быстрая загрузка одной акции"""
    loader = MoexDataLoader()
    
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    print(f"🚀 Загрузка {symbol} за последние {years} лет...")
    
    df = loader.load_single_stock(symbol, start_date, end_date)
    return df

# Примеры использования
if __name__ == "__main__":
    # Пример 1: Быстрая загрузка портфеля
    print("=== ПРИМЕР 1: ПОРТФЕЛЬ ГОЛУБЫХ ФИШЕК ===")
    blue_chips = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'NVTK']
    portfolio_df, symbols = quick_load_portfolio(blue_chips, years=2, portfolio_name="BLUE_CHIPS")
    
    print("\n=== ПРИМЕР 2: ПОРТФЕЛЬ ТЕХНОЛОГИЙ ===")
    tech_stocks = ['YNDX', 'OZON', 'TCSG']
    portfolio_df, symbols = quick_load_portfolio(tech_stocks, years=1, portfolio_name="TECH_STOCKS")
    
    print("\n=== ПРИМЕР 3: ОТДЕЛЬНАЯ АКЦИЯ ===")
    df = quick_load_single('SBERP', years=1)
    
    # Показываем список файлов
    loader = MoexDataLoader()
    loader.list_saved_data()