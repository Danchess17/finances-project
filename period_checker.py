import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
import pytz

class DataPeriodChecker:
    """Класс для проверки доступных периодов данных по акциям"""
    
    def __init__(self, data_dir="data/periods"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _convert_to_naive(self, dt):
        """Конвертирует datetime в timezone-naive"""
        if hasattr(dt, 'tz') and dt.tz is not None:
            return dt.tz_convert(None)
        return dt
    
    def get_ticker_info(self, ticker):
        """Получить полную информацию о доступных данных для тикера"""
        print(f"\n🔍 Анализ тикера: {ticker}")
        print("=" * 50)
        
        try:
            # Создаем объект тикера
            ticker_obj = yf.Ticker(ticker)
            
            # Получаем информацию о компании
            info = ticker_obj.info
            company_name = info.get('longName', 'N/A')
            exchange = info.get('exchange', 'N/A')
            currency = info.get('currency', 'N/A')
            
            print(f"🏢 Компания: {company_name}")
            print(f"📊 Биржа: {exchange}")
            print(f"💰 Валюта: {currency}")
            
            # Получаем исторические данные за максимальный период
            history = ticker_obj.history(period="max")
            
            if history.empty:
                print("❌ Нет исторических данных для этого тикера")
                return None
            
            # Конвертируем даты в timezone-naive для корректного сравнения
            history.index = history.index.tz_localize(None)
            
            # Анализируем данные
            start_date = self._convert_to_naive(history.index.min())
            end_date = self._convert_to_naive(history.index.max())
            total_days = len(history)
            total_years = (end_date - start_date).days / 365.25
            
            print(f"📅 Период данных: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
            print(f"⏱️  Всего дней: {total_days:,}")
            print(f"📈 Лет данных: {total_years:.1f}")
            
            # Анализ по годам
            yearly_data = self._analyze_yearly_data(history, start_date, end_date)
            
            # Анализ пропусков
            self._analyze_data_gaps(history)
            
            # Сохраняем информацию
            self._save_period_info(ticker, start_date, end_date, total_days, yearly_data)
            
            return {
                'ticker': ticker,
                'company_name': company_name,
                'start_date': start_date,
                'end_date': end_date,
                'total_days': total_days,
                'total_years': total_years,
                'yearly_data': yearly_data,
                'history': history
            }
            
        except Exception as e:
            print(f"❌ Ошибка при анализе {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_yearly_data(self, history, start_date, end_date):
        """Анализ данных по годам"""
        print(f"\n📊 ДАННЫЕ ПО ГОДАМ:")
        print("-" * 30)
        
        yearly_data = {}
        current_year = start_date.year
        end_year = end_date.year
        
        for year in range(current_year, end_year + 1):
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31)
            
            # Используем .loc с timezone-naive датами
            year_mask = (history.index >= year_start) & (history.index <= year_end)
            year_data = history.loc[year_mask]
            trading_days = len(year_data)
            
            if trading_days > 0:
                yearly_data[year] = trading_days
                completeness = (trading_days / 252) * 100  # ~252 торговых дней в году
                print(f"  {year}: {trading_days:3d} дней ({completeness:.1f}%)")
        
        return yearly_data
    
    def _analyze_data_gaps(self, history):
        """Анализ пропусков в данных"""
        print(f"\n🔎 АНАЛИЗ ПРОПУСКОВ:")
        print("-" * 25)
        
        try:
            # Проверяем пропуски между днями
            date_diff = history.index.to_series().diff()
            gaps = date_diff[date_diff > timedelta(days=1)]
            
            if len(gaps) == 0:
                print("  ✅ Пропусков нет - данные непрерывны")
            else:
                print(f"  ⚠️  Найдено {len(gaps)} пропусков:")
                for gap in gaps.head(5):  # Показываем первые 5 пропусков
                    print(f"    - Пропуск: {gap.days} дней")
                
                if len(gaps) > 5:
                    print(f"    ... и еще {len(gaps) - 5} пропусков")
        except Exception as e:
            print(f"  ⚠️  Не удалось проанализировать пропуски: {e}")
    
    def _save_period_info(self, ticker, start_date, end_date, total_days, yearly_data):
        """Сохранить информацию о периодах в CSV"""
        filename = f"{ticker}_period_info.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            # Создаем DataFrame с yearly данными
            data = []
            for year, days in yearly_data.items():
                data.append({
                    'ticker': ticker,
                    'year': year,
                    'trading_days': days,
                    'completeness_percent': (days / 252) * 100
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"💾 Информация сохранена: {filepath}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
    
    def check_multiple_tickers(self, tickers):
        """Проверить несколько тикеров"""
        results = []
        
        for ticker in tickers:
            result = self.get_ticker_info(ticker)
            if result:
                results.append(result)
        
        # Сводная таблица
        if results:
            self._print_summary_table(results)
        
        return results
    
    def _print_summary_table(self, results):
        """Вывести сводную таблицу по всем тикерам"""
        print(f"\n🎯 СВОДНАЯ ТАБЛИЦА")
        print("=" * 80)
        print(f"{'Тикер':<10} {'Компания':<25} {'Начало':<12} {'Конец':<12} {'Лет':<6} {'Дней':<8}")
        print("-" * 80)
        
        for result in results:
            ticker = result['ticker']
            company = result['company_name'][:23] + "..." if len(result['company_name']) > 25 else result['company_name']
            start = result['start_date'].strftime('%Y-%m-%d')
            end = result['end_date'].strftime('%Y-%m-%d')
            years = f"{result['total_years']:.1f}"
            days = f"{result['total_days']:,}"
            
            print(f"{ticker:<10} {company:<25} {start:<12} {end:<12} {years:<6} {days:<8}")
    
    def suggest_analysis_periods(self, ticker, min_years=1, max_years=10):
        """Предложить оптимальные периоды для анализа"""
        print(f"\n💡 ПРЕДЛОЖЕНИЯ ДЛЯ АНАЛИЗА {ticker}:")
        print("-" * 40)
        
        result = self.get_ticker_info(ticker)
        if not result:
            return
        
        end_date = result['end_date']
        total_years = result['total_years']
        
        suggestions = []
        
        # Разные периоды для анализа
        periods = [
            ("Краткосрочный", 0.5, "6 месяцев"),
            ("Среднесрочный", 1, "1 год"),
            ("Долгосрочный", 3, "3 года"),
            ("Полный цикл", 5, "5 лет"),
            ("Исторический", min(10, total_years), f"{min(10, int(total_years))} лет")
        ]
        
        for name, years, description in periods:
            if total_years >= years:
                start_date = end_date - timedelta(days=years*365)
                suggestions.append({
                    'name': name,
                    'period': description,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'years': years
                })
                print(f"  ✅ {name}: {description} ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})")
        
        return suggestions

    def get_available_periods(self, ticker):
        """Простая проверка доступных периодов"""
        try:
            ticker_obj = yf.Ticker(ticker)
            history = ticker_obj.history(period="max")
            
            if history.empty:
                return None
            
            # Конвертируем даты
            history.index = history.index.tz_localize(None)
            start_date = history.index.min()
            end_date = history.index.max()
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': len(history),
                'total_years': (end_date - start_date).days / 365.25
            }
        except Exception as e:
            print(f"Ошибка для {ticker}: {e}")
            return None

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Проверка доступных периодов данных для акций')
    parser.add_argument('tickers', nargs='*', help='Тикеры для проверки (например: AAPL MSFT GOOGL)')
    parser.add_argument('--file', help='Файл со списком тикеров')
    parser.add_argument('--suggest', action='store_true', help='Предложить периоды для анализа')
    parser.add_argument('--quick', action='store_true', help='Быстрая проверка')
    
    args = parser.parse_args()
    
    checker = DataPeriodChecker()
    
    # Получаем список тикеров
    tickers = []
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            print(f"📖 Загружено {len(tickers)} тикеров из файла {args.file}")
        except Exception as e:
            print(f"❌ Ошибка чтения файла: {e}")
            return
    
    if args.tickers:
        tickers.extend(args.tickers)
    
    if not tickers:
        # Если тикеры не указаны, используем демо-список
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'SPY']
        print("🔍 Демо-режим: проверка популярных тикеров")
    
    # Проверяем тикеры
    if args.quick:
        # Быстрая проверка
        print("🚀 БЫСТРАЯ ПРОВЕРКА:")
        for ticker in tickers:
            result = checker.get_available_periods(ticker)
            if result:
                print(f"  {ticker}: {result['start_date'].strftime('%Y-%m-%d')} - {result['end_date'].strftime('%Y-%m-%d')} ({result['total_years']:.1f} лет)")
    elif args.suggest:
        # Только предложения периодов для первого тикера
        if tickers:
            checker.suggest_analysis_periods(tickers[0])
    else:
        # Полная проверка всех тикеров
        checker.check_multiple_tickers(tickers)

# Дополнительные утилиты
def quick_check():
    """Быстрая проверка одного тикера"""
    ticker = input("Введите тикер для проверки: ").strip().upper()
    
    checker = DataPeriodChecker()
    
    print(f"\n🚀 БЫСТРАЯ ПРОВЕРКА {ticker}")
    print("=" * 50)
    
    result = checker.get_ticker_info(ticker)
    
    if result:
        # Предлагаем периоды для анализа
        checker.suggest_analysis_periods(ticker)

def create_periods_csv():
    """Создать CSV файл с периодами для списка тикеров"""
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 
               'SPY', 'QQQ', 'GLD', 'TLT', 'VTI', 'IWM']
    
    checker = DataPeriodChecker()
    
    print("📋 Создание общего файла с периодами...")
    all_data = []
    
    for ticker in tickers:
        print(f"Обработка {ticker}...")
        result = checker.get_available_periods(ticker)
        
        if result:
            all_data.append({
                'ticker': ticker,
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'total_days': result['total_days'],
                'total_years': result['total_years']
            })
    
    # Сохраняем общий файл
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv('data/periods/all_tickers_periods.csv', index=False, encoding='utf-8')
        print(f"💾 Общий файл сохранен: data/periods/all_tickers_periods.csv")
        print(f"📊 Обработано {len(all_data)} тикеров")

if __name__ == "__main__":
    # Если запуск без аргументов - интерактивный режим
    import sys
    if len(sys.argv) == 1:
        print("🎯 ПРОВЕРКА ДОСТУПНЫХ ПЕРИОДОВ ДАННЫХ")
        print("=" * 50)
        print("1 - Быстрая проверка одного тикера")
        print("2 - Создать CSV со всеми периодами")
        print("3 - Демо-режим (популярные тикеры)")
        
        choice = input("\nВыберите действие: ").strip()
        
        if choice == '1':
            quick_check()
        elif choice == '2':
            create_periods_csv()
        elif choice == '3':
            main()
        else:
            print("Запуск в демо-режиме...")
            main()
    else:
        main()