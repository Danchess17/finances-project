# run_moex_loader.py
from moex_data_loader import MoexDataLoader
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os


def print_debug_dates(portfolio_df, symbols, start_date, end_date):
    """
    Выводит подробную информацию о реальных датах загрузки для каждой акции
    
    Parameters:
        portfolio_df (pd.DataFrame): Данные портфеля
        symbols (list): Список успешно загруженных символов
        start_date (str): Запрошенная дата начала
        end_date (str): Запрошенная дата окончания
    """
    if portfolio_df is None or portfolio_df.empty:
        print("❌ Нет данных для отладки")
        return
    
    print("\n" + "=" * 60)
    print("🐛 DEBUG: ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О РЕАЛЬНЫХ ДАТАХ")
    print("=" * 60)
    print(f"📅 Запрошенный период: {start_date} - {end_date}")
    print()
    
    # Убеждаемся, что Date в формате datetime
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    for symbol in symbols:
        # Фильтруем данные для текущей акции
        stock_data = portfolio_df[portfolio_df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        if stock_data.empty:
            print(f"❌ {symbol}: Нет данных")
            continue
        
        # Получаем реальные даты
        actual_first_date = stock_data['Date'].min()
        actual_last_date = stock_data['Date'].max()
        num_days = len(stock_data)
        
        # Сравниваем с запрошенными датами
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        
        print(f"📊 {symbol}:")
        print(f"   Запрошено:     {start_date} - {end_date}")
        print(f"   Реальный период:")
        print(f"   • Первая дата:  {actual_first_date.strftime('%Y-%m-%d')} ", end="")
        
        if actual_first_date > requested_start:
            print(f"⚠️ (позже запрошенной на {(actual_first_date - requested_start).days} дн.)")
        elif actual_first_date < requested_start:
            print(f"✅ (раньше запрошенной на {(requested_start - actual_first_date).days} дн.)")
        else:
            print("✅ (совпадает)")
        
        print(f"   • Последняя дата: {actual_last_date.strftime('%Y-%m-%d')} ", end="")
        
        if actual_last_date < requested_end:
            print(f"⚠️ (раньше запрошенной на {(requested_end - actual_last_date).days} дн.)")
        elif actual_last_date > requested_end:
            print(f"✅ (позже запрошенной на {(actual_last_date - requested_end).days} дн.)")
        else:
            print("✅ (совпадает)")
        
        print(f"   • Всего дней данных: {num_days}")
        print()
    
    print("=" * 60)


def plot_high_low_prices(portfolio_df, symbols, start_date, end_date, output_path=None, debug=False):
    """
    Рисует график High и Low для каждой акции из портфеля с течением времени
    
    Parameters:
        portfolio_df (pd.DataFrame): Данные портфеля
        symbols (list): Список успешно загруженных символов
        start_date (str): Дата начала
        end_date (str): Дата окончания
        output_path (str): Путь для сохранения графика
    """
    if portfolio_df is None or portfolio_df.empty:
        if debug:
            print("❌ Нет данных для построения графика")
        return None
    
    if not symbols or len(symbols) == 0:
        if debug:
            print("❌ Нет успешно загруженных символов для построения графика")
        return None
    
    # Создаем график
    n_symbols = len(symbols)
    fig, axes = plt.subplots(n_symbols, 1, figsize=(16, 5 * n_symbols))
    
    if n_symbols == 1:
        axes = [axes]
    
    # Убеждаемся, что Date в формате datetime
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    # Рисуем для каждой акции
    for i, symbol in enumerate(symbols):
        ax = axes[i]
        
        # Фильтруем данные для текущей акции
        stock_data = portfolio_df[portfolio_df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        if stock_data.empty:
            ax.text(0.5, 0.5, f'Нет данных для {symbol}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{symbol} - нет данных')
            continue
        
        # Рисуем High и Low
        dates = stock_data['Date']
        ax.plot(dates, stock_data['High'], label='High', color='green', linewidth=1.5, alpha=0.7)
        ax.plot(dates, stock_data['Low'], label='Low', color='red', linewidth=1.5, alpha=0.7)
        
        # Заливаем область между High и Low
        ax.fill_between(dates, stock_data['Low'], stock_data['High'], 
                       alpha=0.2, color='gray', label='Диапазон')
        
        ax.set_title(f'{symbol} - High и Low цены', fontweight='bold', fontsize=12)
        ax.set_xlabel('Дата')
        ax.set_ylabel('Цена (руб.)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Форматируем даты на оси X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Сохраняем график
    if output_path is None:
        output_path = f"high_low_chart_{start_date}_to_{end_date}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if debug:
        print(f"\n📊 График сохранен в: {output_path}")
    
    # Показываем график (опционально, можно закомментировать для серверных запусков)
    # plt.show()
    plt.close()
    
    return output_path


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


def main_cli():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(
        description='Загрузка данных с MOEX и создание CSV/графиков',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Загрузка портфеля с графиком (автоматически: последние 3 года до вчера)
  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --plot
  
  # С подробной информацией о реальных датах (debug режим)
  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --plot --debug
  
  # Загрузка с указанием периода
  python run_moex_data_loader.py --start 2023-01-01 --end 2024-01-01 --portfolio SBER GAZP LKOH --plot
  
  # Загрузка до определенной даты (start будет автоматически 3 года назад)
  python run_moex_data_loader.py --end 2024-01-01 --portfolio SBER GAZP
  
  # Загрузка с указанием имени портфеля
  python run_moex_data_loader.py --portfolio SBER GAZP --portfolio-name MY_PORTFOLIO --plot
        """
    )
    
    # Параметры дат (опциональные, с значениями по умолчанию)
    parser.add_argument('--start', '--start-date', 
                       dest='start_date',
                       type=str,
                       default=None,
                       help='Дата начала в формате YYYY-MM-DD (если не указано, будет использовано последние 3 года)')
    
    parser.add_argument('--end', '--end-date',
                       dest='end_date',
                       type=str,
                       default=None,
                       help='Дата окончания в формате YYYY-MM-DD (если не указано, будет использован вчерашний день)')
    
    parser.add_argument('--portfolio', '--symbols',
                       dest='symbols',
                       nargs='+',
                       required=True,
                       help='Список тикеров (например: SBER GAZP LKOH)')
    
    # Опциональные параметры
    parser.add_argument('--portfolio-name',
                       dest='portfolio_name',
                       type=str,
                       default=None,
                       help='Название портфеля (если не указано, будет сгенерировано автоматически)')
    
    parser.add_argument('--plot', '--plot-graph',
                       dest='plot',
                       action='store_true',
                       help='Создать график High/Low для каждой акции')
    
    parser.add_argument('--plot-output',
                       dest='plot_output',
                       type=str,
                       default=None,
                       help='Путь для сохранения графика (по умолчанию: high_low_chart_START_to_END.png)')
    
    parser.add_argument('--csv-output',
                       dest='csv_output',
                       type=str,
                       default=None,
                       help='Путь для сохранения CSV (по умолчанию: сохраняется в data/russian_portfolio/)')
    
    parser.add_argument('--debug',
                       dest='debug',
                       action='store_true',
                       help='Показать подробную информацию о реальных датах загрузки для каждой акции')
    
    args = parser.parse_args()
    
    # Устанавливаем значения по умолчанию для дат
    # Если end не указан, используем вчерашний день
    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Если start не указан, используем дату за последние 3 года от end_date
    if args.start_date is None:
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=3*365)
        args.start_date = start_dt.strftime('%Y-%m-%d')
    
    # Валидация дат
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            parser.error("Дата начала должна быть раньше даты окончания")
    except ValueError as e:
        parser.error(f"Неверный формат даты. Используйте YYYY-MM-DD. Ошибка: {e}")
    
    # Загружаем данные
    loader = MoexDataLoader(debug=args.debug)
    
    if args.debug:
        print("=" * 60)
        print("🚀 ЗАГРУЗКА ДАННЫХ С MOEX")
        print("=" * 60)
        print(f"📅 Период: {args.start_date} - {args.end_date}")
        print(f"📊 Портфель: {', '.join(args.symbols)}")
        if args.portfolio_name:
            print(f"📝 Название портфеля: {args.portfolio_name}")
        print("=" * 60)
    
    portfolio_df, successful_symbols = loader.load_portfolio_data(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio_name=args.portfolio_name
    )
    
    if portfolio_df is None or portfolio_df.empty:
        if not args.debug:
            print("❌ Не удалось загрузить данные.")
        return
    
    # Определяем путь к сохраненному CSV файлу (файл уже сохранен в load_portfolio_data)
    if args.csv_output:
        # Если указан отдельный путь, сохраняем туда
        portfolio_df.to_csv(args.csv_output, index=False)
        csv_path = args.csv_output
    else:
        # Используем путь, который был использован в load_portfolio_data
        portfolio_name = args.portfolio_name if args.portfolio_name else "_".join(args.symbols)
        csv_path = os.path.join("data", "russian_portfolio", 
                               f"{portfolio_name}_{args.start_date}_{args.end_date}_moex.csv")
    
    # Показываем debug информацию, если запрошено
    if args.debug:
        print_debug_dates(
            portfolio_df=portfolio_df,
            symbols=successful_symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"\n💾 CSV сохранен в: {csv_path}")
    
    # В обычном режиме выводим только путь к CSV (без лишнего текста)
    if not args.debug:
        print(csv_path)
    
    # Строим график, если нужно
    if args.plot:
        if args.debug:
            print("\n" + "=" * 60)
            print("📊 СОЗДАНИЕ ГРАФИКА")
            print("=" * 60)
        
        plot_output_path = plot_high_low_prices(
            portfolio_df=portfolio_df,
            symbols=successful_symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.plot_output,
            debug=args.debug
        )
        
        if not args.debug and plot_output_path:
            print(plot_output_path)
    
    if args.debug:
        print("\n" + "=" * 60)
        print("✅ ЗАВЕРШЕНО")
        print("=" * 60)


# Примеры использования (для обратной совместимости)
if __name__ == "__main__":
    import sys
    
    # Если запущено без аргументов - показываем примеры старого использования
    if len(sys.argv) == 1:
        print("Использование CLI:")
        print("  # Самый простой вариант (последние 3 года до вчера):")
        print("  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --plot")
        print("\n  # С указанием дат:")
        print("  python run_moex_data_loader.py --start 2023-01-01 --end 2024-01-01 --portfolio SBER GAZP LKOH --plot")
        print("\n" + "=" * 60)
        print("Запуск примеров старого формата:")
        print("=" * 60)
        
        # Пример 1: Быстрая загрузка портфеля
        print("\n=== ПРИМЕР 1: ПОРТФЕЛЬ ГОЛУБЫХ ФИШЕК ===")
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
    else:
        # Запускаем CLI
        main_cli()