import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from liquidity import LiquidityAnalyzer
from risk_calculator import RiskCalculator
from visualizer import RiskVisualizer
from portfolio import PortfolioManager

class LiquidityAdjustedVaRAnalyzer:
    """
    Главный класс для комплексного анализа риска портфеля
    с поправкой на ликвидность
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.risk_calculator = RiskCalculator()
        self.visualizer = RiskVisualizer()
        self.portfolio_manager = PortfolioManager()
        
    def run_analysis(self, tickers, weights, portfolio_value, 
                    start_date, end_date=None, confidence_level=0.95,
                    time_horizon=10, save_report=None):
        """
        Полный запуск анализа
        
        Parameters:
        -----------
        tickers : list
            Список тикеров (например, ['AAPL', 'MSFT', 'GOOGL'])
        weights : list
            Веса активов (сумма должна быть 1.0)
        portfolio_value : float
            Стоимость портфеля
        start_date : str
            Начальная дата анализа 'YYYY-MM-DD'
        end_date : str
            Конечная дата анализа 'YYYY-MM-DD' (по умолчанию сегодня)
        confidence_level : float
            Уровень доверия для VaR (по умолчанию 0.95)
        time_horizon : int
            Временной горизонт в днях (по умолчанию 10)
        save_report : str
            Путь для сохранения отчета (опционально)
        """
        
        # Настройка портфеля
        self.portfolio_manager.set_portfolio(tickers, weights, portfolio_value)
        self.portfolio_manager.set_period(start_date, end_date)
        self.portfolio_manager.set_risk_parameters(confidence_level, time_horizon)
        
        print("=" * 60)
        print("КОМПЛЕКСНЫЙ АНАЛИЗ РИСКА ПОРТФЕЛЯ")
        print("=" * 60)
        
        portfolio_info = self.portfolio_manager.get_portfolio_info()
        print(f"Портфель: {', '.join(tickers)}")
        print(f"Период: {start_date} - {end_date}")
        print(f"Стоимость: {portfolio_value:,.2f}")
        print(f"Доверительный уровень: {confidence_level*100}%")
        print(f"Горизонт: {time_horizon} дней")
        print()
        
        try:
            # 1. Загрузка данных
            portfolio_data = self.data_loader.fetch_data(
                tickers, start_date, end_date
            )
            
            # 2. Анализ ликвидности
            hl_spreads_daily, hl_spreads_avg = self.liquidity_analyzer.calculate_hl_spread_daily(
                portfolio_data
            )
            
            # 3. Временные ряды ликвидности
            liquidity_timeseries = self.liquidity_analyzer.get_liquidity_timeseries(
                portfolio_data, weights
            )
            
            # 4. Расчет рисков
            self.risk_calculator.calculate_returns(portfolio_data)
            var_results = self.risk_calculator.calculate_var(
                weights, portfolio_value, confidence_level, time_horizon
            )
            lvar_results = self.risk_calculator.calculate_lvar(
                weights, portfolio_value, hl_spreads_avg, confidence_level, time_horizon
            )
            
            # 5. Визуализация
            self.visualizer.create_comprehensive_report(
                portfolio_data, weights, portfolio_value,
                var_results, lvar_results, hl_spreads_avg,
                liquidity_timeseries, save_report
            )
            
            # 6. Дополнительные графики ликвидности
            self.visualizer.plot_individual_asset_liquidity(liquidity_timeseries)
            
            # 7. Вывод результатов
            self._print_summary_results(weights, portfolio_value, hl_spreads_avg,
                                      var_results, lvar_results)
            
            return {
                'portfolio_data': portfolio_data,
                'hl_spreads_avg': hl_spreads_avg,
                'hl_spreads_daily': hl_spreads_daily,
                'var_results': var_results,
                'lvar_results': lvar_results,
                'liquidity_timeseries': liquidity_timeseries
            }
            
        except Exception as e:
            print(f"Ошибка при выполнении анализа: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_summary_results(self, weights, portfolio_value, hl_spreads_avg,
                             var_results, lvar_results):
        """Вывод сводных результатов"""
        print("\n" + "=" * 80)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        # Таблица по активам
        print("\nАНАЛИЗ АКТИВОВ:")
        tickers = list(hl_spreads_avg.keys())
        for i, ticker in enumerate(tickers):
            print(f"  {ticker}: вес {weights[i]*100:5.1f}%, "
                  f"HL-спред {hl_spreads_avg[ticker]*100:6.3f}%")
        
        # Сводка по портфелю
        print(f"\nАНАЛИЗ ПОРТФЕЛЯ:")
        print(f"  Стоимость портфеля: {portfolio_value:,.2f}")
        print(f"  Средневзвешенный HL-спред: {lvar_results['portfolio_spread']*100:.4f}%")
        print(f"  Стандартное отклонение: {var_results['portfolio_std']:.4f}")
        print(f"  Standard VaR: {var_results['value']:,.2f} ({var_results['percentage']:.2f}%)")
        print(f"  Liquidity-Adjusted VaR: {lvar_results['value']:,.2f} ({lvar_results['percentage']:.2f}%)")
        print(f"  Дополнительный риск ликвидности: {lvar_results['liquidity_adjustment']:,.2f}")
        
        # Анализ чувствительности
        impact_ratio = (lvar_results['value'] - var_results['value']) / var_results['value'] * 100
        print(f"\nАНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ:")
        print(f"  Влияние ликвидности на общий риск: +{impact_ratio:.2f}%")


# Пример использования
if __name__ == "__main__":
    # Создаем анализатор
    analyzer = LiquidityAdjustedVaRAnalyzer()
    
    # Пример 1: Технологический портфель
    print("ПРИМЕР 1: Технологический портфель")
    results1 = analyzer.run_analysis(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        weights=[0.3, 0.25, 0.2, 0.15, 0.1],
        portfolio_value=1000000,  # 1 млн долларов
        start_date='2022-01-01',
        end_date='2024-01-01',
        confidence_level=0.95,
        time_horizon=10,
        save_report='portfolio_analysis_tech.png'
    )
    
    # Пример 2: Можно проанализировать другой портфель
    print("\n" + "=" * 80)
    print("ПРИМЕР 2: Диверсифицированный портфель")
    results2 = analyzer.run_analysis(
        tickers=['SPY', 'QQQ', 'GLD', 'TLT'],
        weights=[0.4, 0.3, 0.2, 0.1],
        portfolio_value=500000,  # 500 тыс долларов
        start_date='2023-01-01',  # Более короткий период
        confidence_level=0.99,    # Более строгий уровень доверия
        time_horizon=5           # Более короткий горизонт
    )