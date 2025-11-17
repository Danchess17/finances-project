import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings
from scipy.stats import norm
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

class LiquidityAdjustedVaR:
    """
    Модуль для комплексной оценки рыночного риска портфеля 
    с поправкой на риск ликвидности
    """
    
    def __init__(self):
        self.portfolio_data = None
        self.hl_spreads = None
        self.returns = None
        self.cov_matrix = None
        self.var_results = None
        self.lvar_results = None
        
    def fetch_data(self, tickers, start_date, end_date, source='yfinance'):
        """
        Загрузка исторических данных по активам
        """
        print("Загрузка данных...")
        data = {}
        
        if source == 'yfinance':
            for ticker in tickers:
                try:
                    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not stock.empty:
                        data[ticker] = stock[['High', 'Low', 'Close']]
                        print(f"Данные для {ticker} успешно загружены")
                    else:
                        print(f"Ошибка загрузки данных для {ticker}")
                except Exception as e:
                    print(f"Ошибка при загрузке {ticker}: {e}")
        
        elif source == 'moex':
            print("MOEX источник временно недоступен, используем yfinance")
            return self.fetch_data(tickers, start_date, end_date, 'yfinance')
        
        self.portfolio_data = data
        return data
    
    def preprocess_data(self):
        """
        Предобработка данных и расчет доходностей
        """
        if self.portfolio_data is None:
            raise ValueError("Данные не загружены. Сначала выполните fetch_data()")
        
        print("Предобработка данных...")
        
        # Объединение данных по Close ценам
        close_prices = pd.DataFrame()
        for ticker, data in self.portfolio_data.items():
            close_prices[ticker] = data['Close']
        
        # Обработка пропусков
        close_prices = close_prices.ffill().bfill().dropna()
        
        # Расчет логарифмических доходностей
        self.returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        print("Предобработка данных завершена")
        return self.returns
    
    def calculate_hl_spread(self):
        """
        Расчет HL-спреда по модели Корвина-Шульца
        """
        if self.portfolio_data is None:
            raise ValueError("Данные не загружены")
        
        print("Расчет HL-спредов...")
        hl_spreads = {}
        
        for ticker, data in self.portfolio_data.items():
            try:
                # Расчет дневного спреда
                daily_spread = 2 * (data['High'] - data['Low']) / (data['High'] + data['Low'])
                
                # Усреднение за весь период
                avg_spread = float(daily_spread.mean())  # Преобразуем в float
                hl_spreads[ticker] = avg_spread
                print(f"  {ticker}: {avg_spread:.4f}")
            except Exception as e:
                print(f"  Ошибка расчета спреда для {ticker}: {e}")
                hl_spreads[ticker] = 0.01
        
        self.hl_spreads = hl_spreads
        print("Расчет HL-спредов завершен")
        return hl_spreads
    
    def calculate_var(self, weights, portfolio_value, confidence_level=0.95, time_horizon=10):
        """
        Расчет стандартного Value at Risk
        """
        if self.returns is None:
            self.preprocess_data()
        
        print("Расчет VaR...")
        
        weights = np.array(weights)
        
        # Проверка весов
        if abs(np.sum(weights) - 1.0) > 0.01:
            raise ValueError(f"Сумма весов должна быть равна 1.0, получено: {np.sum(weights)}")
        
        # Расчет ковариационной матрицы
        self.cov_matrix = self.returns.cov()
        
        # Стандартное отклонение портфеля
        portfolio_std = float(np.sqrt(weights.T @ self.cov_matrix @ weights))  # Преобразуем в float
        
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
    
    def calculate_lvar(self, weights, portfolio_value, confidence_level=0.95, time_horizon=10):
        """
        Расчет Liquidity-Adjusted VaR
        """
        if self.hl_spreads is None:
            self.calculate_hl_spread()
        
        if self.var_results is None:
            self.calculate_var(weights, portfolio_value, confidence_level, time_horizon)
        
        print("Расчет LVaR...")
        
        weights = np.array(weights)
        tickers = list(self.portfolio_data.keys())
        
        # Средневзвешенный спред портфеля
        portfolio_spread = float(sum(weights[i] * self.hl_spreads[ticker] 
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
    
    def generate_report(self, weights, portfolio_value, confidence_level=0.95, 
                       time_horizon=10, save_path=None):
        """
        Генерация комплексного отчета с визуализацией
        """
        if any(x is None for x in [self.var_results, self.lvar_results, self.hl_spreads]):
            self.calculate_lvar(weights, portfolio_value, confidence_level, time_horizon)
        
        print("Генерация отчета...")
        
        # Создание фигуры с несколькими subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Комплексный анализ риска портфеля с учетом ликвидности', 
                    fontsize=16, fontweight='bold')
        
        # График 1: Сравнение VaR и LVaR
        labels = ['Standard VaR', 'Liquidity-Adjusted VaR']
        values = [self.var_results['value'], self.lvar_results['value']]
        percentages = [self.var_results['percentage'], self.lvar_results['percentage']]
        
        bars = ax1.bar(labels, values, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax1.set_title('Сравнение VaR и LVaR', fontweight='bold')
        ax1.set_ylabel('Стоимостная оценка риска')
        
        # Добавление значений на столбцы
        for bar, value, percentage in zip(bars, values, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,.0f}\n({percentage:.2f}%)', ha='center', va='bottom', 
                    fontweight='bold')
        
        # График 2: HL-спреды по активам (рейтинг ликвидности)
        tickers = list(self.hl_spreads.keys())
        spreads = [self.hl_spreads[ticker] * 100 for ticker in tickers]  # в процентах
        
        # Сортировка по спреду (от более ликвидных к менее ликвидным)
        sorted_indices = np.argsort(spreads)
        sorted_tickers = [tickers[i] for i in sorted_indices]
        sorted_spreads = [spreads[i] for i in sorted_indices]
        
        bars = ax2.barh(sorted_tickers, sorted_spreads, color='lightgreen', alpha=0.7)
        ax2.set_title('HL-спреды активов (рейтинг ликвидности)', fontweight='bold')
        ax2.set_xlabel('HL-спред (%)')
        
        # Добавление значений на столбцы
        for bar, spread in zip(bars, sorted_spreads):
            width = bar.get_width()
            ax2.text(width + max(sorted_spreads)*0.01, bar.get_y() + bar.get_height()/2.,
                    f'{spread:.3f}%', ha='left', va='center', fontweight='bold')
        
        # График 3: Вклад в риск ликвидности
        liquidity_contributions = [weights[i] * self.hl_spreads[ticker] * 100 
                                 for i, ticker in enumerate(tickers)]
        
        ax3.pie(liquidity_contributions, labels=tickers, autopct='%1.1f%%', 
                startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(tickers))))
        ax3.set_title('Вклад активов в риск ликвидности', fontweight='bold')
        
        # График 4: Детализация корректировки LVaR
        components = ['Рыночный риск (VaR)', 'Корректировка на ликвидность']
        component_values = [self.var_results['value'], 
                          self.lvar_results['liquidity_adjustment']]
        
        bars = ax4.bar(components, component_values, color=['lightblue', 'orange'], alpha=0.7)
        ax4.set_title('Декомпозиция LVaR', fontweight='bold')
        ax4.set_ylabel('Стоимостная оценка')
        
        for bar, value in zip(bars, component_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики сохранены в {save_path}")
        
        plt.show()
        
        # Вывод табличных результатов
        self._print_summary_table(weights, portfolio_value)
    
    def _print_summary_table(self, weights, portfolio_value):
        """
        Вывод сводной таблицы результатов
        """
        print("\n" + "="*80)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("="*80)
        
        # Таблица по активам
        assets_data = []
        tickers = list(self.portfolio_data.keys())
        
        for i, ticker in enumerate(tickers):
            assets_data.append({
                'Актив': ticker,
                'Вес (%)': weights[i] * 100,
                'HL-спред (%)': self.hl_spreads[ticker] * 100,
                'Вклад в риск ликвидности': weights[i] * self.hl_spreads[ticker] * 100
            })
        
        assets_df = pd.DataFrame(assets_data)
        print("\nАНАЛИЗ АКТИВОВ:")
        print(assets_df.to_string(index=False, float_format='%.4f'))
        
        # Сводка по портфелю
        print(f"\nАНАЛИЗ ПОРТФЕЛЯ:")
        print(f"Стоимость портфеля: {portfolio_value:,.2f}")
        print(f"Средневзвешенный HL-спред: {self.lvar_results['portfolio_spread']*100:.4f}%")
        print(f"Стандартное отклонение портфеля: {self.var_results['portfolio_std']:.4f}")
        print(f"Standard VaR: {self.var_results['value']:,.2f} ({self.var_results['percentage']:.2f}%)")
        print(f"Liquidity-Adjusted VaR: {self.lvar_results['value']:,.2f} ({self.lvar_results['percentage']:.2f}%)")
        print(f"Дополнительный риск ликвидности: {self.lvar_results['liquidity_adjustment']:,.2f}")
        
        # Анализ чувствительности
        print(f"\nАНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ:")
        impact_ratio = (self.lvar_results['value'] - self.var_results['value']) / self.var_results['value'] * 100
        print(f"Влияние ликвидности на общий риск: +{impact_ratio:.2f}%")

    def run_complete_analysis(self, tickers, weights, portfolio_value, 
                            start_date, end_date, confidence_level=0.95, 
                            time_horizon=10, source='yfinance'):
        """
        Полный запуск анализа
        """
        print("ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА РИСКА")
        print("="*50)
        
        try:
            # Загрузка данных
            self.fetch_data(tickers, start_date, end_date, source)
            
            # Предобработка
            self.preprocess_data()
            
            # Расчет метрик
            self.calculate_hl_spread()
            self.calculate_var(weights, portfolio_value, confidence_level, time_horizon)
            self.calculate_lvar(weights, portfolio_value, confidence_level, time_horizon)
            
            # Генерация отчета
            self.generate_report(weights, portfolio_value, confidence_level, time_horizon)
            
            return {
                'hl_spreads': self.hl_spreads,
                'var': self.var_results,
                'lvar': self.lvar_results
            }
            
        except Exception as e:
            print(f"Ошибка при выполнении анализа: {e}")
            import traceback
            traceback.print_exc()
            return None


# Пример использования
def main():
    """
    Демонстрация работы модуля на примере портфеля
    """
    # Параметры портфеля
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Сумма = 1
    portfolio_value = 1000000  # 1 млн рублей/долларов
    confidence_level = 0.95
    time_horizon = 10
    
    # Период анализа (последние 2 года)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    
    # Создание и запуск анализа
    risk_analyzer = LiquidityAdjustedVaR()
    
    try:
        results = risk_analyzer.run_complete_analysis(
            tickers=tickers,
            weights=weights,
            portfolio_value=portfolio_value,
            start_date=start_date,
            end_date=end_date,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            source='yfinance'
        )
        
        return results
        
    except Exception as e:
        print(f"Ошибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Запуск демонстрации
    results = main()
    
    if results:
        # Дополнительный пример: сравнение портфелей с разной ликвидностью
        print("\n" + "="*80)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ПОРТФЕЛЕЙ С РАЗНОЙ ЛИКВИДНОСТЬЮ")
        print("="*80)
        
        # Портфель с высоколиквидными активами
        liquid_tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ']
        liquid_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # Портфель с низколиквидными активами
        illiquid_tickers = ['TSLA', 'NVDA', 'MRNA', 'ARKK', 'IBB']
        illiquid_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        analyzer1 = LiquidityAdjustedVaR()
        analyzer2 = LiquidityAdjustedVaR()
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        print("\nВысоколиквидный портфель:")
        results1 = analyzer1.run_complete_analysis(
            liquid_tickers, liquid_weights, 1000000, 
            start_date, end_date, confidence_level=0.95, time_horizon=10
        )
        
        print("\nНизколиквидный портфель:")
        results2 = analyzer2.run_complete_analysis(
            illiquid_tickers, illiquid_weights, 1000000,
            start_date, end_date, confidence_level=0.95, time_horizon=10
        )