import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime

class RiskVisualizer:
    """Визуализация результатов анализа рисков"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig_size = (16, 12)
        
    def create_comprehensive_report(self, portfolio_data, weights, portfolio_value,
                                 var_results, lvar_results, hl_spreads_avg,
                                 liquidity_timeseries, save_path=None):
        """Создание комплексного отчета"""
        
        fig = plt.figure(figsize=(16, 14))
        
        # Сетка графиков
        gs = fig.add_gridspec(3, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])  # VaR vs LVaR
        ax2 = fig.add_subplot(gs[0, 1])  # HL-спреды активов
        ax3 = fig.add_subplot(gs[1, 0])  # Вклад в риск ликвидности
        ax4 = fig.add_subplot(gs[1, 1])  # Декомпозиция LVaR
        ax5 = fig.add_subplot(gs[2, :])  # Динамика ликвидности портфеля
        
        fig.suptitle('Комплексный анализ риска портфеля с учетом ликвидности', 
                    fontsize=16, fontweight='bold')
        
        # График 1: Сравнение VaR и LVaR
        self._plot_var_comparison(ax1, var_results, lvar_results)
        
        # График 2: HL-спреды активов
        self._plot_hl_spreads(ax2, hl_spreads_avg)
        
        # График 3: Вклад в риск ликвидности
        self._plot_liquidity_contribution(ax3, weights, hl_spreads_avg)
        
        # График 4: Декомпозиция LVaR
        self._plot_lvar_decomposition(ax4, var_results, lvar_results)
        
        # График 5: Динамика ликвидности портфеля
        self._plot_portfolio_liquidity_timeseries(ax5, liquidity_timeseries)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Отчет сохранен в {save_path}")
        
        plt.show()
    
    def _plot_var_comparison(self, ax, var_results, lvar_results):
        """Сравнение VaR и LVaR"""
        labels = ['Standard VaR', 'Liquidity-Adjusted VaR']
        values = [var_results['value'], lvar_results['value']]
        percentages = [var_results['percentage'], lvar_results['percentage']]
        
        bars = ax.bar(labels, values, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax.set_title('Сравнение VaR и LVaR', fontweight='bold')
        ax.set_ylabel('Стоимостная оценка риска')
        
        for bar, value, percentage in zip(bars, values, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,.0f}\n({percentage:.2f}%)', ha='center', va='bottom', 
                   fontweight='bold')
    
    def _plot_hl_spreads(self, ax, hl_spreads_avg):
        """Рейтинг ликвидности активов"""
        tickers = list(hl_spreads_avg.keys())
        spreads = [hl_spreads_avg[ticker] * 100 for ticker in tickers]
        
        sorted_indices = np.argsort(spreads)
        sorted_tickers = [tickers[i] for i in sorted_indices]
        sorted_spreads = [spreads[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_tickers, sorted_spreads, color='lightgreen', alpha=0.7)
        ax.set_title('HL-спреды активов (рейтинг ликвидности)', fontweight='bold')
        ax.set_xlabel('HL-спред (%)')
        
        for bar, spread in zip(bars, sorted_spreads):
            width = bar.get_width()
            ax.text(width + max(sorted_spreads)*0.01, bar.get_y() + bar.get_height()/2.,
                   f'{spread:.3f}%', ha='left', va='center', fontweight='bold')
    
    def _plot_liquidity_contribution(self, ax, weights, hl_spreads_avg):
        """Вклад активов в риск ликвидности"""
        tickers = list(hl_spreads_avg.keys())
        contributions = [weights[i] * hl_spreads_avg[ticker] * 100 
                        for i, ticker in enumerate(tickers)]
        
        ax.pie(contributions, labels=tickers, autopct='%1.1f%%', 
               startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(tickers))))
        ax.set_title('Вклад активов в риск ликвидности', fontweight='bold')
    
    def _plot_lvar_decomposition(self, ax, var_results, lvar_results):
        """Декомпозиция LVaR"""
        components = ['Рыночный риск (VaR)', 'Корректировка на ликвидность']
        component_values = [var_results['value'], lvar_results['liquidity_adjustment']]
        
        bars = ax.bar(components, component_values, color=['lightblue', 'orange'], alpha=0.7)
        ax.set_title('Декомпозиция LVaR', fontweight='bold')
        ax.set_ylabel('Стоимостная оценка')
        
        for bar, value in zip(bars, component_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_portfolio_liquidity_timeseries(self, ax, liquidity_timeseries):
        """Динамика ликвидности портфеля с течением времени"""
        portfolio_liquidity = liquidity_timeseries['portfolio']
        assets_liquidity = liquidity_timeseries['assets']
        
        # Портфель в целом
        ax.plot(portfolio_liquidity.index, portfolio_liquidity.values * 100, 
               linewidth=2, color='red', label='Портфель (средневзвешенный)')
        
        # Отдельные активы
        for ticker in assets_liquidity.columns:
            ax.plot(assets_liquidity.index, assets_liquidity[ticker].values * 100,
                   alpha=0.6, label=ticker)
        
        ax.set_title('Динамика ликвидности портфеля', fontweight='bold')
        ax.set_ylabel('HL-спред (%)')
        ax.set_xlabel('Дата')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def plot_individual_asset_liquidity(self, liquidity_timeseries, save_path=None):
        """Отдельный график ликвидности для каждого актива"""
        assets_liquidity = liquidity_timeseries['assets']
        
        n_assets = len(assets_liquidity.columns)
        fig, axes = plt.subplots(n_assets, 1, figsize=(14, 4*n_assets))
        
        if n_assets == 1:
            axes = [axes]
        
        for i, ticker in enumerate(assets_liquidity.columns):
            ax = axes[i]
            ax.plot(assets_liquidity.index, assets_liquidity[ticker].values * 100,
                   linewidth=2, color=f'C{i}')
            ax.set_title(f'Динамика ликвидности {ticker}', fontweight='bold')
            ax.set_ylabel('HL-спред (%)')
            ax.grid(True, alpha=0.3)
            
            # Форматирование дат
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики ликвидности сохранены в {save_path}")
        
        plt.show()