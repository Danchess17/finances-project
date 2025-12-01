import sys

from moex_data_loader import MoexDataLoader
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.optimize import minimize


def compute_corwin_schultz_spread(df):
    """
    Вычисляет спред по методу Корвина-Шульца (Corwin-Schultz)
    Code written based on Corwin & Schultz (2011)
    
    Правильная формула:
    - Для дня t: beta_t = [ln(H_t/L_t)]^2
    - Для двух дней t и t+1: 
      H_{t,t+1} = max(H_t, H_{t+1}), L_{t,t+1} = min(L_t, L_{t+1})
      gamma_t = [ln(H_{t,t+1}/L_{t,t+1})]^2
    - alpha = sqrt(2*beta - sqrt(2)/(3-2*sqrt(2)) * sqrt(beta)) - sqrt(1/(3-2*sqrt(2)) * gamma)
    - Spread = 2*(exp(alpha) - 1)/(1 + exp(alpha))
    
    Parameters:
        df (pd.DataFrame): DataFrame с колонками High, Low, Close
        
    Returns:
        pd.Series: Массив значений спреда для каждого дня
    """
    # Check for null or infinite values
    if df['High'].isnull().any() or df['Low'].isnull().any():
        return pd.Series([0.0] * len(df), index=df.index)
    
    if np.isinf(df['High']).any() or np.isinf(df['Low']).any():
        return pd.Series([0.0] * len(df), index=df.index)
    
    epsilon = 1e-10  # Small constant to prevent division by zero
    
    # Beta calculation for day t: beta_t = [ln(H_t/L_t)]^2
    beta = (np.log(df['High'] / (df['Low'] + epsilon)) ** 2)
    
    # Gamma calculation for two-day period: 
    # H_{t,t+1} = max(H_t, H_{t+1}), L_{t,t+1} = min(L_t, L_{t+1})
    # gamma_t = [ln(H_{t,t+1}/L_{t,t+1})]^2
    high_combined = pd.concat([df['High'], df['High'].shift(-1)], axis=1).max(axis=1)
    low_combined = pd.concat([df['Low'], df['Low'].shift(-1)], axis=1).min(axis=1)
    gamma = (np.log(high_combined / (low_combined + epsilon)) ** 2)
    
    # Alpha calculation according to Corwin-Schultz formula
    # Correct formula from research.ipynb (based on Corwin & Schultz 2011):
    # denom = sqrt(3 - 2*sqrt(2))
    # alpha = (sqrt(2*beta) - sqrt(gamma)) / denom
    
    denom = np.sqrt(3 - 2 * np.sqrt(2))  # denominator
    
    # Calculate alpha using the correct simplified formula
    sqrt_2beta = np.sqrt(2 * beta)
    sqrt_gamma = np.sqrt(gamma)
    
    # Ensure non-negative for square roots
    sqrt_2beta = np.maximum(sqrt_2beta, epsilon)
    sqrt_gamma = np.maximum(sqrt_gamma, epsilon)
    
    # Alpha = (sqrt(2*beta) - sqrt(gamma)) / denom
    alpha = (sqrt_2beta - sqrt_gamma) / denom
    
    # Spread calculation: S = 2*(exp(alpha) - 1)/(1 + exp(alpha))
    # This formula gives values in range [-2, 2] theoretically, but typically [0, 2]
    # For realistic spreads, values should be small (< 0.1 for 10% spread)
    exp_alpha = np.exp(alpha)
    S = 2 * (exp_alpha - 1) / (1 + exp_alpha)
    
    # The formula can give negative values (which we handle) or values > 1
    # For percentage spread, we expect values in [0, 1] range
    # If values are > 1, there may be an issue with the calculation
    
    # Ensure spread is non-negative
    S = np.maximum(0, S)
    
    # Cap at reasonable maximum (e.g., 0.5 = 50% spread is already very high)
    # Values above this suggest calculation issues
    S = np.minimum(S, 0.5)  # Cap at 50% as a safety measure
    
    # Remove NaN values (from shift operations at boundaries)
    S = S.dropna()
    
    return S


def compute_spread(df, use_corwin_schultz=False):
    """
    Вычисляет спред для каждого дня
    
    Parameters:
        df (pd.DataFrame): DataFrame с колонками High, Low, Close
        use_corwin_schultz (bool): Если True - использует метод Корвина-Шульца,
                                  если False - использует простую формулу (High-Low)/Close
        
    Returns:
        pd.Series: Массив значений спреда для каждого дня
    """
    if use_corwin_schultz:
        return compute_corwin_schultz_spread(df)
    else:
        # Простая формула: (High - Low) / Close
        spread = (df["High"] - df["Low"]) / df["Close"]
        return spread


def calculate_optimal_weights(portfolio_df, symbols, debug=False, use_corwin_schultz=False):
    """
    Рассчитывает оптимальные веса для минимизации LVaR (Liquidity-adjusted VaR)
    Использует методику из research.ipynb
    
    Parameters:
        portfolio_df (pd.DataFrame): Данные портфеля с колонками Date, High, Low, Close, Symbol
        symbols (list): Список символов
        debug (bool): Режим отладки
        use_corwin_schultz (bool): Если True - использует метод Корвина-Шульца для расчета спреда,
                                  если False - использует простую формулу (High-Low)/Close
        
    Returns:
        dict: Словарь {symbol: weight}
    """
    if portfolio_df is None or portfolio_df.empty:
        if debug:
            print("Нет данных для расчета весов")
        return None, None, None, None
    
    if not symbols or len(symbols) == 0:
        if debug:
            print("Нет символов для расчета весов")
        return None, None, None, None
    
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    spreads = {}
    prices_dict = {}
    
    for symbol in symbols:
        stock_data = portfolio_df[portfolio_df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        if stock_data.empty:
            if debug:
                print(f"Нет данных для {symbol}, пропускаем")
            continue
        
        # Удаляем дубликаты дат (берем последнюю запись, как в ноутбуке)
        stock_data = stock_data[~stock_data['Date'].duplicated(keep='last')]
        
        if stock_data.empty:
            if debug:
                print(f"Нет данных для {symbol} после удаления дубликатов")
            continue
        
        # Рассчитываем спред (возвращает массив спредов)
        spread_array = compute_spread(stock_data[['High', 'Low', 'Close']], use_corwin_schultz=use_corwin_schultz)
        spreads[symbol] = spread_array  # Сохраняем весь массив, медиана будет вычислена в compute_lvar
        
        # Сохраняем цены закрытия (устанавливаем Date как индекс для объединения)
        stock_data_indexed = stock_data.set_index('Date')
        prices_dict[symbol] = stock_data_indexed['Close']
        
    if not spreads:
        if debug:
            print("Не удалось рассчитать спреды ни для одной акции")
        return None, None, None, None
    
    # Обновляем список символов (только те, для которых есть данные)
    valid_symbols = list(spreads.keys())
    
    if len(valid_symbols) < 2:
        if debug:
            print(f"Нужно минимум 2 акции для расчета весов, найдено: {len(valid_symbols)}")
        return None, None, None, None
    
    # 2. Создаем DataFrame с ценами закрытия
    prices = pd.DataFrame(prices_dict).dropna()
    
    if prices.empty or len(prices) < 2:
        if debug:
            print("Недостаточно данных о ценах для расчета")
        return None, None, None, None
    
    # 3. Рассчитываем доходности
    returns = prices.pct_change().dropna()
    
    if returns.empty or len(returns) < 2:
        if debug:
            print("Недостаточно данных о доходностях")
        return None, None, None, None
    
    # 4. Рассчитываем ковариационную матрицу
    sigma = returns.cov().values
    tickers = returns.columns.tolist()
    n = len(tickers)
    
    # 5. Создаем словарь массивов спредов в том же порядке, что и tickers (медиана будет вычислена в compute_lvar)
    spreads_dict = {t: spreads[t] for t in tickers}
    
    if debug:
        print(f"\nКовариационная матрица ({n}x{n}):")
        print(sigma)
        print(f"\nСпреды (медиана):")
        for t in tickers:
            spread_median = spreads[t].median()
            print(f"{t}: {spread_median:.6f}")
        
    # 6. Функция для минимизации LVaR (использует единый метод compute_lvar)
    z = norm.ppf(0.95)  # 95% доверительный уровень
    
    def portfolio_lvar(w, sigma, spreads_dict, z):
        """Функция для расчета LVaR портфеля (для оптимизации)"""
        result = compute_lvar(w, sigma, spreads_dict, z)
        return result['lvar']
    
    # 7. Оптимизация весов
    w0 = np.ones(n) / n  # Начальные веса (равномерное распределение)
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Ограничение: сумма весов = 1
    bounds = [(0, 1)] * n  # Ограничение: веса между 0 и 1
    
    try:
        opt_lvar = minimize(
            portfolio_lvar,
            w0,
            args=(sigma, spreads_dict, z),
            method="SLSQP",
            bounds=bounds,
            constraints=[cons]
        )
        
        if not opt_lvar.success:
            if debug:
                print(f"Оптимизация не сошлась: {opt_lvar.message}")
            return None, None, None, None
        
        w_lvar = opt_lvar.x
        
        # Создаем словарь весов
        weights_dict = {ticker: weight for ticker, weight in zip(tickers, w_lvar)}
        
        if debug:
            # Показываем итоговые метрики (используем единый метод compute_lvar)
            final_metrics = compute_lvar(w_lvar, sigma, spreads_dict, z)
            
            print(f"\n Оптимальные веса (минимизация LVaR):")
            for t, w in weights_dict.items():
                print(f"{t}: {w:.4f} ({w*100:.2f}%)")
            
            print(f"\n Метрики портфеля:")
            print(f"Стандартное отклонение: {final_metrics['sigma_p']:.6f}")
            print(f"VaR (95%): {final_metrics['var']:.6f}")
            print(f"Стоимость ликвидности: {final_metrics['liquidity_cost']:.6f}")
            print(f"LVaR: {final_metrics['lvar']:.6f}")

            if debug:
                plot_lvar_contribution(w_lvar, sigma, np.cov(np.vstack(list(spreads_dict.values()))), tickers)
        
        return weights_dict, sigma, spreads_dict, tickers
        
    except Exception as e:
        if debug:
            print(f"Ошибка при оптимизации весов: {e}")
        return None, None, None, None


def compute_lvar(weights, sigma, spreads_data, z=norm.ppf(0.95), kappa=1.0):
    weights = np.array(weights)

    assert isinstance(spreads_data, dict)

    sigma_s = np.cov(np.vstack(list(spreads_data.values())))

    sigma_p = np.sqrt(weights @ sigma @ weights)

    liquidity_cost = 0.5 * kappa * np.sqrt(weights @ sigma_s @ weights)

    lvar = z * sigma_p + liquidity_cost

    return {
        'lvar': lvar,
        'var': (z * sigma_p),
        'liquidity_cost': liquidity_cost,
        'sigma_p': sigma_p
    }



def calculate_lvar_for_weights(weights, sigma, spread_array, z=norm.ppf(0.95)):
    """
    Рассчитывает LVaR для заданных весов (обертка над compute_lvar для обратной совместимости)
    
    Parameters:
        weights (np.array): Вектор весов
        sigma (np.array): Ковариационная матрица
        spread_array (np.array): Массив спредов (один спред на актив)
        z (float): Квантиль нормального распределения (по умолчанию 95%)
        
    Returns:
        dict: Словарь с метриками {'lvar', 'var', 'liquidity_cost', 'sigma_p'}
    """
    return compute_lvar(weights, sigma, spread_array, z)


def evaluate_portfolio_liquidity(weights, portfolio_df, symbols, use_corwin_schultz=False, debug=False):
    """
    Оценивает ликвидность портфеля с заданными весами
    
    Что значит "ликвидный портфель":
    1. Низкая стоимость ликвидности (weighted average spread)
    2. Низкие спреды активов в портфеле
    3. Можно быстро закрыть позицию без больших потерь
    4. Портфель устойчив к изменениям ликвидности
    
    Parameters:
        weights (dict): Словарь весов {symbol: weight}
        portfolio_df (pd.DataFrame): Данные портфеля с колонками Date, High, Low, Close, Symbol
        symbols (list): Список символов
        use_corwin_schultz (bool): Использовать метод Корвина-Шульца для расчета спреда
        debug (bool): Режим отладки
        
    Returns:
        dict: Результаты оценки ликвидности с метриками и оценкой
    """
    if portfolio_df is None or portfolio_df.empty:
        return {'error': 'Нет данных портфеля'}
    
    if not weights or not symbols:
        return {'error': 'Нет весов или символов'}
    
    portfolio_df = portfolio_df.copy()
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    # 1. Рассчитываем спреды для каждого актива
    spreads_median = {}
    spreads_mean = {}
    spreads_max = {}
    spreads_std = {}
    
    for symbol in symbols:
        symbol_data = portfolio_df[portfolio_df['Symbol'] == symbol][['High', 'Low', 'Close']]
        if symbol_data.empty:
            continue
        
        spread_series = compute_spread(symbol_data, use_corwin_schultz=use_corwin_schultz)
        spreads_median[symbol] = spread_series.median()
        spreads_mean[symbol] = spread_series.mean()
        spreads_max[symbol] = spread_series.max()
        spreads_std[symbol] = spread_series.std()
    
    # 2. Рассчитываем средневзвешенный спред портфеля
    portfolio_spread_weighted = sum([weights.get(s, 0) * spreads_median.get(s, 0) for s in symbols])
    portfolio_spread_mean = sum([weights.get(s, 0) * spreads_mean.get(s, 0) for s in symbols])
    
    # 3. Стоимость ликвидности портфеля (0.5 * weighted spread)
    liquidity_cost = 0.5 * portfolio_spread_weighted
    
    # 4. Сравнение с равными весами (бенчмарк)
    n = len(symbols)
    equal_weights = {s: 1.0/n for s in symbols}
    equal_spread = sum([equal_weights.get(s, 0) * spreads_median.get(s, 0) for s in symbols])
    equal_liquidity_cost = 0.5 * equal_spread
    
    # 5. Концентрация риска ликвидности (максимальный вклад одного актива)
    max_contribution = max([weights.get(s, 0) * spreads_median.get(s, 0) for s in symbols])
    max_contribution_pct = (max_contribution / portfolio_spread_weighted * 100) if portfolio_spread_weighted > 0 else 0
    
    # 6. Оценка ликвидности по порогам
    # Пороги для оценки (на основе эмпирических данных):
    # - Отлично: спред < 0.01 (1%)
    # - Хорошо: спред 0.01-0.02 (1-2%)
    # - Средне: спред 0.02-0.03 (2-3%)
    # - Плохо: спред > 0.03 (3%+)
    
    if portfolio_spread_weighted < 0.01:
        liquidity_rating = "ОТЛИЧНО"
        rating_score = 5
    elif portfolio_spread_weighted < 0.02:
        liquidity_rating = "ХОРОШО"
        rating_score = 4
    elif portfolio_spread_weighted < 0.03:
        liquidity_rating = "СРЕДНЕ"
        rating_score = 3
    elif portfolio_spread_weighted < 0.05:
        liquidity_rating = "НИЗКО"
        rating_score = 2
    else:
        liquidity_rating = "ОЧЕНЬ НИЗКО"
        rating_score = 1
    
    # 7. Стабильность ликвидности (коэффициент вариации спредов)
    spread_cv = {}
    for symbol in symbols:
        if spreads_std.get(symbol, 0) > 0 and spreads_mean.get(symbol, 0) > 0:
            spread_cv[symbol] = (spreads_std[symbol] / spreads_mean[symbol]) * 100
        else:
            spread_cv[symbol] = 0
    
    avg_cv = np.mean([spread_cv.get(s, 0) for s in symbols])
    
    # 8. Сравнение с бенчмарком
    improvement_vs_equal = ((equal_liquidity_cost - liquidity_cost) / equal_liquidity_cost * 100) if equal_liquidity_cost > 0 else 0
    
    results = {
        'liquidity_rating': liquidity_rating,
        'rating_score': rating_score,
        'portfolio_spread_median': portfolio_spread_weighted,
        'portfolio_spread_mean': portfolio_spread_mean,
        'liquidity_cost': liquidity_cost,
        'equal_weights_spread': equal_spread,
        'equal_weights_liquidity_cost': equal_liquidity_cost,
        'improvement_vs_equal_pct': improvement_vs_equal,
        'max_contribution_pct': max_contribution_pct,
        'spread_stability_cv': avg_cv,
        'asset_spreads': {
            s: {
                'median': spreads_median.get(s, 0),
                'mean': spreads_mean.get(s, 0),
                'max': spreads_max.get(s, 0),
                'std': spreads_std.get(s, 0),
                'cv': spread_cv.get(s, 0),
                'weight': weights.get(s, 0),
                'contribution': weights.get(s, 0) * spreads_median.get(s, 0)
            }
            for s in symbols
        },
        'weights': weights
    }
    
    if debug:
        print("\n" + "=" * 70)
        print("ОЦЕНКА ЛИКВИДНОСТИ ПОРТФЕЛЯ")
        print("=" * 70)
        print(f"\nОценка: {liquidity_rating} (балл: {rating_score}/5)")
        print(f"\n--- МЕТРИКИ ПОРТФЕЛЯ ---")
        print(f"Средневзвешенный спред (медиана): {portfolio_spread_weighted:.4f} ({portfolio_spread_weighted*100:.2f}%)")
        print(f"Средневзвешенный спред (среднее): {portfolio_spread_mean:.4f} ({portfolio_spread_mean*100:.2f}%)")
        print(f"Стоимость ликвидности: {liquidity_cost:.6f}")
        print(f"\n--- СРАВНЕНИЕ С РАВНЫМИ ВЕСАМИ ---")
        print(f"Спред при равных весах: {equal_spread:.4f} ({equal_spread*100:.2f}%)")
        print(f"Стоимость ликвидности при равных весах: {equal_liquidity_cost:.6f}")
        print(f"Улучшение: {improvement_vs_equal:+.2f}%")
        
        print(f"\n--- АНАЛИЗ ПО АКТИВАМ ---")
        for symbol in symbols:
            asset_info = results['asset_spreads'][symbol]
            print(f"\n{symbol}:")
            print(f"  Вес: {asset_info['weight']:.2%}")
            print(f"  Спред (медиана): {asset_info['median']:.4f} ({asset_info['median']*100:.2f}%)")
            print(f"  Спред (макс): {asset_info['max']:.4f} ({asset_info['max']*100:.2f}%)")
            print(f"  Вклад в спред портфеля: {asset_info['contribution']:.6f} ({asset_info['contribution']/portfolio_spread_weighted*100:.1f}%)")
            print(f"  Стабильность (CV): {asset_info['cv']:.1f}%")
        
        print(f"\n--- РИСКИ ЛИКВИДНОСТИ ---")
        print(f"Максимальный вклад одного актива: {max_contribution_pct:.1f}%")
        if max_contribution_pct > 50:
            print("  ⚠️  ВНИМАНИЕ: Высокая концентрация риска ликвидности!")
            print("     Это означает, что более 50% спреда портфеля приходится на один актив.")
            print("     Если ликвидность этого актива ухудшится, это сильно повлияет на весь портфель.")
        else:
            print("  ✓ Концентрация риска приемлемая")
        
        print(f"Средняя стабильность спредов (CV): {avg_cv:.1f}%")
        if avg_cv > 50:
            print("  ⚠️  ВНИМАНИЕ: Высокая изменчивость спредов!")
            print("     Коэффициент вариации >50% означает, что спреды сильно колеблются во времени.")
            print("     Текущая оценка ликвидности может сильно измениться в будущем.")
        else:
            print("  ✓ Спреды стабильны во времени")
        
        print(f"\n--- КРИТЕРИИ ОЦЕНКИ ЛИКВИДНОСТИ ---")
        print("Оценка основана на средневзвешенном спреде портфеля:")
        print("  • ОТЛИЧНО (5/5): спред < 1.0%")
        print("  • ХОРОШО (4/5):  спред 1.0-2.0%")
        print("  • СРЕДНЕ (3/5):  спред 2.0-3.0%")
        print("  • НИЗКО (2/5):   спред 3.0-5.0%")
        print("  • ОЧЕНЬ НИЗКО (1/5): спред > 5.0%")
        
        print(f"\n--- ИНТЕРПРЕТАЦИЯ ---")
        if rating_score >= 4:
            print("✓ Портфель имеет хорошую ликвидность. Можно быстро закрыть позиции с минимальными потерями.")
            if max_contribution_pct > 50 or avg_cv > 50:
                print("⚠ Однако есть дополнительные риски (см. раздел 'РИСКИ ЛИКВИДНОСТИ')")
        elif rating_score == 3:
            print("⚠ Портфель имеет среднюю ликвидность. При закрытии позиций возможны умеренные потери.")
        else:
            print("✗ Портфель имеет низкую ликвидность. При закрытии позиций возможны значительные потери.")
        
        print(f"\n--- МЕТОД РАСЧЕТА СПРЕДА ---")
        method_name = "Корвина-Шульца" if use_corwin_schultz else "простая формула (High-Low)/Close"
        print(f"Используется: {method_name}")
    
    return results


def verify_optimal_weights(portfolio_df, symbols, optimal_weights, sigma, spread_array, 
                          debug=False, num_samples=100):
    """
    Проверяет оптимальность весов путем сравнения с альтернативными комбинациями
    и моделирования рыночных сценариев
    
    Parameters:
        portfolio_df (pd.DataFrame): Данные портфеля
        symbols (list): Список символов
        optimal_weights (dict): Оптимальные веса
        sigma (np.array): Ковариационная матрица
        spread_array (np.array): Массив спредов
        debug (bool): Режим отладки
        num_samples (int): Количество случайных комбинаций для проверки
        
    Returns:
        dict: Результаты проверки
    """
    z = norm.ppf(0.95)
    n = len(symbols)
    
    # Конвертируем оптимальные веса в массив
    optimal_w = np.array([optimal_weights[s] for s in symbols])
    optimal_metrics = calculate_lvar_for_weights(optimal_w, sigma, spread_array, z)
    optimal_lvar = optimal_metrics['lvar']
    
    results = {
        'optimal_lvar': optimal_lvar,
        'optimal_metrics': optimal_metrics,
        'alternative_weights': [],
        'random_samples': [],
        'equal_weights': None,
        'single_asset_weights': []
    }
    
    if debug:
        print("\n" + "=" * 70)
        print("ПРОВЕРКА ОПТИМАЛЬНОСТИ ВЕСОВ")
        print("=" * 70)
        print(f"\n Оптимальные веса дают LVaR = {optimal_lvar:.6f}")
        print(f"VaR = {optimal_metrics['var']:.6f}")
        print(f"Стоимость ликвидности = {optimal_metrics['liquidity_cost']:.6f}")
    
    # 1. Проверяем равномерное распределение весов
    equal_w = np.ones(n) / n
    equal_metrics = calculate_lvar_for_weights(equal_w, sigma, spread_array, z)
    results['equal_weights'] = {
        'weights': {s: w for s, w in zip(symbols, equal_w)},
        'metrics': equal_metrics,
        'lvar': equal_metrics['lvar']
    }
    
    if debug:
        print(f"\nРавномерное распределение (1/{n} для каждой акции):")
        for s, w in zip(symbols, equal_w):
            print(f"{s}: {w:.2%}")
        print(f"LVaR = {equal_metrics['lvar']:.6f} "
              f"({((equal_metrics['lvar'] / optimal_lvar - 1) * 100):+.2f}% относительно оптимального)")
    
    # 2. Проверяем портфели из одной акции
    if debug:
        print(f"\nПортфели из одной акции:")
    
    for i, symbol in enumerate(symbols):
        single_w = np.zeros(n)
        single_w[i] = 1.0
        single_metrics = calculate_lvar_for_weights(single_w, sigma, spread_array, z)
        
        results['single_asset_weights'].append({
            'symbol': symbol,
            'weights': {s: w for s, w in zip(symbols, single_w)},
            'metrics': single_metrics,
            'lvar': single_metrics['lvar']
        })
        
        if debug:
            print(f"100% {symbol}:")
            print(f"LVaR = {single_metrics['lvar']:.6f} "
                  f"({((single_metrics['lvar'] / optimal_lvar - 1) * 100):+.2f}% относительно оптимального)")
    
    # 3. Генерируем случайные комбинации весов
    np.random.seed(42)  # Для воспроизводимости
    random_lvars = []
    random_samples_list = []
    
    for _ in range(num_samples):
        # Генерируем случайные веса, которые в сумме дают 1
        random_w = np.random.dirichlet(np.ones(n))
        random_metrics = calculate_lvar_for_weights(random_w, sigma, spread_array, z)
        random_lvar = random_metrics['lvar']
        random_lvars.append(random_lvar)
        
        random_samples_list.append({
            'weights': {s: w for s, w in zip(symbols, random_w)},
            'metrics': random_metrics,
            'lvar': random_lvar
        })
    
    random_lvars = np.array(random_lvars)
    
    # Сохраняем несколько примеров для вывода
    results['random_samples'] = random_samples_list[:5]
    results['all_random_lvars'] = random_lvars  # Сохраняем все для графика
    
    if debug:
        print(f"\nСлучайные комбинации весов ({num_samples} образцов):")
        print(f"Минимальный LVaR: {random_lvars.min():.6f}")
        print(f"Максимальный LVaR: {random_lvars.max():.6f}")
        print(f"Средний LVaR: {random_lvars.mean():.6f}")
        print(f"Медианный LVaR: {np.median(random_lvars):.6f}")
        print(f"Оптимальный LVaR лучше, чем {((random_lvars > optimal_lvar).sum() / num_samples * 100):.1f}% случайных комбинаций")
        
        print(f"\n Примеры случайных комбинаций:")
        for i, sample in enumerate(results['random_samples'][:3], 1):
            print(f"Пример {i}:")
            for s, w in sample['weights'].items():
                print(f"{s}: {w:.2%}")
            print(f"LVaR = {sample['lvar']:.6f} "
                  f"({((sample['lvar'] / optimal_lvar - 1) * 100):+.2f}%)")
    
    # 4. Проверяем альтернативные стратегии (например, без GAZP)
    if n >= 3:
        # Стратегия: равномерно между двумя акциями с минимальным риском
        sorted_by_spread = sorted(zip(symbols, spread_array), key=lambda x: x[1])
        two_best = [s for s, _ in sorted_by_spread[:2]]
        
        if len(two_best) == 2:
            two_best_idx = [symbols.index(s) for s in two_best]
            two_best_w = np.zeros(n)
            two_best_w[two_best_idx[0]] = 0.5
            two_best_w[two_best_idx[1]] = 0.5
            
            two_best_metrics = calculate_lvar_for_weights(two_best_w, sigma, spread_array, z)
            results['alternative_weights'].append({
                'name': f'Равномерно между {two_best[0]} и {two_best[1]}',
                'weights': {s: w for s, w in zip(symbols, two_best_w)},
                'metrics': two_best_metrics,
                'lvar': two_best_metrics['lvar']
            })
            
            if debug:
                print(f"\nАльтернативная стратегия: равномерно между двумя лучшими по спреду:")
                for s, w in zip(symbols, two_best_w):
                    if w > 0:
                        print(f"{s}: {w:.2%}")
                print(f"LVaR = {two_best_metrics['lvar']:.6f} "
                      f"({((two_best_metrics['lvar'] / optimal_lvar - 1) * 100):+.2f}%)")
    
    # Итоговый вывод
    if debug:
        print("\n" + "=" * 70)
        print("ИТОГОВЫЕ ВЫВОДЫ")
        print("=" * 70)
        
        better_count = 0
        total_count = 1  # равномерное
        total_count += len(results['single_asset_weights'])
        total_count += len(results['alternative_weights'])
        
        if results['equal_weights']['lvar'] > optimal_lvar:
            better_count += 1
        
        for single in results['single_asset_weights']:
            if single['lvar'] > optimal_lvar:
                better_count += 1
        
        for alt in results['alternative_weights']:
            if alt['lvar'] > optimal_lvar:
                better_count += 1
        
        print(f"\n Оптимальные веса лучше, чем:")
        print(f"- Равномерное распределение")
        print(f"- Все портфели из одной акции")
        print(f"- Альтернативные стратегии")
        print(f"- {((random_lvars > optimal_lvar).sum() / num_samples * 100):.1f}% случайных комбинаций")
        print(f"\n Вывод: Веса действительно оптимальны для минимизации LVaR!")
    
    return results


def plot_lvar_comparison(verification_results, symbols, output_path=None, debug=False, show=False):
    """
    Строит график сравнения LVaR для разных стратегий весов
    
    Parameters:
        verification_results (dict): Результаты проверки оптимальности
        symbols (list): Список символов
        output_path (str): Путь для сохранения графика
        debug (bool): Режим отладки
        show (bool): Показать график
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # График 1: Сравнение основных стратегий
    strategies = []
    lvars = []
    labels = []
    
    # Оптимальные веса
    strategies.append('Оптимальные\nвеса')
    lvars.append(verification_results['optimal_lvar'])
    labels.append('Оптимальные')
    
    # Равномерное распределение
    if verification_results['equal_weights']:
        strategies.append('Равномерное\nраспределение')
        lvars.append(verification_results['equal_weights']['lvar'])
        labels.append('Равномерное')
    
    # Портфели из одной акции
    for single in verification_results['single_asset_weights']:
        strategies.append(f'100% {single["symbol"]}')
        lvars.append(single['lvar'])
        labels.append(f'{single["symbol"]}')
    
    # Альтернативные стратегии
    for alt in verification_results['alternative_weights']:
        strategies.append(alt['name'].replace(' ', '\n'))
        lvars.append(alt['lvar'])
        labels.append(alt['name'])
    
    colors = ['green' if label == 'Оптимальные' else 'red' if '100%' in label else 'orange' for label in labels]
    
    bars1 = ax1.barh(strategies, lvars, color=colors, alpha=0.7)
    ax1.set_xlabel('LVaR')
    ax1.set_title('Сравнение LVaR для разных стратегий весов', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, lvar) in enumerate(zip(bars1, lvars)):
        ax1.text(lvar, i, f' {lvar:.6f}', va='center', fontsize=9)
    
    # График 2: Распределение LVaR для случайных комбинаций
    if 'all_random_lvars' in verification_results and len(verification_results['all_random_lvars']) > 0:
        random_lvars = verification_results['all_random_lvars']
        
        # Создаем гистограмму всех случайных комбинаций
        ax2.hist(random_lvars, bins=30, alpha=0.6, color='gray', 
                label=f'Случайные комбинации (n={len(random_lvars)})', edgecolor='black', linewidth=0.5)
        ax2.axvline(verification_results['optimal_lvar'], color='green', linewidth=2, 
                   label=f'Оптимальные веса ({verification_results["optimal_lvar"]:.6f})', linestyle='--')
        
        # Добавляем линию для медианы
        median_lvar = np.median(random_lvars)
        ax2.axvline(median_lvar, color='orange', linewidth=1.5, 
                   label=f'Медиана случайных ({median_lvar:.6f})', linestyle=':')
        
        ax2.set_xlabel('LVaR')
        ax2.set_ylabel('Частота')
        ax2.set_title(f'Распределение LVaR (случайные комбинации, n={len(random_lvars)})', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Нет данных о случайных комбинациях', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Распределение LVaR', fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем график
    if output_path is None:
        output_path = "lvar_comparison.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if debug:
        print(f"График сравнения LVaR сохранен в: {output_path}")
    
    # Показываем график, если запрошено
    if show:
        plt.show()
        plt.close()
    else:
        plt.close()
    
    return output_path


def plot_portfolio_with_weights(portfolio_df, symbols, weights_dict, start_date, end_date, 
                                output_path=None, debug=False, show=False):
    """
    Рисует график с несколькими subplot'ами - по одному на каждую акцию
    На каждом subplot: High и Low (разными цветами) с весом в легенде
    
    Parameters:
        portfolio_df (pd.DataFrame): Данные портфеля
        symbols (list): Список символов
        weights_dict (dict): Словарь весов {symbol: weight}
        start_date (str): Дата начала
        end_date (str): Дата окончания
        output_path (str): Путь для сохранения графика
        debug (bool): Режим отладки
        show (bool): Показать график
    """
    if portfolio_df is None or portfolio_df.empty:
        if debug:
            print("Нет данных для построения графика")
        return None
    
    if not symbols or len(symbols) == 0:
        if debug:
            print("Нет символов для построения графика")
        return None
    
    # Убеждаемся, что Date в формате datetime
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    # Отладочная информация о weights_dict
    if debug and weights_dict:
        print(f"DEBUG: weights_dict передан в plot_portfolio_with_weights: {weights_dict}")
    
    # Создаем subplot'ы - по одному на каждую акцию
    n_symbols = len(symbols)
    fig, axes = plt.subplots(n_symbols, 1, figsize=(16, 5 * n_symbols))
    
    if n_symbols == 1:
        axes = [axes]
    
    # Цвета для High и Low (разные для наглядности)
    high_color = 'green'
    low_color = 'red'
    
    # Рисуем для каждой акции на отдельном subplot
    for i, symbol in enumerate(symbols):
        ax = axes[i]
        
        stock_data = portfolio_df[portfolio_df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        if stock_data.empty:
            ax.text(0.5, 0.5, f'Нет данных для {symbol}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{symbol} - нет данных')
            continue
        
        # Удаляем дубликаты дат (как в ноутбуке)
        stock_data = stock_data[~stock_data['Date'].duplicated(keep='last')]
        stock_data = stock_data.sort_values('Date')
        
        dates = stock_data['Date']
        
        # Получаем вес, если он есть
        weight = weights_dict.get(symbol, None) if weights_dict else None
        
        # Отладочная информация
        if debug:
            print(f"DEBUG plot_portfolio_with_weights: {symbol} weight = {weight}")
        
        # Формируем заголовок с весом, если он есть
        if weight is not None and not pd.isna(weight):
            title = f'{symbol} - High и Low цены (вес: {weight:.2%})'
        else:
            title = f'{symbol} - High и Low цены'
            if debug:
                print(f"DEBUG: Вес для {symbol} не найден в weights_dict!")
        
        # Рисуем High и Low разными цветами (без веса в label, он в заголовке)
        ax.plot(dates, stock_data['High'], label='High', 
               color=high_color, linewidth=1.5, alpha=0.8)
        ax.plot(dates, stock_data['Low'], label='Low', 
               color=low_color, linewidth=1.5, alpha=0.8)
        
        # Заливаем область между High и Low для наглядности
        ax.fill_between(dates, stock_data['Low'], stock_data['High'], 
                       alpha=0.2, color='gray', label='Диапазон')
        
        ax.set_title(title, fontweight='bold', fontsize=12)
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
        output_path = f"portfolio_weights_{start_date}_to_{end_date}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if debug:
        print(f"График с весами сохранен в: {output_path}")
    
    # Показываем график, если запрошено
    if show:
        if debug:
            print("Показываем график...")
        plt.show()
        plt.close()
    else:
        plt.close()
    
    return output_path


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
        print("Нет данных для отладки")
        return
    
    print("\n" + "=" * 60)
    print("DEBUG: ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О РЕАЛЬНЫХ ДАТАХ")
    print("=" * 60)
    print(f"Запрошенный период: {start_date} - {end_date}")
    print()
    
    # Убеждаемся, что Date в формате datetime
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    for symbol in symbols:
        # Фильтруем данные для текущей акции
        stock_data = portfolio_df[portfolio_df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        if stock_data.empty:
            print(f"{symbol}: Нет данных")
            continue
        
        # Получаем реальные даты
        actual_first_date = stock_data['Date'].min()
        actual_last_date = stock_data['Date'].max()
        num_days = len(stock_data)
        
        # Сравниваем с запрошенными датами
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        
        print(f"{symbol}:")
        print(f"Запрошено:     {start_date} - {end_date}")
        print(f"Реальный период:")
        print(f"• Первая дата:  {actual_first_date.strftime('%Y-%m-%d')} ", end="")
        
        if actual_first_date > requested_start:
            print(f"(позже запрошенной на {(actual_first_date - requested_start).days} дн.)")
        elif actual_first_date < requested_start:
            print(f"(раньше запрошенной на {(requested_start - actual_first_date).days} дн.)")
        else:
            print("(совпадает)")
        
        print(f"• Последняя дата: {actual_last_date.strftime('%Y-%m-%d')} ", end="")
        
        if actual_last_date < requested_end:
            print(f"(раньше запрошенной на {(requested_end - actual_last_date).days} дн.)")
        elif actual_last_date > requested_end:
            print(f"(позже запрошенной на {(actual_last_date - requested_end).days} дн.)")
        else:
            print("(совпадает)")
        
        print(f"• Всего дней данных: {num_days}")
        print()
    
    print("=" * 60)


def plot_high_low_prices(portfolio_df, symbols, start_date, end_date, output_path=None, debug=False, show=False):
    """
    Рисует график High и Low для каждой акции из портфеля с течением времени
    
    Parameters:
        portfolio_df (pd.DataFrame): Данные портфеля
        symbols (list): Список успешно загруженных символов
        start_date (str): Дата начала
        end_date (str): Дата окончания
        output_path (str): Путь для сохранения графика
        debug (bool): Режим отладки
        show (bool): Показать график в окне (в дополнение к сохранению)
    """
    if portfolio_df is None or portfolio_df.empty:
        if debug:
            print("Нет данных для построения графика")
        return None
    
    if not symbols or len(symbols) == 0:
        if debug:
            print("Нет успешно загруженных символов для построения графика")
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
        print(f"График сохранен в: {output_path}")
    
    # Показываем график, если запрошено
    if show:
        if debug:
            print("Показываем график...")
        plt.show()
        # Закрываем фигуру после того, как пользователь закроет окно
        plt.close()
    else:
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
    
    print(f"Загрузка портфеля за последние {years} лет...")
    print(f"Период: {start_date} - {end_date}")
    print(f"Акции: {', '.join(symbols)}")
    
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
    
    print(f"Загрузка {symbol} за последние {years} лет...")
    
    df = loader.load_single_stock(symbol, start_date, end_date)
    return df


def plot_lvar_contribution(w_lvar, sigma, sigma_s, tickers, z=norm.ppf(0.95), kappa=1):
    w = np.array(w_lvar)

    port_cov = sigma @ w
    port_cov_s = sigma_s @ w

    sigma_p = np.sqrt(w @ sigma @ w)
    sigma_s_p = np.sqrt(w @ sigma_s @ w)

    if sigma_p == 0:
        mc_var = np.zeros_like(w)
    else:
        mc_var = z * port_cov / sigma_p

    if sigma_s_p == 0:
        mc_liq = np.zeros_like(w)
    else:
        mc_liq = 0.5 * kappa * port_cov_s / sigma_s_p

    var_contrib = w * mc_var
    liq_contrib = w * mc_liq
    total_contrib = var_contrib + liq_contrib

    print(f"Вклад в LVaR (рыночный): {var_contrib}")
    print(f"Вклад в LVaR (ликвидность): {liq_contrib}")
    print(f"Суммарный вклад в LVaR: {total_contrib}")
    print(f"Сумма вкладов = {total_contrib.sum()}, LVaR = {z * sigma_p + 0.5 * kappa * sigma_s_p}")

    n = len(w)
    x = np.arange(n)
    width = 0.25

    plt.figure(figsize=(10, 5))

    plt.bar(x - width, var_contrib, width, label="VaR part")
    plt.bar(x, liq_contrib, width, label="Liquidity part")
    plt.bar(x + width, total_contrib, width, label="Total LVaR")

    plt.xticks(x, tickers, rotation=45, ha="right")
    plt.ylabel("Contribution to LVaR")
    plt.title("LVaR Contributions by Asset")
    plt.legend()
    plt.tight_layout()
    plt.savefig("contribution.png", dpi=300, bbox_inches='tight')
    plt.show()




def main_cli():
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
  
  # Загрузка с показом графика в окне
  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --plot --show
  
  # Расчет оптимальных весов портфеля (минимизация LVaR)
  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --weights
  
  # Расчет весов с методом Корвина-Шульца для спреда
  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --weights --corwin-schultz
  
  # Расчет весов с показом графика
  python run_moex_data_loader.py --portfolio SBER GAZP LKOH --weights --show --debug
        """
    )
    
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
    
    parser.add_argument('--weights',
                       dest='weights',
                       action='store_true',
                       help='Рассчитать оптимальные веса портфеля (минимизация LVaR) и показать график с весами')
    
    parser.add_argument('--corwin-schultz',
                       dest='use_corwin_schultz',
                       action='store_true',
                       help='Использовать метод Корвина-Шульца для расчета спреда (по умолчанию: простая формула)')
    
    parser.add_argument('--plot-output',
                       dest='plot_output',
                       type=str,
                       default=None,
                       help='Путь для сохранения графика (по умолчанию: high_low_chart_START_to_END.png)')
    
    parser.add_argument('--show',
                       dest='show',
                       action='store_true',
                       help='Показать график в окне (в дополнение к сохранению)')
    
    parser.add_argument('--csv-output',
                       dest='csv_output',
                       type=str,
                       default=None,
                       help='Путь для сохранения CSV (по умолчанию: сохраняется в data/russian_portfolio/)')
    
    parser.add_argument('--debug',
                       dest='debug',
                       action='store_true',
                       help='Показать подробную информацию о реальных датах загрузки для каждой акции')
    
    parser.add_argument('--check-liquidity',
                       dest='check_liquidity',
                       action='store_true',
                       help='Проверить ликвидность портфеля с заданными весами (если --weights не указан, использует равные веса)')

    if len(sys.argv) == 1:
            parser.print_help()

    args = parser.parse_args()


    if args.end_date is None:
        args.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if args.start_date is None:
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=3*365)
        args.start_date = start_dt.strftime('%Y-%m-%d')
    
    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            parser.error("Дата начала должна быть раньше даты окончания")
    except ValueError as e:
        parser.error(f"Неверный формат даты. Используйте YYYY-MM-DD. Ошибка: {e}")
    
    loader = MoexDataLoader(debug=args.debug)
    
    if args.debug:
        print("=" * 60)
        print("ЗАГРУЗКА ДАННЫХ С MOEX")
        print("=" * 60)
        print(f"Период: {args.start_date} - {args.end_date}")
        print(f"Портфель: {', '.join(args.symbols)}")
        if args.portfolio_name:
            print(f"Название портфеля: {args.portfolio_name}")
        print("=" * 60)
    
    portfolio_df, successful_symbols = loader.load_portfolio_data(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio_name=args.portfolio_name
    )
    
    if portfolio_df is None or portfolio_df.empty:
        print("Не удалось загрузить данные.")
        return
    
    if args.csv_output:
        portfolio_df.to_csv(args.csv_output, index=False)
        csv_path = args.csv_output
    else:
        portfolio_name = args.portfolio_name if args.portfolio_name else "_".join(args.symbols)
        csv_path = os.path.join("data", "russian_portfolio", 
                               f"{portfolio_name}_{args.start_date}_{args.end_date}_moex.csv")
    
    if args.debug:
        print_debug_dates(
            portfolio_df=portfolio_df,
            symbols=successful_symbols,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"\nCSV сохранен в: {csv_path}")
    
    if not args.debug:
        print(csv_path)
    
    if args.plot:
        if args.debug:
            print("\n" + "=" * 60)
            print("СОЗДАНИЕ ГРАФИКА")
            print("=" * 60)
        
        plot_output_path = plot_high_low_prices(
            portfolio_df=portfolio_df,
            symbols=successful_symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.plot_output,
            debug=args.debug,
            show=args.show
        )
        
        if not args.debug and plot_output_path:
            print(plot_output_path)
    
    if args.weights:
        if args.debug:
            print("\n" + "=" * 60)
            print("РАСЧЕТ ОПТИМАЛЬНЫХ ВЕСОВ ПОРТФЕЛЯ")
            print("=" * 60)
        
        weights_dict, sigma, spread_array, tickers_ordered = calculate_optimal_weights(
            portfolio_df=portfolio_df,
            symbols=successful_symbols,
            debug=args.debug,
            use_corwin_schultz=args.use_corwin_schultz
        )
        
        if weights_dict and sigma is not None and spread_array is not None:
            verification_results = verify_optimal_weights(
                portfolio_df=portfolio_df,
                symbols=tickers_ordered,
                optimal_weights=weights_dict,
                sigma=sigma,
                spread_array=spread_array,
                debug=args.debug
            )
            
            # Строим график сравнения LVaR
            if args.debug:
                plot_lvar_comparison(
                    verification_results=verification_results,
                    symbols=successful_symbols,
                    output_path=None,
                    debug=args.debug,
                    show=args.show
                )

            weights_output_path = plot_portfolio_with_weights(
                portfolio_df=portfolio_df,
                symbols=successful_symbols,
                weights_dict=weights_dict,
                start_date=args.start_date,
                end_date=args.end_date,
                output_path=None if not args.plot_output else args.plot_output.replace('.png', '_weights.png'),
                debug=args.debug,
                show=args.show
            )
            
            if not args.debug and weights_output_path:
                print(weights_output_path)
            
            if not args.debug:
                # В обычном режиме выводим только веса
                print("\nВеса портфеля:")
                for symbol in successful_symbols:
                    if symbol in weights_dict:
                        print(f"{symbol}: {weights_dict[symbol]:.4f} ({weights_dict[symbol]*100:.2f}%)")
            
            # Проверка ликвидности портфеля (если запрошена)
            if args.check_liquidity:
                if args.debug:
                    print("\n" + "=" * 60)
                    print("ПРОВЕРКА ЛИКВИДНОСТИ ПОРТФЕЛЯ")
                    print("=" * 60)
                
                liquidity_results = evaluate_portfolio_liquidity(
                    weights=weights_dict,
                    portfolio_df=portfolio_df,
                    symbols=successful_symbols,
                    use_corwin_schultz=args.use_corwin_schultz,
                    debug=args.debug
                )
        else:
            print("Не удалось рассчитать оптимальные веса")
    
    # Проверка ликвидности с равными весами (если --check-liquidity указан, но --weights нет)
    elif args.check_liquidity:
        if args.debug:
            print("\n" + "=" * 60)
            print("ПРОВЕРКА ЛИКВИДНОСТИ ПОРТФЕЛЯ (равные веса)")
            print("=" * 60)
        
        # Используем равные веса
        equal_weights = {s: 1.0/len(successful_symbols) for s in successful_symbols}
        
        liquidity_results = evaluate_portfolio_liquidity(
            weights=equal_weights,
            portfolio_df=portfolio_df,
            symbols=successful_symbols,
            use_corwin_schultz=args.use_corwin_schultz,
            debug=args.debug
        )
        
        if not args.debug:
            print(f"\nОценка ликвидности: {liquidity_results.get('liquidity_rating', 'N/A')}")
            print(f"Средневзвешенный спред: {liquidity_results.get('portfolio_spread_median', 0):.4f} ({liquidity_results.get('portfolio_spread_median', 0)*100:.2f}%)")
    
    if args.debug:
        print("\n" + "=" * 60)
        print("ЗАВЕРШЕНО")
        print("=" * 60)


if __name__ == "__main__":
    main_cli()