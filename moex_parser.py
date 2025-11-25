# moex_parser.py
import requests
import pandas as pd
import time
from tqdm import tqdm
from config import REQUEST_HEADERS, MOEX_BASE_URL, REQUEST_TIMEOUT

class MoexParser:
    def __init__(self, debug=False):
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
    
    def _print(self, *args, **kwargs):
        """Выводит сообщение только если включен debug режим"""
        if self.debug:
            print(*args, **kwargs)
    
    def check_security_exists(self, symbol):
        """Проверяем существует ли ценная бумага"""
        url = f"{MOEX_BASE_URL}/securities/{symbol}.json"
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            data = response.json()
            
            # Если есть описание или площадки - бумага существует
            has_description = 'description' in data and data['description']['data']
            has_boards = 'boards' in data and data['boards']['data']
            
            return has_description or has_boards
            
        except Exception as e:
            self._print(f"Ошибка при проверке {symbol}: {e}")
            return False
    
    def get_security_boards(self, symbol):
        """Получаем доступные торговые площадки для бумаги"""
        url = f"{MOEX_BASE_URL}/securities/{symbol}.json"
        
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            data = response.json()
            
            boards = []
            if 'boards' in data:
                for board in data['boards']['data']:
                    if len(board) >= 4:
                        board_id = board[0]
                        board_group = board[2] if board[2] else 'UNKNOWN'
                        board_name = board[3] if board[3] else 'UNKNOWN'
                        
                        boards.append({
                            'id': board_id,
                            'group': board_group,
                            'name': board_name
                        })
            
            return boards
            
        except Exception as e:
            self._print(f"Ошибка при получении площадок для {symbol}: {e}")
            return []
    
    def parse_stock_data(self, symbol, start_date, end_date):
        """Парсим исторические данные с MOEX"""
        self._print(f"\nПоиск данных для {symbol}...")
        
        # Проверяем существует ли бумага
        if not self.check_security_exists(symbol):
            self._print(f"Ценная бумага {symbol} не найдена на MOEX")
            return None
        
        self._print(f"Бумага {symbol} найдена на MOEX")
        
        # Получаем информацию о площадках
        boards = self.get_security_boards(symbol)
        if boards:
            self._print(f"Найдено торговых площадок: {len(boards)}")
            for board in boards[:3]:  # Показываем первые 3
                self._print(f"- {board['id']} ({board['group']}): {board['name']}")
        
        # Форматируем даты
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self._print(f"Период: {start_str} - {end_str}")
        
        # Получаем исторические данные с пагинацией
        # MOEX API возвращает максимум 100 записей за запрос, нужна пагинация
        url = f"{MOEX_BASE_URL}/history/engines/stock/markets/shares/securities/{symbol}.json"
        
        all_rows = []
        columns = None
        start_index = 0
        page_size = 100  # MOEX возвращает максимум 100 записей за раз
        
        try:
            self._print(f"Запрос к MOEX API (с пагинацией)...")
            
            # Инициализируем прогресс-бар
            pbar = None
            last_count = 0
            if self.debug:
                # Создаем прогресс-бар без общего количества (будет обновлен позже)
                pbar = tqdm(total=None, desc=f"Загрузка {symbol}", unit=" записей", 
                           bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                           disable=False, ncols=100, dynamic_ncols=True)
            
            while True:
                params = {
                    'from': start_str,
                    'till': end_str,
                    'iss.meta': 'on',  # Включаем метаданные для получения информации о пагинации
                    'limit': page_size,
                    'start': start_index
                }
                
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                data = response.json()
                
                if 'history' not in data:
                    if start_index == 0:
                        self._print("В ответе нет исторических данных")
                        if pbar:
                            pbar.close()
                        return None
                    break  # Если на первой странице нет данных - ошибка, иначе закончили пагинацию
                
                # Получаем колонки (они одинаковые для всех страниц)
                if columns is None:
                    columns = data['history']['columns']
                
                page_rows = data['history']['data']
                
                if not page_rows:
                    break  # Больше нет данных
                
                all_rows.extend(page_rows)
                
                # Обновляем прогресс-бар
                if pbar is not None:
                    # Обновляем на количество новых записей
                    new_count = len(all_rows) - last_count
                    if new_count > 0:
                        pbar.update(new_count)
                        last_count = len(all_rows)
                
                # Проверяем, есть ли еще данные и общее количество
                total_records = None
                if 'history.cursor' in data and 'data' in data['history.cursor']:
                    cursor_data = data['history.cursor']['data']
                    if cursor_data and len(cursor_data) > 0:
                        # Формат: [INDEX, TOTAL, PAGESIZE]
                        total_records = cursor_data[0][1] if len(cursor_data[0]) > 1 else None
                        
                        # Обновляем общее количество в прогресс-баре
                        if pbar is not None and total_records:
                            if pbar.total is None or pbar.total != total_records:
                                pbar.total = total_records
                
                # Если получили меньше записей, чем запросили - это последняя страница
                if len(page_rows) < page_size:
                    break
                
                start_index += page_size
                
                # Небольшая задержка между запросами страниц (чтобы не перегружать API)
                if len(page_rows) >= page_size:
                    time.sleep(0.5)
                
                # Защита от бесконечного цикла
                if len(all_rows) >= 10000:  # Максимум 10000 записей
                    self._print("Достигнут максимум 10000 записей, прекращаем загрузку")
                    break
            
            # Закрываем прогресс-бар
            if pbar is not None:
                # Обновляем до финального значения, если нужно
                final_count = len(all_rows) - last_count
                if final_count > 0:
                    pbar.update(final_count)
                if pbar.total is None:
                    pbar.total = len(all_rows)
                pbar.close()
            
            rows = all_rows
            
            # Не выводим "Всего получено" если был прогресс-бар (он уже показал финальное значение)
            if not self.debug:
                self._print(f"Всего получено {len(rows)} записей")
            
            if not rows:
                self._print("Нет данных за указанный период")
                return None
            
            # Парсим данные
            df_data = []
            for row in rows:
                if len(row) != len(columns):
                    continue
                    
                row_dict = dict(zip(columns, row))
                
                # Безопасно извлекаем данные
                try:
                    trade_date = row_dict.get('TRADEDATE')
                    open_price = row_dict.get('OPEN')
                    high_price = row_dict.get('HIGH')
                    low_price = row_dict.get('LOW')
                    close_price = row_dict.get('CLOSE')
                    
                    # Проверяем что все основные поля есть
                    if all([trade_date, open_price is not None, high_price is not None, 
                           low_price is not None, close_price is not None]):
                        df_data.append({
                            'Date': trade_date,
                            'Open': float(open_price) if open_price else 0,
                            'High': float(high_price) if high_price else 0,
                            'Low': float(low_price) if low_price else 0,
                            'Close': float(close_price) if close_price else 0,
                            'Volume': int(row_dict.get('VOLUME', 0)) if row_dict.get('VOLUME') else 0,
                            'Value': float(row_dict.get('VALUE', 0)) if row_dict.get('VALUE') else 0,
                        })
                except (ValueError, TypeError) as e:
                    continue  # Пропускаем проблемные записи
            
            if not df_data:
                self._print("Не удалось извлечь корректные данные")
                return None
            
            df = pd.DataFrame(df_data)
            df['Symbol'] = symbol
            df['Source'] = 'MOEX'
            
            # Сортируем по дате
            df = df.sort_values('Date')
            
            self._print(f"Успешно обработано {len(df)} записей")
            self._print(f"Диапазон дат: {df['Date'].min()} - {df['Date'].max()}")
            
            return df
            
        except Exception as e:
            self._print(f"Ошибка при получении данных: {e}")
            return None
    
    def test_popular_tickers(self):
        """Тестируем популярные тикеры"""
        popular_tickers = [
            'SBER', 'SBERP', 'GAZP', 'LKOH', 'ROSN', 'GMKN', 
            'NLMK', 'TATN', 'TATNP', 'MTSS', 'NVTK', 'MGNT',
            'YNDX', 'TCSG', 'OZON', 'VTBR', 'ALRS', 'POLY'
        ]
        
        print("Проверка популярных тикеров на MOEX:")
        
        working_tickers = []
        for ticker in popular_tickers:
            exists = self.check_security_exists(ticker)
            status = "" if exists else ""
            print(f"{status} {ticker}")
            
            if exists:
                working_tickers.append(ticker)
        
        print(f"\n Итого: {len(working_tickers)} из {len(popular_tickers)} тикеров работают")
        return working_tickers