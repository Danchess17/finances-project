# moex_parser.py
import requests
import pandas as pd
from config import REQUEST_HEADERS, MOEX_BASE_URL, REQUEST_TIMEOUT

class MoexParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
    
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
            print(f"Ошибка при проверке {symbol}: {e}")
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
            print(f"Ошибка при получении площадок для {symbol}: {e}")
            return []
    
    def parse_stock_data(self, symbol, start_date, end_date):
        """Парсим исторические данные с MOEX"""
        print(f"\n🔍 Поиск данных для {symbol}...")
        
        # Проверяем существует ли бумага
        if not self.check_security_exists(symbol):
            print(f"❌ Ценная бумага {symbol} не найдена на MOEX")
            return None
        
        print(f"✅ Бумага {symbol} найдена на MOEX")
        
        # Получаем информацию о площадках
        boards = self.get_security_boards(symbol)
        if boards:
            print(f"📊 Найдено торговых площадок: {len(boards)}")
            for board in boards[:3]:  # Показываем первые 3
                print(f"   - {board['id']} ({board['group']}): {board['name']}")
        
        # Форматируем даты
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"📅 Период: {start_str} - {end_str}")
        
        # Получаем исторические данные
        url = f"{MOEX_BASE_URL}/history/engines/stock/markets/shares/securities/{symbol}.json"
        params = {
            'from': start_str,
            'till': end_str,
            'iss.meta': 'off',
            'limit': 1000
        }
        
        try:
            print(f"🌐 Запрос к MOEX API...")
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            data = response.json()
            
            if 'history' not in data:
                print("❌ В ответе нет исторических данных")
                return None
            
            rows = data['history']['data']
            columns = data['history']['columns']
            
            print(f"📈 Получено {len(rows)} записей")
            
            if not rows:
                print("⚠️ Нет данных за указанный период")
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
                print("❌ Не удалось извлечь корректные данные")
                return None
            
            df = pd.DataFrame(df_data)
            df['Symbol'] = symbol
            df['Source'] = 'MOEX'
            
            # Сортируем по дате
            df = df.sort_values('Date')
            
            print(f"✅ Успешно обработано {len(df)} записей")
            print(f"📊 Диапазон дат: {df['Date'].min()} - {df['Date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка при получении данных: {e}")
            return None
    
    def test_popular_tickers(self):
        """Тестируем популярные тикеры"""
        popular_tickers = [
            'SBER', 'SBERP', 'GAZP', 'LKOH', 'ROSN', 'GMKN', 
            'NLMK', 'TATN', 'TATNP', 'MTSS', 'NVTK', 'MGNT',
            'YNDX', 'TCSG', 'OZON', 'VTBR', 'ALRS', 'POLY'
        ]
        
        print("🔍 Проверка популярных тикеров на MOEX:")
        
        working_tickers = []
        for ticker in popular_tickers:
            exists = self.check_security_exists(ticker)
            status = "✅" if exists else "❌"
            print(f"  {status} {ticker}")
            
            if exists:
                working_tickers.append(ticker)
        
        print(f"\n📊 Итого: {len(working_tickers)} из {len(popular_tickers)} тикеров работают")
        return working_tickers