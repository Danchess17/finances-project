# config.py
import requests

# Общие настройки
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Российские акции для автоматического определения рынка
RUSSIAN_SYMBOLS = [
    'GAZP', 'SBER', 'LKOH', 'ROSN', 'GMKN', 'MTSS', 'NVTK', 
    'TATN', 'MGNT', 'PLZL', 'AFKS', 'VTBR', 'ALRS', 'POLY', 'MOEX',
    'YNDX', 'TCSG', 'OZON', 'POSI', 'PHOR', 'DSKY', 'LSRG'
]

# URL-адреса API
INVESTING_BASE_URL = "https://api.investing.com/api/financialdata"
INVESTING_SEARCH_URL = "https://www.investing.com/search/service/search"
MOEX_BASE_URL = "https://iss.moex.com/iss"

# Настройки запросов
REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 1