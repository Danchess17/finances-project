# data_manager.py
import pandas as pd
import os
from datetime import datetime

class DataManager:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self._create_output_dir()
    
    def _create_output_dir(self):
        """Создаем директорию для выходных файлов если её нет"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_to_csv(self, df, symbol, market):
        """Сохраняем DataFrame в CSV файл"""
        if df is not None and not df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{market}_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"Данные сохранены в файл: {filepath}")
            return filepath
        else:
            print("Нет данных для сохранения")
            return None
    
    def merge_multiple_stocks(self, dataframes, output_filename=None):
        """Объединяет данные по нескольким акциям в один CSV"""
        if not dataframes:
            print("Нет данных для объединения")
            return None
        
        valid_dfs = [df for df in dataframes if df is not None and not df.empty]
        
        if not valid_dfs:
            print("Нет валидных данных для объединения")
            return None
        
        merged_df = pd.concat(valid_dfs, ignore_index=True)
        
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"merged_stocks_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, output_filename)
        merged_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Объединенные данные сохранены в: {filepath}")
        return filepath