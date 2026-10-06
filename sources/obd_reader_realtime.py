# sources/obd_reader.py
import os
import time

class RealtimeOBDReader:
    """Читает последнюю запись скорости из CSV файла, генерируемого симулятором."""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.last_speed = 0.0
        self.last_timestamp = 0.0
        
    def get_current_speed(self):
        """Возвращает последнюю известную скорость из файла."""
        if not os.path.exists(self.csv_path):
            return 0.0
            
        try:
            # Читаем файл с конца для эффективности
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 2: # Только заголовок или пусто
                    return 0.0
                
                last_line = lines[-1].strip()
                parts = last_line.split(',')
                
                if len(parts) >= 3:
                    self.last_timestamp = float(parts[0])
                    self.last_speed = float(parts[2])
                    
        except Exception as e:
            # Если файл занят симулятором, возвращаем последнее известное значение
            pass
            
        return self.last_speed