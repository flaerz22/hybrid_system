#!/usr/bin/env python3
"""
OBD Stream Simulator
Симулирует поступление данных скорости с автомобиля в реальном времени.
Читает данные из Excel и записывает их в CSV с соблюдением временных интервалов.
"""
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

class OBDStreamSimulator:
    """Симулятор потока данных OBD в реальном времени."""
    
    def __init__(self, input_file, output_folder, clean_start=True, verbose=False):
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)
        self.clean_start = clean_start
        self.verbose = verbose
        self.data = []
        self.running = False
        
    def prepare_data(self):
        """Загружает данные из Excel файла."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.input_file}")
            
        print(f"📂 Чтение данных из {self.input_file}...")
        try:
            df = pd.read_excel(self.input_file)
            # Ожидаемые колонки: timestamp_unix, time_msk, speed_kmh
            required_cols = ['timestamp_unix', 'time_msk', 'speed_kmh']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Файл должен содержать колонки: {required_cols}")
                
            self.data = df.to_dict('records')
            if self.verbose:
                print(f"   Загружено записей: {len(self.data)}")
                if len(self.data) > 0:
                    print(f"   Начало: {self.data[0]['time_msk']}")
                    print(f"   Конец:  {self.data[-1]['time_msk']}")
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения Excel: {e}")

    def _clean_output_folder(self):
        """Очищает папку вывода или удаляет старый лог."""
        if self.output_folder.exists():
            # Удаляем только файлы логов скорости, чтобы не трогать другие данные
            for item in self.output_folder.iterdir():
                if item.is_file() and item.name.startswith("obd_sim_log"):
                    item.unlink()
                    if self.verbose:
                        print(f"🗑️  Удален старый лог: {item.name}")
        
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Запускает симуляцию записи данных."""
        self.prepare_data()
        
        if self.clean_start:
            self._clean_output_folder()
            
        # Формируем имя нового файла с таймстампом запуска
        start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_folder / f"obd_sim_log.csv"
        
        self.running = True
        print(f"\n🚀 Запуск симуляции OBD потока...")
        print(f"   Вход:  {self.input_file}")
        print(f"   Выход: {output_file}")
        print(f"   Режим: {'Чистый старт' if self.clean_start else 'Добавление'}")
        
        start_time_real = time.time()
        
        # Если данных нет, выходим
        if not self.data:
            print("⚠️ Нет данных для симуляции.")
            return

        # Базовое время из первой записи данных
        base_data_time = float(self.data[0]['timestamp_unix'])
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                # Записываем заголовок
                f.write("timestamp_unix,time_msk,speed_kmh\n")
                
                total_records = len(self.data)
                
                for i, row in enumerate(self.data):
                    if not self.running:
                        break
                    
                    # 1. Вычисляем, когда должна появиться эта запись относительно старта
                    # target_offset = (время_записи - время_первой_записи)
                    target_offset = float(row['timestamp_unix']) - base_data_time
                    
                    # 2. Ждем нужного момента
                    current_elapsed = time.time() - start_time_real
                    sleep_time = target_offset - current_elapsed
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    # 3. Записываем данные в файл
                    # Используем оригинальные данные из Excel
                    line = f"{row['timestamp_unix']},{row['time_msk']},{row['speed_kmh']}\n"
                    f.write(line)
                    f.flush() # Принудительно сбрасываем буфер на диск
                    
                    # 4. Вывод в консоль (каждые 10 записей или последняя)
                    if self.verbose and (i % 10 == 0 or i == total_records - 1):
                        print(f"[{i+1:4d}/{total_records}] "
                              f"Time: {row['time_msk']} | "
                              f"Speed: {row['speed_kmh']:5.1f} km/h")
                              
        except KeyboardInterrupt:
            print("\n⏹️  Симуляция остановлена пользователем.")
        finally:
            self.running = False
            total_time = time.time() - start_time_real
            print(f"\n✅ Симуляция завершена!")
            print(f"   Обработано записей: {len(self.data)}")
            print(f"   Длительность симуляции: {total_time:.1f}с")
            print(f"   Файл сохранен: {os.path.abspath(output_file)}")

    def stop(self):
        self.running = False

if __name__ == '__main__':
    # === НАСТРОЙКИ ===
    INPUT_FILE = 'data/obd_speed_sync_log_3.xlsx'  # Путь к вашему Excel файлу
    OUTPUT_FOLDER = 'output'                     # Папка для сохранения логов
    CLEAN_START = True                           # Удалять старые логи при старте
    VERBOSE = True                               # Подробный вывод в консоль
    
    simulator = OBDStreamSimulator(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_FOLDER,
        clean_start=CLEAN_START,
        verbose=VERBOSE
    )
    
    try:
        simulator.start()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")