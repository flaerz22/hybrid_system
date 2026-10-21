#!/usr/bin/env python3
"""
IMU Stream Simulator (Real-time Sync + Pass-through)
Симулирует поступление данных гироскопа в реальном времени 
с ТОЧНЫМ механизмом sleep из OBD-симулятора.
Сохраняет исходную структуру данных (сплюснутый CSV).
"""

import os
import time
from pathlib import Path


class IMUStreamSimulator:
    def __init__(self, input_file, output_folder, clean_start=True, verbose=False):
        self.input_file = Path(input_file)
        self.output_folder = Path(output_folder)
        self.clean_start = clean_start
        self.verbose = verbose
        self.running = False

    def _clean_output_folder(self):
        """Очищает папку вывода от старых логов IMU."""
        if self.output_folder.exists():
            for item in self.output_folder.iterdir():
                if item.is_file() and item.name.startswith("imu_stream_log"):
                    try:
                        item.unlink()
                    except PermissionError:
                        pass
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _parse_elapsed_from_line(self, line):
        """
        Надежно извлекает seconds_elapsed из 'сплюснутой' строки.
        Формат: "time_ns",elapsed,z,y,x
        Учитывает, что time_ns может быть разбит запятой внутри кавычек.
        """
        # Убираем кавычки и пробелы, разбиваем по запятым
        parts = line.strip().replace('"', '').split(',')
        
        # Собираем все валидные числа из строки
        nums = []
        for p in parts:
            try:
                # Заменяем запятую на точку на случай локали, хотя split уже разбил
                val = float(p.replace(',', '.'))
                nums.append(val)
            except ValueError:
                continue
        
        # Эвристика для вашего формата:
        # Если time_ns разбит запятой -> [part1, part2, elapsed, z, y, x] (len >= 6)
        # Если time_ns целое -> [unix, elapsed, z, y, x] (len == 5)
        if len(nums) >= 6:
            return nums[2]  # elapsed - третье число
        elif len(nums) >= 4:
            return nums[1]  # elapsed - второе число
        return None

    def start(self):
        """Запускает симуляцию с точной привязкой к реальному времени."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.input_file}")

        if self.clean_start:
            self._clean_output_folder()

        output_file = self.output_folder / "imu_stream_log.csv"
        self.running = True
        
        # === ФИКСАЦИЯ ВРЕМЕНИ СТАРТА (как в OBD-симуляторе) ===
        start_time_real = time.time()
        base_data_time = None
        
        print(f"🚀 Запуск симуляции IMU потока (Real-time Sync)...")
        print(f"   Вход:  {self.input_file}")
        print(f"   Выход: {output_file}")

        records_count = 0

        try:
            with open(self.input_file, 'r', encoding='utf-8') as f_in, \
                 open(output_file, 'w', encoding='utf-8') as f_out:

                # Копируем заголовок без изменений
                header_line = f_in.readline()
                if not header_line:
                    raise ValueError("Исходный файл пуст")
                f_out.write(header_line)

                # Обрабатываем строки данных по одной
                for line_num, raw_line in enumerate(f_in, start=2):
                    if not self.running:
                        break
                    
                    stripped_line = raw_line.strip()
                    if not stripped_line:
                        continue

                    # --- Парсинг ТОЛЬКО для вычисления sleep ---
                    current_elapsed = self._parse_elapsed_from_line(stripped_line)
                    
                    if current_elapsed is not None:
                        # Устанавливаем базовое время по первой точке
                        if base_data_time is None:
                            base_data_time = current_elapsed
                        
                        # === МЕХАНИЗМ SLEEP ИЗ OBD-СИМУЛЯТОРА ===
                        # target_offset = сколько времени прошло от начала данных
                        target_offset = current_elapsed - base_data_time
                        # current_elapsed_real = сколько времени прошло от запуска скрипта
                        current_elapsed_real = time.time() - start_time_real
                        # sleep_time = разница между целевым и фактическим временем
                        sleep_time = target_offset - current_elapsed_real
                        
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        # ==========================================

                    # --- ЗАПИСЬ ДАННЫХ КАК ЕСТЬ (Pass-through) ---
                    f_out.write(raw_line)
                    f_out.flush()
                    
                    records_count += 1
                    
                    if self.verbose and (records_count % 50 == 0 or current_elapsed is not None):
                        actual_drift = (time.time() - start_time_real) - (current_elapsed - base_data_time if base_data_time else 0)
                        print(f"[{records_count}] t={current_elapsed:.4f}s | Sleep={sleep_time:.4f}s | Drift={actual_drift*1000:.1f}ms")

        except KeyboardInterrupt:
            print("\n⏹️ Симуляция остановлена пользователем.")
        finally:
            self.running = False
            
            # === ВЫВОД ИТОГОВОГО ВРЕМЕНИ РАБОТЫ ===
            total_duration = time.time() - start_time_real
            print(f"\n✅ Симуляция завершена!")
            print(f"   Обработано записей: {records_count}")
            print(f"   ⏱️ Реальное время работы: {total_duration:.2f} сек ({total_duration/60:.1f} мин)")
            
            if base_data_time is not None and current_elapsed is not None:
                data_duration = current_elapsed - base_data_time
                error_pct = abs(total_duration - data_duration) / max(data_duration, 0.001) * 100
                print(f"   📊 Длительность данных: {data_duration:.2f} сек")
                print(f"   🎯 Точность синхронизации: {error_pct:.3f}%")
                
            print(f"   Файл сохранен: {os.path.abspath(output_file)}")

    def stop(self):
        self.running = False


if __name__ == '__main__':
    simulator = IMUStreamSimulator(
        input_file='data/imu/Gyroscope.csv',
        output_folder='output',
        clean_start=True,
        verbose=True
    )
    try:
        simulator.start()
    except Exception as e:
        print(f"❌ Ошибка: {e}")