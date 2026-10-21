import csv
import os

class IMUFileReader:
    def __init__(self, filepath):
        self.data = []
        self.last_processed_index = 0
        self.time_offset = 0.0
        
        if not os.path.exists(filepath):
            print(f"⚠️ Файл IMU не найден: {filepath}")
            return

        with open(filepath, 'r') as f:
            # Читаем файл построчно, игнорируя стандартный CSV-парсер для данных
            lines = f.readlines()
            
            # Пропускаем заголовок (первая строка)
            for line in lines[1:]:
                line = line.strip()
                if not line: 
                    continue
                    
                try:
                    # 1. Разбиваем строку по запятым (даже если они внутри кавычек или чисел)
                    parts = line.split(',')

                    # 2. Собираем ВСЕ валидные числа из этой строки в один список
                    nums = []
                    for p in parts:
                        p = p.strip().replace('"', '') # Убираем возможные кавычки
                        if p:
                            try:
                                # Пытаемся превратить кусок в число
                                val = float(p.replace(',', '.'))
                                nums.append(val)
                            except ValueError:
                                pass

                    if len(nums) >= 4:
                        z_val = nums[-3]      # Gyro Z
                        elapsed_val = nums[-4] # Seconds Elapsed
                        self.data.append({'raw_t': elapsed_val, 'yaw_rate': z_val})
                    break
                except Exception as e:
                    # print(f"Ошибка парсинга строки: {line[:50]}... {e}")
                    continue
        
        # === АВТОМАТИЧЕСКАЯ КАЛИБРОВКА ВРЕМЕНИ ===
        if self.data:
            # Берем время первой записи как точку отсчета (t=0 для видео)
            self.time_offset = self.data[0]['raw_t']
            
            # Пересчитываем все времена относительно начала
            for item in self.data:
                item['t'] = item['raw_t'] - self.time_offset
                
            print(f"✅ Загружено {len(self.data)} точек гироскопа из {filepath}")
            print(f"   ⏱️ Автоматическая синхронизация времени:")
            print(f"      Начало в файле: {self.data[0]['raw_t']:.4f} сек")
            print(f"      Применен сдвиг (offset): {self.time_offset:.4f} сек")
            print(f"      Новое начало для SLAM: {self.data[0]['t']:.4f} сек")
        else:
            print("⚠️ Данные IMU пусты. Проверьте формат CSV.")

    def get_yaw_delta(self, start_time, end_time):
        """
        Возвращает СУММАРНЫЙ угол поворота (в радианах) 
        за интервал [start_time, end_time].
        """
        if not self.data:
            print('пусто')
            return 0.0
            
        total_yaw_rad = 0.0
        i = self.last_processed_index
        
        while i < len(self.data):
            point_t = self.data[i]['t'] # Используем уже скорректированное время
            
            # Если точка еще не наступила (будущее), выходим
            if point_t > end_time:
                break
                
            # Если точка уже прошла предыдущий кадр, пропускаем её
            if point_t <= start_time:
                i += 1
                continue
                
            # Точка находится внутри текущего интервала.
            # Вычисляем dt (разницу с предыдущей точкой в файле для точности)
            if i > 0:
                dt = self.data[i]['t'] - self.data[i-1]['t']
            else:
                dt = 0.01 # Заглушка для самой первой точки
                
            # Интегрируем: Угол = Скорость * Время
            total_yaw_rad += self.data[i]['yaw_rate'] * dt
            i += 1
            
        # Обновляем индекс, чтобы следующий вызов начался с этого места
        self.last_processed_index = i
        return total_yaw_rad