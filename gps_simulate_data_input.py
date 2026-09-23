#!/usr/bin python3
"""
GPS Realtime Simulator
Читает CSV с GPS-данными и записывает их в новый файл с реальными временными интервалами,
имитируя работу GPS-приемника в реальном времени.
"""

import csv
import time
from datetime import datetime
from pathlib import Path


def parse_gps_file(input_file, target_id=None):
    """
    Читает исходный CSV файл и возвращает список записей с вычисленными интервалами.
    
    :param input_file: путь к исходному CSV
    :param target_id: фильтр по id_eqitem (если None — берем все)
    :return: список словарей с полями id_eqitem, date_creation, pos, delay
    """
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Фильтрация по id_eqitem
            if target_id is not None and int(row['id_eqitem']) != target_id:
                continue
            
            # Парсим время
            try:
                dt = datetime.strptime(row['date_creation'], '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"⚠️  Ошибка парсинга времени: {row['date_creation']} — {e}")
                continue
            
            records.append({
                'id_eqitem': row['id_eqitem'],
                'date_creation': dt,
                'pos': row['pos'],
                'date_str': row['date_creation'],
                'lat': row['lat'],
                'lon': row['lon']
            })
    
    # Сортируем по времени (на случай если данные перемешаны)
    records.sort(key=lambda x: x['date_creation'])
    
    # Вычисляем интервалы между записями
    for i in range(len(records)):
        if i == 0:
            records[i]['delay'] = 0  # Первая запись — без задержки
        else:
            delta = (records[i]['date_creation'] - records[i-1]['date_creation']).total_seconds()
            records[i]['delay'] = delta
    
    return records


def simulate_gps_recording(input_file, output_file, target_id=None, speed_multiplier=1.0):
    """
    Симулирует запись GPS-данных в реальном времени.
    
    :param input_file: путь к исходному CSV
    :param output_file: путь для записи симулированных данных
    :param target_id: фильтр по id_eqitem
    :param speed_multiplier: множитель скорости (1.0 = реальное время, 2.0 = в 2 раза быстрее)
    """
    print("=" * 60)
    print("GPS REALTIME SIMULATOR")
    print("=" * 60)
    print(f" Исходный файл: {input_file}")
    print(f"📝 Выходной файл: {output_file}")
    if target_id:
        print(f"🎯 Фильтр id_eqitem: {target_id}")
    print(f"⚡ Множитель скорости: {speed_multiplier}x")
    print()
    
    # Читаем данные
    records = parse_gps_file(input_file, target_id)
    
    if not records:
        print("❌ Нет данных для обработки!")
        return
    
    print(f"📊 Загружено записей: {len(records)}")
    
    # Статистика по интервалам
    delays = [r['delay'] for r in records if r['delay'] > 0]
    if delays:
        print(f" Статистика интервалов:")
        print(f"   Минимальный: {min(delays):.1f}с")
        print(f"   Максимальный: {max(delays):.1f}с")
        print(f"   Средний: {sum(delays)/len(delays):.1f}с")
    print()
    
    # Создаем выходной файл с заголовком
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id_eqitem', 'date_creation', 'pos', 'lat', 'lon'])
    
    # Запускаем симуляцию
    print("🚀 Запуск симуляции...")
    print(f"   Начало: {records[0]['date_str']}")
    print(f"   Конец:  {records[-1]['date_str']}")
    print()
    
    start_time = time.time()
    simulated_start = records[0]['date_creation']
    
    for i, record in enumerate(records):
        # Ждем нужный интервал (с учетом множителя скорости)
        if record['delay'] > 0:
            wait_time = record['delay'] / speed_multiplier
            time.sleep(wait_time)
        
        # Вычисляем "симулированное" текущее время
        elapsed = time.time() - start_time
        
        # Записываем в файл
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record['id_eqitem'],
                record['date_str'],
                record['pos'],
                record['lat'],
                record['lon']
            ])
            f.flush()  # Принудительная запись на диск
        
        # Выводим прогресс
        print(f"[{i+1:4d}/{len(records)}] "
              f"Время: {record['date_str']} | "
              f"ID: {record['id_eqitem']} | "
              f"Интервал: {record['delay']:.1f}с | "
              f"Pos: {record['pos'][:30]}...")
    
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("✅ Симуляция завершена!")
    print(f"   Записано точек: {len(records)}")
    print(f"   Реальное время работы: {total_time:.1f}с")
    print(f"   Симулированный период: {records[-1]['delay'] if len(records) > 1 else 0:.0f}с")
    print(f"   Файл сохранен: {output_file}")
    print("=" * 60)


# =============================================================================
# ЗАПУСК
# =============================================================================

if __name__ == '__main__':
    INPUT_FILE = 'data/test_gps_filtered_123682.csv'           # Исходный файл
    OUTPUT_FILE = 'data/gps_realtime_sim_123682.csv'  # Выходной файл
    TARGET_ID = 5                          # Фильтр по id_eqitem (None = все)
    SPEED_MULTIPLIER = 1.0                 # 1.0 = реальное время, 10.0 = в 10 раз быстрее
    
    simulate_gps_recording(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        target_id=TARGET_ID,
        speed_multiplier=SPEED_MULTIPLIER
    )