#!/usr/bin python3
"""
GPS Monitor Module
Модуль мониторинга GPS-потока с автоматическим переключением на vSLAM.

Использование:
    from gps_monitor import GPSMonitor
    
    monitor = GPSMonitor('gps_data.csv', timeout_seconds=3.0)
    monitor.start()
    
    # В главном цикле:
    channel = monitor.get_channel()  # 'GPS' или 'VSLAM'
    if channel == 'GPS':
        # Записываем GPS данные
        gps_data = monitor.get_last_gps()
    else:
        # Запускаем vSLAM
        pass
    
    monitor.stop()
"""

import time
import threading
from pathlib import Path


class GPSMonitor:
    """
    Монитор GPS-потока с автоматическим переключением на vSLAM.
    
    Атрибуты:
        state: текущее состояние ('WAITING_FIRST_GPS', 'GPS_OK', 'VSLAM_ACTIVE')
        last_gps_record: последняя GPS запись (словарь с полями id_eqitem, date_creation, pos)
    """
    
    def __init__(self, gps_file, timeout_seconds=4.0, check_interval=0.5, verbose=True):
        """
        :param gps_file: путь к файлу с GPS данными
        :param timeout_seconds: таймаут отсутствия GPS (секунды)
        :param check_interval: интервал проверки файла (секунды)
        :param verbose: выводить ли логи в консоль
        """
        self.gps_file = gps_file
        self.timeout = timeout_seconds
        self.check_interval = check_interval
        self.verbose = verbose
        
        # Состояние системы
        self.state = 'WAITING_FIRST_GPS'
        self.last_gps_time = None
        self.last_gps_record = None
        self.file_position = 0
        
        # Флаги управления
        self.running = False
        self.monitor_thread = None
        
        # Callback (опционально)
        self.on_switch = None
        
        # Статистика
        self.total_gps_records = 0
        self.switch_count = 0
    
    def start(self):
        """Запускает мониторинг в отдельном потоке"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.verbose:
            print(f"🟢 GPS Monitor запущен (файл: {self.gps_file}, таймаут: {self.timeout}с)")
    
    def stop(self):
        """Останавливает мониторинг"""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if self.verbose:
            print(f"🔴 GPS Monitor остановлен")
    
    def get_channel(self):
        """
        Возвращает активный канал для записи.
        
        :return: 'GPS' или 'VSLAM'
        """
        if self.state == 'GPS_OK':
            return 'GPS'
        else:
            return 'VSLAM'
    
    def get_state(self):
        """Возвращает текущее состояние системы"""
        return self.state
    
    def get_last_gps(self):
        """
        Возвращает последнюю GPS запись.
        
        :return: словарь {'id_eqitem': ..., 'date_creation': ..., 'pos': ...} или None
        """
        return self.last_gps_record
    
    def get_stats(self):
        """Возвращает статистику работы"""
        return {
            'state': self.state,
            'total_gps_records': self.total_gps_records,
            'switch_count': self.switch_count,
            'last_gps': self.last_gps_record
        }
    
    def _monitor_loop(self):
        """Основной цикл мониторинга (работает в отдельном потоке)"""
        if self.verbose:
            print(f"📖 Мониторинг файла: {self.gps_file}")
        
        # Проверяем существование файла
        while self.running and not Path(self.gps_file).exists():
            if self.verbose:
                print(f"⏳ Ожидание появления файла: {self.gps_file}")
            time.sleep(1)
        
        if not self.running:
            return
        
        if self.verbose:
            print(f"✅ Файл найден. Начинаем чтение...")
        
        while self.running:
            # Читаем новые записи из файла
            new_records = self._read_new_records()
            
            if new_records:
                # GPS данные появились
                for record in new_records:
                    self._handle_gps_record(record)
            else:
                # Нет новых записей - проверяем таймаут
                self._check_timeout()
            
            # Ждем перед следующей проверкой
            time.sleep(self.check_interval)
    
    def _read_new_records(self):
        """Читает новые записи из файла"""
        records = []
        
        try:
            with open(self.gps_file, 'r', encoding='utf-8') as f:
                f.seek(self.file_position)
                lines = f.readlines()
                self.file_position = f.tell()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('id_eqitem'):  # Пропускаем заголовок
                    continue
                
                parts = line.split(',', 2)
                if len(parts) >= 3:
                    records.append({
                        'id_eqitem': parts[0].strip(),
                        'date_creation': parts[1].strip(),
                        'pos': parts[2].strip()
                    })
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Ошибка чтения файла: {e}")
        
        return records
    
    def _handle_gps_record(self, record):
        """Обработка новой GPS записи"""
        self.total_gps_records += 1
        self.last_gps_time = time.time()
        self.last_gps_record = record
        
        # Если были в режиме vSLAM - переключаемся на GPS
        if self.state == 'VSLAM_ACTIVE':
            old_state = self.state
            self.state = 'GPS_OK'
            self.switch_count += 1
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"✅ GPS ВОССТАНОВЛЕН!")
                print(f"   Время: {record['date_creation']}")
                print(f"   ID: {record['id_eqitem']}")
                print(f"   Позиция: {record['pos']}")
                print(f"   Переключение: {old_state} → {self.state}")
                print(f"   Команда: ИСПОЛЬЗОВАТЬ GPS")
                print(f"{'='*60}\n")
            
            # Вызываем callback если есть
            if self.on_switch:
                self.on_switch('GPS_OK', record)
        
        elif self.state == 'WAITING_FIRST_GPS':
            self.state = 'GPS_OK'
            if self.verbose:
                print(f"\n Первая GPS запись получена: {record['date_creation']}")
                print(f"   Команда: ИСПОЛЬЗОВАТЬ GPS\n")
    
    def _check_timeout(self):
        """Проверяет таймаут отсутствия GPS"""
        if self.state != 'GPS_OK' or self.last_gps_time is None:
            return
        
        elapsed = time.time() - self.last_gps_time
        
        if elapsed > self.timeout:
            old_state = self.state
            self.state = 'VSLAM_ACTIVE'
            self.switch_count += 1
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"⚠️  GPS СИГНАЛ ПОТЕРЯН!")
                print(f"   Последняя запись: {self.last_gps_record['date_creation']}")
                print(f"   Прошло времени: {elapsed:.1f}с")
                print(f"   Переключение: {old_state} → {self.state}")
                print(f"   Команда: ИСПОЛЬЗОВАТЬ vSLAM")
                print(f"{'='*60}\n")
            
            # Вызываем callback если есть
            if self.on_switch:
                self.on_switch('VSLAM_ACTIVE', None)


# =============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (запускается только при прямом вызове модуля)
# =============================================================================

if __name__ == '__main__':
    # НАСТРОЙКИ
    GPS_FILE = 'data/gps_realtime_sim.csv'
    TIMEOUT = 3.0
    CHECK_INTERVAL = 0.5
    
    print("=" * 60)
    print("GPS MONITOR - ТЕСТОВЫЙ ЗАПУСК")
    print("=" * 60)
    
    # Создаем монитор
    monitor = GPSMonitor(GPS_FILE, timeout_seconds=TIMEOUT, check_interval=CHECK_INTERVAL)
    
    # Запускаем мониторинг
    monitor.start()
    
    # Главный цикл
    try:
        print("🚀 Мониторинг активен. Нажмите Ctrl+C для остановки.\n")
        
        while True:
            time.sleep(1)
            
            # Получаем активный канал
            channel = monitor.get_channel()
            last_gps = monitor.get_last_gps()
            
            if channel == 'GPS':
                if last_gps:
                    print(f"[GPS] {last_gps['date_creation']} | {last_gps['pos']}")
            else:
                print(f"[vSLAM] Работает визуальная одометрия...")
    
    except KeyboardInterrupt:
        print("\n⏹️  Остановка")
    
    finally:
        monitor.stop()
        
        # Вывод статистики
        stats = monitor.get_stats()
        print(f"\n{'='*60}")
        print("📊 СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"   GPS записей: {stats['total_gps_records']}")
        print(f"   Переключений: {stats['switch_count']}")
        print(f"{'='*60}")