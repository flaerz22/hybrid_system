#!/usr/bin python3
"""
Video Stream Simulator
Симулирует поступление видеопотока в реальном времени.
Теперь с точной синхронизацией по времени (компенсация задержек).
"""

import os
import time
import shutil
from pathlib import Path
from datetime import datetime


class VideoStreamSimulator:
    """Симулятор видеопотока в реальном времени с компенсацией времени."""
    
    def __init__(self, input_folder, output_folder, fps=10, 
                 clean_start=True, verbose=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.fps = fps
        self.clean_start = clean_start
        self.verbose = verbose
        
        self.frame_interval = 1.0 / fps
        self.frame_files = []
        self.current_index = 0
        self.running = False
    
    def prepare_frames(self):
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Папка с кадрами не найдена: {self.input_folder}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.frame_files = sorted([
            f for f in self.input_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not self.frame_files:
            raise ValueError(f"В папке не найдено изображений: {self.input_folder}")
        
        if self.verbose:
            print(f"📹 Найдено кадров: {len(self.frame_files)}")
            print(f"⏱️  FPS: {self.fps} (интервал: {self.frame_interval*1000:.0f} мс)")
    
    def _clean_output_folder(self):
        if self.output_folder.exists():
            try:
                shutil.rmtree(self.output_folder)
                if self.verbose:
                    print(f"🗑️  Удалена старая папка: {self.output_folder}")
            except PermissionError:
                # Если папка занята, удаляем только содержимое
                for item in self.output_folder.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                if self.verbose:
                    print(f"✅ Содержимое папки очищено")
    
    def start(self):
        self.prepare_frames()
        
        if self.clean_start:
            self._clean_output_folder()
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.current_index = 0
        
        if self.verbose:
            print(f"\n🚀 Запуск симуляции видеопотока...")
            print(f"   Вход:  {self.input_folder}")
            print(f"   Выход: {self.output_folder}")
            print(f"   Режим: {'Чистый старт' if self.clean_start else 'Добавление к существующим'}")
            print(f"   Начало: {datetime.now().strftime('%H:%M:%S')}\n")
        
        start_time = time.time()
        next_frame_time = start_time  # планируемое время следующего кадра
        
        total_frames = len(self.frame_files)
        
        while self.running and self.current_index < total_frames:
            frame_file = self.frame_files[self.current_index]
            output_file = self.output_folder / frame_file.name
            
            # Копируем кадр (используем copyfile – быстрее, чем copy2)
            shutil.copyfile(frame_file, output_file)
            
            # Промежуточный вывод только если verbose=True
            if self.verbose and (self.current_index % 10 == 0 or 
                                 self.current_index == total_frames - 1):
                elapsed = time.time() - start_time
                print(f"[{self.current_index+1:4d}/{total_frames}] "
                      f"Кадр: {frame_file.name} | "
                      f"Время: {elapsed:.1f}с")
            
            self.current_index += 1
            
            # Если не последний кадр – планируем следующий
            if self.running and self.current_index < total_frames:
                next_frame_time += self.frame_interval
                current_time = time.time()
                sleep_time = next_frame_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # если sleep_time <= 0, значит мы отстаём – пропускаем ожидание,
                # чтобы наверстать упущенное время
        
        total_time = time.time() - start_time
        actual_fps = self.current_index / total_time if total_time > 0 else 0
        
        # Вывод итоговой статистики (всегда, независимо от verbose)
        print(f"\n✅ Симуляция завершена!")
        print(f"   Обработано кадров: {self.current_index}")
        print(f"   Общее время: {total_time:.1f}с")
        print(f"   Заданный FPS: {self.fps}")
        print(f"   Фактический FPS: {actual_fps:.1f}")
        
        if self.verbose:
            print(f"   Дата и время окончания: {datetime.now().strftime('%H:%M:%S')}")
    
    def stop(self):
        self.running = False
        if self.verbose:
            print(f"\n⏹️  Симуляция остановлена на кадре {self.current_index}")


if __name__ == '__main__':
    # НАСТРОЙКИ
    INPUT_FOLDER = 'data/test_50kmh_2_qhd' # 'data/video_10fps'
    OUTPUT_FOLDER = 'video_stream'
    FPS = 30
    CLEAN_START = True
    VERBOSE = False   # ← выключен подробный вывод, только итог
    
    simulator = VideoStreamSimulator(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        fps=FPS,
        clean_start=CLEAN_START,
        verbose=VERBOSE
    )
    
    try:
        simulator.start()
    except KeyboardInterrupt:
        simulator.stop()