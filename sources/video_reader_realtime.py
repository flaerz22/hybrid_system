#!/usr/bin/env python3
"""
FolderVideoCapture с умным таймаутом
Таймаут срабатывает только если в папке НЕ ПОЯВЛЯЮТСЯ новые кадры,
а не если основной цикл редко вызывает read().
"""

import cv2 as cv
import time
from pathlib import Path


class FolderVideoCapture:
    """
    Виртуальный видео-захват из папки с кадрами.
    
    Особенности:
    - При первом вызове read() находит последний кадр в папке как стартовый
    - Читает только НОВЫЕ кадры, которые появляются после стартового
    - УМНЫЙ ТАЙМАУТ: срабатывает только если в папке нет новых кадров,
      а не если основной цикл занят обработкой
    - Поддерживает многократный запуск через reset()
    """
    
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    
    def __init__(self, folder_path, fps=10, timeout=5.0):
        """
        :param folder_path: путь к папке с кадрами
        :param fps: ожидаемая частота кадров
        :param timeout: таймаут бездействия (сек).
                        Срабатывает только если в папке нет новых кадров.
        """
        self.folder_path = Path(folder_path)
        self.fps = fps
        self.timeout = timeout
        
        # Состояние
        self._is_opened = False
        self._frame_interval = 1.0 / fps if fps > 0 else 0.1
        self._supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        # Список кадров и позиция
        self._all_frames = []
        self._current_index = -1
        self._last_processed_frame = None
        
        # 🔑 УМНЫЙ ТАЙМАУТ: время последнего НОВОГО кадра в папке
        # 🔑 ИСПРАВЛЕНИЕ: Инициализируется None, а не time.time()
        self._last_new_frame_in_folder_time = None
        
        # Свойства видео
        self._width = 0
        self._height = 0
        
        # Статистика
        self._frames_read = 0
        self._session_count = 0
        
        self._open()
    
    def _open(self):
        """Открывает папку и находит последний доступный кадр."""
        if not self.folder_path.exists():
            print(f"❌ Папка не найдена: {self.folder_path}")
            self._is_opened = False
            return
        
        self._all_frames = sorted([
            f for f in self.folder_path.iterdir()
            if f.suffix.lower() in self._supported_extensions
        ])
        
        if not self._all_frames:
            print(f"⚠️ В папке нет изображений: {self.folder_path}")
            self._is_opened = False
            return
        
        sample = cv.imread(str(self._all_frames[0]))
        if sample is not None:
            self._height, self._width = sample.shape[:2]
        else:
            print(f"⚠️ Не удалось прочитать первый кадр")
            self._is_opened = False
            return
        
        self._current_index = len(self._all_frames) - 1
        self._last_processed_frame = None
        
        # 🔑 ИСПРАВЛЕНИЕ: НЕ инициализируем таймер здесь!
        # self._last_new_frame_in_folder_time = None  # Уже None
        
        self._is_opened = True
        self._session_count += 1
        #print(f"✅ FolderVideoCapture открыт (сессия #{self._session_count}):")
        #print(f"   Папка: {self.folder_path}")
        #print(f"   Кадров: {len(self._all_frames)}")
        #print(f"   Размеры: {self._width}x{self._height}")
        #print(f"   FPS: {self.fps}")
        #print(f"   Таймаут: {self.timeout}с")
        #print(f"   🎯 Стартовый кадр: {self._all_frames[self._current_index].name}")
    
    def reset(self):
        """Сбрасывает состояние и находит НОВЫЙ последний кадр."""
        if not self.folder_path.exists():
            return False
        
        self._all_frames = sorted([
            f for f in self.folder_path.iterdir()
            if f.suffix.lower() in self._supported_extensions
        ])
        
        if not self._all_frames:
            return False
        
        self._current_index = len(self._all_frames) - 1
        self._last_processed_frame = None
        
        # 🔑 ИСПРАВЛЕНИЕ: Сбрасываем таймер в None
        self._last_new_frame_in_folder_time = None
        self._frames_read = 0
        
        self._is_opened = True
        self._session_count += 1
        
        #print(f"\n🔄 FolderVideoCapture сброшен (сессия #{self._session_count})")
        #print(f"   🎯 Новый стартовый кадр: {self._all_frames[self._current_index].name}") 
        return True
    
    def isOpened(self):
        return self._is_opened
    
    def read(self):
        """
        Читает следующий кадр.
        
        УМНЫЙ ТАЙМАУТ: срабатывает только если в папке не появляются новые кадры,
        а не если основной цикл редко вызывает read().
        """
        if not self._is_opened:
            return False, None
        
        # Первый вызов — возвращаем якорный кадр
        if self._last_processed_frame is None:
            anchor_frame = self._all_frames[self._current_index]
            frame = cv.imread(str(anchor_frame))
            if frame is None:
                return False, None
            
            self._last_processed_frame = anchor_frame
            
            # 🔑 ИСПРАВЛЕНИЕ: Инициализируем таймер при первом вызове read()
            self._last_new_frame_in_folder_time = time.time()
            
            self._frames_read += 1
            return True, frame
        
        # Последующие вызовы — ищем новые кадры
        while True:
            # 🔑 УМНЫЙ ТАЙМАУТ: проверяем, появляются ли НОВЫЕ кадры в папке
            current_frames = sorted([
                f for f in self.folder_path.iterdir()
                if f.suffix.lower() in self._supported_extensions
            ])
            
            # Проверяем, есть ли новые кадры после последнего обработанного
            has_new_frames = False
            next_frame_path = None
            next_index = -1
            
            for i, frame_path in enumerate(current_frames):
                if frame_path > self._last_processed_frame:
                    has_new_frames = True
                    if next_frame_path is None:  # Берём самый старый новый кадр
                        next_frame_path = frame_path
                        next_index = i
            
            # 🔑 Если появились новые кадры — сбрасываем таймер
            if has_new_frames:
                self._last_new_frame_in_folder_time = time.time()
            
            # 🔑 Проверяем таймаут: если новых кадров нет дольше timeout
            # 🔑 ИСПРАВЛЕНИЕ: Проверяем только если таймер инициализирован
            if self.timeout is not None and self._last_new_frame_in_folder_time is not None:
                elapsed = time.time() - self._last_new_frame_in_folder_time
                if elapsed > self.timeout:
                    #print(f"\n⏱️ Таймаут: новых кадров в папке не было {elapsed:.1f}с > {self.timeout}с")
                    self._is_opened = False
                    return False, None
            
            # Нашли новый кадр — возвращаем его
            if next_frame_path is not None:
                frame = cv.imread(str(next_frame_path))
                if frame is None:
                    self._last_processed_frame = next_frame_path
                    continue
                
                self._last_processed_frame = next_frame_path
                self._current_index = next_index
                self._last_new_frame_in_folder_time = time.time()  # 🔑 Обновляем
                self._frames_read += 1
                return True, frame
            
            # Новых кадров нет — ждём
            time.sleep(self._frame_interval / 2)
    
    def get(self, prop_id):
        if prop_id == self.CAP_PROP_FRAME_WIDTH or prop_id == cv.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        elif prop_id == self.CAP_PROP_FRAME_HEIGHT or prop_id == cv.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        elif prop_id == self.CAP_PROP_FPS or prop_id == cv.CAP_PROP_FPS:
            return float(self.fps)
        elif prop_id == self.CAP_PROP_FRAME_COUNT or prop_id == cv.CAP_PROP_FRAME_COUNT:
            return -1.0
        else:
            return 0.0
    
    def release(self):
        self._is_opened = False
        self._all_frames = []
        self._last_processed_frame = None
        #print(f"🔴 FolderVideoCapture закрыт. Сессий: {self._session_count}")
    
    def get_current_frame_name(self):
        if self._last_processed_frame:
            return self._last_processed_frame.name
        return None
    
    def get_frames_count(self):
        return len(self._all_frames)
    
    def get_frames_read(self):
        return self._frames_read
    
    def get_session_count(self):
        return self._session_count