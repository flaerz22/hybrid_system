"""
Утилита для загрузки параметров камеры из файла калибровки.
Поддерживает форматы .npz (OpenCV save) и .pkl (pickle).
"""

import numpy as np
import os


def load_camera_calibration(calibration_file='data/camera_calibration.npz'):
    """
    Загружает матрицу камеры и коэффициенты дисторсии из файла.
    
    Параметры:
        calibration_file — путь к файлу калибровки (.npz или .pkl)
    
    Возвращает:
        dict с ключами:
            'camera_matrix' — матрица 3x3 (K)
            'dist_coeffs' — коэффициенты дисторсии
            'focal_x' — фокусное расстояние по X
            'focal_y' — фокусное расстояние по Y
            'cx' — оптический центр X
            'cy' — оптический центр Y
            'image_size' — размер кадра (если есть)
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"❌ Файл калибровки не найден: {calibration_file}")
    
    result = {}
    
    # Определяем формат по расширению
    ext = os.path.splitext(calibration_file)[1].lower()
    
    if ext == '.npz':
        # Формат OpenCV (np.savez)
        data = np.load(calibration_file)
        
        print("=" * 70)
        print(" 📷 ЗАГРУЗКА КАЛИБРОВКИ КАМЕРЫ")
        print("=" * 70)
        print(f"    Файл: {os.path.abspath(calibration_file)}")
        print(f"   📦 Доступные массивы: {data.files}")
        
        # Извлекаем матрицу камеры
        if 'camera_matrix' in data.files:
            result['camera_matrix'] = data['camera_matrix'].astype(np.float64)
        elif 'K' in data.files:
            result['camera_matrix'] = data['K'].astype(np.float64)
        else:
            raise KeyError("❌ В файле нет 'camera_matrix' или 'K'")
        
        # Извлекаем коэффициенты дисторсии
        if 'dist_coeffs' in data.files:
            dist_coeffs = data['dist_coeffs'].astype(np.float64).flatten()
        elif 'dist' in data.files:
            dist_coeffs = data['dist'].astype(np.float64).flatten()
        else:
            print(f"   ⚠️ Коэффициенты дисторсии не найдены, используем нулевые")
            dist_coeffs = np.zeros(5, dtype=np.float64)
        
        # # Нормализуем до 5 коэффициентов (стандарт OpenCV)
        # if len(dist_coeffs) < 5:
        #     dist_coeffs = np.pad(dist_coeffs, (0, 5 - len(dist_coeffs)))
        # elif len(dist_coeffs) > 5:
        #     print(f"   ⚠️ Найдено {len(dist_coeffs)} коэффициентов, используем первые 5")
        #     dist_coeffs = dist_coeffs[:5]
        
        result['dist_coeffs'] = dist_coeffs
        
        # Извлекаем размер изображения (если есть)
        if 'image_size' in data.files:
            result['image_size'] = tuple(data['image_size'])
        
    elif ext in ['.pkl', '.pickle']:
        # Формат pickle
        import pickle
        with open(calibration_file, 'rb') as f:
            data = pickle.load(f)
        
        result['camera_matrix'] = np.array(data['camera_matrix'], dtype=np.float64)
        dist_coeffs = np.array(data.get('dist_coeffs', np.zeros(5)), dtype=np.float64).flatten()
        
        # # Нормализуем до 5 коэффициентов
        # if len(dist_coeffs) < 5:
        #     dist_coeffs = np.pad(dist_coeffs, (0, 5 - len(dist_coeffs)))
        # elif len(dist_coeffs) > 5:
        #     dist_coeffs = dist_coeffs[:5]
        
        result['dist_coeffs'] = dist_coeffs
        
    else:
        raise ValueError(f"❌ Неизвестный формат файла: {ext}")
    
    # Извлекаем параметры из матрицы камеры
    K = result['camera_matrix']
    result['focal_x'] = float(K[0, 0])
    result['focal_y'] = float(K[1, 1])
    result['cx'] = float(K[0, 2])
    result['cy'] = float(K[1, 2])
    
    # Выводим информацию
    print(f"\n   📐 Матрица камеры K:")
    print(f"      {K}")
    print(f"\n    Фокусное расстояние: focal_x={result['focal_x']:.2f}, focal_y={result['focal_y']:.2f}")
    print(f"    Оптический центр: ({result['cx']:.1f}, {result['cy']:.1f})")
    print(f"   🔧 Коэффициенты дисторсии: {result['dist_coeffs']}")
    
    if 'image_size' in result:
        print(f"   📐 Размер изображения: {result['image_size']}")
    
    print("=" * 70)
    
    return result


def get_default_calibration(width=1280, height=720, focal_x=811.27, focal_y=811.27):
    """
    Возвращает калибровку по умолчанию (если файл не найден).
    Используется как fallback.
    """
    cx = width / 2
    cy = height / 2
    
    camera_matrix = np.array([
        [focal_x, 0,      cx],
        [0,      focal_y, cy],
        [0,      0,      1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros(5, dtype=np.float64)
    
    return {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'focal_x': focal_x,
        'focal_y': focal_y,
        'cx': cx,
        'cy': cy
    }


def undistort_frame(frame, camera_matrix, dist_coeffs, balance=0.5):
    """
    Корректирует дисторсию кадра используя FISHEYE-модель.
    
    Параметры:
        frame — входной кадр (numpy array)
        camera_matrix — матрица камеры 3x3
        dist_coeffs — коэффициенты дисторсии [k1, k2, k3, k4]
        balance — баланс между FOV и качеством (0.0-1.0)
                  0.0 = максимальное качество, меньше FOV
                  1.0 = максимальный FOV, больше искажений по краям
                  0.5 = компромисс
    
    Возвращает:
        Кадр без дисторсии
    """
    import cv2
    
    # Если дисторсия нулевая — возвращаем кадр как есть
    if dist_coeffs is None or np.all(dist_coeffs == 0):
        return frame
    
    h, w = frame.shape[:2]
    
    # === FISHEYE МОДЕЛЬ (как в скрипте калибровки) ===
    
    # 1. Вычисляем новую матрицу камеры с балансом FOV/качество
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, 
        dist_coeffs, 
        (w, h), 
        np.eye(3),  # R — единичная матрица (без дополнительного поворота)
        balance=balance
    )
    
    # 2. Вычисляем карты преобразования
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, 
        dist_coeffs, 
        np.eye(3),  # R
        new_camera_matrix, 
        (w, h), 
        cv2.CV_16SC2
    )
    
    # 3. Применяем преобразование
    undistorted = cv2.remap(
        frame, 
        map1, 
        map2, 
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    
    return undistorted