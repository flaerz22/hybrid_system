#!/usr/bin/env python3
"""
Гибридная навигационная система GPS + vSLAM
Главный оркестратор системы.
"""

import os
import csv
import cv2 as cv
import numpy as np
import time
import pygame
import math
import matplotlib.pyplot as plt

# === Импорты из sources ===
from sources.slam_scale import Slam
from sources.render import Renderer3D
from sources.gps_monitor import GPSMonitor
from sources.video_reader_realtime import FolderVideoCapture

# === Импорты из utils ===
from utils.geo_utils import extract_lat_lon_from_pos, gps_to_meters
from utils.trajectory_utils import align_and_merge_trajectories, save_and_plot_trajectories
from utils.visualization_utils_rotate import draw_stats, draw_status
from utils.camera_utils1 import load_camera_calibration, get_default_calibration, undistort_frame, get_undistort_setup


# =============================================================================
# КОНФИГУРАЦИЯ ПУТЕЙ И ФАЙЛОВ
# =============================================================================
# Входные данные
INPUT_VIDEO_FOLDER = 'video_stream'
VIDEO_FILE = 'data/car.mp4'
GPS_REALTIME_FILE = 'data/gps_realtime_sim_123682.csv'
REFERENCE_GPS_FILE = 'data/test_gps_123682_2min.csv'
# Калибровка камеры
CALIBRATION_FILE = 'data/camera_calibration_qhd.npz'

# Выходные данные
OUTPUT_DIR = 'output'
OUTPUT_CSV = f"{OUTPUT_DIR}/merged_trajectory.csv"
OUTPUT_PLOT_MERGED = f"{OUTPUT_DIR}/merged_trajectory_plot.png"
OUTPUT_PLOT_COMPARISON = f"{OUTPUT_DIR}/hybrid_vs_full_reference.png"

# Параметры обработки
FPS = 10
FRAME_SKIP_RATE = 3
GPS_TIMEOUT_SECONDS = 4.0
# =============================================================================

USE_MASKER = True
USE_GPS = False  # ← False = только vSLAM, True = GPS + vSLAM
ENABLE_VISUALIZATION = True # False = только обработка, без окон

# =============================================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ СИСТЕМЫ
# =============================================================================
slam = None
renderer = None
last_frame_pixels = None
# Калибровка камеры
camera_matrix = None
dist_coeffs = None
slam_camera_matrix = None
undistort_map1 = None
undistort_map2 = None

# GPS привязка
current_gps_anchor = None
last_gps_state = None
gps_ref_lat = None
gps_ref_lon = None
last_gps_meters = (0.0, 0.0)

# Траектория vSLAM
trajectory_x = []
trajectory_z = []
trajectory_gps_anchors = []

# Счетчики и флаги
frame_id = 0
vslam_reset_count = 0
vslam_trajectory_num = 0
vslam_recording_active = False


# =============================================================================
# ФУНКЦИИ УПРАВЛЕНИЯ vSLAM
# =============================================================================
def reset_vslam_with_gps_anchor(gps_point, width, height):
    """
    Сбрасывает vSLAM и привязывает его к новой GPS точке.
    """
    global slam, renderer, last_frame_pixels, vslam_reset_count
    global current_gps_anchor, trajectory_x, trajectory_z
    global gps_ref_lat, gps_ref_lon, last_gps_meters
    global camera_matrix, dist_coeffs
    global slam_camera_matrix

    vslam_reset_count += 1
    current_gps_anchor = gps_point.copy()

    # Извлекаем координаты из строки pos
    pos_string = gps_point.get('pos')
    lat, lon = extract_lat_lon_from_pos(pos_string)

    if lat is not None and lon is not None:
        if gps_ref_lat is None:
            gps_ref_lat = lat
            gps_ref_lon = lon

        last_gps_meters = gps_to_meters(lat, lon, gps_ref_lat, gps_ref_lon)
        print(f"    GPS якорь (метрические): X={last_gps_meters[0]:.2f}, Z={last_gps_meters[1]:.2f}")
        print(f"   📍 GPS якорь (градусы): lat={lat:.6f}, lon={lon:.6f}")
    else:
        print(f"   ⚠️  Не удалось извлечь координаты из pos: {pos_string}")

    # Освобождаем старый vSLAM
    if slam is not None:
        if hasattr(slam.vision, 'masker') and slam.vision.masker is not None:
            slam.vision.masker.stop()

    # Создаём НОВЫЙ объект Slam
    slam = Slam(width, height,
                camera_matrix=slam_camera_matrix,   # ← K выпрямленного кадра
                dist_coeffs=np.zeros(4),            # ← кадр УЖЕ выпрямлен
                use_masker=USE_MASKER)
    slam.vision.step_dt = FRAME_SKIP_RATE / FPS
    renderer = Renderer3D(pov_=90, cam_distance=1200)

    # Обнуляем траекторию
    trajectory_x = []
    trajectory_z = []
    last_frame_pixels = None

    print(f"   ✅ vSLAM сброшен, траектория обнулена")


def save_current_segment():
    """
    Сохраняет текущий сегмент vSLAM в список trajectory_gps_anchors.
    Вызывается при восстановлении GPS или в конце видео.
    """
    global vslam_recording_active, vslam_trajectory_num, trajectory_x, trajectory_z

    if not vslam_recording_active or len(trajectory_x) == 0:
        return

    print(f"\n💾 Сохраняем траекторию vSLAM (GPS пропадал)")
    print(f"   Точек в сегменте: {len(trajectory_x)}")

    # СМЕЩАЕМ траекторию vSLAM относительно метрических координат GPS
    offset_x = last_gps_meters[0] - trajectory_x[0]
    offset_z = last_gps_meters[1] - trajectory_z[0]

    shifted_x = [x + offset_x for x in trajectory_x]
    shifted_z = [z + offset_z for z in trajectory_z]

    vslam_trajectory_num += 1
    trajectory_gps_anchors.append({
        'reset_num': vslam_trajectory_num,
        'gps_anchor': current_gps_anchor,
        'trajectory_x': shifted_x,
        'trajectory_z': shifted_z,
        'start_gps_meters': last_gps_meters
    })
    print(f"   ✅ Сегмент {vslam_trajectory_num} смещён и сохранён.")

    # Обнуляем текущую траекторию
    trajectory_x = []
    trajectory_z = []
    vslam_recording_active = False


def load_reference_trajectory(ref_file, ref_lat, ref_lon):
    """Загружает эталонную траекторию для сравнения."""
    ref_trajectory = []
    try:
        with open(ref_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_line = 2 if len(lines) > 1 and not lines[1].strip()[0].isdigit() else 1

            for line in lines[start_line:]:
                parts = line.strip().split(',', 2)
                if len(parts) >= 3:
                    lat, lon = extract_lat_lon_from_pos(parts[2].strip())
                    if lat is not None and lon is not None:
                        x, z = gps_to_meters(lat, lon, ref_lat, ref_lon)
                        ref_trajectory.append({
                            'time': parts[1].strip(),
                            'x_meters': x,
                            'z_meters': z,
                            'source': 'REFERENCE'
                        })
        print(f"✅ Эталон загружен: {len(ref_trajectory)} точек")
    except FileNotFoundError:
        print(f"⚠️ Файл эталона {ref_file} не найден.")
    return ref_trajectory


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================
def main():
    global slam, renderer, last_frame_pixels, vslam_reset_count
    global current_gps_anchor, trajectory_x, trajectory_z
    global gps_ref_lat, gps_ref_lon, last_gps_meters
    global trajectory_gps_anchors, vslam_trajectory_num, vslam_recording_active
    global last_gps_state, frame_id, frame_skip_counter
    global camera_matrix, dist_coeffs
    global slam_camera_matrix, undistort_map1, undistort_map2

    last_frame_for_zupt = None

    start_time = time.time()
    print(f"\n⏱️ Старт обработки: {time.strftime('%H:%M:%S')}")

    # === 0. ЗАГРУЗКА КАЛИБРОВКИ КАМЕРЫ ===

    calibration_loaded_from_file = False

    try:
        calibration = load_camera_calibration(CALIBRATION_FILE)
        camera_matrix = calibration['camera_matrix']
        dist_coeffs = calibration['dist_coeffs']
        calibration_loaded_from_file = True
        print(f"✅ Калибровка камеры загружена из файла")
    except (FileNotFoundError, KeyError) as e:
        print(f"⚠️ {e}")
        print(f"   Используем калибровку по умолчанию")
        default_calib = get_default_calibration(width=1280, height=720)
        camera_matrix = default_calib['camera_matrix']
        dist_coeffs = default_calib['dist_coeffs']

    # === 1. Инициализация источников данных ===
    video = FolderVideoCapture(f'./{INPUT_VIDEO_FOLDER}/', fps=FPS, timeout=5.0)

    # video = cv.VideoCapture(VIDEO_FILE)
    
    if not video.isOpened():
        print("Failed to read video")
        return

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_dim = (width, height)

    # Если калибровка была по умолчанию — пересоздаём с реальными размерами
    if not calibration_loaded_from_file:
        default_calib = get_default_calibration(width=width, height=height)
        camera_matrix = default_calib['camera_matrix']
        dist_coeffs = default_calib['dist_coeffs']
        print(f"   📐 Калибровка по умолчанию обновлена под размер кадра: {width}x{height}")

    undistort_map1, undistort_map2, slam_camera_matrix = get_undistort_setup(camera_matrix, dist_coeffs, width, height)
    

    # === УСЛОВНЫЙ ЗАПУСК GPS MONITOR ===
    gps_monitor = None
    if USE_GPS:
        gps_monitor = GPSMonitor(GPS_REALTIME_FILE, timeout_seconds=GPS_TIMEOUT_SECONDS, verbose=True)
        gps_monitor.start()
        print("📡 GPS Monitor запущен")
    else:
        print("📡 GPS Monitor ОТКЛЮЧЕН — работаем в режиме чистого vSLAM")
    # ======================================

    # === 2. Настройка окон ===
    if ENABLE_VISUALIZATION:
        cv.namedWindow('vSLAM', cv.WINDOW_NORMAL)
        cv.resizeWindow('vSLAM', video_dim[0], video_dim[1])
        cv.namedWindow('Mask Debug', cv.WINDOW_NORMAL)
        cv.resizeWindow('Mask Debug', 320, 240)
    else:
        print(" Визуализация ОТКЛЮЧЕНА — режим максимальной производительности")

    print("🟢 Starting hybrid navigation system...")
    if USE_GPS:
        print("🎯 vSLAM работает ВСЕГДА, GPS сбрасывает его при каждой новой точке")
    else:
        print("🎯 Режим чистого vSLAM (без GPS)")
    print("Press 'q' to stop.\n")

    # === 3. Главный цикл ===
    frame_skip_counter = 0

    while True:
        # --- ПРОВЕРКА СОСТОЯНИЯ GPS (только если USE_GPS=True) ---
        if USE_GPS and gps_monitor is not None:
            gps_state = gps_monitor.get_state()
            last_gps = gps_monitor.get_last_gps()

            # === ОБРАБОТКА GPS СОБЫТИЙ ===
            if gps_state == 'GPS_OK' and last_gps is not None:
                if (last_gps_state != 'GPS_OK' or
                    current_gps_anchor is None or
                    current_gps_anchor['date_creation'] != last_gps['date_creation']):
                    save_current_segment()
                    reset_vslam_with_gps_anchor(last_gps, width, height)
            elif gps_state == 'VSLAM_ACTIVE' and slam is not None:
                if not vslam_recording_active:
                    print("\n📝 GPS пропал! Начинаем использовать vSLAM как основной источник.")
                    vslam_recording_active = True

            last_gps_state = gps_state
        else:
            # === РЕЖИМ БЕЗ GPS: vSLAM работает всегда ===
            if slam is None:
                # Инициализируем vSLAM при первом кадре
                slam = Slam(width, height,
                            camera_matrix=slam_camera_matrix,   # ← K выпрямленного кадра
                            dist_coeffs=np.zeros(4),            # ← кадр УЖЕ выпрямлен
                            use_masker=USE_MASKER)
                slam.vision.step_dt = FRAME_SKIP_RATE / FPS
                renderer = Renderer3D(pov_=90, cam_distance=1200)
                print("✅ vSLAM инициализирован (без GPS)")

        # === ЧТЕНИЕ И ОБРАБОТКА КАДРОВ ===
        ret, frame_pixels = video.read()

        if not ret:
            print("\n⏹️ Видеопоток завершён")
            break

        frame_id += 1
        frame_pixels = cv.resize(frame_pixels, video_dim, interpolation=cv.INTER_AREA)

        # === КОРРЕКЦИЯ ДИСТОРСИИ ===
        if undistort_map1 is not None:
            frame_pixels = cv.remap(frame_pixels, undistort_map1, undistort_map2,
                                    cv.INTER_LINEAR)

        if slam is not None:
            slam.vision.update_motion_state(last_frame_for_zupt, frame_pixels)
        last_frame_for_zupt = frame_pixels.copy()

        # Если vSLAM ещё не инициализирован — просто ждём
        if slam is None:
            if ENABLE_VISUALIZATION:
                cv.imshow('vSLAM', frame_pixels)
                if cv.waitKey(1) == ord('q'):
                    break
            else:
                # Без визуализации — просто продолжаем
                if cv.waitKey(1) == ord('q'):
                    break
            continue

        # --- ЛОГИКА ПРОПУСКА КАДРОВ ---
        frame_skip_counter += 1
        if frame_skip_counter % FRAME_SKIP_RATE != 0:
            if ENABLE_VISUALIZATION:
                cv.putText(frame_pixels,
                        f"FRAME SKIPPED ({frame_skip_counter % FRAME_SKIP_RATE}/{FRAME_SKIP_RATE})",
                        (30, frame_pixels.shape[0] - 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.imshow('vSLAM', frame_pixels)
                if cv.waitKey(1) == ord('q'):
                    break
            else:
                # Без визуализации — просто проверяем выход
                if cv.waitKey(1) == ord('q'):
                    break
            continue

        # === ПОЛНАЯ ОБРАБОТКА vSLAM ===
        if slam.vision.masker is not None:
            slam.vision.masker.submit_frame(frame_pixels)

        if last_frame_pixels is not None and not (frame_pixels == last_frame_pixels).all():
            slam.update_frame_pixels(current_frame_pixels=frame_pixels,
                                     last_frame_pixels=last_frame_pixels)

            if ENABLE_VISUALIZATION:
                matches, frame_pixels = slam.get_vision_matches(frame_pixels)
            else:
                # Без отрисовки — просто находим матчи и считаем позу
                matches = slam.vision.find_matching_points(slam.vision.current_frame)
                if matches is not None:
                    slam.vision.get_camera_pose(matches)

            if matches is not None:
                if ENABLE_VISUALIZATION:
                    points = slam.triangulate(matches)
                    if points is not None:
                        renderer.render3dSpace(points, slam.get_camera_poses())
                else:
                    slam.triangulate(matches)

                R_total, T_total = slam.vision.get_pose_cumulation()
                t_flat = T_total.flatten()
                current_x = float(t_flat[0])
                current_z = float(t_flat[2])

                # Записываем относительную траекторию
                trajectory_x.append(current_x)
                trajectory_z.append(current_z)

        last_frame_pixels = frame_pixels.copy()

        # === ВИЗУАЛИЗАЦИЯ ===
        if ENABLE_VISUALIZATION:
            try:
                frame_with_mask = slam.vision.apply_mask_visualization(frame_pixels.copy(), mode="black_fill")
                frame_with_stats = draw_stats(frame_with_mask, slam)
                frame_with_status = draw_status(frame_with_stats, slam)
                cv.imshow('vSLAM', frame_with_status)
            except Exception:
                cv.imshow('vSLAM', frame_pixels)
            
            slam.vision.debug_show_mask(frame_pixels)
            renderer.render()
            
            if cv.waitKey(1) == ord('q'):
                print("\n️ Остановка по команде пользователя")
                break
        else:
            # Без визуализации — просто печатаем прогресс каждые 100 кадров
            if frame_id % 100 == 0:
                print(f"   ⚙️  Кадр #{frame_id} | точек в траектории: {len(trajectory_x)}")
            # Всё равно проверяем выход
            if cv.waitKey(1) == ord('q'):
                break

    # =============================================================================
    # 4. ЗАВЕРШЕНИЕ РАБОТЫ
    # =============================================================================
    # Сохраняем последнюю траекторию
    if USE_GPS:
        save_current_segment()
    else:
        # В режиме без GPS — сохраняем всю траекторию как один сегмент
        if len(trajectory_x) > 0:
            vslam_trajectory_num += 1
            trajectory_gps_anchors.append({
                'reset_num': vslam_trajectory_num,
                'gps_anchor': {'date_creation': 'vslam_only'},
                'trajectory_x': trajectory_x.copy(),
                'trajectory_z': trajectory_z.copy(),
                'start_gps_meters': (0.0, 0.0)
            })
            print(f"✅ Сохранён vSLAM-сегмент: {len(trajectory_x)} точек")

    # Освобождаем ресурсы vSLAM
    if slam is not None:
        if hasattr(slam.vision, 'masker') and slam.vision.masker is not None:
            slam.vision.masker.stop()
        if hasattr(slam.vision, 'print_mask_statistics'):
            slam.vision.print_mask_statistics()

    video.release()
    cv.destroyAllWindows()
    
    if gps_monitor is not None:
        gps_monitor.stop()
    
    pygame.quit()

    print("\n🛑 Hybrid Navigation System Finished")
    print(f"   Всего кадров обработано: {frame_id}")
    print(f"   Всего сбросов vSLAM: {vslam_reset_count}")
    print(f"   Всего сегментов траектории сохранено: {len(trajectory_gps_anchors)}")

    # =============================================================================
    # 5. ПОСТ-ОБРАБОТКА
    # =============================================================================
    if USE_GPS:
        print(f"\n📍 Формирование единой непрерывной траектории с векторным выравниванием...")

        # Читаем исходный GPS файл
        gps_data_raw = []
        try:
            with open(GPS_REALTIME_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(',', 2)
                    if len(parts) >= 3:
                        date_creation = parts[1].strip()
                        pos_str = parts[2].strip()
                        lat, lon = extract_lat_lon_from_pos(pos_str)
                        if lat is not None and lon is not None:
                            x, z = gps_to_meters(lat, lon, gps_ref_lat, gps_ref_lon)
                            gps_data_raw.append({
                                'time': date_creation,
                                'x': x,
                                'z': z,
                                'source': 'GPS'
                            })
        except FileNotFoundError:
            print(f"⚠️ Файл {GPS_REALTIME_FILE} не найден. Пропускаем чтение GPS.")

        # Слияние траекторий
        merged_trajectory = align_and_merge_trajectories(gps_data_raw, trajectory_gps_anchors)

        # Загрузка эталона
        ref_trajectory = []
        if gps_ref_lat is not None:
            ref_trajectory = load_reference_trajectory(REFERENCE_GPS_FILE, gps_ref_lat, gps_ref_lon)

        # Сохранение CSV и построение графиков
        save_and_plot_trajectories(
            merged_trajectory=merged_trajectory,
            ref_trajectory=ref_trajectory,
            gps_ref_lat=gps_ref_lat,
            gps_ref_lon=gps_ref_lon,
            csv_filename=OUTPUT_CSV,
            plot_merged_filename=OUTPUT_PLOT_MERGED,
            plot_comparison_filename=OUTPUT_PLOT_COMPARISON
        )

        # Итоговая статистика
        stats = gps_monitor.get_stats()
        print(f"\n📊 Итоговая статистика:")
        print(f"   GPS записей в исходном файле: {stats['total_gps_records']}")
        print(f"   Переключений (разрывов): {stats['switch_count']}")
        print(f"   Сегментов vSLAM сохранено: {len(trajectory_gps_anchors)}")
        print(f"   Всего точек в единой траектории: {len(merged_trajectory) if merged_trajectory else 0}")
    
    else:
        # === РЕЖИМ БЕЗ GPS: просто сохраняем траекторию vSLAM ===
        print(f"\n📍 Режим чистого vSLAM — сохраняем траекторию без GPS-привязки...")
        
        # 1. Сохраняем CSV с траекторией vSLAM
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'x_meters', 'z_meters', 'source'])
            for i, (x, z) in enumerate(zip(trajectory_x, trajectory_z)):
                writer.writerow([i, x, z, 'vSLAM'])
        print(f"✅ Траектория vSLAM сохранена: {os.path.abspath(OUTPUT_CSV)}")
        
        # 2. Строим и сохраняем график траектории vSLAM
        if len(trajectory_x) > 0:
            plt.figure(figsize=(14, 10))
            
            # Основная линия траектории
            plt.plot(trajectory_x, trajectory_z, color='blue', linewidth=2, label='vSLAM траектория', zorder=1)
            
            # Отмечаем старт и финиш
            plt.scatter([trajectory_x[0]], [trajectory_z[0]], color='green', s=200, marker='^', label='Старт (0, 0)', zorder=5, edgecolors='black')
            plt.scatter([trajectory_x[-1]], [trajectory_z[-1]], color='red', s=200, marker='v', label='Финиш', zorder=5, edgecolors='black')
            
            # Настройки графика
            plt.title('Траектория vSLAM (без GPS-привязки)', fontsize=16, fontweight='bold')
            plt.xlabel('X (метры, относительные)')
            plt.ylabel('Z (метры, относительные)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axis('equal')  # Важно: одинаковый масштаб осей для корректного отображения поворотов
            plt.legend(fontsize=12)
            
            # Сохраняем график
            vslam_plot_filename = f"{OUTPUT_DIR}/vslam_only_trajectory_plot.png"
            plt.savefig(vslam_plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ График траектории vSLAM сохранён: {os.path.abspath(vslam_plot_filename)}")
            
            # Вычисляем общую длину траектории
            total_distance = math.sqrt((trajectory_x[-1] - trajectory_x[0])**2 + (trajectory_z[-1] - trajectory_z[0])**2)
            
            print(f"\n📊 Итоговая статистика (чистый vSLAM):")
            print(f"   Всего кадров обработано: {frame_id}")
            print(f"   Точек в траектории vSLAM: {len(trajectory_x)}")
            print(f"   Прямое расстояние старт-финиш: {total_distance:.2f} м")
        else:
            print("⚠️ Траектория vSLAM пуста, график не построен.")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Итого: {elapsed:.2f} сек ({elapsed/60:.1f} мин)")
    print(f"   Обработано кадров: {frame_id}")
    print(f"   Средняя скорость: {frame_id/elapsed:.1f} FPS (входной поток: {FPS} FPS)")
    print(f"   Ускорение относительно реального времени: {FPS*frame_id/elapsed:.1f}x")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == '__main__':
    main()