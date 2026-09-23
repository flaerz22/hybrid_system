#!/usr/bin/env python3
"""
Гибридная навигационная система GPS + vSLAM
Главный оркестратор системы.
"""

import os
import csv
import cv2 as cv
import numpy as np
import pygame
import math

# === Импорты из sources ===
from sources.slam import Slam
from sources.render import Renderer3D
from sources.gps_monitor import GPSMonitor
from sources.video_reader_realtime import FolderVideoCapture

# === Импорты из utils ===
from utils.geo_utils import extract_lat_lon_from_pos, gps_to_meters
from utils.trajectory_utils import align_and_merge_trajectories, save_and_plot_trajectories
from utils.visualization_utils import draw_stats, draw_status


# =============================================================================
# КОНФИГУРАЦИЯ ПУТЕЙ И ФАЙЛОВ
# =============================================================================
# Входные данные
INPUT_VIDEO_FOLDER = 'video_stream_live_123682'
GPS_REALTIME_FILE = 'data/gps_realtime_sim_123682.csv'
REFERENCE_GPS_FILE = 'data/test_gps_123682_2min.csv'

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


# =============================================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ СИСТЕМЫ
# =============================================================================
slam = None
renderer = None
last_frame_pixels = None

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
        if hasattr(slam.vision, 'masker'):
            slam.vision.masker.stop()
        if hasattr(slam.vision, 'close_logger'):
            slam.vision.close_logger()

    # Создаём НОВЫЙ объект Slam
    slam = Slam(width, height)
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

    # === 1. Инициализация источников данных ===
    video = FolderVideoCapture(f'./{INPUT_VIDEO_FOLDER}/', fps=FPS, timeout=5.0)
    if not video.isOpened():
        print("Failed to read video")
        return

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    video_dim = (width, height)

    gps_monitor = GPSMonitor(GPS_REALTIME_FILE, timeout_seconds=GPS_TIMEOUT_SECONDS, verbose=True)
    gps_monitor.start()

    # === 2. Настройка окон ===
    cv.namedWindow('vSLAM', cv.WINDOW_NORMAL)
    cv.resizeWindow('vSLAM', video_dim[0], video_dim[1])
    cv.namedWindow('Mask Debug', cv.WINDOW_NORMAL)
    cv.resizeWindow('Mask Debug', 320, 240)

    print("🟢 Starting hybrid navigation system...")
    print("🎯 vSLAM работает ВСЕГДА, GPS сбрасывает его при каждой новой точке")
    print("Press 'q' to stop.\n")

    # === 3. Главный цикл ===
    frame_skip_counter = 0

    while True:
        # --- ПРОВЕРКА СОСТОЯНИЯ GPS ---
        gps_state = gps_monitor.get_state()
        last_gps = gps_monitor.get_last_gps()

        # === ОБРАБОТКА GPS СОБЫТИЙ ===
        if gps_state == 'GPS_OK' and last_gps is not None:
            if (last_gps_state != 'GPS_OK' or
                current_gps_anchor is None or
                current_gps_anchor['date_creation'] != last_gps['date_creation']):

                # Сохраняем старый сегмент ПЕРЕД сбросом
                save_current_segment()
                # Сбрасываем vSLAM и привязываем к новой GPS точке
                reset_vslam_with_gps_anchor(last_gps, width, height)

        elif gps_state == 'VSLAM_ACTIVE' and slam is not None:
            if not vslam_recording_active:
                print("\n📝 GPS пропал! Начинаем использовать vSLAM как основной источник.")
                vslam_recording_active = True

        last_gps_state = gps_state

        # === ЧТЕНИЕ И ОБРАБОТКА КАДРОВ ===
        ret, frame_pixels = video.read()
        if not ret:
            print("\n⏹️ Видеопоток завершён")
            break

        frame_id += 1
        frame_pixels = cv.resize(frame_pixels, video_dim, interpolation=cv.INTER_AREA)

        # Если vSLAM ещё не инициализирован — просто ждём
        if slam is None:
            cv.imshow('vSLAM', frame_pixels)
            if cv.waitKey(1) == ord('q'):
                break
            continue

        # --- ЛОГИКА ПРОПУСКА КАДРОВ ---
        frame_skip_counter += 1
        if frame_skip_counter % FRAME_SKIP_RATE != 0:
            cv.putText(frame_pixels,
                       f"FRAME SKIPPED ({frame_skip_counter % FRAME_SKIP_RATE}/{FRAME_SKIP_RATE})",
                       (30, frame_pixels.shape[0] - 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow('vSLAM', frame_pixels)
            if cv.waitKey(1) == ord('q'):
                break
            continue

        # === ПОЛНАЯ ОБРАБОТКА vSLAM ===
        slam.vision.masker.submit_frame(frame_pixels)

        if last_frame_pixels is not None and not (frame_pixels == last_frame_pixels).all():
            slam.update_frame_pixels(current_frame_pixels=frame_pixels,
                                     last_frame_pixels=last_frame_pixels)

            matches, frame_pixels = slam.get_vision_matches(frame_pixels)

            if matches is not None:
                points = slam.triangulate(matches)
                if points is not None:
                    renderer.render3dSpace(points, slam.get_camera_poses())

                    R_total, T_total = slam.vision.get_pose_cumulation()
                    t_flat = T_total.flatten()
                    current_x = float(t_flat[0])
                    current_z = float(t_flat[2])

                    # Записываем относительную траекторию
                    trajectory_x.append(current_x)
                    trajectory_z.append(current_z)

        last_frame_pixels = frame_pixels.copy()

        # === ВИЗУАЛИЗАЦИЯ ===
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
            print("\n⏹️ Остановка по команде пользователя")
            break

    # =============================================================================
    # 4. ЗАВЕРШЕНИЕ РАБОТЫ
    # =============================================================================
    # Сохраняем последнюю траекторию (если видео закончилось, а GPS так и не вернулся)
    save_current_segment()

    # Освобождаем ресурсы vSLAM
    if slam is not None:
        if hasattr(slam.vision, 'masker'):
            slam.vision.masker.stop()
        if hasattr(slam.vision, 'close_logger'):
            slam.vision.close_logger()
        if hasattr(slam.vision, 'print_mask_statistics'):
            slam.vision.print_mask_statistics()

    video.release()
    cv.destroyAllWindows()
    gps_monitor.stop()
    pygame.quit()

    print("\n🛑 Hybrid Navigation System Finished")
    print(f"   Всего кадров обработано: {frame_id}")
    print(f"   Всего сбросов vSLAM: {vslam_reset_count}")
    print(f"   Всего сегментов траектории сохранено: {len(trajectory_gps_anchors)}")

    # =============================================================================
    # 5. ПОСТ-ОБРАБОТКА: слияние и графики
    # =============================================================================
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


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == '__main__':
    main()