import os
import csv
import math
import matplotlib.pyplot as plt


def calculate_scale_factor(seg, gps_anchor_pt, next_gps_pt):
    """
    Вычисляет коэффициент масштабирования для сегмента vSLAM.
    
    :param seg: Словарь сегмента траектории vSLAM.
    :param gps_anchor_pt: GPS точка начала разрыва (якорь).
    :param next_gps_pt: Первая GPS точка после восстановления сигнала.
    :return: Коэффициент масштаба (float).
    """
    # Расстояние между GPS якорем и следующей GPS точкой (ИСТИННЫЙ МАСШТАБ)
    dist_gps = math.sqrt(
        (next_gps_pt['x'] - gps_anchor_pt['x'])**2 + 
        (next_gps_pt['z'] - gps_anchor_pt['z'])**2
    )
    
    # Расстояние от начала до конца сегмента vSLAM (ЛОКАЛЬНЫЙ МАСШТАБ)
    last_idx = len(seg['trajectory_x']) - 1
    dist_vslam = math.sqrt(
        (seg['trajectory_x'][last_idx] - seg['trajectory_x'][0])**2 + 
        (seg['trajectory_z'][last_idx] - seg['trajectory_z'][0])**2
    )
    
    # Защита от деления на ноль и слишком коротких сегментов
    if dist_vslam < 0.001:
        print(f"   ⚠️ Сегмент {seg['reset_num']} слишком короткий для расчета масштаба.")
        return 1.0
        
    scale = dist_gps / dist_vslam
    
    print(f"   📏 Масштаб сегмента {seg['reset_num']}: {scale:.4f} "
          f"(GPS: {dist_gps:.2f}м / vSLAM: {dist_vslam:.2f}м)")
          
    return scale


def align_and_merge_trajectories(gps_data_raw, trajectory_gps_anchors):
    """
    Объединяет траектории GPS и vSLAM, применяя векторное выравнивание 
    и согласование масштаба через отдельную функцию.
    """
    merged_trajectory = []
    
    vslam_segments_by_anchor = {}
    for seg in trajectory_gps_anchors:
        anchor_time = seg['gps_anchor'].get('date_creation')
        vslam_segments_by_anchor[anchor_time] = seg

    for idx, gps_pt in enumerate(gps_data_raw):
        # Добавляем саму GPS точку
        merged_trajectory.append({
            'time': gps_pt['time'],
            'x_meters': gps_pt['x'],
            'z_meters': gps_pt['z'],
            'source': 'GPS'
        })
        
        if gps_pt['time'] in vslam_segments_by_anchor:
            seg = vslam_segments_by_anchor[gps_pt['time']]
            
            # --- ОРИЕНТАЦИЯ (Поворот) ---
            if idx > 0:
                prev_gps = gps_data_raw[idx - 1]
                vec_gps_x = gps_pt['x'] - prev_gps['x']
                vec_gps_z = gps_pt['z'] - prev_gps['z']
            else:
                vec_gps_x, vec_gps_z = 1.0, 0.0
                
            if len(seg['trajectory_x']) >= 2:
                vec_vslam_x = seg['trajectory_x'][1] - seg['trajectory_x'][0]
                vec_vslam_z = seg['trajectory_z'][1] - seg['trajectory_z'][0]
            else:
                vec_vslam_x, vec_vslam_z = 1.0, 0.0

            angle_gps = math.atan2(vec_gps_z, vec_gps_x)
            angle_vslam = math.atan2(vec_vslam_z, vec_vslam_x)
            delta_angle = angle_gps - angle_vslam
            
            cos_a = math.cos(delta_angle)
            sin_a = math.sin(delta_angle)
            
            anchor_x = seg['trajectory_x'][0]
            anchor_z = seg['trajectory_z'][0]

            # --- МАСШТАБ (Выносим логику в отдельную функцию) ---
            scale_factor = 1.0
            if idx + 1 < len(gps_data_raw):
                next_gps = gps_data_raw[idx + 1]
                scale_factor = calculate_scale_factor(seg, gps_pt, next_gps)
            else:
                print(f"   ⚠️ Нет следующей GPS точки для масштаба сегмента {seg['reset_num']}.")

            # --- ТРАНСФОРМАЦИЯ ТОЧЕК (Поворот + Масштаб) ---
            for i in range(1, len(seg['trajectory_x'])):
                rel_x = seg['trajectory_x'][i] - anchor_x
                rel_z = seg['trajectory_z'][i] - anchor_z
                
                # Поворот
                rot_x = rel_x * cos_a - rel_z * sin_a
                rot_z = rel_x * sin_a + rel_z * cos_a
                
                # Применение масштаба
                final_x = (rot_x * scale_factor) + anchor_x
                final_z = (rot_z * scale_factor) + anchor_z
                
                merged_trajectory.append({
                    'time': f"{gps_pt['time']}_vslam_{i}",
                    'x_meters': final_x,
                    'z_meters': final_z,
                    'source': 'vSLAM'
                })
                
    return merged_trajectory


def save_and_plot_trajectories(merged_trajectory, ref_trajectory, gps_ref_lat, gps_ref_lon,
    csv_filename = "output/merged_trajectory.csv",
    plot_merged_filename = "output/merged_trajectory_plot.png",
    plot_comparison_filename = "output/hybrid_vs_full_reference.png"
):
    """Сохраняет CSV и строит сравнительные графики. Имена файлов задаются из main.py"""
    if not merged_trajectory:
        print("❌ Нет данных для построения траектории.")
        return

    # Создаем директорию на основе пути к CSV файлу
    output_dir = os.path.dirname(csv_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Сохранение CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['time', 'x_meters', 'z_meters', 'source'])
        writer.writeheader()
        writer.writerows(merged_trajectory)
    print(f"✅ Единая траектория сохранена: {os.path.abspath(csv_filename)}")

    # 2. График объединенной траектории
    plt.figure(figsize=(14, 10))
    x_all = [p['x_meters'] for p in merged_trajectory]
    z_all = [p['z_meters'] for p in merged_trajectory]
    plt.plot(x_all, z_all, color='gray', linewidth=1.5, label='Непрерывная траектория', zorder=1)
    
    x_gps = [p['x_meters'] for p in merged_trajectory if p['source'] == 'GPS']
    z_gps = [p['z_meters'] for p in merged_trajectory if p['source'] == 'GPS']
    plt.scatter(x_gps, z_gps, color='green', s=60, marker='o', edgecolors='black', label='GPS точки', zorder=3)
    
    plt.title('Единая непрерывная траектория (GPS + vSLAM)', fontsize=16, fontweight='bold')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Z Position (meters)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend()
    plt.savefig(plot_merged_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График объединенной траектории сохранен: {os.path.abspath(plot_merged_filename)}")

    # 3. Сравнение с эталоном (если есть)
    if ref_trajectory:
        plt.figure(figsize=(16, 12))
        x_ref, z_ref = [p['x_meters'] for p in ref_trajectory], [p['z_meters'] for p in ref_trajectory]
        plt.plot(x_ref, z_ref, color='green', linestyle='--', linewidth=2, alpha=0.7, label='GPS-эталон', zorder=1)
        plt.scatter(x_ref[0], z_ref[0], s=180, c='lime', edgecolors='black', marker='^', label='Старт', zorder=6)
        plt.scatter(x_ref[-1], z_ref[-1], s=180, c='red', edgecolors='black', marker='v', label='Финиш', zorder=6)
        
        plt.plot(x_all, z_all, color='blue', linewidth=2, alpha=0.9, label='Гибридная система', zorder=2)
        plt.scatter(0, 0, color='orange', s=150, marker='o', zorder=5, label='Опорная точка (0,0)', edgecolors='black')
        
        plt.title('Сравнение: Гибридная система vs GPS-эталон', fontsize=16, fontweight='bold')
        plt.xlabel('X (м)')
        plt.ylabel('Z (м)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.legend()
        plt.savefig(plot_comparison_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Сравнительный график сохранен: {os.path.abspath(plot_comparison_filename)}")