#!/usr/bin/env python3
"""
Интерактивное сравнение траектории vSLAM с эталонной GPS-траекторией.
1. Читает эталон из .xlsx
2. Читает vSLAM из .csv
3. Автоматически масштабирует vSLAM по отношению длин старт→финиш
4. Позволяет ВРУЧНУЮ подобрать угол поворота через ползунок
5. Сохраняет результат
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# =============================================================================
# 1. ЧТЕНИЕ ДАННЫХ
# =============================================================================
def load_reference(xlsx_path):
    """Загружает эталонную GPS-траекторию из .xlsx"""
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Файл эталона не найден: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    print(f"📂 Эталон: {os.path.abspath(xlsx_path)}")
    print(f"   Записей: {len(df)}")
    return df['x_m'].values, df['z_m'].values

def load_vslam(csv_path):
    """Загружает траекторию vSLAM из .csv"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл vSLAM не найден: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📂 vSLAM: {os.path.abspath(csv_path)}")
    print(f"   Записей: {len(df)}")
    return df['x_meters'].values, df['z_meters'].values

# =============================================================================
# 2. ГЕОМЕТРИЯ
# =============================================================================
def apply_rotation(x, z, angle_deg, pivot_x, pivot_z):
    """Поворачивает траекторию вокруг точки (pivot_x, pivot_z) на угол в градусах."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    x_rel = x - pivot_x
    z_rel = z - pivot_z
    
    x_rot = x_rel * cos_a - z_rel * sin_a
    z_rot = x_rel * sin_a + z_rel * cos_a
    
    return x_rot + pivot_x, z_rot + pivot_z

def align_origin(x_vslam, z_vslam, x_ref, z_ref):
    """Смещает первую точку vSLAM в первую точку эталона."""
    dx = x_ref[0] - x_vslam[0]
    dz = z_ref[0] - z_vslam[0]
    return x_vslam + dx, z_vslam + dz

def compute_scale_factor(x_ref, z_ref, x_vslam, z_vslam):
    """
    Вычисляет коэффициент масштабирования как отношение
    прямого расстояния старт→финиш эталона к прямому расстоянию vSLAM.
    """
    len_ref = np.hypot(x_ref[-1] - x_ref[0], z_ref[-1] - z_ref[0])
    len_vslam = np.hypot(x_vslam[-1] - x_vslam[0], z_vslam[-1] - z_vslam[0])
    
    if len_vslam < 1e-9:
        print("⚠️ Длина vSLAM ≈ 0, масштабирование невозможно")
        return 1.0
    
    scale = len_ref / len_vslam
    print(f"\n📐 АВТОМАТИЧЕСКОЕ МАСШТАБИРОВАНИЕ:")
    print(f"   Длина эталона (старт→финиш): {len_ref:.2f} м")
    print(f"   Длина vSLAM (старт→финиш):   {len_vslam:.2f} м")
    print(f"   Коэффициент масштаба:        {scale:.4f}")
    return scale

def apply_scale(x, z, scale, pivot_x, pivot_z):
    """Масштабирует траекторию относительно точки pivot."""
    x_scaled = (x - pivot_x) * scale + pivot_x
    z_scaled = (z - pivot_z) * scale + pivot_z
    return x_scaled, z_scaled

# =============================================================================
# 3. ИНТЕРАКТИВНЫЙ ПОДБОР
# =============================================================================
class InteractiveAligner:
    def __init__(self, x_ref, z_ref, x_vslam, z_vslam):
        self.x_ref = x_ref
        self.z_ref = z_ref
        self.x_vslam_raw = x_vslam
        self.z_vslam_raw = z_vslam
        
        # Автоматическое масштабирование
        self.scale = compute_scale_factor(x_ref, z_ref, x_vslam, z_vslam)
        self.x_vslam_scaled, self.z_vslam_scaled = apply_scale(
            x_vslam, z_vslam, self.scale, x_vslam[0], z_vslam[0]
        )
        
        # Начальный угол (вычисляется для МАСШТАБИРОВАННОЙ траектории)
        self.initial_angle = self._guess_initial_angle()
        self.current_angle = self.initial_angle
        
        # Финальные данные
        self.final_x_aligned = None
        self.final_z_aligned = None
        self.is_saved = False

    def _guess_initial_angle(self):
        """Примерный расчет начального угла для масштабированной траектории."""
        dx_ref = self.x_ref[-1] - self.x_ref[0]
        dz_ref = self.z_ref[-1] - self.z_ref[0]
        angle_ref = np.arctan2(dz_ref, dx_ref)
        
        dx_vslam = self.x_vslam_scaled[-1] - self.x_vslam_scaled[0]
        dz_vslam = self.z_vslam_scaled[-1] - self.z_vslam_scaled[0]
        angle_vslam = np.arctan2(dz_vslam, dx_vslam)
        
        return np.degrees(angle_ref - angle_vslam)

    def update_plot(self, val):
        """Обновляет график при движении ползунка."""
        self.current_angle = val
        
        # 1. Поворот МАСШТАБИРОВАННОЙ траектории вокруг первой точки
        x_rot, z_rot = apply_rotation(
            self.x_vslam_scaled, self.z_vslam_scaled, 
            self.current_angle, 
            self.x_vslam_scaled[0], self.z_vslam_scaled[0]
        )
        
        # 2. Смещение к началу эталона
        x_aligned, z_aligned = align_origin(x_rot, z_rot, self.x_ref, self.z_ref)
        
        # Обновляем линию на графике
        self.line_vslam.set_data(x_aligned, z_aligned)
        
        # Обновляем точку финиша vSLAM
        self.scatter_finish.set_offsets([[x_aligned[-1], z_aligned[-1]]])
        
        # Обновляем текст с углом
        self.text_angle.set_text(f'Угол: {self.current_angle:.2f}°\nМасштаб: {self.scale:.4f}')
        
        self.fig.canvas.draw_idle()

    def save_result(self, event):
        """Сохраняет текущее состояние."""
        print(f"\n💾 Сохранение результата: угол={self.current_angle:.2f}°, масштаб={self.scale:.4f}")
        self.is_saved = True
        plt.close(self.fig)

    def run(self):
        # Создаем фигуру
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Рисуем эталон
        self.ax.plot(self.x_ref, self.z_ref, 'g--', linewidth=2.5, label='Эталон (GPS)')
        
        # Рисуем начальную позицию МАСШТАБИРОВАННОЙ vSLAM
        x_rot, z_rot = apply_rotation(
            self.x_vslam_scaled, self.z_vslam_scaled, 
            self.initial_angle, 
            self.x_vslam_scaled[0], self.z_vslam_scaled[0]
        )
        x_init, z_init = align_origin(x_rot, z_rot, self.x_ref, self.z_ref)
        
        self.line_vslam, = self.ax.plot(x_init, z_init, 'b-', linewidth=2.5, label='vSLAM (масштаб+поворот)')
        
        # Точки старта/финиша
        self.ax.scatter([self.x_ref[0]], [self.z_ref[0]], color='lime', s=200, marker='^', edgecolors='k', label='Старт', zorder=5)
        self.ax.scatter([self.x_ref[-1]], [self.z_ref[-1]], color='red', s=200, marker='v', edgecolors='k', label='Финиш GPS', zorder=5)
        self.scatter_finish = self.ax.scatter([x_init[-1]], [z_init[-1]], color='cyan', s=100, marker='o', edgecolors='k', label='Финиш vSLAM', zorder=5)
        
        self.ax.set_title('Интерактивный подбор (масштаб + поворот)')
        self.ax.set_xlabel('X (метры)')
        self.ax.set_ylabel('Z (метры)')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.axis('equal')
        self.ax.legend()
        
        # Текст с текущим углом и масштабом
        self.text_angle = self.ax.text(0.05, 0.95, 
                                       f'Угол: {self.initial_angle:.2f}°\nМасштаб: {self.scale:.4f}', 
                                       transform=self.ax.transAxes, fontsize=14, 
                                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # === ПОЛЗУНОК ===
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Угол (°)', -180.0, 180.0, valinit=self.initial_angle, valstep=0.1)
        self.slider.on_changed(self.update_plot)
        
        # === КНОПКА СОХРАНЕНИЯ ===
        ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(ax_button, 'Сохранить', color='lightgreen', hovercolor='0.975')
        self.button.on_clicked(self.save_result)
        
        print("\n🎮 ИНСТРУКЦИЯ:")
        print("   1. Траектория vSLAM уже масштабирована автоматически.")
        print("   2. Крутите ползунок, чтобы совместить синюю линию с зеленой.")
        print("   3. Нажмите 'Сохранить' или закройте окно.")
        
        plt.show()
        
        # После закрытия окна возвращаем финальные данные
        x_rot, z_rot = apply_rotation(
            self.x_vslam_scaled, self.z_vslam_scaled, 
            self.current_angle, 
            self.x_vslam_scaled[0], self.z_vslam_scaled[0]
        )
        self.final_x_aligned, self.final_z_aligned = align_origin(x_rot, z_rot, self.x_ref, self.z_ref)
            
        return self.final_x_aligned, self.final_z_aligned, self.current_angle, self.scale

# =============================================================================
# 4. ВЫЧИСЛЕНИЕ ОШИБКИ
# =============================================================================
def compute_error(x_ref, z_ref, x_aligned, z_aligned, scale):
    """Вычисляет метрики ошибки и длины траекторий."""
    # Длины траекторий (прямое расстояние старт → финиш)
    len_ref = np.hypot(x_ref[-1] - x_ref[0], z_ref[-1] - z_ref[0])
    len_vslam = np.hypot(x_aligned[-1] - x_aligned[0], z_aligned[-1] - z_aligned[0])
    length_diff_pct = abs(len_ref - len_vslam) / len_ref * 100 if len_ref > 0 else 0.0
    
    # Накопленная длина пути
    total_len_ref = np.sum(np.hypot(np.diff(x_ref), np.diff(z_ref)))
    total_len_vslam = np.sum(np.hypot(np.diff(x_aligned), np.diff(z_aligned)))
    path_diff_pct = abs(total_len_ref - total_len_vslam) / total_len_ref * 100 if total_len_ref > 0 else 0.0
    
    # Интерполяция по длине пути
    s_ref = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_ref), np.diff(z_ref)))])
    s_vslam = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_aligned), np.diff(z_aligned)))])
    
    s_ref_norm = s_ref / s_ref[-1]
    s_vslam_norm = s_vslam / s_vslam[-1]
    
    x_vslam_interp = np.interp(s_ref_norm, s_vslam_norm, x_aligned)
    z_vslam_interp = np.interp(s_ref_norm, s_vslam_norm, z_aligned)
    
    errors = np.hypot(x_ref - x_vslam_interp, z_ref - z_vslam_interp)
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_err = np.max(errors)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ВЫРАВНИВАНИЯ:")
    print(f"   Масштаб:                      {scale:.4f}")
    print(f"   Прямое расстояние (эталон):   {len_ref:.2f} м")
    print(f"   Прямое расстояние (vSLAM):    {len_vslam:.2f} м  (разница: {length_diff_pct:.2f}%)")
    print(f"   Длина пути (эталон):          {total_len_ref:.2f} м")
    print(f"   Длина пути (vSLAM):           {total_len_vslam:.2f} м  (разница: {path_diff_pct:.2f}%)")
    print(f"   Средняя ошибка (MAE):         {mae:.2f} м")
    print(f"   RMSE:                         {rmse:.2f} м")
    print(f"   Максимальная ошибка:          {max_err:.2f} м")
    
    return mae, rmse, max_err

# =============================================================================
# 5. ФИНАЛЬНЫЙ ГРАФИК
# =============================================================================
def plot_final(x_ref, z_ref, x_aligned, z_aligned, angle, scale, output_dir):
    """Строит итоговый статичный график."""
    plt.figure(figsize=(12, 9))
    
    plt.plot(x_ref, z_ref, 'g--', linewidth=2.5, label='Эталон (GPS)')
    plt.plot(x_aligned, z_aligned, 'b-', linewidth=2.5, 
             label=f'vSLAM (масштаб={scale:.3f}, угол={angle:.1f}°)')
    
    plt.scatter([x_ref[0]], [z_ref[0]], color='lime', s=250, marker='^', edgecolors='k', label='Старт')
    plt.scatter([x_ref[-1]], [z_ref[-1]], color='red', s=250, marker='v', edgecolors='k', label='Финиш')
    plt.scatter([x_aligned[-1]], [z_aligned[-1]], color='cyan', s=150, marker='o', edgecolors='k', label='Финиш vSLAM')
    
    plt.title(f'Сравнение траекторий (масштаб={scale:.3f}, угол={angle:.2f}°)')
    plt.xlabel('X (метры)')
    plt.ylabel('Z (метры)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend()
    
    out_path = os.path.join(output_dir, 'manual_alignment_result_3_scale.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Итоговый график сохранён: {os.path.abspath(out_path)}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Пути к файлам
    XLSX_FILE = 'output/test_obd_3.xlsx'
    CSV_FILE = 'output/merged_trajectory.csv'
    OUTPUT_DIR = 'output'
    
    try:
        # 1. Загрузка
        x_ref, z_ref = load_reference(XLSX_FILE)
        x_vslam, z_vslam = load_vslam(CSV_FILE)
        
        # 2. Интерактивный подбор (с автоматическим масштабированием)
        aligner = InteractiveAligner(x_ref, z_ref, x_vslam, z_vslam)
        x_aligned, z_aligned, final_angle, final_scale = aligner.run()
        
        if x_aligned is None:
            print("❌ Операция отменена.")
            return

        # 3. Расчет ошибок
        compute_error(x_ref, z_ref, x_aligned, z_aligned, final_scale)
        
        # 4. Финальный график
        plot_final(x_ref, z_ref, x_aligned, z_aligned, final_angle, final_scale, OUTPUT_DIR)
        
        # 5. Сохранение CSV
        aligned_csv = os.path.join(OUTPUT_DIR, 'vslam_manual_aligned.csv')
        pd.DataFrame({
            'x_meters_raw': x_vslam,
            'z_meters_raw': z_vslam,
            'x_meters_aligned': x_aligned,
            'z_meters_aligned': z_aligned,
            'manual_angle_deg': [final_angle] * len(x_vslam),
            'scale_factor': [final_scale] * len(x_vslam)
        }).to_csv(aligned_csv, index=False)
        print(f"✅ Данные сохранены: {os.path.abspath(aligned_csv)}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()