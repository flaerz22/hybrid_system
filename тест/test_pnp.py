import numpy as np
import cv2


# ----------------------------------------
# 1. Известные 3D-точки в мировых координатах
# ----------------------------------------

# Например, несколько точек в метрах
object_points = np.array([
    [-0.5, -0.5, 0.0],
    [ 0.5, -0.5, 0.0],
    [-0.5,  0.5, 0.0],
    [ 0.0,  1.0, 0.5]
], dtype=np.float32)


# ----------------------------------------
# 2. Параметры виртуальной камеры
# ----------------------------------------

image_width = 1280
image_height = 720

fx = 800.0
fy = 800.0

cx = image_width / 2.0
cy = image_height / 2.0

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Идеальная камера без дисторсии
dist_coeffs = np.zeros(5, dtype=np.float64)


# ----------------------------------------
# 3. Истинное положение и ориентация камеры
# ----------------------------------------

# Поворот камеры в виде вектора Родригеса
rvec_true = np.array([0.15, -0.25, 0.05], dtype=np.float64)

# Трансляция
# Формула OpenCV:
# X_camera = R * X_world + t
tvec_true = np.array([0.1, -0.2, 4.0], dtype=np.float64)


# Получим матрицу поворота для истинной позы
R_true, _ = cv2.Rodrigues(rvec_true)

# Истинный центр камеры в мировой системе координат:
# C = -R^T * t
camera_center_true = -R_true.T @ tvec_true

print("Истинный центр камеры:")
print(camera_center_true)


# ----------------------------------------
# 4. "Фотографируем" точки этой камерой
# ----------------------------------------

image_points, _ = cv2.projectPoints(
    object_points.reshape(-1, 1, 3),
    rvec_true,
    tvec_true,
    K,
    dist_coeffs
)

image_points = image_points.reshape(-1, 2)

print("\nСпроецированные пиксельные координаты точек:")
print(image_points)


# ----------------------------------------
# 5. Можно добавить шум, как на реальном снимке
# ----------------------------------------

noise_stddev_pixels = 0.3

image_points_noisy = image_points + np.random.normal(
    0.0,
    noise_stddev_pixels,
    image_points.shape
)

image_points_noisy = image_points_noisy.astype(np.float32)


# ----------------------------------------
# 6. Восстанавливаем позу камеры по 3D-2D точкам
# ----------------------------------------

success, rvec_est, tvec_est = cv2.solvePnP(
    object_points.reshape(-1, 1, 3),
    image_points.reshape(-1, 1, 2),
    K,
    dist_coeffs,
    flags=cv2.SOLVEPNP_EPNP
)

if not success:
    raise RuntimeError("solvePnP не смог найти решение")


# ----------------------------------------
# 7. Получаем восстановленный центр камеры
# ----------------------------------------

R_est, _ = cv2.Rodrigues(rvec_est)

camera_center_est = -R_est.T @ tvec_est

print("\nВосстановленный центр камеры:")
print(camera_center_est)