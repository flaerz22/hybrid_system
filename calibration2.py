import cv2
import numpy as np
import os
import glob

IMAGE_FOLDER = "calib_frames"
CHESSBOARD_SIZE = (6, 4)
SQUARE_SIZE = 50.0
FILE_EXTENSION = "*.jpg"

def calibrate_camera():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Папка '{IMAGE_FOLDER}' не найдена!")
        return None, None
    
    images_path = glob.glob(os.path.join(IMAGE_FOLDER, FILE_EXTENSION))
    images_path = [f for f in images_path if "vis_" not in f and "undistorted" not in f]
    
    print(f"📁 Найдено изображений: {len(images_path)}")
    
    if len(images_path) < 30:
        print(f"⚠ Для fisheye нужно минимум 30-40 кадров. У вас {len(images_path)}.")
    
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    
    objpoints = []
    imgpoints = []
    img_shape = None
    
    print(f"\n🔍 Поиск углов...")
    print("-" * 60)
    
    for i, fname in enumerate(images_path):
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        
        # Улучшенные флаги
        flags_find = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                      cv2.CALIB_CB_FILTER_QUADS)
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags_find)
        
        if ret:
            # Увеличенное окно для точности
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            corners2 = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            vis_img = img.copy()
            cv2.drawChessboardCorners(vis_img, CHESSBOARD_SIZE, corners2, ret)
            vis_path = os.path.join(IMAGE_FOLDER, f"vis_{os.path.basename(fname)}")
            cv2.imwrite(vis_path, vis_img)
            
            print(f"✓ [{i+1}] {os.path.basename(fname)}")
        else:
            print(f"✗ [{i+1}] {os.path.basename(fname)} - доска НЕ найдена")
    
    print("-" * 60)
    print(f"\n📊 Обработано: {len(objpoints)} из {len(images_path)} кадров")
    
    if len(objpoints) < 20:
        print(f"\n❌ Слишком мало кадров ({len(objpoints)}). Нужно минимум 25-30.")
        return None, None
    
    print(f"\n🎯 Калибровка fisheye...")
    
    objpoints_f = [pts.reshape(-1, 1, 3).astype(np.float32) for pts in objpoints]
    imgpoints_f = [pts.reshape(-1, 1, 2).astype(np.float32) for pts in imgpoints]
    
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_CHECK_COND +
             cv2.fisheye.CALIB_FIX_SKEW)
    
    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_f, imgpoints_f, img_shape, None, None, flags=flags
        )
    except cv2.error as e:
        print(f"\n❌ Ошибка: {e}")
        return None, None
    
    print("\n" + "=" * 60)
    print("✓ КАЛИБРОВКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"\n Средняя ошибка репроекций: {rms:.4f} пикселей")
    
    if rms < 1.0:
        print("  ✅ Отличный результат!")
    elif rms < 1.5:
        print("  ✅ Хороший результат.")
    elif rms < 2.0:
        print("  ⚠ Приемлемый результат.")
    else:
        print(f"   Плохой результат ({rms:.2f} px). Нужно больше кадров!")
    
    print(f"\n📐 Матрица камеры:")
    print(camera_matrix)
    print(f"\n📐 Коэффициенты дисторсии:")
    print(len(dist_coeffs))
    print(dist_coeffs.ravel())
    
    np.savez("camera_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             image_size=img_shape)
    
    print(f"\n✓ Сохранено в: camera_calibration.npz")
    
    return camera_matrix, dist_coeffs


def undistort_images(camera_matrix, dist_coeffs):
    output_folder = os.path.join(IMAGE_FOLDER, "undistorted")
    os.makedirs(output_folder, exist_ok=True)
    
    images_path = glob.glob(os.path.join(IMAGE_FOLDER, FILE_EXTENSION))
    images_path = [f for f in images_path if "vis_" not in f and "undistorted" not in f]
    
    print(f"\n Коррекция {len(images_path)} изображений...")
    
    balance = 0.5
    processed = 0
    
    for fname in images_path:
        img = cv2.imread(fname)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        new_cam_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), None, balance=balance
        )
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, np.eye(3), new_cam_matrix, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)
        
        out_path = os.path.join(output_folder, os.path.basename(fname))
        cv2.imwrite(out_path, undistorted)
        processed += 1
    
    print(f"✓ Исправлено: {processed} изображений")
    print(f"📁 Папка: {output_folder}")


if __name__ == "__main__":
    print("=" * 60)
    print("🎥 КАЛИБРОВКА ВИДЕОРЕГИСТРАТОРА 170°")
    print("=" * 60)
    
    camera_matrix, dist_coeffs = calibrate_camera()
    
    if camera_matrix is not None:
        answer = input("\nПрименить коррекцию? (y/n): ")
        if answer.lower() == 'y':
            undistort_images(camera_matrix, dist_coeffs)
        
        print("\n✅ ГОТОВО!")