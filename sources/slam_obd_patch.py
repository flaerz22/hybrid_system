"""
SLAM
"""

#!/usr/bin python3

import numpy as np
import cv2 as cv
from typing import Tuple, List, Optional, Dict, Any
import math
import copy
from collections import deque  # <-- Для motion_history

from sources.masker import DynamicMasker

# ========================================
# rappel matrice:
# Camera matrix (K) - encodes the intrinsic parameters of a camera, including the focal length and principal point, relates points in the world to points in the images
# Essential matrix (E) - Contains information about the relative rotation and translation between the two cameras
# Fundamental matrix (F) - similar to the essential matrix, but it is not used in this case 

class Frame():
    def __init__(self) -> None:
        self.pixels = None
        self.kps = None
        self.des = None
        self.E = None
        self.pose = dict()
        self.pose['R'] = np.eye(3)
        self.pose['t'] = np.zeros((3, 1))
    
    def copy(self, that_frame, pixels):
        self.pixels = pixels.copy()
        self.kps = that_frame.kps
        self.des = that_frame.des
    
    def __str__(self) -> str:
        return f"Data: {len(self.pixels)}, kps: {self.kps}, des: {self.des}, E: {self.E}, pose: {self.pose}"

class Vision():
    def __init__(self, video_dim: Tuple[int, int], 
                 _focal: float = None,
                 _focal_x: float = None,
                 _focal_y: float = None,
                 camera_matrix: np.ndarray = None,
                 dist_coeffs: np.ndarray = None,
                 use_masker: bool = True) -> None:
        # =========================================================================
        # 1. КАЛИБРОВКА КАМЕРЫ
        # =========================================================================
        if camera_matrix is not None:
            # Используем реальную калибровку
            self.K = camera_matrix.astype(np.float64)
            self.focal_x = float(self.K[0, 0])
            self.focal_y = float(self.K[1, 1])
            self.cx = float(self.K[0, 2])
            self.cy = float(self.K[1, 2])
            self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
            print(f"    SLAM: используем реальную калибровку камеры")
            print(f"      focal_x={self.focal_x:.2f}, focal_y={self.focal_y:.2f}")
            print(f"      cx={self.cx:.1f}, cy={self.cy:.1f}")
        else:
            # Fallback: используем фокусное расстояние
            if _focal_x is not None and _focal_y is not None:
                self.focal_x = _focal_x
                self.focal_y = _focal_y
            elif _focal is not None:
                self.focal_x = _focal
                self.focal_y = _focal
            else:
                self.focal_x = 811.27
                self.focal_y = 811.27
            
            self.cx = video_dim[0] // 2
            self.cy = video_dim[1] // 2
            self.K = np.array([[self.focal_x, 0, self.cx],
                               [0, self.focal_y, self.cy],
                               [0, 0, 1]])
            self.dist_coeffs = np.zeros(4)
            self.dist_coeffs[0] = -0.3
            print(f"    SLAM: используем калибровку по умолчанию (focal_x={self.focal_x}, focal_y={self.focal_y})")
        
        # =========================================================================
        # 2. DETECTOR & MATCHER (ORB)
        # =========================================================================
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.feats = None
        
        # =========================================================================
        # 3. FRAME OBJECTS
        # =========================================================================
        self.current_frame = Frame()
        self.last_frame = Frame()
        self.matches = None
        
        # =========================================================================
        # 4. POSE ACCUMULATION
        # =========================================================================
        self.camera_poses = []
        self.T_total = np.zeros((3, 1))
        self.R_total = np.eye(3)
        
        # =========================================================================
        # 5. YOLO MASKER (Dynamic Object Filtering)
        # =========================================================================
        self.USE_MASKER = use_masker

        if self.USE_MASKER:
            self.masker = DynamicMasker(
                model_path='yolov8n-seg.pt',
                input_size=(480, 480),
                conf_threshold=0.25
            )
            self.masker.start()
        else:
            self.masker = None
            print("    ️ Masker ОТКЛЮЧЕН — динамические объекты не фильтруются")
        
        # =========================================================================
        # 6. STATIONARY DETECTION
        # =========================================================================
        self.stationary_frame_count = 0
        self.is_stationary = False
        self.PIXEL_MOVE_THRESHOLD = 1.5
        self.PIXEL_STOP_THRESHOLD = 0.5
        self.STATIONARY_FRAMES_REQUIRED = 5 # 5
        self.motion_history = deque(maxlen=5)
        self.last_T_total = np.zeros((3, 1))
        
        # =========================================================================
        # 7. STATISTICS
        # =========================================================================
        self.points_total = 0
        self.points_filtered_by_mask = 0
        self.current_motion_magnitude = 0.0
        self.current_inlier_ratio = 0.0
        self.current_mask_coverage = 0.0
        self.total_matches_all_frames = 0
        self.filtered_by_mask_all_frames = 0
        
        # =========================================================================
        # 8. MISCELLANEOUS
        # =========================================================================
        self.frame_id = 0

    def update_motion_state(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """
        Вычисляет среднее пиксельное смещение между двумя кадрами 
        с помощью sparse optical flow (Lucas-Kanade) и обновляет состояние остановки.
        Вызывается для каждого кадра видеопотока (даже если SLAM его пропускает).
        """
        if prev_frame is None or curr_frame is None:
            return

        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

        # Находим угловые точки в предыдущем кадре (мало, чтобы было быстро)
        pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
        if pts is None:
            return
        pts = pts.reshape(-1, 1, 2)

        # Вычисляем оптический поток
        next_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None,
                                                    winSize=(15, 15), maxLevel=2)

        # Отбираем успешные треки
        good_old = pts[status == 1]
        good_new = next_pts[status == 1]
        if len(good_old) == 0:
            return

        # Вычисляем евклидово расстояние между точками (по последней оси)
        displacements = np.linalg.norm(good_new - good_old, axis=-1).flatten()
        avg_motion = np.mean(displacements)

        # Обновляем историю движения и сглаженное значение
        self.motion_history.append(avg_motion)
        smoothed_motion = np.mean(self.motion_history)

        # Детекция остановки (ZUPT)
        if not self.is_stationary:
            if smoothed_motion < self.PIXEL_STOP_THRESHOLD:
                self.stationary_frame_count += 1
                if self.stationary_frame_count >= self.STATIONARY_FRAMES_REQUIRED:
                    self.is_stationary = True
            else:
                self.stationary_frame_count = 0
        else:
            if smoothed_motion > self.PIXEL_MOVE_THRESHOLD:
                self.is_stationary = False
                self.stationary_frame_count = 0

        # (опционально) сохраняем текущее значение для статистики
        self.current_motion_magnitude = avg_motion
    
    def get_camera_poses(self):
        return self.camera_poses
    
    def camera_pose_to_opengl(self, T_total, R_total):
        pose = dict()
        pose['R'] = R_total
        pose['t'] = T_total
        corrected_pose = copy.deepcopy(pose)
        corrected_pose['t'][1] = 0
        corrected_pose['t'][2] *= -1
        return corrected_pose

    def get_camera_pose(self, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                    frame_id: int = 0, static_mask: Optional[np.ndarray] = None,
                    current_speed_kmh: float = 0.0, speed_scale_factor: float = 0.01,
                    delta_time_sec=0.033):
        """
        Оценивает относительную позу...
        t_scaled = t_unit * (speed * k)
        """
        assert matches is not None, "No matches given"
        
        # ... (код извлечения c1, c2, K_contiguous, findEssentialMat - без изменений) ...
        c1 = [pt1 for pt1, pt2 in matches]
        c2 = [pt2 for pt1, pt2 in matches]
        c1 = np.ascontiguousarray(np.array(c1, dtype=np.float64)).reshape(-1, 1, 2)
        c2 = np.ascontiguousarray(np.array(c2, dtype=np.float64)).reshape(-1, 1, 2)
        K_contiguous = np.ascontiguousarray(self.K, dtype=np.float64)

        E, mask = cv.findEssentialMat(c1, c2, K_contiguous, cv.RANSAC, 0.999, 1.0)
        
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
            self.current_inlier_ratio = inlier_ratio

        _, R, t, _ = cv.recoverPose(E, c1, c2, K_contiguous)

        # === ЛОГИКА МАСШТАБИРОВАНИЯ ПО СКОРОСТИ (y = kx) ===
        if self.is_stationary:
            scaled_t = np.zeros((3, 1))
        else:
            speed_ms = current_speed_kmh / 3.6
            step_length = speed_ms * delta_time_sec * speed_scale_factor
            
            if step_length < 0.005 and current_speed_kmh > 0:
                step_length = 0.005
                
            scaled_t = t * step_length

        pose = {'R': R, 't': scaled_t}
        
        # ... (сохранение позы, накопление T_total - без изменений) ...
        self.current_frame.E = E
        self.current_frame.pose = pose
        self.camera_poses.append(self.camera_pose_to_opengl(self.T_total, self.R_total))

        if not self.is_stationary:
            self.T_total += self.R_total @ pose['t']
            self.R_total = self.R_total @ pose['R']
            self.last_T_total = self.T_total.copy()
        
        # Статистика по маске (если передана)
        if static_mask is not None:
            total_px = static_mask.size
            masked_px = np.sum(static_mask == 0)
            self.current_mask_coverage = (masked_px / total_px) * 100 if total_px > 0 else 0

    def get_pose_cumulation(self):
        return self.R_total, self.T_total

    def distance_between_points(self, pt1: float, pt2: float):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def find_matching_points(self, current_frame: Frame):
        assert current_frame.pixels is not None, "No frame passed"
        
        # === Получаем маску от YOLO (только если включен) ===
        static_mask = None
        if self.USE_MASKER and self.masker is not None:
            static_mask = self.masker.get_mask_for_frame(current_frame.pixels.shape)
        
        # Копируем кадр для детекции
        frame_for_detection = current_frame.pixels.copy()
        
        # Если маска есть - закрашиваем маскированные зоны серым
        if static_mask is not None:
            frame_for_detection[static_mask == 0] = 128
        
        match = np.mean(frame_for_detection, axis=2).astype(np.uint8)
        feats = cv.goodFeaturesToTrack(match, maxCorners=3000, qualityLevel=0.01, minDistance=7)
        
        if feats is None:
            return None
        
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(frame_for_detection, kps)
        
        self.feats = feats
        self.current_frame.kps = kps
        self.current_frame.des = des
        
        if self.last_frame.kps is None or self.last_frame.des is None:
            return None
        
        frame_total = 0
        frame_filtered = 0
        self.matches = []
        
        for m in self.matcher.match(des, self.last_frame.des):
            kp1 = self.current_frame.kps[m.queryIdx].pt
            kp2 = self.last_frame.kps[m.trainIdx].pt
            frame_total += 1
            
            if self.distance_between_points(kp1, kp2) > 200:
                continue
            
            if kp1 != kp2:
                # === Фильтрация по маске (только если включен) ===
                if static_mask is not None:
                    x, y = int(kp1[0]), int(kp1[1])
                    h, w = static_mask.shape
                    if 0 <= y < h and 0 <= x < w:
                        if static_mask[y, x] == 0:
                            frame_filtered += 1
                            continue
                
                self.matches.append((kp1, kp2))
        
        self.total_matches_all_frames += frame_total
        self.filtered_by_mask_all_frames += frame_filtered
        
        self.points_total = frame_total
        self.points_filtered_by_mask = frame_filtered
        
        if len(self.matches) == 0:
            return None
        
        return self.matches

    def view_interest_points(self, frame: Frame, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        assert matches != None, "No matches passed"
        assert frame.pixels is not None, "No frame passed"

        for i, (pt1, pt2) in enumerate(matches):
            assert pt1 != pt2, "Points are the same"
            # current frame
            cv.circle(frame.pixels, (int(pt1[0]), int(pt1[1])), color=(0, 255, 255), radius=4)
            # previous frame
            cv.circle(frame.pixels, (int(pt2[0]), int(pt2[1])), color=(255, 0, 255), radius=3)
            # line
            cv.line(frame.pixels, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=(38, 207, 63), thickness=1)
        return frame.pixels

    def apply_mask_visualization(self, frame: np.ndarray, mode: str = "black_fill") -> np.ndarray:
        """
        Накладывает визуализацию маски на кадр.
        mode: 'black_fill' | 'red_overlay' | 'border'
        """
        if not self.USE_MASKER or self.masker is None:
            return frame

        static_mask = self.masker.get_mask_for_frame(frame.shape)
        if static_mask is None:
            return frame
            
        result = frame.copy()
        
        if mode == "black_fill":
            result[static_mask == 0] = [0, 0, 0]
            cv.putText(result, "MASKED: Person/Car (BLACK)", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif mode == "red_overlay":
            overlay = frame.copy()
            overlay[static_mask == 0] = [0, 0, 255]
            result = cv.addWeighted(overlay, 0.4, frame, 0.6, 0)
            cv.putText(result, "MASKED: Person/Car (RED)", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result

    def debug_show_mask(self, frame: np.ndarray, window_name: str = "Mask Debug"):
        """Показывает сырую бинарную маску в отдельном окне"""
        if not self.USE_MASKER or self.masker is None:
            return

        static_mask = self.masker.get_mask_for_frame(frame.shape)
        
        if static_mask is None:
            debug_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            cv.putText(debug_img, "NO MASK YET", (50, 100), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.imshow(window_name, debug_img)
            return
            
        mask_bgr = cv.cvtColor(static_mask, cv.COLOR_GRAY2BGR)
        
        total_pixels = static_mask.size
        masked_pixels = np.sum(static_mask == 0)
        mask_percent = (masked_pixels / total_pixels) * 100
        
        info_text = [
            f"Masked: {masked_pixels}/{total_pixels} px",
            f"Coverage: {mask_percent:.1f}%",
            f"Classes: person(0), car(2)"
        ]
        
        for i, text in enumerate(info_text):
            cv.putText(mask_bgr, text, (10, 30 + i * 25), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv.imshow(window_name, mask_bgr)
        
    def print_mask_statistics(self):
        """Выводит итоговую статистику работы маски в консоль"""
        if self.total_matches_all_frames > 0:
            percent = (self.filtered_by_mask_all_frames / self.total_matches_all_frames) * 100
        else:
            percent = 0.0
            
        print("\n" + "="*60)
        print("📊 YOLO MASK STATISTICS")
        print("="*60)
        print(f"Total matches found:     {self.total_matches_all_frames}")
        print(f"Filtered by mask:        {self.filtered_by_mask_all_frames}")
        print(f"Filter rate:             {percent:.2f}%")
        print(f"Used for SLAM:           {self.total_matches_all_frames - self.filtered_by_mask_all_frames}")
        print("="*60 + "\n")


class Slam():
    def __init__(self, width: int, height: int, 
                 camera_matrix: np.ndarray = None,
                 dist_coeffs: np.ndarray = None,
                 use_masker: bool = True) -> None:
        # Извлекаем focal из матрицы камеры, если она есть
        focal_x = None
        focal_y = None
        if camera_matrix is not None:
            focal_x = float(camera_matrix[0, 0])
            focal_y = float(camera_matrix[1, 1])
        
        self.vision = Vision(
            video_dim=(width, height), 
            _focal_x=focal_x,
            _focal_y=focal_y,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            use_masker=use_masker
        )
        self._projection_matrix = None
        self._past_projection_matrix = None
        self.E_buffer = None
        self.pose_buffer = None
        self.points_centroid = None
        self.points3Dcumulative = []
    
    @property
    def projection_matrix(self):
        return self._projection_matrix
    
    @property
    def past_projection_matrix(self):
        return self._past_projection_matrix
    
    def get_camera_poses(self):
        return self.vision.get_camera_poses()

    def update_frame_pixels(self, current_frame_pixels: np.ndarray, last_frame_pixels: np.ndarray):
        assert current_frame_pixels is not None, "No frame passed"
        assert last_frame_pixels is not None, "No last frame passed"
        assert (current_frame_pixels==last_frame_pixels).all() == False, "Frames are the same"
        if last_frame_pixels is not None:
            self.vision.last_frame.copy(that_frame=self.vision.current_frame, pixels=last_frame_pixels)
        self.vision.current_frame.pixels = current_frame_pixels.copy()
    
    def get_vision_matches(self, render_frame, current_speed_kmh: float = 0.0,
                            speed_scale_factor: float = 0.01, delta_time_sec=0.033):
        assert render_frame is not None, "No frame for rendering"
        assert self.vision.current_frame.pixels is not None, "No frame passed"
        assert self.vision.last_frame.pixels is not None, "No last frame"
        assert (self.vision.current_frame.pixels==self.vision.last_frame.pixels).all() == False, "Frames are the same"
        matches = self.vision.find_matching_points(self.vision.current_frame)
        if matches is not None:
            render_frame = self.vision.view_interest_points(self.vision.current_frame, matches)
            # Передаем скорость дальше
            self.vision.get_camera_pose(matches, current_speed_kmh=current_speed_kmh,
                                         speed_scale_factor=speed_scale_factor, delta_time_sec=delta_time_sec)
            return matches, render_frame
        print("No matches found")
        return None, render_frame

    def hand_rule_change(self, points3D):
        assert points3D is not None, "points3D None"
        assert points3D.shape[0] > 4, "Points4D not 4xN"
        # change coordinate opencv to openGL
        points3D[:, 1] *= -1
        points3D[:, 2] *= -1
        return points3D
    
    def transform_points_3D_openGL(self, points3D):
        assert points3D is not None, "points3D None"
        assert points3D.shape[0] == 3, "Points3D not 3D"
        R_total, t_total = self.vision.get_pose_cumulation()
        pose_corrected = self.vision.camera_pose_to_opengl(t_total, R_total)
        return np.dot(points3D.T, -pose_corrected['R']) + pose_corrected['t'].T
    
    def project_points(self, points3D):
        inv = np.linalg.inv(self.vision.K)
        points3D_projected = np.dot(inv, points3D)
        return points3D_projected
    
    def triangulate(self, matches: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        assert matches != None, "matches is None"
        assert len(matches) > 0, "No matches passed"
        assert self.vision.current_frame.E is not None, "current essential matrix is None"
        assert self.vision.current_frame.pose is not None, "current pose is None"
        if self.vision.last_frame.E is None:
            self.vision.last_frame.E = self.vision.current_frame.E
            self.vision.last_frame.pose = self.vision.current_frame.pose
        self._projection_matrix = np.hstack((self.vision.current_frame.pose['R'],
                                       self.vision.current_frame.pose['t']))
        self._past_projection_matrix = np.hstack((self.vision.last_frame.pose['R'],
                                            self.vision.last_frame.pose['t']))
        projPoints1 = []
        projPoints2 = []
        for kp1, kp2 in matches:
            projPoints1.append([kp1[0], kp1[1]])
            projPoints2.append([kp2[0], kp2[1]])
        projPoints1 = np.array(projPoints1).T  # (2, N)
        projPoints2 = np.array(projPoints2).T
        K = self.vision.K
        projMat1 = K @ cv.hconcat([self.vision.current_frame.pose['R'], self.vision.current_frame.pose['t']])
        projMat2 = K @ cv.hconcat([self.vision.last_frame.pose['R'], self.vision.last_frame.pose['t']])
        points1u = cv.undistortPoints(projPoints1, K, 1, None, K)
        points2u = cv.undistortPoints(projPoints2, K, 1, None, K)
        points4D = cv.triangulatePoints(projMat1, projMat2, points1u, points2u)
        # 3d
        points3D = (points4D[:3] / points4D[3]) # normalize and go 3d
        goods = np.abs(points4D[3, :]) > 0.005 # check w good
        goods &= points3D[1, :] > 0 # check point are not underground
        goods &= points3D[2, :] > 0 # check points are in front of camera
        goods &= points3D[2, :] < 500 # check points z are not too far
        goods &= np.abs(points3D[0, :]) < 500 # check points x are not too far
        if len(goods[goods == True]) < 25:
            return None
        points3D = points3D[:, goods]
        points3D = self.transform_points_3D_openGL(points3D)
        self.points_centroid = sum([v for v in points3D]) / len(points3D)
        self.vision.last_frame.E = self.vision.current_frame.E
        self.vision.last_frame.pose = self.vision.current_frame.pose
        point_info = (points3D, self.points_centroid)
        self.points3Dcumulative.append(point_info)
        return self.points3Dcumulative