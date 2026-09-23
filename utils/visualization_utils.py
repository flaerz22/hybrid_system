import cv2 as cv


def draw_stats(frame, slam):
    """Рисует статистику фильтрации точек в левом верхнем углу"""
    total = getattr(slam.vision, 'points_total', 0)
    filtered = getattr(slam.vision, 'points_filtered_by_mask', 0)
    used = total - filtered
    
    stats = [
        f"Total matches: {total}",
        f"Filtered by mask: {filtered}",
        f"Used for SLAM: {used}",
    ]
    
    overlay = frame.copy()
    cv.rectangle(overlay, (5, 5), (260, 80), (0, 0, 0), -1)
    frame = cv.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    for i, text in enumerate(stats):
        color = (0, 255, 0) if i == 2 else (255, 255, 255)
        cv.putText(frame, text, (15, 30 + i * 22), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_status(frame, slam):
    """Рисует статус движения в правом верхнем углу"""
    is_stationary = getattr(slam.vision, 'is_stationary', False)
    count = getattr(slam.vision, 'stationary_frame_count', 0)
    motion = getattr(slam.vision, 'current_motion_magnitude', 0)
    
    status_text = "STOPPED" if is_stationary else "MOVING"
    color = (0, 0, 255) if is_stationary else (0, 255, 0)
    
    cv.rectangle(frame, (frame.shape[1] - 220, 10), (frame.shape[1] - 10, 70), color, -1)
    cv.putText(frame, status_text, (frame.shape[1] - 210, 35), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"Frames: {count}", (frame.shape[1] - 210, 60), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(frame, f"Pixel motion: {motion:.2f}", (frame.shape[1] - 315, 100), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame