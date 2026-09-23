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
    """Рисует статус движения и режим одометра в правом верхнем углу"""
    is_stationary = getattr(slam.vision, 'is_stationary', False)
    is_turning = getattr(slam.vision, 'is_turning', False)
    count = getattr(slam.vision, 'stationary_frame_count', 0)
    motion = getattr(slam.vision, 'current_motion_magnitude', 0)
    step_len = getattr(slam.vision, 'current_step_len', 0.0)  # Длина текущего шага (если добавили)

    # Определяем режим и цвет
    if is_stationary:
        status_text = "STOP"
        color = (0, 0, 255)       # Красный
    elif is_turning:
        status_text = "TURN"
        color = (0, 165, 255)     # Оранжевый
    else:
        status_text = "STRAIGHT"
        color = (0, 255, 0)       # Зеленый

    # Рисуем плашку статуса
    cv.rectangle(frame, (frame.shape[1] - 220, 10), (frame.shape[1] - 10, 70), color, -1)
    cv.putText(frame, status_text, (frame.shape[1] - 210, 35), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"Frames: {count}", (frame.shape[1] - 210, 60), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Рисуем дополнительную информацию (поток и длина шага)
    y_offset = frame.shape[1] - 315
    cv.putText(frame, f"Pixel motion: {motion:.2f}", (y_offset, 100), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv.putText(frame, f"Step length: {step_len:.3f} m", (y_offset, 125), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
               
    return frame