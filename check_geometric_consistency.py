#=====================================================================
# ШАГ 2: Точнаяn (Weak Geometric Consistency)
# =====================================================================

import cv2
import numpy as np

def extract_sift_features(image):
    """Извлекает ключевые точки и дескрипторы SIFT"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def check_geometric_consistency(kp_query, des_query, kp_candidate, des_candidate, threshold_ratio=0.3):
    """
    Проверяет согласованность угла и масштаба (WGC) между двумя картинками.
    
    Args:
        threshold_ratio: минимальная доля совпадений с одинаковым углом/масштабом (по умолчанию 0.3 = 30%)
    """
    if des_query is None or des_candidate is None:
        return False, 0, 0, 0, 0
        
    # Находим соответствия через Brute-Force матчер
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_candidate, k=2)
    
    # Тест Лоу (Ratio test) для фильтрации шума
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 10:
        return False, 0, 0, 0, 0  # Too few common points
        
    angles = []
    scales = []
    
    for m in good_matches:
        # Точка на картинке-запросе
        pt_q = kp_query[m.queryIdx]
        # Точка на картинке-кандидате из базы
        pt_c = kp_candidate[m.trainIdx]
        
        # Разница углов (в градусах)
        angle_diff = (pt_q.angle - pt_c.angle) % 360
        angles.append(angle_diff)
        
        # Отношение масштабов
        if pt_c.size > 0:
            scale_ratio = pt_q.size / pt_c.size
            scales.append(scale_ratio)

    # WGC: Проверяем, образуют ли углы и масштабы пик в гистограмме
    # Если картинка просто повернута/обрезана, большинство точек покажут одинаковую разницу
    
    # 1. Проверка по углу (допуск ±15 градусов)
    hist_angles, bins_angles = np.histogram(angles, bins=24, range=(0, 360))
    max_angle_votes = np.max(hist_angles)
    
    # 2. Проверка по масштабу (допуск по логарифмической шкале)
    log_scales = np.log2(scales)
    hist_scales, bins_scales = np.histogram(log_scales, bins=20, range=(-3, 3))
    max_scale_votes = np.max(hist_scales)
    
    # Условие прохождения WGC: порог определяется параметром threshold_ratio
    threshold = len(good_matches) * threshold_ratio
    
    is_valid = (max_angle_votes > threshold) and (max_scale_votes > threshold)
    
    if is_valid:
        # Считаем средний угол и масштаб из доминирующего бина
        best_angle_bin = np.argmax(hist_angles)
        avg_angle = (bins_angles[best_angle_bin] + bins_angles[best_angle_bin+1]) / 2
        
        best_scale_bin = np.argmax(hist_scales)
        avg_scale = 2**((bins_scales[best_scale_bin] + bins_scales[best_scale_bin+1]) / 2)
        
        return True, avg_angle, avg_scale, max_angle_votes, max_scale_votes
        
    return False, 0, 0, 0, 0
