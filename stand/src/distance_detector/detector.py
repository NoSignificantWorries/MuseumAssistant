import os
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


class InteractiveStandDetector:
    """
    Класс для детекции человека и оценки дистанции до него на видео.
    Используется модель YOLOv8 Pose, которая определяет ключевые точки тела человека.
    Оценка дистанции происходит на основе размера bounding box, при условии, что
    обнаружены ключевые точки головы (нос) и хотя бы одного плеча.

    Атрибуты:
        model_path (str): Путь к локальной модели YOLOv8.
        model (YOLO): Загруженная модель YOLOv8 для pose-detection.
        activation_distance (float): Порог активации дистанции (в метрах).
        speed_history (deque): История скоростей центра bbox для анализа замедления.
        last_center (tuple): Последняя позиция центра человека.

    Методы:
        get_person_depth(frame): Принимает кадр видео, возвращает расстояние до человека,
                                если определена голова и плечи с достаточной уверенностью.
                                Возвращает None, если человек не обнаружен или confidence низкий.
        _analyze_speed(self, center): Анализирует скорость движения центра bbox.
    """

    def __init__(self):

        self.model_path = Path("../shared/yolo_weights/yolov8n-pose.pt").resolve().absolute()

        print(self.model_path, self.model_path.exists())
    
        # Создаем директорию для модели, если ее нет
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Если локальная модель отсутствует — скачиваем и сохраняем локально
        if not os.path.exists(self.model_path):
            print("Скачивание YOLOv8 Pose...")
            temp_model = YOLO('yolov8n-pose.pt')  # Скачивается в кэш Ultralytics
            temp_model.save(self.model_path)      # Сохраняем в локальную папку
            print(f"Модель сохранена: {self.model_path}")
        else:
            print(f"Модель найдена локально: {self.model_path}")
        
        # Загружаем локальную модель
        self.model = YOLO(self.model_path)
        self.activation_distance = 2.0
        
        # Анализ замедления шага
        self.speed_history = deque(maxlen=10)  # Последние 10 скоростей
        self.last_center = None
        self.slowing_down = False
        
        print("Детектор готов.")

    def _analyze_speed(self, center):
        """
        Анализирует скорость движения центра bbox для определения замедления шага.
        
        Параметры:
            center (tuple): Координаты центра bbox (x, y).
            
        Возвращает:
            bool: True если человек замедляется, False иначе.
        """
        if self.last_center is None:
            self.last_center = center
            return False
        
        # Скорость в пикселях за кадр (при 30 FPS)
        dx = abs(center[0] - self.last_center[0])
        self.speed_history.append(dx)
        
        if len(self.speed_history) >= 10:
            avg_speed = np.mean(list(self.speed_history)[-5:])
            self.last_center = center
            
            # Замедление: скорость < 0.8 пикселя/кадр
            if avg_speed < 0.8:
                self.slowing_down = True
                return True
            else:
                self.slowing_down = False
        
        return False

    def get_person_depth(self, frame) -> Optional[tuple[float, np.array]]:
        """
        Анализирует кадр и возвращает оценку дистанции до человека,
        при условии что модель уверенно видит голову и плечи.

        Параметры:
            frame (np.ndarray): Кадр видео (BGR формат).

        Возвращает:
            float или None: Расстояние до человека в метрах, либо None если
                           человек не обнаружен или недостаточно данных.
        """
        results = self.model(frame, verbose=False)
        
        # Если людей нет, возвращаем None
        if len(results[0].boxes) == 0:
            self.last_center = None  # Сброс скорости
            return None
        
        keypoints = results[0].keypoints
        
        # Если нет ключевых точек — fallback на bbox без проверки pose
        # if len(keypoints) == 0:
        #     box = results[0].boxes.xyxy[0].cpu().numpy()
        #     box_height = box[3] - box[1]
        #     distance = max(0.5, 300 / box_height)
        #     
        #     # Анализ скорости даже без позы
        #     center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        #     slowing_down = self._analyze_speed(center)
        #     
        #     x1, y1, x2, y2 = map(int, box)
        #     color = (0, 255, 0) if slowing_down else (0, 255, 255)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #     return distance, keypoints.xy[0].cpu().numpy()[0]
        
        # Получаем confidence ключевых точек первой детекции
        conf = keypoints.conf[0].cpu().numpy()
        
        # Индексы ключевых точек YOLOv8 Pose:
        # 0 - нос (голова)
        # 5 - левое плечо
        # 6 - правое плечо
        nose_conf = conf[0]
        left_shoulder_conf = conf[5]
        right_shoulder_conf = conf[6]
        
        # Порог уверенности для головы и плеч
        threshold = 0.5
        
        # Проверяем, что голова и хотя бы одно плечо с достаточной уверенностью
        if nose_conf > threshold and (left_shoulder_conf > threshold or right_shoulder_conf > threshold):
            box = results[0].boxes.xyxy[0].cpu().numpy()
            box_height = box[3] - box[1]
            distance = max(0.5, 300 / box_height)
            
            # Анализ скорости
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            slowing_down = self._analyze_speed(center)
            
            x1, y1, x2, y2 = map(int, box)
            # Зелёный если полноценный человек + замедление
            color = (0, 255, 0) if slowing_down else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            return distance, keypoints.xy[0].cpu().numpy()[0]
        else:
            # Если недостаточно данных — красный прямоугольник и None
            box = results[0].boxes.xyxy[0].cpu().numpy()
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            self._analyze_speed(center)  # Всё равно отслеживаем скорость
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            return None
