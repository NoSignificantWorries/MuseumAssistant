from detector import InteractiveStandDetector
import cv2
import time


def activate_stand():
    print("СТЕНД ВКЛЮЧЕН!")  # Тут будет запуск видео/аудио


def deactivate_stand():
    print("СТЕНД ВЫКЛЮЧЕН!")  # Остановить воспроизведение

detector = InteractiveStandDetector()
activation_distance = 1.5

cap = cv2.VideoCapture(0)

stand_active = False
time_start = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    distance = detector.get_person_depth(frame)
    
    if distance and (distance < activation_distance or detector.slowing_down):
        color_text = "SLOWING DOWN!" if detector.slowing_down else "Person nearby"

        cv2.putText(frame, f"{color_text} {distance:.1f}m", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if not stand_active:
            activate_stand()
            stand_active = True
            time_start = time.time()
        elapsed = time.time() - time_start
        
        cv2.putText(frame, f"{distance:.1f}m Time: {int(elapsed)}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        if stand_active:
            deactivate_stand()
            stand_active = False
            total_time = time.time() - time_start
            print(f"Посетитель ушёл. Время у стенда: {int(total_time)} секунд")
        cv2.putText(frame, "Ожидание...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    cv2.imshow('Distance Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if stand_active:
            deactivate_stand()
        break

cap.release()
cv2.destroyAllWindows()
