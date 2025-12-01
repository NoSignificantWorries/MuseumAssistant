from datetime import datetime

import cv2

# custom modules
from distance_detector import InteractiveStandDetector
from demographics_detector import DemographicsEstimator


def main():
    movement_distance_detector = InteractiveStandDetector()
    demographic_detector = DemographicsEstimator()

    activation_distance = 1.5

    cap = cv2.VideoCapture(0)

    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y %H:%M:%S")
    print(f"Форматированная дата: {formatted_date}")

    while True:
        ret, frame = cap.read()

        distance = movement_distance_detector.get_person_depth(frame)

        if distance and distance <= activation_distance:
            frame_analization = demographic_detector.analyze_frame(frame)
            print(frame_analization)
        
        if not ret:
            raise RuntimeError("Error while reading video capture")

        # cv2.imshow('Distance Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(err)

