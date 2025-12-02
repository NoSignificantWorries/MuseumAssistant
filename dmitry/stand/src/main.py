from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# custom modules
from distance_detector import InteractiveStandDetector
from demographics_detector import DemographicsEstimator


def analyze_frame(dem_detector: DemographicsEstimator, frame: np.array, nose_pos: np.array) -> Optional[dict]:
    bboxes = dem_detector._detect_faces(frame)

    nx, ny = nose_pos
    
    dist_face = None
    min_dist = float("inf")
    max_conf = 0

    for face in bboxes:
        x1, y1, x2, y2, conf = face
        x0 = (x1 + x2) / 2
        y0 = (y1 + y2) / 2
        dist = np.sqrt((x0 - nx) ** 2 + (y0 - ny) ** 2)
        
        if dist < min_dist and conf > max_conf:
            dist_face = face
            min_dist = dist
            max_conf = conf

    if not dist_face:
        return None

    x1, y1, x2, y2, conf = dist_face
    face_img = frame[max(0, y1 - 20):y2 + 20, max(0, x1 - 20):x2 + 20]
    if face_img.size == 0:
        return None
    age_str, gender = dem_detector._predict_age_gender(face_img)
    bucket = dem_detector._map_age_bucket(age_str)

    return {"gender": gender, "group": bucket, "age": age_str}


def activate(human_info: Optional[dict]):
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y %H:%M:%S")
    print(f"Форматированная дата: {formatted_date}")


def main():
    movement_distance_detector = InteractiveStandDetector()
    demographic_detector = DemographicsEstimator()

    activation_distance = 1.5

    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()

        dist = movement_distance_detector.get_person_depth(frame)

        if dist and dist[0] <= activation_distance:
            human_info = analyze_frame(demographic_detector, frame, dist[1])
            activate(human_info)
        
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
    # except Exception as err:
    #     print(err)

