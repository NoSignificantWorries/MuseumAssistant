import time
from datetime import datetime
from typing import Optional
import threading

import cv2
import numpy as np

# custom modules
from distance_detector import InteractiveStandDetector
from demographics_detector import DemographicsEstimator


class Time:
    def __init__(self) -> None:
        self.start = 0
        self.end = 0

    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()

    def get(self):
        return self.end - self.start


class Pipeline:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread = None

        self._movement_distance_detector = InteractiveStandDetector()
        self._demographic_detector = DemographicsEstimator()

        self.activation_distance = 1.5
        self.activated = False

        self._cap = cv2.VideoCapture(0)

        self.human_info = None

        self._started_at = 0
        self._finished_at = 0
        self._time_activated = None
    
    def _loop(self):
        while True:
            ret, frame = self._cap.read()
            
            if not ret:
                raise RuntimeError("Error while reading video capture")

            dist = self._movement_distance_detector.get_person_depth(frame)

            if dist and dist[0] <= self.activation_distance and not self.activated:
                self.human_info = self._analyze_frame(frame, dist[1])
                self._activate()
                print("Welcome!")
            elif dist and dist[0] > self.activation_distance and self.activated:
                self._deactivate()
                time_elapsed = (self._finished_at - self._started_at) / 60
                stats = self.human_info
                stats["datetime"] = self._time_activated
                stats["time_elapsed"] = time_elapsed
                print(stats)
                print("Good bye!")

            if self._stop_event.wait(timeout=0.01):
                break

            # cv2.imshow('Distance Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._cap.release()
        cv2.destroyAllWindows()
            
    def _analyze_frame(self, frame: np.array, nose_pos: np.array) -> Optional[dict]:
        bboxes = self._demographic_detector._detect_faces(frame)

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
        age_str, gender = self._demographic_detector._predict_age_gender(face_img)
        bucket = self._demographic_detector._map_age_bucket(age_str)

        return {"gender": gender, "group": bucket, "age": age_str}

    def _activate(self):
        self._started_at = time.time()
        self.activated = True
        now = datetime.now()
        self._time_activated = now.strftime("%d.%m.%Y %H:%M:%S")

    def _deactivate(self):
        self._finished_at = time.time()
        self.activated = False

    def _is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if not self._is_running():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop)
            self._thread.daemon = True
            self._thread.start()
            return True
        return False

    def stop(self) -> bool:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                pass
        
        return True


def main():
    pipeline = Pipeline()

    try:
        started = pipeline.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pipeline.stop()
    except BaseException:
        pipeline.stop()


if __name__ == "__main__":
    main()
    # except Exception as err:
    #     print(err)

