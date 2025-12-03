from pathlib import Path

import cv2


FACE_PROTO = Path("../shared/demographics/opencv_face_detector.pbtxt").resolve().absolute()
FACE_MODEL = Path("../shared/demographics/opencv_face_detector_uint8.pb").resolve().absolute()
AGE_PROTO = Path("../shared/demographics/age_deploy.prototxt").resolve().absolute()
AGE_MODEL = Path("../shared/demographics/age_net.caffemodel").resolve().absolute()
GENDER_PROTO = Path("../shared/demographics/gender_deploy.prototxt").resolve().absolute()
GENDER_MODEL = Path("../shared/demographics/gender_net.caffemodel").resolve().absolute()

AGE_LIST = ['0-2', '4-6', '8-12', '15-20',
            '21-24', '25-32', '33-43', '44-53', '60-100'] 
GENDER_LIST = ['Male', 'Female', "Unknown"]

# Mean values used to normalize input images for the DNNs
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


class DemographicsEstimator:
    """
    Lightweight demographics estimator based on OpenCV DNN models.

    The estimator detects faces in a frame and, for each face, predicts
    an age range and gender label using pre‑trained Caffe models.
    It also maps age ranges into coarse buckets (child / young / adult / senior).

    Args:
        conf_threshold (float): Minimum confidence for a face detection
            to be considered valid.
    """

    def __init__(self, conf_threshold: float = 0.7):
        self.conf_threshold = conf_threshold
        self.face_net   = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        self.age_net    = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        self.gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    def _detect_faces(self, frame):
        """
        Run face detection on a BGR frame.

        Uses an OpenCV DNN face detector and returns bounding boxes with confidence.

        Args:
            frame (np.ndarray): Input frame in BGR format.

        Returns:
            list[tuple[int, int, int, int, float]]:
                List of (x1, y1, x2, y2, confidence) for each detected face.
        """

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     [104, 117, 123], swapRB=False, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2, y2, conf))
        return faces

    def _predict_age_gender(self, face_img):
        """
        Predict age range and gender for a single face crop.

        Args:
            face_img (np.ndarray): Face image in BGR format.

        Returns:
            tuple[str, str]: (age_range, gender_label).
        """

        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227, 227),
            MODEL_MEAN_VALUES, swapRB=False
        )
        self.gender_net.setInput(blob)
        g_pred = self.gender_net.forward()
        gender_id = g_pred[0].argmax()
        if g_pred[0][gender_id] < 0.65:
            gender_id = -1
        gender = GENDER_LIST[gender_id]

        self.age_net.setInput(blob)
        a_pred = self.age_net.forward()
        age_str = AGE_LIST[a_pred[0].argmax()]

        return age_str, gender

    def _map_age_bucket(self, age_str):
        """
        Map a model age range into a coarse age bucket.

        Args:
            age_str (str): Age range string from AGE_LIST, e.g. "25-32".

        Returns:
            str: One of "child", "young", "adult", "senior".
        """

        if "-" in age_str:
            lo = int(age_str.split("-")[0])
        else:
            lo = 60
        if lo < 18:
            return "child"
        elif 18 <= lo < 30:
            return "young"
        elif 40 <= lo <= 60:
            return "adult"
        else:
            return "senior"

    def analyze_frame(self, frame_bgr):
        """
        Run demographics analysis for all faces in a BGR frame.

        Args:
            frame_bgr (np.ndarray): Input frame in BGR format (OpenCV).

        Returns:
            list[dict]: One dict per detected face with the following keys:
                - "bbox": (x1, y1, x2, y2)
                - "confidence": face detection confidence (float)
                - "age_range": model age range string, e.g. "25-32"
                - "age_bucket": mapped bucket: "child" / "young" / "adult" / "senior"
                - "gender": gender label from GENDER_LIST
        """
        
        results = []
        faces = self._detect_faces(frame_bgr)
        for (x1, y1, x2, y2, conf) in faces:
            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                continue
            age_str, gender = self._predict_age_gender(face)
            bucket = self._map_age_bucket(age_str)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "age_range": age_str,
                "age_bucket": bucket,
                "gender": gender,
            })
        return results

