import cv2

from src.collision_detection import CollisionDetector
from src.lane_detection import detect_lanes
from src.traffic_sign_recognition import TrafficSignRecognizer
from utils.config import CAMERA_INDEX


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    sign_recognizer = TrafficSignRecognizer()
    collision_detector = CollisionDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sign = sign_recognizer.recognize(frame)
        cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = detect_lanes(frame)

        detections = collision_detector.detect_objects(frame)

        cv2.imshow("Driver Assistance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
