import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class CollisionDetector:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

    def detect_objects(self, frame):
        input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
        detections = self.model(input_tensor)
        return detections


def main():
    cap = cv2.VideoCapture(0)
    detector = CollisionDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_objects(frame)

        cv2.imshow("Collision Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
