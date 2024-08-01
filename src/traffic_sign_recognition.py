import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.config import TRAFFIC_SIGN_MODEL_PATH
from utils.preprocess_data import preprocess_image


class TrafficSignRecognizer:
    def __init__(self):
        self.model = load_model(TRAFFIC_SIGN_MODEL_PATH)
        self.class_names = {
            0: "Speed Limit 20",
            1: "Speed Limit 30",
            2: "Speed Limit 50",
            3: "Speed Limit 60",
            4: "Speed Limit 70",
            5: "Speed Limit 80",
            6: "End of Speed Limit 80",
            7: "Speed Limit 100",
            8: "Speed Limit 120",
            9: "No passing",
            10: "No passing for vehicles over 3.5 metric tons",
            11: "Right-of-way at the next intersection",
            12: "Priority road",
            13: "Yield",
            14: "Stop",
            15: "No vehicles",
            16: "Vehicles over 3.5 metric tons prohibited",
            17: "No entry",
            18: "General caution",
            19: "Dangerous curve to the left",
            20: "Dangerous curve to the right",
            21: "Double curve",
            22: "Bumpy road",
            23: "Slippery road",
            24: "Road narrows on the right",
            25: "Road work",
            26: "Traffic signals",
            27: "Pedestrians",
            28: "Children crossing",
            29: "Bicycles crossing",
            30: "Beware of ice/snow",
            31: "Wild animals crossing",
            32: "End of all speed and passing limits",
            33: "Turn right ahead",
            34: "Turn left ahead",
            35: "Ahead only",
            36: "Go straight or right",
            37: "Go straight or left",
            38: "Keep right",
            39: "Keep left",
            40: "Roundabout mandatory",
            41: "End of no passing",
            42: "End of no passing by vehicles over 3.5 metric tons",
        }

    def recognize(self, frame):
        image = preprocess_image(frame)
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        class_id = np.argmax(prediction)
        return self.class_names[class_id]


def main():
    cap = cv2.VideoCapture(0)
    recognizer = TrafficSignRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sign = recognizer.recognize(frame)
        cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Traffic Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
