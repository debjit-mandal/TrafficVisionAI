import cv2
import numpy as np

from utils.helpers import draw_lane_lines


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = edges[300:, :]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    frame = draw_lane_lines(frame, lines)
    return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_lanes(frame)
        cv2.imshow("Lane Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
