import cv2
import numpy as np


def draw_lane_lines(frame, lines):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return frame


def distance_to_camera(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width
