import cv2
import numpy as np


def resize_frame(frame, target_width):
    if not target_width:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def preprocess_for_detection(frame, blur_kernel=5, adaptive_block_size=11, adaptive_c=2, morph_kernel_size=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure blur kernel is odd
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    # Adaptive threshold isolates small bright/dark objects regardless of lighting variation
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned
