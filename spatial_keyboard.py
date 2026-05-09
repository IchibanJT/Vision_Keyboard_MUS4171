import cv2
import numpy as np
import pygame
import os

# Get the directory of this script for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Initialize Pygame Mixer
pygame.mixer.init()
pygame.mixer.set_num_channels(8) # Allow multiple sounds to play at exactly the same time

# Load our pentatonic scale
sounds = {
    0: pygame.mixer.Sound(os.path.join(BASE_DIR, "thumb.wav")),
    1: pygame.mixer.Sound(os.path.join(BASE_DIR, "index.wav")),
    2: pygame.mixer.Sound(os.path.join(BASE_DIR, "middle.wav")),
    3: pygame.mixer.Sound(os.path.join(BASE_DIR, "ring.wav")),
    4: pygame.mixer.Sound(os.path.join(BASE_DIR, "pinky.wav"))
}

# 2. Create hand detector   
def build_skin_mask(frame):
    """Build a more stable skin mask using HSV and YCrCb with broader ranges."""
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    # Broader HSV ranges to catch more skin tones
    # Red hues (0-20) and light reds (160-180)
    lower_hsv1 = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_hsv2 = np.array([160, 20, 40], dtype=np.uint8)
    upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    
    # YCrCb ranges for skin detection
    lower_ycrcb = np.array([0, 130, 80], dtype=np.uint8)
    upper_ycrcb = np.array([255, 185, 140], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def get_largest_hand_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:  # Lowered from 3000 for better sensitivity
        return None

    return largest


def get_zone_activations(contour, threshold_y, zone_width, min_zone_points):
    hull = cv2.convexHull(contour)
    points = hull.reshape(-1, 2)

    zone_counts = [0] * 5
    for x, y in points:
        if y > threshold_y:
            zone_id = min(x // zone_width, 4)
            zone_counts[zone_id] += 1

    return [count >= min_zone_points for count in zone_counts], zone_counts


# We need to track the state of each region independently so they don't stutter
HISTORY_LENGTH = 6
region_states = {i: False for i in range(5)}
region_history = {i: [False] * HISTORY_LENGTH for i in range(5)}

# Calibration controls
def nothing(x):
    pass

cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 420, 90)
cv2.createTrackbar('Threshold %', 'Controls', 60, 90, nothing)
cv2.createTrackbar('Min Points', 'Controls', 3, 10, nothing)
cv2.createTrackbar('Smooth', 'Controls', 3, 10, nothing)

# 3. Start Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    height, width, _ = image.shape

    threshold_pct = cv2.getTrackbarPos('Threshold %', 'Controls')
    min_zone_points = max(1, cv2.getTrackbarPos('Min Points', 'Controls'))
    smooth_frames = max(1, cv2.getTrackbarPos('Smooth', 'Controls'))

    threshold_y = int(height * max(30, threshold_pct) / 100)
    cv2.line(image, (0, threshold_y), (width, threshold_y), (0, 255, 0), 2)
    cv2.putText(
        image,
        f'Threshold {threshold_pct}%  Points {min_zone_points}  Smooth {smooth_frames}',
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Divide screen into 5 zones for 5 fingers
    zone_width = width // 5

    mask = build_skin_mask(image)
    hand_contour = get_largest_hand_contour(mask)

    if hand_contour is not None:
        cv2.drawContours(image, [hand_contour], -1, (255, 0, 0), 2)
        active_zones, counts = get_zone_activations(hand_contour, threshold_y, zone_width, min_zone_points)
    else:
        active_zones = [False] * 5
        counts = [0] * 5

    for zone_id in range(5):
        region_history[zone_id].pop(0)
        region_history[zone_id].append(active_zones[zone_id])

        is_active = sum(region_history[zone_id]) >= smooth_frames

        if is_active and not region_states[zone_id]:
            sounds[zone_id].play()

        region_states[zone_id] = is_active

        zone_x_start = zone_id * zone_width
        zone_color = (0, 255, 0) if is_active else (100, 100, 100)
        cv2.rectangle(image, (zone_x_start, 0), (zone_x_start + zone_width, height), zone_color, 1)
        cv2.putText(
            image,
            str(counts[zone_id]),
            (zone_x_start + 10, threshold_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            zone_color,
            1,
            cv2.LINE_AA,
        )

    cv2.imshow('Vision-Tracked Spatial Keyboard', image)
    cv2.imshow('Skin Mask', mask)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()