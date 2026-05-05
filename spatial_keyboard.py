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

# 2. Create hand detector using skin color detection
def get_hand_contours(frame):
    """Detect hand regions using skin color in HSV"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Skin color range in HSV
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([20, 40, 200], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# We need to track the state of each region independently so they don't stutter
region_states = {0: False, 1: False, 2: False, 3: False, 4: False}

# 3. Start Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    height, width, _ = image.shape
    
    # 4. Draw the Invisible Keyboard Line
    threshold_y = int(height * 0.6) 
    cv2.line(image, (0, threshold_y), (width, threshold_y), (0, 255, 0), 2)
    
    # Divide screen into 5 zones for 5 fingers
    zone_width = width // 5

    # 5. Get hand contours from skin color detection
    contours, mask = get_hand_contours(image)

    # 6. Process each zone
    for zone_id in range(5):
        zone_x_start = zone_id * zone_width
        zone_x_end = (zone_id + 1) * zone_width
        
        # Check if any hand contour crosses threshold in this zone
        touched = False
        
        for contour in contours:
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is in this zone and below threshold
            if zone_x_start <= x <= zone_x_end and y > threshold_y:
                touched = True
                break
        
        # Play sound if zone is touched and wasn't before
        if touched:
            if not region_states[zone_id]:
                sounds[zone_id].play()
                region_states[zone_id] = True
        else:
            region_states[zone_id] = False
        
        # Draw zone boundaries
        cv2.line(image, (zone_x_start, 0), (zone_x_start, height), (100, 100, 100), 1)

    cv2.imshow('Vision-Tracked Spatial Keyboard', image)
    
    if cv2.waitKey(5) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()