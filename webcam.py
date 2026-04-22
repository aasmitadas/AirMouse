import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import math
import time

# Webcam
cap = cv2.VideoCapture(0)

# Detector
detector = HandDetector(maxHands=1, detectionCon=0.7)

screen_w, screen_h = pyautogui.size()

click_delay = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        # Index finger tip
        x1, y1 = lmList[8][0], lmList[8][1]
        # Thumb tip
        x2, y2 = lmList[4][0], lmList[4][1]

        # Move mouse
        screen_x = screen_w * (x1 / 640)
        screen_y = screen_h * (y1 / 480)
        pyautogui.moveTo(screen_x, screen_y)

        # Distance for click
        dist = math.hypot(x2 - x1, y2 - y1)

        if dist < 40 and time.time() - click_delay > 1:
            pyautogui.click()
            click_delay = time.time()

    cv2.imshow("Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()