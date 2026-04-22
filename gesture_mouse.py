import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

paused = False
prev_x = None
prev_y = None
click_delay = 0
pause_delay = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape

        def get_pos(id):
            return int(lm.landmark[id].x * w), int(lm.landmark[id].y * h)

        index_x, index_y = get_pos(8)
        middle_x, middle_y = get_pos(12)
        thumb_x, thumb_y = get_pos(4)

        # Finger states
        fingers = []
        fingers.append(1 if lm.landmark[4].x < lm.landmark[3].x else 0)  # thumb
        fingers.append(1 if lm.landmark[8].y < lm.landmark[6].y else 0)
        fingers.append(1 if lm.landmark[12].y < lm.landmark[10].y else 0)
        fingers.append(1 if lm.landmark[16].y < lm.landmark[14].y else 0)
        fingers.append(1 if lm.landmark[20].y < lm.landmark[18].y else 0)

        total_fingers = sum(fingers)

        # ✊ Fist → pause toggle (with delay)
        if total_fingers == 0 and time.time() - pause_delay > 1:
            paused = not paused
            pause_delay = time.time()

        if not paused:
            # 👉 Move cursor (smooth)
            screen_x = screen_w * lm.landmark[8].x
            screen_y = screen_h * lm.landmark[8].y
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # 🤏 Click (pinch)
            dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
            if dist < 40 and time.time() - click_delay > 1:
                pyautogui.click()
                click_delay = time.time()

            # ✌️ Scroll (based on vertical movement)
            if fingers[1] == 1 and fingers[2] == 1 and total_fingers == 2:
                if prev_y is not None:
                    dy = index_y - prev_y

                    if abs(dy) > 10:
                        pyautogui.scroll(int(-dy * 2))

                prev_y = index_y
            else:
                prev_y = None

            # 👉 Swipe detection (left/right)
            if prev_x is not None:
                dx = index_x - prev_x

                if dx > 60:
                    pyautogui.press("right")
                    time.sleep(0.3)

                elif dx < -60:
                    pyautogui.press("left")
                    time.sleep(0.3)

            prev_x = index_x

        # Status text
        status = "PAUSED" if paused else "ACTIVE"
        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Touchless Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()