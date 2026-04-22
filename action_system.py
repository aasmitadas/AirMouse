import cv2
import mediapipe as mp
import pyautogui
import math

# ---------------- CONFIG ----------------
HOLD_THRESHOLD = 6

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = True  # move mouse to top-left to abort

# ---------------- SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_gesture = None
hold_count = 0

# ---------------- FUNCTIONS ----------------

def fingers_up(lm):
    tips = [8, 12, 16, 20]
    mids = [6, 10, 14, 18]

    up = []
    for tip, mid in zip(tips, mids):
        up.append(lm[tip].y < lm[mid].y)

    # thumb (x-axis)
    up.insert(0, lm[4].x < lm[3].x)
    return up


def classify_gesture(lm):
    f = fingers_up(lm)

    # pinch detection
    dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
    if dist < 0.05:
        return "pinch"

    if f == [0,1,0,0,0]:
        return "point"
    elif f == [0,0,0,0,0]:
        return "fist"
    elif f == [1,0,0,0,0]:
        return "thumbsup"
    else:
        return "unknown"


def execute_action(gesture, lm):
    x = int(lm[8].x * SCREEN_W)
    y = int(lm[8].y * SCREEN_H)

    if gesture == "point":
        pyautogui.moveTo(x, y, duration=0.05)

    elif gesture == "fist":
        pyautogui.click()

    elif gesture == "thumbsup":
        pyautogui.press('right')

    elif gesture == "pinch":
        pyautogui.mouseDown()
    else:
        pyautogui.mouseUp()


# ---------------- MAIN LOOP ----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "none"

    if results.multi_hand_landmarks:
        lm_obj = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm_obj, mp_hands.HAND_CONNECTIONS)

        lm = lm_obj.landmark
        gesture = classify_gesture(lm)

        # debounce logic
        if gesture == last_gesture:
            hold_count += 1

            if hold_count == HOLD_THRESHOLD:
                execute_action(gesture, lm)

        else:
            hold_count = 0
            last_gesture = gesture

    else:
        pyautogui.mouseUp()

    # display
    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Control (pyautogui)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()