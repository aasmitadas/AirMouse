import cv2
import mediapipe as mp
import pyautogui
import math

# ---------------- CONFIG ----------------
HOLD_THRESHOLD = 8  # frames to hold same gesture before firing

GESTURE_MAP = {
    "point":    ("move_cursor", None),
    "fist":     ("click",       None),
    "peace":    ("right_click", None),
    "open":     ("scroll",      "up"),
    "thumbsup": ("key",         "right"),
    "pinch":    ("drag",        None),
}

# --------------- SETUP ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# debounce state
last_gesture = None
hold_count = 0
dragging = False

# --------------- HELPERS ----------------
def fingers_up(lm):
    tips = [8, 12, 16, 20]
    mids = [6, 10, 14, 18]

    up = []
    for tip, mid in zip(tips, mids):
        up.append(lm[tip].y < lm[mid].y)

    # thumb (x-axis)
    up.insert(0, lm[4].x < lm[3].x)
    return up  # [thumb, index, middle, ring, pinky]


def classify_gesture(lm):
    f = fingers_up(lm)

    # pinch (thumb + index close)
    x1, y1 = lm[8].x, lm[8].y
    x2, y2 = lm[4].x, lm[4].y
    dist = math.hypot(x1 - x2, y1 - y2)

    if dist < 0.05:
        return "pinch"

    if f == [0,1,0,0,0]:
        return "point"
    elif f == [0,1,1,0,0]:
        return "peace"
    elif f == [0,0,0,0,0]:
        return "fist"
    elif f == [1,1,1,1,1]:
        return "open"
    elif f == [1,0,0,0,0]:
        return "thumbsup"
    else:
        return "unknown"


def execute_action(gesture, lm):
    global dragging

    action, param = GESTURE_MAP.get(gesture, (None, None))
    if action is None:
        return

    # 👉 move cursor
    if action == "move_cursor":
        x = lm[8].x * screen_w
        y = lm[8].y * screen_h
        pyautogui.moveTo(x, y, duration=0.05)

    # 👊 left click
    elif action == "click":
        pyautogui.click()

    # ✌️ right click
    elif action == "right_click":
        pyautogui.rightClick()

    # ✋ scroll
    elif action == "scroll":
        pyautogui.scroll(100 if param == "up" else -100)

    # 👍 key press
    elif action == "key":
        pyautogui.press(param)

    # 🤏 drag
    elif action == "drag":
        x = lm[8].x * screen_w
        y = lm[8].y * screen_h
        pyautogui.moveTo(x, y, duration=0.05)

        if not dragging:
            pyautogui.mouseDown()
            dragging = True

    else:
        pass


def release_drag_if_needed(gesture):
    global dragging
    if gesture != "pinch" and dragging:
        pyautogui.mouseUp()
        dragging = False


# --------------- MAIN LOOP ----------------
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

        release_drag_if_needed(gesture)

    else:
        release_drag_if_needed("none")

    # display
    cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Control (Debounced)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()