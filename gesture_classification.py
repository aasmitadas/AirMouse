import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 🔹 Finger detection function
def fingers_up(landmarks):
    tips = [8, 12, 16, 20]
    mids = [6, 10, 14, 18]

    up = []
    for tip, mid in zip(tips, mids):
        up.append(landmarks[tip].y < landmarks[mid].y)

    # Thumb (x-axis check)
    up.insert(0, landmarks[4].x < landmarks[3].x)

    return up  # [thumb, index, middle, ring, pinky]


# 🔹 Gesture classification
def classify_gesture(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        return "POINT"
    elif fingers == [0, 1, 1, 0, 0]:
        return "SCROLL"
    elif fingers == [0, 0, 0, 0, 0]:
        return "FIST"
    elif fingers == [1, 1, 1, 1, 1]:
        return "OPEN HAND"
    elif fingers == [0, 1, 1, 1, 0]:
        return "THREE"
    else:
        return "UNKNOWN"


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture_name = "No Hand"

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Get landmarks list
        landmarks = handLms.landmark

        # Detect fingers
        fingers = fingers_up(landmarks)

        # Classify gesture
        gesture_name = classify_gesture(fingers)

        # Show finger states (optional debug)
        cv2.putText(frame, f"{fingers}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display gesture
    cv2.putText(frame, gesture_name, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()