import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Define screen size for mouse movement
screen_size = pyautogui.size()

# Initialize click flag
clicked = False

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    # Convert BGR image to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])
            thumb_up = (landmarks[4][1] < landmarks[3][1] < landmarks[2][1] < landmarks[1][1] < landmarks[0][1])
            index_up = (landmarks[8][2] < landmarks[7][2])
            middle_up = (landmarks[12][2] < landmarks[11][2])
            ring_up = (landmarks[16][2] < landmarks[15][2])
            pinky_up = (landmarks[20][2] < landmarks[19][2])

            # Move mouse cursor if all fingers are up
            if thumb_up and index_up and middle_up and ring_up and pinky_up:
                x = int(landmarks[8][0] * screen_size.width)
                y = int(landmarks[8][1] * screen_size.height)
                pyautogui.moveTo(x, y)
                # Reset click flag
                clicked = False

            # Perform left click if all fingers are closed and click flag is False
            elif thumb_up and not clicked:
                pyautogui.click()
                # Set click flag
                clicked = True

    # Show the frame
    cv2.imshow('Hand Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
