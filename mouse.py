import cv2
import mediapipe as mp
import pyautogui
import time

# Disable PyAutoGUI fail-safe mechanism
pyautogui.FAILSAFE = False

class Tracking:
    def __init__(self, stable_threshold=5):
        self.hands = mp.solutions.hands.Hands()
        self.draw = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.stable_frames = 0
        self.stable_time = 0
        self.stable_threshold = stable_threshold
        self.cap = cv2.VideoCapture(0)

    def get_index_finger_position(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_x, index_finger_y = int(index_finger_tip.x * frame.shape[1]), int(
                    index_finger_tip.y * frame.shape[0])

                screen_x = int(index_finger_x * self.screen_width / frame.shape[1])
                screen_y = int(index_finger_y * self.screen_height / frame.shape[0])

                return screen_x, screen_y

        return None

    def move_mouse_cursor(self, screen_x, screen_y):
        pyautogui.moveTo(screen_x, screen_y)

    def check_hand_stability(self, screen_x, screen_y):
        if self.stable_frames > 0 and time.time() - self.stable_time > self.stable_threshold:
            pyautogui.click(button='left')
            #cv2.circle(frame, (screen_x, screen_y), 20, (0, 255, 0), -1)
            self.stable_frames = 0

        if pyautogui.position() == (screen_x, screen_y):
            self.stable_frames += 1
        else:
            self.stable_frames = 0

        if self.stable_frames == 1:
            self.stable_time = time.time()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1) # add this line to flip the frame horizontally
            if not ret:
                break

            screen_pos = self.get_index_finger_position(frame)
            if screen_pos is not None:
                screen_x, screen_y = screen_pos
                self.move_mouse_cursor(screen_x, screen_y)
                self.check_hand_stability(screen_x, screen_y)

            cv2.imshow('Hand Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# uncomment to run directly
gc1 = Tracking()
gc1.run()