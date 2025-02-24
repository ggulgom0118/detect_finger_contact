import cv2
import mediapipe as mp
import pyautogui
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ModuleNotFoundError:
    import os
    os.system('pip install webdriver-manager')
    from webdriver_manager.chrome import ChromeDriverManager

def open_youtube_shorts():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://www.youtube.com/shorts")
    return driver

def detect_finger_contact():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_pos = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))

                distance = ((thumb_pos[0] - index_pos[0]) ** 2 + (thumb_pos[1] - index_pos[1]) ** 2) ** 0.5
                
                if distance < 10:
                    print("Next video")
                    pyautogui.press('down')

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    driver = open_youtube_shorts()
    time.sleep(5)  
    detect_finger_contact()
