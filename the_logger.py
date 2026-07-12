import cv2
import mediapipe as mp
import winsound
import csv
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

log_file = open('gaze_log.csv', mode='w', newline='')
log_writer = csv.writer(log_file)

log_writer.writerow(['Timestamp', 'Gaze State', 'Alarm Triggered'])
log_file.flush() 

COUNTER = 0
ALARM_THRESHOLD = 30
last_state = ""

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    current_state = "CENTER"
    alarm_active = "NO"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            upper = face_landmarks.landmark[159]
            lower = face_landmarks.landmark[145]
            
            if (lower.y - upper.y) < 0.012:
                COUNTER += 1
                current_state = "SLEEPING"
                if COUNTER >= ALARM_THRESHOLD:
                    alarm_active = "YES"
                    winsound.Beep(2500, 200)
            else:
                COUNTER = 0
                left_pupil = face_landmarks.landmark[468]
                l_corner = face_landmarks.landmark[33]
                r_corner = face_landmarks.landmark[133]
                h_ratio = (left_pupil.x - l_corner.x) / (r_corner.x - l_corner.x)
                
                if h_ratio < 0.4: current_state = "LEFT"
                elif h_ratio > 0.6: current_state = "RIGHT"
                else: current_state = "CENTER"

            if current_state != last_state:
                timestamp = datetime.now().strftime('%H:%M:%S')
                log_writer.writerow([timestamp, current_state, alarm_active])
                
                log_file.flush() 
                
                print(f"New Log Entry: {current_state}")
                last_state = current_state

            cv2.putText(image, f"STATE: {current_state}", (30, 50), 2, 1, (255, 255, 0), 2)

    cv2.imshow('Separate Logger System', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()