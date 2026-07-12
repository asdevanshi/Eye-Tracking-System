import cv2
import mediapipe as mp
import winsound

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# Alarm Variables
COUNTER = 0
ALARM_THRESHOLD = 30 

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Check Eye Openness (Distance between top and bottom eyelid)
            upper = face_landmarks.landmark[159]
            lower = face_landmarks.landmark[145]
            distance = lower.y - upper.y

            # Drowsiness Logic
            if distance < 0.012:  # You might need to adjust this number
                COUNTER += 1
                if COUNTER >= ALARM_THRESHOLD:
                    cv2.putText(image, "ALARM: WAKE UP!", (50, 250), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                    winsound.Beep(2500, 500) # Frequency: 2500Hz, Duration: 500ms
            else:
                COUNTER = 0

            # Gaze Logic (Your previous working code)
            left_pupil = face_landmarks.landmark[468]
            l_corner = face_landmarks.landmark[33]
            r_corner = face_landmarks.landmark[133]
            h_ratio = (left_pupil.x - l_corner.x) / (r_corner.x - l_corner.x)

            status = "CENTER"
            if h_ratio < 0.4: status = "LEFT"
            elif h_ratio > 0.6: status = "RIGHT"

            cv2.putText(image, f"GAZE: {status}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(image, f"Closed Frames: {COUNTER}", (30, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Drowsiness Alarm System', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()