import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_pupil = face_landmarks.landmark[468]
            right_pupil = face_landmarks.landmark[473]
            
            upper_eye = face_landmarks.landmark[159]
            lower_eye = face_landmarks.landmark[145]
            
            eye_height = (lower_eye.y - upper_eye.y)
            v_ratio = (left_pupil.y - upper_eye.y) / eye_height if eye_height > 0 else 0.5

            l_corner = face_landmarks.landmark[33]
            r_corner = face_landmarks.landmark[133]
            
            eye_width = (r_corner.x - l_corner.x)
            h_ratio = (left_pupil.x - l_corner.x) / eye_width if eye_width > 0 else 0.5

            if h_ratio < 0.4: h_text = "LEFT"
            elif h_ratio > 0.6: h_text = "RIGHT"
            else: h_text = "CENTER"

            if v_ratio < 0.4: v_text = "UP"
            elif v_ratio > 0.8: v_text = "DOWN"
            else: v_text = "CENTER"

            for p in [left_pupil, right_pupil]:
                cv2.circle(image, (int(p.x * width), int(p.y * height)), 2, (0, 255, 0), -1)

            cv2.putText(image, f"GAZE: {v_text} {h_text}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.putText(image, f"H-Ratio: {round(h_ratio, 2)} V-Ratio: {round(v_ratio, 2)}", 
                        (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Final 2D Gaze Tracker', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





