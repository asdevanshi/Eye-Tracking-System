import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
# "refine_landmarks=True" is what unlocks the Iris/Pupil tracking
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
            # 1. GET THE PUPIL (Landmark 468 is the center of the left iris)
            pupil = face_landmarks.landmark[468]
            
            # 2. GET THE EYE CORNERS (Left eye)
            left_corner = face_landmarks.landmark[33]
            right_corner = face_landmarks.landmark[133]

            # 3. CALCULATE GAZE (How far is pupil from the left corner?)
            # We convert normalized coordinates to pixel values
            pupil_x = int(pupil.x * width)
            left_x = int(left_corner.x * width)
            right_x = int(right_corner.x * width)

            eye_width = right_x - left_x
            if eye_width > 0:
                relative_pos = (pupil_x - left_x) / eye_width
                
                if relative_pos < 0.4:
                    text = "LOOKING LEFT"
                elif relative_pos > 0.6:
                    text = "LOOKING RIGHT"
                else:
                    text = "CENTER"
                
                cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.circle(image, (pupil_x, int(pupil.y * height)), 2, (255, 0, 0), -1)

    cv2.imshow('Pupil Tracking v2', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



