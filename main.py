import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            upper = face_landmarks.landmark[159]
            lower = face_landmarks.landmark[145]
            
            distance = get_distance(upper, lower)

            if distance < 0.008: 
                status = "SLEEPING / BLINKING"
                color = (0, 0, 255) 
            else:
                status = "EYES OPEN"
                color = (0, 255, 0) 

            cv2.putText(image, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=1)
            )

    cv2.imshow('Eye Tracker v1.0', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()