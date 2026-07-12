import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# CALIBRATION: Adjust this 'K' value. 
# Smaller K = shorter distance, Larger K = longer distance.
K = 550 

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. Get Pupil Points
            lp = face_landmarks.landmark[468]
            rp = face_landmarks.landmark[473]

            # 2. Calculate pixel distance (dist)
            lx, ly = lp.x * width, lp.y * height
            rx, ry = rp.x * width, rp.y * height
            dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)

            # 3. High Accuracy Math
            if dist > 0:
                # The Physics: distance is inversely proportional to pupil pixel-width
                estimated_meters = K / dist
                
                # Logic for "Infinity" focus
                if estimated_meters > 20:
                    display_text = "Range: Infinity (>20m)"
                    color = (0, 255, 0)
                else:
                    display_text = f"Range: {round(estimated_meters, 2)}m"
                    color = (0, 255, 255)

                cv2.putText(image, display_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, f"Pixel Dist: {round(dist, 1)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Precision Range Finder', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()