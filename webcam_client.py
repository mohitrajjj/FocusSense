# webcam_client.py
import cv2
import mediapipe as mp
from focus_inference import FocusSenseInference
from feature_extractor import extract_features

mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

model = FocusSenseInference(
    model_path="focussense_model.h5",
    threshold=0.6
)

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            feature = extract_features(landmarks)

            state, score = model.update(feature)

            if state:
                cv2.putText(
                    frame,
                    f"{state} ({score}%)",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if state == "Focused" else (0, 0, 255),
                    2
                )

        cv2.imshow("FocusSense – Desktop Mode", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
