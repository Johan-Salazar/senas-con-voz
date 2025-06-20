from flask import Flask, Response, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__, static_folder='img')
CORS(app)  # Habilita CORS para todas las rutas

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

def is_finger_extended(landmarks, tip_idx, dip_idx, wrist):
    tip = landmarks.landmark[tip_idx]
    dip = landmarks.landmark[dip_idx]
    
    tip_dip_dist = np.hypot(tip.x - dip.x, tip.y - dip.y)
    tip_wrist_dist = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
    
    return tip_dip_dist < tip_wrist_dist * 0.5

def is_hand_facing_up(landmarks):
    wrist = landmarks.landmark[0]
    middle_mcp = landmarks.landmark[9]
    middle_tip = landmarks.landmark[12]
    
    finger_vector_y = middle_mcp.y - middle_tip.y
    angle = math.atan2(-finger_vector_y, 1) * 180 / math.pi
    normalized_angle = (angle + 360) % 360
    
    is_up = math.fabs(normalized_angle - 0) < 45 or math.fabs(normalized_angle - 360) < 45
    
    return is_up

def detect_letter_from_hand(landmarks):
    if not is_hand_facing_up(landmarks):
        return ''

    wrist = landmarks.landmark[0]
    
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    middle_tip = landmarks.landmark[12]
    ring_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]
    
    thumb_index_dist = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    index_middle_dist = np.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
    
    thumb_extended = is_finger_extended(landmarks, 4, 3, wrist)
    index_extended = is_finger_extended(landmarks, 8, 7, wrist)
    middle_extended = is_finger_extended(landmarks, 12, 11, wrist)
    ring_extended = is_finger_extended(landmarks, 16, 15, wrist)
    pinky_extended = is_finger_extended(landmarks, 20, 19, wrist)

    # Detección de vocales
    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return 'A'
    elif not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return 'E'
    elif not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return 'I'
    elif thumb_index_dist < 0.1 and not middle_extended and not ring_extended and not pinky_extended:
        return 'O'
    elif not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended and index_middle_dist < 0.15:
        return 'U'
    
    return ''

def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            detected_letter = ''
            orientation_warning = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    if is_hand_facing_up(hand_landmarks):
                        detected_letter = detect_letter_from_hand(hand_landmarks)
                    else:
                        orientation_warning = True

            if detected_letter:
                cv2.putText(frame, f"Letra: {detected_letter}", (50, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            elif orientation_warning:
                cv2.putText(frame, "Gira tu mano hacia arriba", (50, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Mostrando la mano", (50, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

@app.route("/")
def home():
    return send_file('prueba.html')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Para desarrollo (accesible desde otros dispositivos en la misma red)
    #app.run(host='0.0.0.0', port=5000, debug=True)
    
    # Para HTTPS (puede dar problemas con certificado autofirmado)
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', debug=True) 