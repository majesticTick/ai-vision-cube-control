import cv2
import mediapipe as mp
import numpy as np

#초기 설정 손 인식 + 얼굴 인식
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection 
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

#  3D 큐브 데이터
points = np.array([
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]
], dtype=float)
edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

angle_x, angle_y = 0, 0
prev_pos = None

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #  얼굴 블러 처리 
    face_results = face_detection.process(rgb_image)
    if face_results.detections:
        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            
            # 얼굴 영역 추출
            face_roi = image[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)]
            if face_roi.size > 0:
                #블러 강도 조절 (35, 35) 숫자를 키우면 더 뿌예집니다.
                blurred_face = cv2.GaussianBlur(face_roi, (55, 55), 0)
                image[max(0, y):min(h, y+fh), max(0, x):min(w, x+fw)] = blurred_face

    # 손 인식 및 큐브 제어 
    hand_results = hands.process(rgb_image)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            t = hand_landmarks.landmark[4] # 엄지
            i = hand_landmarks.landmark[8] # 검지
            
            dist = np.linalg.norm(np.array([t.x - i.x, t.y - i.y]))
            curr_pos = np.array([t.x * w, t.y * h])

            if dist < 0.05: # 핀치 시
                if prev_pos is not None:
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    angle_y += dx * 0.01 
                    angle_x -= dy * 0.01 # 상하 쓸어올리기로 조절
                prev_pos = curr_pos
                cv2.circle(image, (int(curr_pos[0]), int(curr_pos[1])), 10, (0, 255, 0), -1)
            else:
                prev_pos = None

    # 3D 투영, 큐브 그리기
    rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    projected_points = []
    for p in points:
        rotated = ry @ (rx @ p)
        z = 1 / (4 - rotated[2]) 
        px = int(rotated[0] * z * 600 + w/2)
        py = int(rotated[1] * z * 600 + h/2)
        projected_points.append((px, py))

    for edge in edges:
        cv2.line(image, projected_points[edge[0]], projected_points[edge[1]], (0, 255, 255), 2)

    cv2.imshow('Face Blur + 3D Control', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()