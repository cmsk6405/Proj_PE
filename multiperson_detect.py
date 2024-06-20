import cv2
import numpy as np
import torch
import mediapipe as mp
from ultralytics import YOLO


# YOLOv8 모델 로드
model = YOLO('yolov8l-pose.pt')

# MediaPipe 포즈 추정 모델 로드
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path ="/mnt/d/Data/multi_person/국민체조 교육용 영상.mp4_20240603_142338.mkv"
# video_path = "/mnt/d/Data/multi_person/Sample (2)/Sample/01.원천데이터/MP4/M060_M061_M062/M060_M061_M062_16_02/M060_M061_M062_16_02-01.mp4"
# video_path = "/mnt/d/Data/multi_person/Sample/Sample/01.원천데이터/1. 동영상/DATA_B경기342.mp4"
# 동영상 파일 또는 웹캠에서 프레임 읽기
cap = cv2.VideoCapture(video_path)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8을 사용하여 사람 탐지
    results = model(frame)[0]

    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result

        if int(cls) == 0:  # 사람 클래스일 경우
            # 바운딩 박스 영역 추출
            person_img = frame[int(y1):int(y2), int(x1):int(x2)]

            # MediaPipe를 사용하여 포즈 추정
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            result = pose.process(person_rgb)

            if result.pose_landmarks:
                # 원본 프레임에 포즈 랜드마크 및 연결선 그리기
                mp_drawing.draw_landmarks(frame[int(y1):int(y2), int(x1):int(x2)], 
                                          result.pose_landmarks, 
                                          mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    # 결과 프레임 보여주기
    cv2.imshow('Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()