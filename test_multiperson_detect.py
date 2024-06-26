import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

import argparse

def main():
    # YOLOv8 모델 로드
    model = YOLO('yolov8l-pose.pt')

    # MediaPipe 포즈 추정 모델 로드
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # when choose webcam or video with def get_args_parser():
    # if args.webcam:
    #     cap = cv2.VideoCapture(0)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력
    #     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # elif args.video:
    #     video_path = "./data/yoga_test_namas.mp4"
    #     cap = cv2.VideoCapture(video_path)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력


    # 비디오와 캠의 내용을 비교해야 하므로 둘 다 사용하게 함
    cap = cv2.VideoCapture(0)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    video_path = "./data/yoga_test_namas.mp4"
    vid = cv2.VideoCapture(video_path)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력


    while cap.isOpened():
        cam_ret, cam_frame = cap.read()
        cam_frame = cv2.flip(cam_frame, 1)
        if not cam_ret:
            break
        
        vid_ret, vid_frame = vid.read()
        if not vid_ret:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue


        # YOLOv8을 사용하여 사람 탐지
        results = model(cam_frame)[0]

        for result in results.boxes.data:
            # 박스점, 신뢰도, 분류 클래스
            x1, y1, x2, y2, conf, cls = result

            if int(cls) == 0:  # 사람 클래스일 경우
                # 바운딩 박스 영역 추출
                person_img = cam_frame[int(y1):int(y2), int(x1):int(x2)]

                # MediaPipe를 사용하여 포즈 추정
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                mp_result = pose.process(person_rgb)

                if mp_result.pose_landmarks:
                    # 원본 프레임에 포즈 랜드마크 및 연결선 그리기
                    mp_drawing.draw_landmarks(cam_frame[int(y1):int(y2), int(x1):int(x2)], 
                                            mp_result.pose_landmarks, 
                                            mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        # TODO : 결과는 동영상, 웹캠, 자세평가와 횟수가 나와야함
        # 결과 프레임 보여주기
        # cv2.imshow('Pose Estimation', frame)
        # cv2.imshow('Pose Estimation', frame2)
        frame_webcam = cv2.resize(cam_frame, (vid_frame.shape[1], vid_frame.shape[0]))
    
        # 두 프레임을 가로로 연결
        combined_frame = np.hstack((vid_frame, frame_webcam))

        # 크기 설정
        cv2.namedWindow('Video and Webcam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video and Webcam', 1280, 720)  # 원하는 창 크기로 설정
        # 결과 표시
        cv2.imshow('Video and Webcam', combined_frame)
    


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# 둘 중에 선택 시
# def get_args_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--webcam",  default=False, action="store_true", help="using webcam")
#     parser.add_argument("--video",  default=False, action="store_true", help="using video")
#     return parser

if __name__ == "__main__":
    # args = get_args_parser().parse_args()
    main()