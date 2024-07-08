import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from typing import Dict, List, Tuple, Optional

import math

import time


kpts_angle = {
    "right_shoulder": [11, 12, 14],
    "right_arm": [12, 14, 16],
    "left_shoulder": [13, 11, 12],
    "left_arm": [15, 13, 11],
    "right_hip": [23, 24, 26],
    "right_leg": [24, 26, 28],
    "left_hip": [25, 23, 24],
    "left_leg": [23, 25, 27]
}
def get_angle(kpts_coord: List[Tuple[int, int]], angle_kpts: List[Tuple[int, int]]):
    """Get angle from 3 keypoints

    This function will calculate the angle between 3 points.

    Args:
        kpts_coord: keypoints cooradiate, should be 17 length long
        angle_kpts: List of keypoints to get angle from. It should be 3 length
            long with the angle point in the middle
    """

    
    # Get coords from the 3 points
    a = kpts_coord[angle_kpts[0]]
    print(f"-------------{a}")
    b = kpts_coord[angle_kpts[1]]
    c = kpts_coord[angle_kpts[2]]

    # Calculate angle
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    # Sanity check, angle must be between 0-180
    ang = int(ang + 360 if ang < 0 else ang)
    ang = int(ang - 180 if ang > 270 else ang)

    return ang

def calc_angle_diff(angle_ref, angle_trgt):
    """Calculate target and reference's angle difference

    Returns:
            A dictionary containing the joint as key, angle difference as the value
    """
    angle_diff = {k: angle_ref[k] - angle_trgt[k] for k in angle_ref}

    return angle_diff
    
def load_img(frame):

    # YOLOv8 모델 로드
    model = YOLO('yolov8l-pose.pt')

    # MediaPipe 포즈 추정 모델 로드
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

     # YOLOv8을 사용하여 사람 탐지
    results = model(frame)[0]

    for result in results.boxes.data:
        # bbox, 신뢰도, 분류 클래스
        x1, y1, x2, y2, conf, cls = result
 
        bbox = []

        if int(cls) == 0:  # 사람 클래스일 경우
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            # 바운딩 박스 영역 추출
            person_img = frame[int(y1):int(y2), int(x1):int(x2)]

            # MediaPipe를 사용하여 포즈 추정
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            mp_result = pose.process(person_rgb)

            landmarks = []
            for i in range(33):
                landmarks.append([mp_result.pose_landmarks.landmark[i].x,
                                    mp_result.pose_landmarks.landmark[i].y,
                                    mp_result.pose_landmarks.landmark[i].visibility])

            if mp_result.pose_landmarks:
                # 원본 프레임에 포즈 랜드마크 및 연결선 그리기
                mp_drawing.draw_landmarks(frame[int(y1):int(y2), int(x1):int(x2)], 
                                        mp_result.pose_landmarks, 
                                        mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        angles = {}
        for k, v in kpts_angle.items():
            angles[k] = get_angle(landmarks, v)

    return angles

def main():

    #ret
    video_path = "./data/yoga_test_namas.mp4"
    vid = cv2.VideoCapture(video_path)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps


    #trgt
    cap = cv2.VideoCapture(0)  # 0은 웹캠을 의미, 파일을 사용하려면 파일 경로를 입력
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    
    while cap.isOpened():
        start_time = time.time()
        vid_ret, vid_frame = vid.read()
        if not vid_ret:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cam_ret, cam_frame = cap.read()
        cam_frame = cv2.flip(cam_frame, 1)
        if not cam_ret:
            break
        
        
        angle_ref = load_img(vid_frame)
        angle_trgt = load_img(cam_frame)
        
        # offset = 20

        # angle_diff = calc_angle_diff(angle_ref, angle_trgt)
        # angle_diff_str = [f"Maximum Angle Diff: {offset}"]

        # all_ok = True
        # for k, v in angle_diff.items():
        #     if abs(v) < offset:
        #         status = f"OK ({v})"
        #     else:
        #         status = f"NOT OK ({v})"
        #         # If at least one is not OK then all_ok will be False
        #         all_ok = False

        #     angle_diff[k] = status

        # angle_diff_str.extend([f"{k}: {v}" for k, v in angle_diff.items()])
        # angle_diff_str.append(f"ALL OK: {all_ok}")
        # angle_diff_str = "\n".join(angle_diff_str)

        # print(angle_diff_str)
        # 결과 프레임 보여주기
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

        # 프레임 처리 시간 계산
        process_time = time.time() - start_time
        
        # 필요한 경우 대기
        if process_time < frame_time:
            time.sleep(frame_time - process_time)

    cap.release()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()