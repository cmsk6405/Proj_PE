from util.pose_compare import PoseCompare
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import cv2
import time
from torchvision.io import read_image
from util.helpers import transforms

"""Webcam compare with still image

    This script will output 3 windows:
        1. Reference: reference inference
        2. Target: Target inference
        3. Compare: Comparison between two pose
"""
def create_model():
    yolo = YOLO('yolov8l-pose.pt')
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    return yolo, pose, mp_pose

if __name__ == "__main__":

    # ==== Setup ==== #
    # Initialize PoseCompare
	pose = PoseCompare()

    # Load Video
	video_path = "../data/yoga_test_namas.mp4"
	vid = cv2.VideoCapture(video_path)

    # ==== CV2 ==== #
	cam = cv2.VideoCapture(0)
	cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # FPS 
	prev_frame_time = 0
	new_frame_time = 0

	# model create
	video_models = create_model()
	webcam_models = create_model()

	while True:
        # Read from video
		vid_ret, vid_frame = vid.read()
        # Inference video image
		# 비디오 끝나도 계속 재생되게 하기 필요에 따라 삭제
		if not vid_ret:
			vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue
		print(f"-------------캠프레임 타입 - {type(vid_frame)}")

		pose.load_img(frame=vid_frame, model=video_models, dest="ref") 

        # Read from camera
		_, cam_frame = cam.read()
		# 캠영상 좌우 반전
		cam_frame = cv2.flip(cam_frame, 1)
        # Inference webcam image
		print(f"-------------캠프레임 타입 - {type(cam_frame)}")
		pose.load_img(frame=cam_frame, model=webcam_models, dest="trgt") 

		# Calculate FPS
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time

		# Compare Image 기존 출ㅇ력
		# compare_img = pose.draw_compare(fps=fps, offset=20)
		# compare_img = np.array(compare_img)

		# cv2를 이용한 출력
		compare_img = pose.compare(offset=20)

		# Show Image
		cv2.imshow("Compare", compare_img)

        # Breakaway condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break