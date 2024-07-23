import mediapipe as mp
from ultralytics import YOLO
import cv2

from util.pose_compare import PoseCompare

# TODO: 추후 필요에 따라 삭제
import numpy as np
import time


def create_model():
	"""
	비디오와 웹캠이 각각의 모델을 사용할 수 있도록 하는 모델 생성 함수

	Returns:
		yolo와 media pipe 모델과 설정
	"""

	yolo = YOLO('yolov8l-pose.pt')
	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

	return yolo, pose, mp_pose

def main():
	"""
	비디오와 웹캠 읽기
	모델 생성
	자세비교
	결과출력
	"""

    # Initialize PoseCompare
	pose = PoseCompare()

    # Load Video
	video_path = "../data/yoga_test_namas.mp4"
	vid = cv2.VideoCapture(video_path)

    # Load Webcam
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
		# 비디오 끝나도 계속 재생되게 하기 필요에 따라 삭제
		if not vid_ret:
			vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue
        # Inference video image
		pose.load_img(frame=vid_frame, model=video_models, dest="ref") 

        # Read from webcam
		_, cam_frame = cam.read()
		# 캠영상 좌우 반전
		cam_frame = cv2.flip(cam_frame, 1)
        # Inference webcam image
		pose.load_img(frame=cam_frame, model=webcam_models, dest="trgt") 

		# Calculate FPS
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time

		# pose compare
		compare_img = pose.compare(offset=20)

		# Show Result
		cv2.imshow("Compare", compare_img)

        # Breakaway condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


# 추후 config 필요시
# import argparse

# def get_args_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--config", default="./configs.py", type=str, help="configuration file")
#     return parser

if __name__ == "__main__":
	# 추후 config 필요시
	# args = get_args_parser().parse_args()
    # exec(open(args.config).read())
	main()