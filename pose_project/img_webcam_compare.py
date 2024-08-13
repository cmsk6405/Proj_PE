import mediapipe as mp
from ultralytics import YOLO
import cv2

from util.pose_compare import PoseCompare

import numpy as np

import torch



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model():

	yolo = YOLO("yolov8n.pt")
	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

	return yolo, pose, mp_pose

def main():

    # Initialize PoseCompare
	pose = PoseCompare()

    # Load Video
	video_path = "../data/yoga_test_namas.mp4"
	vid = cv2.VideoCapture(video_path)

    # Load Webcam
	cam = cv2.VideoCapture(0)
	cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
	if not cam.isOpened():
		print("Cannot open camera")
		exit()

	# model create
	video_models = create_model()
	webcam_models = create_model()

	skip_frames = 3
	frame_count = 0

	while True:
        # Read from video
		vid_ret, vid_frame = vid.read()

		# if video end, play an it again
		if not vid_ret:
			vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue

		if frame_count % skip_frames == 0:
	        # Inference video image
			pose.load_img(frame=vid_frame, model=video_models, dest="ref") 

			# Read from webcam
			cam_ret, cam_frame = cam.read()
			# cam flip
			cam_frame = cv2.flip(cam_frame, 1)
			# Inference webcam image
			if not cam_ret:
				print("Can't receive frame (stream end?). Exiting ...")
				break
			pose.load_img(frame=cam_frame, model=webcam_models, dest="trgt")

			# # pose counting
			pose.counting()
			# # pose compare
			pose.compare(offset=20)

			# frame 사이즈 조정
			vid_frame = cv2.resize(vid_frame, (960 , 720))
			cam_frame = cv2.resize(cam_frame, (960, 720))
			# 비디오와 웹캠 합치기
			hcombine_frame = np.hstack((vid_frame,cam_frame))
			
			# Show Result
			cv2.imshow("Compare", hcombine_frame)

		frame_count += 1

        # Breakaway condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	vid.release()
	cam.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()