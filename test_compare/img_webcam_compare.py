import mediapipe as mp
from ultralytics import YOLO
import cv2

from util.pose_compare import PoseCompare

# TODO: 추후 필요에 따라 삭제
import numpy as np
import time

from threading import Thread

def create_model():
	yolo = YOLO('yolov8l-pose.pt')
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
    # FPS 
	prev_frame_time = 0
	new_frame_time = 0

	# model create
	video_models = create_model()
	webcam_models = create_model()


	while True:
        # Read from video
		vid_ret, vid_frame = vid.read()

		# 비디오 끝나도 계속 다시 재생
		if not vid_ret:
			vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
			continue

        # Inference video image
		# pose.load_img(frame=vid_frame, model=video_models, dest="ref") 
		vid_thrd = Thread(target=pose.load_img, args=(vid_frame, video_models, "ref"))
		vid_thrd.start()
		vid_thrd.join()

        # Read from webcam
		cam_ret, cam_frame = cam.read()
		# 캠영상 좌우 반전
		cam_frame = cv2.flip(cam_frame, 1)
        # Inference webcam image
		if not cam_ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break
		# pose.load_img(frame=cam_frame, model=webcam_models, dest="trgt")
		cam_thrd = Thread(target=pose.load_img, args=(cam_frame, webcam_models, "trgt"))
		cam_thrd.start()
		cam_thrd.join()


		# Calculate FPS
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time

		# pose.counting()
		cnt_thrd = Thread(target = pose.counting)
		cnt_thrd.start()
		cnt_thrd.join()


		# pose compare
		compare_img = pose.compare(offset=20)

		# Show Result
		cv2.imshow("Compare", compare_img)

        # Breakaway condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	vid.release()
	cam.release()
	cv2.destroyAllWindows()


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