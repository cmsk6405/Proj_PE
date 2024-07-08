from util.pose_compare import PoseCompare
import numpy as np
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

if __name__ == "__main__":

    # ==== Setup ==== #
    # Initialize PoseCompare
	pose = PoseCompare()

    # Load Image
	img_path = "/mnt/d/Data/yoga/TRAIN/goddess/00000101.jpg"
	pose.load_img(frame=img_path, dest="ref") 
	# img_path = "./data/yoga_test_namas.mp4"
	# vid = cv2.VideoCapture(img_path)

    # ==== CV2 ==== #
	cam = cv2.VideoCapture(0)
	cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # FPS 
	prev_frame_time = 0
	new_frame_time = 0

	while True:
        # Read from camera
		_, cam_frame = cam.read()

		print(f"-------------캠프레임 타입 - {type(cam_frame)}")
		# _, vid_frame = vid.read()
		# pose.load_img(frame=vid_frame, dest="ref") 

        # Inference webcam image
		pose.load_img(frame=cam_frame, dest="trgt") 

		# Calculate FPS
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time

		# Compare Image
		compare_img = pose.draw_compare(fps=fps, offset=20)
		compare_img = np.array(compare_img)

		# Show Image
		cv2.imshow("Compare", compare_img)

		# Reference Image
		# ref_img = pose.draw_one("ref")
		# ref_img = np.array(ref_img)
		# trgt_img = pose.draw_one("trgt")
		# trgt_img = np.array(trgt_img)
		# Show Image
		# cv2.imshow("Reference", ref_img)
		# cv2.imshow("Target", trgt_img)

        # Breakaway condition
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break