import math
from typing import List, Tuple


# 각도 횟수
import mediapipe as mp
mp_pose = mp.solutions.pose

joint_pairs = [
    ('Left Shoulder', mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ('Right Shoulder', mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ('Left Elbow', mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ('Right Elbow', mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ('Left Hip', mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    ('Right Hip', mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ('Left Knee', mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    ('Right Knee', mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

# ==== Variable ==== #
# media pipe Keypoints mapping
kpts_name = {
    "0" :" nose",
    "1" : "left eye (inner)",
    "2" : "left eye",
    "3" : "left eye (outer)",
    "4" : "right eye (inner)",
    "5" : "right eye",
    "6" : "right eye (outer)",
    "7" : "left ear",
    "8" : "right ear",
    "9" : "mouth:(left)",
    "10" : "mouth:(right)",
    "11" : "left shoulder",
    "12" : "right shoulder",
    "13" : "left elbow",
    "14" : "right elbow",
    "15" : "left wrist",
    "16" : "right wrist",
    "17" : "left pinky",
    "18" : "right pinky",
    "19" : "left index",
    "20" : "right index",
    "21" : "left thumb",
    "22" : "right thumb",
    "23" : "left hip",
    "24" : "right hip",
    "25" : "left knee",
    "26" : "right knee",
    "27" : "left ankle",
    "28" : "right ankle",
    "29" : "left heel",
    "30" : "right heel",
    "31" : "left foot index",
    "32" : "right foot index",
}

# keypoints id required to calculate angle
# 각 부분별로 각도 계산시 필요한 mp의 lm번호
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

# ==== Functions ==== #
def get_angle(kpts_coord: List[Tuple[int, int]], angle_kpts: List[Tuple[int, int]]):
    """
    kpts_angle을 통해 전달받은 각도 계산시 필요한 mp의 lm번호들로 위치를 구해 각도를 구한다

    Args:
        kpts_coord: keypoints cooradiate, should be 33 length long - mp로 찍히는 lm의 수
        angle_kpts: List of keypoints to get angle from. It should be 3 length
            long with the angle point in the middle
    """

    # Get coords from the 3 points
    a = [kpts_coord[angle_kpts[0]].x, kpts_coord[angle_kpts[0]].y]
    b = [kpts_coord[angle_kpts[1]].x, kpts_coord[angle_kpts[1]].y]
    c = [kpts_coord[angle_kpts[2]].x, kpts_coord[angle_kpts[2]].y]

    # print(f"각도 체크용 {a, b, c}")

    # Calculate angle
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    # Sanity check, angle must be between 0-180
    ang = int(ang + 360 if ang < 0 else ang)
    ang = int(ang - 180 if ang > 270 else ang)


    return ang


