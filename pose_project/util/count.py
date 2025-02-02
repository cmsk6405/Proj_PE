
# 이차함수
def count_repetition(previous_pose, current_pose, previous_state, flag, tolerance=90):
    """
    Determine the number of repetitions of the pose by using the x and y coordinates of the landmarks.

    Args:
        previous_pose : landmarks of the previous frame
        current_pose : person land mark
        previous_state : Check the changes in each part
        flag : pose change flag
        tolerance : threshold. Defaults to 90.

    Returns:
        current_pose : be the previous_pose
        current_state.copy() : copy the current_pose
        flag : flag after calculation
    """

    if current_pose is None or len(current_pose) == 0:
        return previous_pose, previous_state, flag
    else:
        current_state = previous_state.copy()
        sdx, sdy = 0, 0

        # MediaPipe uses 33 landmarks.
        for i in range(33):  
            dx = current_pose[i].x - previous_pose[i].x
            dy = current_pose[i].y - previous_pose[i].y

            # Normalize the tolerance by dividing it by 100 to fit it into a 0-1 scale
            if abs(dx) < tolerance / 100:  
                dx = 0
            if abs(dy) < tolerance / 100:
                dy = 0

            sdx += dx
            sdy += dy

        # Update the current_state with pose changes based on the variations in x and y
        if sdx > (tolerance * 3 / 100):
            current_state[0] = 1
        elif sdx < (tolerance * -3 / 100):
            current_state[0] = 0
        if sdy > (tolerance * 3 / 100):
            current_state[1] = 1
        elif sdy < (tolerance * -3 / 100):
            current_state[1] = 0
        
        # Calculate the flag if a pose change is detected.
        if current_state != previous_state:
            flag = (flag + 1) % 2
        
        return current_pose, current_state.copy(), flag