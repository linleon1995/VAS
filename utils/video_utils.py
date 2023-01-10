from typing import List

import cv2


def read_video(filename: str) -> List:
    """Read the video through cv2

    e.g.,
    vid_path = r'video.mp4'
    frames = read_video(vid_path)

    Args:
        filename (str): Input video.

    Returns:
        List: List of video frames
    """
    video_capture = cv2.VideoCapture(filename)
    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames
