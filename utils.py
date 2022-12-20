
import cv2


def get_frames_from_video(vid_path):
    """AI is creating summary for get_frames_from_video

    # vid_path = r'video.mp4'
    # frames = get_frames_from_video(vid_path)
    Args:
        vid_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    video_capture = cv2.VideoCapture(vid_path)
    frames = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames
