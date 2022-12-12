from multiprocessing import Process, Queue
from datetime import datetime
from typing import Tuple, Callable

from matplotlib import cm
import numpy as np
import cv2


class CRDE_detector():
    """Coffee room door event detector"""

    def __init__(self, image_loc, match_target):
        self.image_loc = image_loc
        self.match_target = match_target

    def __call__(self, image):
        # detect door
        if self.door_event_detect(image, self.image_loc, self.match_target):
            return True

        # detect people

    def door_event_detect(self, image, image_loc, match_target):
        # TODO: locate
        pic = image[image_loc]
        return self.matching(pic, match_target)

    def matching(self, img1, img2):
        # TODO: complete
        dist = np.sum(np.abs(img1-img2))
        if dist > 100:
            return True
        else:
            return False


class VideoProcess():
    def __init__(
        self,
        proc_func: Callable,
        foucc: str = 'mp4v',
    ):
        self.proc_func = proc_func
        self.foucc = cv2.VideoWriter_fourcc(*f'{foucc}')

    def get_vid_info(self, vidCap):
        width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidCap.get(cv2.CAP_PROP_FPS)
        return height, width, fps

    def __call__(self, video_ref, save_path):
        vidCap = cv2.VideoCapture(video_ref)
        height, width, fps = self.get_vid_info(vidCap)

        taskqueue = Queue()

        frame_counter = 0

        proc = Process(target=self.image_vis, args=(
            taskqueue, width, height, fps, frames_per_file, save_path
        ))
        proc.start()
        while frame_counter < frames_per_file:
            ret, image = vidCap.read()

            if ret:
                taskqueue.put((image, frame_counter))
                frame_counter += 1
            else:
                break

        taskqueue.put((None, None))
        proc.join()
        vidCap.release()

    def image_vis(self, taskqueue, width, height, fps, frames_per_file, save_path):
        writer = None
        image_idx = 0
        while True:
            image, frame_counter = taskqueue.get()
            if image is None:
                break

            if frame_counter % frames_per_file == 0:
                if writer:
                    writer.release()

                # now = datetime.now()
                # timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
                # TODO: suffix
                writer = cv2.VideoWriter(
                    save_path, self.foucc, fps, (width, height))

            self.proc_func(image)
            image_idx += 1

        writer.release()


def main():
    CaptureLength = 15
    image_loc = []
    match_target = cv2.imread(r'')

    EventDet = CRDE_detector(image_loc, match_target)
    vid_proc = VideoProcess(proc_func=EventDet)

    raw_vid_ref = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\door\20221101142355_coffee_video.mp4'
    vid_proc(video_ref=raw_vid_ref)


if __name__ == '__main__':
    main()
