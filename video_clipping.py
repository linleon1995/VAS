from collections import deque
from glob import glob
import math
from os import makedirs
from os.path import join, basename

import cv2
from tqdm import tqdm
import numpy as np


def get_capture_attribute(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, frames_num


def get_difference_score(ref_image, frame):
    yxyx = np.argwhere(ref_image != 0)[[0, -1], :2].flatten()
    difference = cv2.subtract(src1=ref_image[yxyx[0]:yxyx[2], yxyx[1]:yxyx[3]],
                              src2=frame[yxyx[0]:yxyx[2], yxyx[1]:yxyx[3]])
    difference_score = np.sum(difference)
    return difference_score


def save_frames(target_path, filepath, fps, width, height, event_indices,
                frames):
    start_idx, end_idx = event_indices
    filename = join(target_path,
                    basename(filepath)[:-4] + f'_{start_idx}_{end_idx}.mp4')
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (width, height))
    for f in frames:
        writer.write(f)
    writer.release()


class DoorEventClipper():
    def __init__(self, target_path, buffer_seconds, freq, ref_image, threshold=10000):
        self.target_path = target_path
        self.buffer_seconds = buffer_seconds
        self.freq = freq
        self.ref_image = ref_image
        self.threshold = threshold

    def __call__(self, filepath):
        print(filepath)
        self.video_capture = cv2.VideoCapture(filepath)
        self.fps, self.width, self.height, self.frames_num = get_capture_attribute(
            capture=self.video_capture)
        self.maxlen = self.buffer_seconds * math.ceil(self.fps)
        self.event_indices = []
        self.frames = deque(maxlen=self.maxlen*2)
        self.get_event_indices(filepath)

        self.frame_idx = tqdm()
        select_event_idx = 0
        self.video_capture = cv2.VideoCapture(filepath)
        if self.event_indices:
            while self.video_capture.isOpened():
                self.frame_idx.update(1)
                ret = self.video_capture.grab()
                if not ret:
                    break

                if select_event_idx >= len(self.event_indices):
                    break

                event_start = self.event_indices[select_event_idx]
                event_end = event_start + 2*self.maxlen
                if self.frame_idx.n < event_start or self.frame_idx.n > event_end:
                    continue

                if select_event_idx == 0:
                    last_event_start = 0
                else:
                    last_event_start = self.event_indices[select_event_idx-1]

                new_add_len = min(event_start-last_event_start, 2*self.maxlen)
                self.push_frames(new_add_len)
                save_frames(self.target_path, filepath, self.fps, self.width, self.height,
                            (event_start, event_end), self.frames)
                select_event_idx += 1

    def get_event_indices(self, filepath):
        self.video_capture = cv2.VideoCapture(filepath)
        self.frame_idx = tqdm()
        while self.video_capture.isOpened():
            self.frame_idx.update(1)
            # ret, frame = video_capture.read()
            ret = self.video_capture.grab()
            if not ret:
                break

            if self.frame_idx.n % self.freq != 0:
                continue

            ret, frame = self.video_capture.retrieve()
            if frame is None:    # exist broken frame
                break

            if ret:
                difference_score = get_difference_score(ref_image=self.ref_image,
                                                        frame=frame)
                last_event_start = 0 if not self.event_indices else self.event_indices[-1]
                last_event_end = last_event_start + 2*self.maxlen
                if difference_score <= self.threshold and self.frame_idx.n > last_event_end:
                    self.event_indices.append(self.frame_idx.n-self.maxlen)

    def push_frames(self, num_frames):
        for _ in range(num_frames):
            self.frame_idx.update(1)
            ret, frame = self.video_capture.read()
            if ret:
                self.frames.append(frame)


if __name__ == '__main__':
    # parameters
    root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\door\more_data'
    buffer_seconds = 15
    reference_image_path = 'data/ref.png'
    threshold = 10000
    target_path = 'data/events_door'
    freq = 10

    # create folder
    makedirs(target_path, exist_ok=True)

    # load reference image
    ref_image = cv2.imread(reference_image_path)

    # get video files
    video_files = sorted(glob(join(root, '*.mp4')))

    # run
    raw_vid_processer = DoorEventClipper(
        target_path, buffer_seconds, freq, ref_image, threshold)
    for filepath in tqdm(video_files):
        raw_vid_processer(filepath)
