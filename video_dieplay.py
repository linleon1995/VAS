# TODO: argparse
# TODO: main function
from multiprocessing import Process, Queue
from datetime import datetime

import cv2


class VAS_visualizer():
    def __init__(
        self,
        height,
        width,
        font_color=(255, 255, 255),
        progress_line_color=(0, 0, 0),
    ):
        self.font_color = font_color
        self.progress_line_color = progress_line_color
        # self.progress_cmap = progress_cmap

    def vis(self, video, predict, correct):
        pass

    def draw(self, image):
        pass


def draw_progress_bar(image, image_idx, width, height):
    colors = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'green': (0, 255, 0),
    }

    # Predict class
    predict = 'Tag'
    cv2.putText(image, 'Predict:', (10, 175), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, colors['white'], 1, cv2.LINE_AA)
    cv2.putText(image, predict, (60, 175), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, colors['white'], 1, cv2.LINE_AA)

    # Progress bar
    cv2.putText(image, 'Predict', (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, colors['white'], 1, cv2.LINE_AA)
    cv2.putText(image, 'Correct', (10, 215), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, colors['white'], 1, cv2.LINE_AA)

    for i in range(80):
        if i % 2 == 0:
            color = colors['green']
        else:
            color = colors['white']
        cv2.rectangle(image, (60+i, 194), (61+i, 200),
                      color, -1)
        cv2.rectangle(image, (60+i, 209), (61+i, 215),
                      color, -1)

    start_point = (60+image_idx, 194)
    end_point = (60+image_idx, 215)

    cv2.line(image, start_point, end_point, color=colors['black'], thickness=1)
    return image


def image_vis(taskqueue, width, height, fps, frames_per_file):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = None

    image_idx = 0
    while True:
        image, frame_counter = taskqueue.get()

        if image is None:
            break

        if frame_counter % frames_per_file == 0:

            if writer:
                writer.release()

            # index = int(frame_counter // frames_per_file)
            # writer = cv2.VideoWriter(f'output-{index}.mp4', fourcc, fps, (width, height))

            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
            writer = cv2.VideoWriter(
                f'videos/output-{timestamp}.mp4', fourcc, fps, (width, height))

        draw_progress_bar(image, image_idx, width, height)
        writer.write(image)
        image_idx += 1

    writer.release()


if __name__ == '__main__':
    f = r'C:\Users\test\Desktop\Leon\Datasets\Breakfast\BreakfastII_15fps_qvga_sync\P03\cam01\P03_cereals.avi'

    vidCap = cv2.VideoCapture(f)

    width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = vidCap.get(cv2.CAP_PROP_FPS)

    taskqueue = Queue()

    frame_counter = 0

    total_time = 100 * 3600  # 30 seconds per process
    file_time = 10 * 60  # 10 seconds per file

    total_frames = fps * total_time

    frames_per_file = fps * file_time

    proc = Process(target=image_vis, args=(
        taskqueue, width, height, fps, frames_per_file))
    proc.start()

    while frame_counter < total_frames:
        ret, image = vidCap.read()

        if ret:
            taskqueue.put((image, frame_counter))
            frame_counter += 1
        else:
            break

    # 傳入 None 終止工作行程
    taskqueue.put((None, None))

    # 等待工作行程結束
    proc.join()

    vidCap.release()
