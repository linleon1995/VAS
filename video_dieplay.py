import os
from pathlib import Path
from multiprocessing import Process, Queue
from datetime import datetime

from matplotlib import cm
import numpy as np
import cv2


def get_cmap_for_cv2(num_classes: int, cmap_name: str = 'jet'):
    indices = np.linspace(0, 1, num_classes)
    cmap = cm.get_cmap(cmap_name)
    ind_a, ind_b = indices[:num_classes//2], indices[num_classes//2:][::-1]
    indices = np.reshape(np.stack([ind_a, ind_b]), num_classes, order='F')
    colors = cmap(indices)
    colors = np.int32(255*colors)
    return colors


def get_actions(data_root, dataset: str):
    mapping_file = os.path.join(data_root, dataset, 'mapping.txt')
    with open(mapping_file, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]

    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


class VAS_visualizer():
    def __init__(
        self,
        dataset,
        data_root,
        cmap_name='turbo',
        foucc='mp4v',
        text_color=(255, 255, 255),
        progress_line_color=(0, 0, 0),
        total_time=3600*100,
        file_time=60*10,
        font_size=0.3,
    ):
        self.data_root = data_root
        self.dataset = dataset
        self.cmap_name = cmap_name
        self.foucc = cv2.VideoWriter_fourcc(*f'{foucc}')
        self.text_color = text_color
        self.progress_line_color = progress_line_color

        self.total_time = total_time
        self.file_time = file_time
        self.font_size = font_size
        self.color_map = self.get_action_color_mapping()

    def __call__(self, video_ref, vid_correct, vid_predict, save_path):
        vidCap = cv2.VideoCapture(video_ref)
        width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidCap.get(cv2.CAP_PROP_FPS)

        taskqueue = Queue()

        frame_counter = 0
        # total_frames = fps * self.total_time
        # frames_per_file = fps * self.file_time
        frames_per_file = len(vid_correct)

        proc = Process(target=self.image_vis, args=(
            taskqueue, width, height, fps, frames_per_file, save_path,
            vid_correct, vid_predict
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

    def image_vis(self, taskqueue, width, height, fps, frames_per_file, save_path,
                  vid_correct, vid_predict):
        writer = None

        # writer = cv2.VideoWriter(
        #     str(Path(save_path).joinpath(f'output.mp4')), self.foucc, fps, (width, height))

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

            # TODO: predict, correct tag
            if image_idx % 2 == 0:
                self.draw(image, image_idx, vid_correct, vid_predict)
                writer.write(image)
            image_idx += 1

        writer.release()

    def draw(self, image, image_idx, vid_correct, vid_predict):
        # Predict class
        # TODO: layout
        predict = vid_predict[image_idx]
        correct = vid_correct[image_idx]
        text_params = (cv2.FONT_HERSHEY_SIMPLEX, self.font_size,
                       self.text_color, 1, cv2.LINE_AA)

        cv2.putText(image, 'Predict:', (10, 175), *text_params)
        cv2.putText(image, predict, (60, 175), *text_params)
        cv2.putText(image, 'Correct', (10, 185), *text_params)
        cv2.putText(image, correct, (60, 185), *text_params)

        # Progress bar
        cv2.putText(image, 'Predict', (10, 205), *text_params)
        cv2.putText(image, 'Correct', (10, 215), *text_params)

        r = 16
        for i in range(len(vid_correct)//r):
            correct = vid_correct[i*r]
            correct_color = self.color_map[correct].tolist()
            correct_color = tuple([int(v) for v in correct_color])
            cv2.rectangle(image, (60+i, 200), (61+i, 205),
                          correct_color, -1)

            predict = vid_predict[i*r]
            predict_color = self.color_map[predict].tolist()
            predict_color = tuple([int(v) for v in predict_color])
            cv2.rectangle(image, (60+i, 210), (61+i, 215),
                          predict_color, -1)

        start_point = (60+image_idx//r, 200)
        end_point = (60+image_idx//r, 215)

        cv2.line(image, start_point, end_point,
                 color=self.progress_line_color, thickness=1)

    def get_action_color_mapping(self):
        actions = get_actions(self.data_root, self.dataset)
        num_classes = len(actions)
        colors = get_cmap_for_cv2(num_classes, cmap_name=self.cmap_name)
        action_to_color = dict()
        for action, color in zip(actions, colors):
            action_to_color[action] = color[:-1]
        return action_to_color


def test_coffee():
    # cap = cv2.VideoCapture("videos/P03_webcam01_P03_tea.txt.mp4")
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(length)

    ground_truth_path = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data\coffee_room\groundTruth'
    recog_path = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\results\breakfast\split_1'
    # vid = 'P03_cam01_P03_coffee.txt'
    video_files = Path(ground_truth_path).glob('*.txt')
    video_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room_door_event_dataset'
    data_root = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data'
    dataset = 'coffee_room'
    V = VAS_visualizer(data_root, dataset, cmap_name='turbo')

    for vid_path in video_files:
        vid = vid_path.name
        keys = vid.split('_')
        keys = [keys[0], keys[1], f'{keys[2]}_{keys[3][:-4]}.avi']
        video_ref = os.path.join(video_root, *keys)

        gt_file = os.path.join(ground_truth_path, vid)
        gt_content = read_file(gt_file).split('\n')[0:-1]
        recog_file = os.path.join(recog_path, vid.split('.')[0])
        recog_content = read_file(recog_file).split('\n')[1].split()
        V(video_ref, vid_correct=gt_content,
          vid_predict=recog_content, save_path=f'videos/{vid}.mp4')


if __name__ == '__main__':
    test_coffee()
    # # f = r'C:\Users\test\Desktop\Leon\Projects\video_features\output\i3d'
    # f = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data\breakfast\features'
    # for ff in Path(f).glob('*.npy'):
    #     data = np.load(ff)
    #     print(ff.name, data.shape, f'max {data.max()} min {data.min()}')
    # # import cv2

    # # cap = cv2.VideoCapture("videos/P03_webcam01_P03_tea.txt.mp4")
    # # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # # print(length)

    # ground_truth_path = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data\breakfast\groundTruth'
    # recog_path = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\results\breakfast\split_1'
    # vid = 'P03_cam01_P03_coffee.txt'
    # video_files = Path(ground_truth_path).glob('*.txt')
    # video_root = r'C:\Users\test\Desktop\Leon\Datasets\Breakfast\BreakfastII_15fps_qvga_sync'
    # data_root = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data'
    # V = VAS_visualizer(cmap_name='turbo')

    # for vid_path in video_files:
    #     vid = vid_path.name
    #     keys = vid.split('_')
    #     keys = [keys[0], keys[1], f'{keys[2]}_{keys[3][:-4]}.avi']
    #     video_ref = os.path.join(video_root, *keys)

    #     gt_file = os.path.join(ground_truth_path, vid)
    #     gt_content = read_file(gt_file).split('\n')[0:-1]
    #     recog_file = os.path.join(recog_path, vid.split('.')[0])
    #     recog_content = read_file(recog_file).split('\n')[1].split()
    #     V(video_ref, vid_correct=gt_content,
    #       vid_predict=recog_content, save_path=f'videos/{vid}.mp4')
