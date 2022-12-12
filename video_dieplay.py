import os
from pathlib import Path
from multiprocessing import Process, Queue
from datetime import datetime
from typing import Tuple

from matplotlib import cm
import numpy as np
import cv2


def get_cmap_for_cv2(num_classes: int, cmap_name: str = 'jet'):
    indices = np.linspace(0, 1, num_classes)
    cmap = cm.get_cmap(cmap_name)
    # ind_a, ind_b = indices[:num_classes//2], indices[num_classes//2:][::-1]
    # indices = np.reshape(np.stack([ind_a, ind_b]), num_classes, order='F')
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


class AssignGridPosition():
    def __init__(self, height: int, width: int, grid_height_num: int = 9, grid_width_num: int = 16):
        self.height = height
        self.width = width
        self.grid_height_num = grid_height_num
        self.grid_width_num = grid_width_num
        self.cell_height = height / self.grid_height_num
        self.cell_width = width / self.grid_width_num
        self.grid_to_pixs = self.build_grid_to_pixs()

    def __call__(self, grid_height_idx: int, grid_width_idx: int) -> Tuple:
        assert grid_height_idx < self.grid_height_num, f'Grid height {grid_height_idx} out of range'
        assert grid_width_idx < self.grid_width_num, f'Grid width {grid_width_idx} out of range'
        return self.grid_to_pixs[(grid_height_idx, grid_width_idx)]

    def build_grid_to_pixs(self) -> dict:
        grid_to_pixs = {}
        for grid_h in range(self.grid_height_num):
            for grid_w in range(self.grid_width_num):
                grid_to_pixs[(grid_h, grid_w)] = (
                    int(grid_h*self.cell_height), int(grid_w*self.cell_width))
        return grid_to_pixs


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
        # TODO: height and width reverse
        self.point = AssignGridPosition(height=1920, width=1080,
                                        grid_height_num=64, grid_width_num=36)

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

            self.draw_single_frame(writer, image, image_idx, vid_correct,
                                   vid_predict, width, height)
            image_idx += 1

        writer.release()

    def draw_single_frame(self, writer, image, image_idx, vid_correct,
                          vid_predict, width, height):
        # TODO: predict, correct tag
        if image_idx % 2 == 0:
            self.draw(image, image_idx, vid_correct,
                      vid_predict, width, height)
            writer.write(image)
        return writer

    def draw(self, image, image_idx, vid_correct, vid_predict, width, height):
        # 240 X 320 -> 1080 X 1920
        # Predict class
        predict = vid_predict[image_idx]
        correct = vid_correct[image_idx]
        text_params = (cv2.FONT_HERSHEY_SIMPLEX, self.font_size,
                       self.text_color, 1, cv2.LINE_AA)
        cv2.putText(image, 'Predict:', self.point(*(10, 26)), *text_params)
        cv2.putText(image, predict, self.point(*(16, 26)), *text_params)
        cv2.putText(image, 'Correct', self.point(*(10, 28)), *text_params)
        cv2.putText(image, correct, self.point(*(16, 28)), *text_params)

        # Progress bar
        cv2.putText(image, 'Predict', self.point(*(10, 31)), *text_params)
        cv2.putText(image, 'Correct', self.point(*(10, 33)), *text_params)
        predict_bar_start = self.point(*(16, 30))
        correct_bar_start = self.point(*(16, 32))

        num_frame = len(vid_correct)
        # XXX: Check the behave on long sequence. e.g., frame=15000, width=1920
        # How to perform each action without exceed window width
        r = 1
        rec_h = height // 36
        bar_w = int(width*0.6)
        rec_w = bar_w / num_frame

        for i in range(num_frame//r):
            correct = vid_correct[i*r]
            correct_color = self.color_map[correct].tolist()
            correct_color = tuple([int(v) for v in correct_color])
            cv2.rectangle(image,
                          (int(correct_bar_start[0]+i*rec_w),
                           correct_bar_start[1]),
                          (int(correct_bar_start[0]+(i+1)*rec_w),
                           correct_bar_start[1]+rec_h),
                          correct_color, -1)

            predict = vid_predict[i*r]
            predict_color = self.color_map[predict].tolist()
            predict_color = tuple([int(v) for v in predict_color])
            cv2.rectangle(image,
                          (int(predict_bar_start[0]+i*rec_w),
                           predict_bar_start[1]),
                          (int(predict_bar_start[0]+(i+1)*rec_w),
                           predict_bar_start[1]+rec_h),
                          predict_color, -1)
        # line
        line_start = (correct_bar_start[0]+int(image_idx*rec_w/r),
                      predict_bar_start[1])
        line_end = (correct_bar_start[0]+int(image_idx*rec_w/r),
                    correct_bar_start[1]+rec_h)
        cv2.line(image, line_start, line_end,
                 color=self.progress_line_color, thickness=2)

    def get_action_color_mapping(self):
        actions = get_actions(self.data_root, self.dataset)
        num_classes = len(actions)
        colors = get_cmap_for_cv2(num_classes, cmap_name=self.cmap_name)
        action_to_color = dict()
        for action, color in zip(actions, colors):
            action_to_color[action] = color[:-1]
        return action_to_color


def get_result(video_root, gt_root, pred_root):
    video_files = list(Path(video_root).rglob('*.mp4'))
    gt_files = list(Path(gt_root).rglob('*.txt'))
    pred_files = Path(pred_root).rglob('*.txt')
    data_samples = []
    for pred_f in pred_files:
        for video_f in video_files:
            if video_f.stem == pred_f.stem:
                break

        for gt_f in gt_files:
            if gt_f.stem == pred_f.stem:
                break

        sample = (video_f, gt_f, pred_f)
        data_samples.append(sample)
    return data_samples


def vas_videos(dataset, data_root, video_root, gt_root, pred_root, save_root='./'):
    save_root = Path(save_root)
    V = VAS_visualizer(dataset, data_root, cmap_name='turbo', font_size=1.2)

    data_samples = get_result(video_root, gt_root, pred_root)
    for video_ref, gt_file, recog_file in data_samples:
        vid = video_ref.stem
        gt_content = read_file(gt_file).split('\n')[0:-1]
        recog_content = read_file(recog_file).split('\n')[0:-1]
        # XXX: temporally pad recog_content by last element
        recog_content.append(recog_content[-1])
        # print(gt_file.name, len(gt_content), len(recog_content))
        V(str(video_ref), vid_correct=gt_content,
          vid_predict=recog_content, save_path=str(save_root.joinpath(f'{vid}.mp4')))


if __name__ == '__main__':
    dataset = 'coffee_room'
    gt_root = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data\coffee_room\groundTruth'
    recog_root = r'C:\Users\test\Desktop\Leon\Projects\UVAST\coffee_room2\inference'
    video_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room_door_event_dataset'
    data_root = r'C:\Users\test\Desktop\Leon\Projects\MS-TCN2\data'
    vas_videos(dataset, data_root, video_root,
               gt_root, recog_root, save_root='results')

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
