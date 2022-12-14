from multiprocessing import Process, Queue
from datetime import datetime
from typing import Tuple, Callable

from matplotlib import cm
import numpy as np
import cv2


import os
from pathlib import Path

import numpy as np


def main(data_root, save_root, stack_size, method='i3d'):
    # save_dir = Path(save_root).joinpath(method)
    save_dir = Path(save_root)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ff = '['
    # for idx, path in enumerate(Path(data_root).rglob('*.mp4')):
    #     if idx > 3:
    #         break
    #     ff = ff + str(path) + ', '
    # ff = ff[:-2] + ']'
    # print(ff)
    # print(datetime.now())
    # os.system(
    #     'python main.py '
    #     f'feature_type={method} '
    #     'device="cuda:0" '
    #     f'output_path="{str(save_dir)}" '
    #     f'video_paths="{ff}" '
    #     'on_extraction="save_numpy" '
    #     f'stack_size={stack_size} '
    #     'step_size=1 '
    #     'batch_size=4 '
    # )
    # print(datetime.now())

    for idx, vid_f in enumerate(Path(data_root).rglob('*.mp4')):
        # data_dir = vid_f.parent.parts[-2:]
        # data_dir = Path('output').joinpath(*data_dir)
        # data_dir.mkdir(parents=True, exist_ok=True)
        # os.system('cd ..')
        # os.system('cd video_features')
        os.system(
            'python main.py '
            f'feature_type={method} '
            'device="cuda:0" '
            f'output_path="{str(save_dir)}" '
            f'video_paths="[{vid_f}]" '
            'on_extraction="save_numpy" '
            f'stack_size={stack_size} '
            'step_size=1 '
        )

    # feature_file = data_dir.with_stem(vid_f.stem).with_suffix('.npy')
    # a = np.load(feature_file)
    # print(a.shape, a.min(), a.max())
    return save_dir


def process_feature(data_root, stack_size: int = 16):
    # TODO: call by main
    total_feats = {}
    data_root = Path(data_root)
    save_dir = data_root.parent.joinpath(f'{data_root.name}_vid_f')
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, vid_f in enumerate(data_root.rglob('*.npy')):
        parts = vid_f.stem.split('_')
        filename, feat_type = '_'.join(parts[:-1]), parts[-1]
        if feat_type in ['ms', 'fps']:
            continue

        if filename not in total_feats:
            total_feats[filename] = {
                'rgb': vid_f.with_name(f'{filename}_rgb.npy'),
                'flow': vid_f.with_name(f'{filename}_flow.npy'),
            }
        else:
            continue

    for idx, (filename, feats) in enumerate(total_feats.items(), 1):
        if 'rgb' in feats and 'flow' in feats:
            print(f'{idx}/{len(total_feats)} Process feature')
            rgb_f = np.load(feats['rgb'])
            flow_f = np.load(feats['flow'])

            video_feature = np.concatenate([rgb_f, flow_f], axis=1)
            # XXX: Copy 1st frame currently
            _1st_frame_feature = np.tile(
                video_feature[0:1], reps=(stack_size-1, 1))
            video_feature = np.concatenate(
                [_1st_frame_feature, video_feature], axis=0)

            video_feature = np.transpose(video_feature)
            video_feature = np.float32(video_feature)
            np.save(save_dir.joinpath(f'{filename}.npy'), video_feature)
    pass


if __name__ == '__main__':
    # # TODO: speed up
    # data_root = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door'
    # save_dir = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature'
    # stack_size = 16
    # method = 'i3d'
    # feature_save_dir = main(data_root, save_dir, stack_size)

    feature_save_dir = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature\i3d'
    feature_save_dir = r'C:\Users\test\Desktop\Leon\Projects\video_features\output\exp\New folder\i3d'
    process_feature(feature_save_dir)
