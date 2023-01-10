from pathlib import Path
import time

from tqdm import tqdm
import torch
import torchvision
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import numpy as np

from models.video_feat_extractor import PytorchVideoModel, get_transform_temp, get_vid_temp


def get_transform(model_name):
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    model_transform_params = {
        "x3d_xs": {
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 4,
            "sampling_rate": 12,
        },
        "x3d_s": {
            "side_size": 270,
            "num_frames": 16,
            "sampling_rate": 1,
        },
        "x3d_m": {
            "side_size": 270,
            "num_frames": 16,
            "sampling_rate": 1,
        },
        "x3d_l": {
            "side_size": 270,
            "num_frames": 16,
            "sampling_rate": 1,
        },
    }

    # Get transform parameters based on model
    transform_params = model_transform_params[model_name]
    transform = Compose(
        [
            # UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda window_size: window_size/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            Lambda(lambda window_size: torch.unsqueeze(window_size, dim=0))
        ]
    )
    return transform, transform_params


def main():
    vid_ext_model = 'x3d_m'
    pretrained = True
    device = 'cuda'
    training = False
    skip_exist = True
    vid_feat_transform, transform_params = get_transform(vid_ext_model)
    vid_feat_pipe = PytorchVideoModel(model_name=vid_ext_model, pretrained=pretrained,
                                      device=device, training=training,
                                      transform=vid_feat_transform)

    # data
    data_dir = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room'
    data_dir = Path(data_dir)
    save_dir = r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature'
    save_dir = Path(save_dir)
    save_dir = save_dir.joinpath(vid_ext_model)
    save_dir.mkdir(parents=True, exist_ok=True)

    if skip_exist:
        save_feat_files = list(save_dir.rglob('*.npy'))

    video_files = list(data_dir.rglob('*.mp4'))
    for video_path in tqdm(video_files):
        save_path = save_dir.joinpath(f'{video_path.stem}.npy')
        # if save_path in
        if skip_exist:
            if save_path in save_feat_files:
                print(f'Skip exist feature {save_path.stem}')

                continue

        video = torchvision.io.read_video(str(video_path))
        video_data = video[0]
        video_data = torch.transpose(
            torch.unsqueeze(video_data, dim=0), 0, -1)
        video_data = torch.squeeze(video_data, dim=-1)
        fps = video[2]['video_fps']
        num_frames = video_data.shape[1]

        window_size = 16
        step_size = 1

        try:
            video_feat = []
            t1 = time.time()
            for idx in range(0, num_frames, step_size):
                idx = min(num_frames-window_size, idx)
                video_clip = video_data[:, idx:idx+window_size]
                frame_feat = vid_feat_pipe.run(video_clip)
                # Prevent CUDA out of memory
                frame_feat = frame_feat.detach().cpu().numpy()
                video_feat.append(frame_feat)

            t2 = time.time()
            print(f'Process time {t2-t1}')

            # out_feature = torch.cat(video_feat, dim=0)
            out_feature = np.stack(video_feat, axis=2)
            out_feature = np.squeeze(out_feature, axis=0)
            out_feature = out_feature[:, :, 0, 0, 0]
            np.save(save_path, out_feature)
            print(
                f'Saving feature {video_path.stem}  with shape {out_feature.shape} {video_clip.shape}')

        except:
            print(f'Fail on {video_path.stem}')


if __name__ == '__main__':
    main()
