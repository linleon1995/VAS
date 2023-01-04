from typing import Callable

import torch


def get_vas_model():
    pass


def get_post_process():
    pass


class PytorchVideoModel():
    def __init__(self, model_name: str, pretrained: bool = True,
                 device: str = 'cpu', training: bool = False, transform: Callable = None):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.training = training
        self.model = self.get_video_extractor()
        self.transform = transform

    def run(self, video: torch.Tensor) -> torch.Tensor:
        """AI is creating summary for run

        Args:
            video (torch.Tensor): video tensor with shape (1, C, T, H, W).

        Returns:
            torch.Tensor: [description]
        """
        if self.transform is not None:
            input_data = self.transform(video)
        else:
            input_data = video

        pred = self.model(input_data)
        return pred

    def get_model(self, with_transform: bool):
        if with_transform:
            return None
        else:
            return self.model

    def get_video_extractor(self):
        # Choose the model
        # TODO: 'facebookresearch/pytorchvideo' -> Var?
        model = torch.hub.load('facebookresearch/pytorchvideo',
                               self.model_name, pretrained=self.pretrained)

        # Set to GPU or CPU
        model = model.train() if self.training else model.eval()
        model = model.to(self.device)
        return model


class VAS_inference():
    def __init__(self, source: str, get_vid_feat_extractor: Callable,
                 get_vas_model: Callable, **kwargs):
        if source == 'video':
            self.vid_feat_extractor = get_vid_feat_extractor(**kwargs)
        elif source == 'feature':
            self.vid_feat_extractor = None
        else:
            raise ValueError(f'Unknown type of input data source')

        self.vas_model = get_vas_model()
        self.post_process = get_post_process()

    def __call__(self, input_data):
        if self.vid_feat_extractor is not None:
            vid_feat = self.vid_feat_extractor(input_data)
        else:
            vid_feat = input_data

        vas_raw_pred = self.vas_model(vid_feat)

        vas_pred = self.post_process(vas_raw_pred)

        return vas_pred


def get_transform_temp():
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

    model_name = 'x3d_s'

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
            "side_size": 182,
            "crop_size": 182,
            "num_frames": 13,
            "sampling_rate": 6,
        },
        "x3d_m": {
            "side_size": 256,
            "crop_size": 256,
            "num_frames": 16,
            "sampling_rate": 5,
        }
    }

    # Get transform parameters based on model
    transform_params = model_transform_params[model_name]

    # # Note that this transform is specific to the slow_R50 model.
    # transform = ApplyTransformToKey(
    #     key="video",
    #     transform=Compose(
    #         [
    #             UniformTemporalSubsample(transform_params["num_frames"]),
    #             Lambda(lambda x: x/255.0),
    #             NormalizeVideo(mean, std),
    #             ShortSideScale(size=transform_params["side_size"]),
    #             CenterCropVideo(
    #                 crop_size=(
    #                     transform_params["crop_size"], transform_params["crop_size"])
    #             ),
    #             # Lambda(lambda x: torch.unsqueeze(x, dim=0))
    #         ]
    #     ),
    # )
    transform = Compose(
        [
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCropVideo(
                crop_size=(
                    transform_params["crop_size"], transform_params["crop_size"])
            ),
            Lambda(lambda x: torch.unsqueeze(x, dim=0))
        ]
    )
    return transform, transform_params


def get_vid_temp(transform_params):
    import json
    import urllib
    from pytorchvideo.data.encoded_video import EncodedVideo

    frames_per_second = 30
    # The duration of the input clip is also specific to the model.
    clip_duration = (
        transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

    url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
    video_path = 'archery.mp4'
    try:
        urllib.URLopener().retrieve(url_link, video_path)
    except:
        urllib.request.urlretrieve(url_link, video_path)

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class and load the video
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = video_data["video"]
    return video_data


if __name__ == '__main__':
    transform, transform_params = get_transform_temp()
    video = get_vid_temp(transform_params)

    model = PytorchVideoModel(model_name='x3d_s', pretrained=True,
                              device='cpu', training=False, transform=transform)
    pred = model.run(video)
    vid_model = model.get_model(with_transform=True)

    import json
    json_filename = "kinetics_classnames.json"
    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    pred = post_act(pred)
    pred_classes = pred.topk(k=5).indices[0]

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
    pass

    # inferencer = VAS_inference(
    #     source='feature',
    #     get_vid_feat_extractor=get_vid_extractor,
    #     get_vas_model=get_vas_model,
    # )

    # kwargs = {
    #     'model_name': 'x3d_s',
    #     'pretrained': True,
    #     'device': 'cpu',
    #     'training': False,
    # }
    # model = get_vid_extractor(**kwargs)
    # print(model)
