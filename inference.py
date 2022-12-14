from typing import Callable, Dict

import torch


def get_vas_model(**kwargs):
    pass


def get_post_process(**kwargs):
    pass


class VAS_inference():
    def __init__(self, get_vas_model: Callable, get_vid_feat_extractor: Callable = None,
                 **kwargs):
        self.vas_model = get_vas_model(**kwargs)
        if get_vid_feat_extractor is not None:
            self.vid_feat_extractor = get_vid_feat_extractor(**kwargs)
        else:
            self.vid_feat_extractor = None

        self.post_process = get_post_process

    def __call__(self, input_data: torch.Tensor) -> torch.Tensor:
        if self.vid_feat_extractor is not None:
            vid_feat = self.vid_feat_extractor(input_data)
        else:
            vid_feat = input_data

        vas_raw_pred = self.vas_model(vid_feat)

        vas_pred = self.post_process(vas_raw_pred)

        return vas_pred


if __name__ == '__main__':
    from models.video_feat_extractor import PytorchVideoModel, get_transform_temp, get_vid_temp

    def get_vid_ext_callback(**kargs):
        model_name = kargs.get('model_name', 'x3d_s')
        pretrained = kargs.get('pretrained', True)
        device = kargs.get('device', 'cpu')
        training = kargs.get('training', False)
        transform = kargs.get('vid_feat_transform', None)

        model = PytorchVideoModel(model_name=model_name, pretrained=pretrained,
                                  device=device, training=training, transform=transform)
        return model.run

    def get_vas_model():
        from ..MS_TCN2.model import MS_TCN2

    transform, transform_params = get_transform_temp()
    # TODO: better args passing way.
    inferencer = VAS_inference(
        model_name='x3d_m',
        pretrained=True,
        device='cpu',
        training=False,
        vid_feat_transform=transform,
        get_vid_feat_extractor=get_vid_ext_callback,
        get_vas_model=get_vas_model,
    )

    # video = get_vid_temp(transform_params)

    import torchvision
    video_path = r'C:\Users\test\Desktop\Leon\Projects\VAS\exp\New folder\output-2023-01-03-14-41-06-1.mp4'
    video = torchvision.io.read_video(str(video_path))
    video_data = video[0]
    video_data = torch.transpose(
        torch.unsqueeze(video_data, dim=0), 0, -1)
    video_data = torch.squeeze(video_data, dim=-1)

    inferencer(video)
    pass
