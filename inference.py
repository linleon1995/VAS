from typing import Callable

import torch


def get_vid_extractor(model_name: str, pretrained: bool = True,
                      device: str = 'cpu', training: bool = False):
    """AI is creating summary for get_vid_extractor

    Args:
        model_name (str): [description]
        pretrained (bool, optional): [description]. Defaults to True.
        device (str, optional): [description]. Defaults to 'cpu'.
        training (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # Choose the model
    model = torch.hub.load('facebookresearch/pytorchvideo',
                           model_name, pretrained=pretrained)

    # Set to GPU or CPU
    model = model.train() if training else model.eval()
    model = model.to(device)
    return model


def get_vas_model():
    pass


def get_post_process():
    pass


class PytorchVideoInference():
    def __init__(self, model_name: str, pretrained: bool = True,
                 device: str = 'cpu', training: bool = False):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.training = training
        self.model = self.get_vid_extractor(
            self.model_name, self.pretrained, self.device, self.training)
        self.transform = None

    def __call__(self, video):
        pass


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


if __name__ == '__main__':
    # inferencer = VAS_inference(
    #     source='feature',
    #     get_vid_feat_extractor=get_vid_extractor,
    #     get_vas_model=get_vas_model,
    # )
    kwargs = {
        'model_name': 'x3d_s',
        'pretrained': True,
        'device': 'cpu',
        'training': False,
    }
    model = get_vid_extractor(**kwargs)
    print(model)
