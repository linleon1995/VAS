
# from model import MS_TCN2
from ...ASFormer.model import MyTransformer


def import_by_name(model_name):
    pass


def get_model_params(model_name):
    pass

def get_model(model_name):
    params = get_model_params(model_name)
    model_cls = import_by_name(model_name)
    model = model_cls(params)
    return model
