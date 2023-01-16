import torch.nn
from collections import OrderedDict


def get_model_weights_and_biases(model):
    weights = {}
    biases = {}
    i = 0
    for key, value in model.state_dict().items():
        if i % 2 == 0:
            weights[str(key.split('.weight')[0])] = value.numpy()
        else:
            biases[str(key.split('.bias')[0])] = value.numpy()
        i += 1

    return weights, biases


def get_loss(original_predictions, model_predictions, loss):
    loss_value = loss(torch.Tensor(original_predictions), torch.Tensor(model_predictions)).mean()
    return loss_value


def get_model_layer_names(model):
    layer_names = OrderedDict()
    for name, layer in model.named_modules():
        layer_names[name] = layer
    return layer_names


def get_sample(observations, labels, sample_size):
    return observations[:sample_size], labels[:sample_size]
