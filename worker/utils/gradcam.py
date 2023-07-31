import cv2
import torch
from torch import nn


class Hook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


class HookBwd:
    def __init__(self, m):
        self.hook = m.register_full_backward_hook(self.hook_func)

    def hook_func(self, m, gi, go):
        self.stored = go[0].detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


def get_gradcam(model, x, backward_class=-1, location=-1):
    conv_layers = []
    idx = backward_class
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)

    with HookBwd(conv_layers[location]) as hookg:
        with Hook(conv_layers[location]) as hook:
            output = model.eval()(x)
            if idx == -1:
                idx = torch.argmax(output)
            act = hook.stored
        output[0, idx].backward()
        grad = hookg.stored

    w = grad[0].mean(dim=[1, 2], keepdim=True)
    heatmap = (w * act[0]).sum(0)

    return heatmap, output


def apply_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.normalize(
        heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return blended
