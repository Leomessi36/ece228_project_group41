from itertools import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

Mean = [0.418, 0.423, 0.489]
STD = [0.223, 0.228, 0.229]


def gram_matrix(y):
    (b, c, h, w) = y.size()
    fea = y.view(b, c, w * h)
    feature = fea.transpose(1, 2)
    gram = features.bmm(feature) / (c * h * w)
    return gram
class Visualizer():
    
    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''


    def plot(self, name, y):

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

def get_style_data(path):

    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=Mean, std=STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):

    mean = batch.data.new(Mean).view(1, -1, 1, 1)
    std = batch.data.new(STD).view(1, -1, 1, 1)
    mean = (mean.expand_as(batch.data))
    std = (std.expand_as(batch.data))
    return (batch / 255.0 - mean) / std