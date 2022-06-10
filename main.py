import torch as t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm
import os
import ipdb

class Config(object):
    
    use_gpu = True
    model_path = None 
    img_size = 256 
    batch_size = 8  
    data_root = 'data/' 
    num_workers = 4 
    
    lr = 1e-3
    epoches = 2 
    content_weight = 1e5   
    style_weight = 1e10 

    style_path= 'style.jpg' 
    env = 'neural-style' 
    plot_every = 10 

    debug_file = '/tmp/debugnn' 

    
    content_path = 'input.png' 
    result_path = 'output.png' 


def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils.Visualizer(opt.env)

    
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.img_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    transformer.to(device)

    
    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    
    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)


    
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

            
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            
            style_loss = 0.
            sw=[1,1,1,1]
            i=0
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss +=sw[i]* F.mse_loss(gram_y, gm_s.expand_as(gram_y))
                i=i+1
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
            
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)

@t.no_grad()
def stylize(**kwargs):
   
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    
    
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()
