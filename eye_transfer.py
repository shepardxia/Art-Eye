from ctypes import sizeof
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import copy

from os import path



device = torch.device("cuda")
content_layers = ['relu4_1']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
layerList = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
}


def main():
    torch.autograd.set_detect_anomaly(True)
    content_img, W, H = load_as_tensor("images\starry_night.jpg", [230, 350])
    style_img, W, H = load_as_tensor("images\seated_nude.jpg", [W, H])

    img = torch.randn(content_img.shape).type(torch.cuda.FloatTensor)
    img.requires_grad_(True)
    cnn = initialize_caffe_model('vgg16', 'avg').features.cuda()
    cnn = copy.deepcopy(cnn)
    optimizer = torch.optim.LBFGS([img])
    net, content_losses, style_losses = generate_model(cnn)

    switch_stage(content_losses, 'in')

    
    net(content_img)

    switch_stage(content_losses, 'none')
    
    switch_stage(style_losses, 'in')

    net(style_img)

    switch_stage(content_losses, 'out')
    switch_stage(style_losses, 'out')

    run = [0]

    for param in net.parameters():
        param.requires_grad = False

    def closure():
        run[0] += 1
        optimizer.zero_grad()
        net(img)
        s_loss, c_loss = 0, 0
        for sl in style_losses:
            s_loss += sl.loss.to(device)
        for cl in content_losses:
            c_loss += cl.loss.to(device)
        
        # if epoch % 10 == 0:
        #     print(f"style loss is {s_loss}")
        #     print(f"content loss is {c_loss}")
        loss = s_loss + c_loss
        loss.backward()
        return loss.detach()




    while run[0] < 1000:
        print(run[0])
        optimizer.step(closure)

    image = unload(img, [H, W])
    print(image)
    plt.figure()
    plt.imshow(image)
    plt.pause(1000)
    





# Load jpg file as tensor and normalize
def load_as_tensor(image_name, image_size=[0, 0]):
    image = Image.open(image_name).convert('RGB')
    if image_size == [0, 0]:
        W, H = image.size
    else:
        [W, H] = image_size
    long = max(W, H)
    image = image.resize([long, long], resample=Image.LANCZOS)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
    tensor = Normalize(rgb2bgr(to_tensor(image) * 255)).unsqueeze(0)
    return tensor.type(torch.cuda.FloatTensor), W, H

# Transforms torch tensor back into PIL image
def unload(img, image_size):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    output_tensor = bgr2rgb(Normalize(img.squeeze(0).cpu())) / 255
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image.resize(image_size, resample=Image.LANCZOS)


# Generate layers in VGG module for loading pretrained caffe model
def generate_layer(model_type, pooling_type, p_k_size=2, p_s_size=2, c_k_size=3, c_s_size=1):

    if model_type.lower() != 'vgg16':
        raise RuntimeError(f'Only VGG19 model is implemented')
    if pooling_type.lower() != 'max' and pooling_type.lower() != 'avg':
        raise RuntimeError(f'Only max/avg pooling are available')
    
    layers = []
    in_channel = 3
    vgg16_channels = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P']

    if pooling_type.lower() == 'max':
        pool = nn.MaxPool2d(kernel_size=p_k_size, stride=p_s_size)
    else:
        pool = nn.AvgPool2d(kernel_size=p_k_size, stride=p_s_size)

    for channel in vgg16_channels:
        if channel == 'P':
            layers += [pool]
        else:
            conv2d = nn.Conv2d(in_channel, channel, kernel_size=c_k_size, stride=c_s_size, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channel = channel
    
    return layers


# Load trainable parameters from .pth file
def initialize_caffe_model(model_type, pooling_type, p_k_size=2, p_s_size=2, c_k_size=3, c_s_size=1):
    layers = generate_layer(model_type, pooling_type, p_k_size, p_s_size, c_k_size, c_s_size)
    cnn = VGG(layers)

    cnn.load_state_dict(torch.load('models/vgg16.pth'))
    return cnn


# Generate actual model from the loaded VGG16/19 architecture
def generate_model(cnn):
    content_losses, style_losses = [], []
    conv_id, re_id = 0, 0
    content_id, style_id = 0, 0

    model = nn.Sequential()
    for layer in cnn.children():
        if content_id <= len(content_layers) - 1 or style_id <= len(style_layers) - 1:
            if isinstance(layer, nn.Conv2d):
                model.add_module(str(len(model)), layer)

                if layerList['C'][conv_id] in content_layers:
                    module = ContentLoss()
                    model.add_module(str(len(model)), module)
                    content_losses.append(module)
                    content_id += 1

                if layerList['C'][conv_id] in style_layers:
                    module = StyleLoss()
                    model.add_module(str(len(model)), module)
                    style_losses.append(module)
                    style_id += 1

                conv_id += 1

            if isinstance(layer, nn.ReLU):
                model.add_module(str(len(model)), layer)

                if layerList['R'][re_id] in content_layers:
                    module = ContentLoss()
                    model.add_module(str(len(model)), module)
                    content_losses.append(module)
                    content_id += 1

                if layerList['R'][re_id] in style_layers:
                    module = StyleLoss()
                    model.add_module(str(len(model)), module)
                    style_losses.append(module)
                    style_id += 1

                re_id += 1

            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                model.add_module(str(len(model)), layer)

    return model, content_losses, style_losses


# Used to switch content and style loss modules state
def switch_stage(layers, stage):
    for layer in layers:
        layer.stage = stage



# VGG module for loading caffe VGG16/19 pretrained model
class VGG(nn.Module):
    def __init__(self, layers):
        super(VGG, self).__init__()
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )


# Compute content MSE between input and target feature layer
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.stage = 'none'

    def forward(self, input):
        if self.stage == 'in':
            self.target = input.detach()
        if self.stage == 'out':
            loss = F.mse_loss(input, self.target)
            self.loss = ScaleGradients.apply(loss, 5) * 5

        return input


# Normalized gram matrix for extracting style from feature layers
class GramMatrix(nn.Module):
    def forward(self, input):
        B, C, H, W = input.shape
        view = input.view(B * C, H * W)
        return torch.mm(view, view.t()).div(input.nelement())


# Compute style MSE between input and target feature layer
class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.stage = 'none'
        self.gram = GramMatrix()

    def forward(self, input):
        if self.stage == 'in':
            self.target = self.gram(input).detach()
        if self.stage == 'out':
            loss = F.mse_loss(self.gram(input), self.target)
            self.loss = ScaleGradients.apply(loss, 100) * 100
        return input


# Scale gradients in the backward pass
class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.strength * self.strength, None

main()