import torch
from os import path
from torch.utils.model_zoo import load_url
from collections import OrderedDict


# VGG19 if GPU memory is big enough
# caffe_model = load_url("https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth")
# map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
# caffe_model = OrderedDict([(map[k] if k in map else k,v) for k,v in caffe_model.items()])
# torch.save(caffe_model, path.join("models", "vgg19.pth"))


print("Downloading the VGG-16 model")
sd = load_url("https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("models", "vgg16.pth"))