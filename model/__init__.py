import model
from model.resnet import GraphMapResNet18, GraphMapResNet50, GraphMapResNet101,GraphMapResNet152
from model.VGG16 import GraphMapVGG16
from model.vgg19 import GraphMapVGG19
__all__ = [
    "GraphMapResNet18",
    "GraphMapResNet50",
    "GraphMapResNet101",
    "GraphMapResNet152",
    "GraphMapVGG16",
    "GraphMapVGG19"
]
