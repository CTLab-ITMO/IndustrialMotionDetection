from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


def get_train_transform():
    clip_len = 16
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pixel_max_value = 255.0
    
    transform = Compose([
        UniformTemporalSubsample(clip_len),
        Lambda(lambda x: x/pixel_max_value),
        NormalizeVideo(mean, std),
    ])

    return transform


def get_val_transform():
    clip_len = 16
    crop_size = 224
    side_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pixel_max_value = 255.0
    
    transform = Compose([
        UniformTemporalSubsample(clip_len),
        Lambda(lambda x: x/pixel_max_value),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size=(crop_size, crop_size))
    ])
    
    return transform
