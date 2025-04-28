from transforms_impl.transforms import (
    Compose,
    Lambda,
    NormalizeVideo,
    ShortSideScaleWithBoxes,
    UniformTemporalSubsample,
    RandomCropVideoWithBoxes
)


def get_transform(clip_len: int, clip_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pixel_max_value = 255.0
    return Compose([
        UniformTemporalSubsample(clip_len),
        Lambda(lambda x: x/pixel_max_value),
        NormalizeVideo(mean, std),
        ShortSideScaleWithBoxes(size=clip_size),
        RandomCropVideoWithBoxes(size=clip_size),
    ])
    