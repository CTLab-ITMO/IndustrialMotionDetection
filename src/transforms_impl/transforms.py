# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
import src.transforms_impl.functional
import torchvision.transforms
from torchvision.transforms import _functional_video as F
from typing import Callable, Dict, List, Optional, Tuple


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        for t in self.transforms:
            target = t(target)
        return target


class Lambda:
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(lambd).__name__)}")
        self.lambd = lambd

    def __call__(
        self,
        target: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        target['video'] = self.lambd(target['video'])
        return target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(
        self,
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        target['video'] = F.normalize(target['video'], self.mean, self.std, self.inplace)
        return target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(
        self,
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        target[self._key] = self._transform(target[self._key])
        return target


class RemoveKey(torch.nn.Module):
    """
    Removes the given key from the input dict. Useful for removing modalities from a
    video clip that aren't needed.
    """

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def __call__(
        self,  
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        if self._key in target:
            del target[self._key]
        return target


class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(
        self, 
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        target['video'] = transforms_impl.functional.uniform_temporal_subsample(
            target['video'], self._num_samples, self._temporal_dim)
        return target


class UniformTemporalSubsampleRepeated(torch.nn.Module):
    """
    ``nn.Module`` wrapper for
    ``transforms_impl.functional.uniform_temporal_subsample_repeated``.
    """

    def __init__(self, frame_ratios: Tuple[int], temporal_dim: int = -3):
        super().__init__()
        self._frame_ratios = frame_ratios
        self._temporal_dim = temporal_dim

    def forward(
        self, 
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        target['video'] = transforms_impl.functional.uniform_temporal_subsample_repeated(
            target['video'], self._frame_ratios, self._temporal_dim
        )
        return target


class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(
        self,
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        target['video'] = transforms_impl.functional.short_side_scale(
            target['video'], self._size, self._interpolation, self._backend
        )
        return target
        

class ShortSideScaleWithBoxes(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(
        self,
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        target['video'], target['bbox'] = transforms_impl.functional.short_side_scale_with_boxes(
            target['video'], target['bbox'], self._size, self._interpolation, self._backend
        )
        return target


class RandomCropVideoWithBoxes(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(
        self, 
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        x = target['video']
        boxes = target['bbox']
        
        # Calculate original box areas
        original_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        cropped_x, cropped_boxes = transforms_impl.functional.random_crop_with_boxes(
            x, self._size, boxes
        )
        
        # Calculate new box areas
        new_areas = (cropped_boxes[:, 2] - cropped_boxes[:, 0]) * (cropped_boxes[:, 3] - cropped_boxes[:, 1])
        
        # Filter boxes where area is reduced by <= 75%
        # Also handle cases where original_area is 0 (though this shouldn't happen with valid boxes)
        valid_indices = (new_areas / (original_areas + 1e-6)) > 0.75
        
        target['video'] = cropped_x
        target['target'] = target['target'][valid_indices]
        target['bbox'] = cropped_boxes[valid_indices]
        
        return target


class BoxDependentCropVideoWithBoxes(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(
        self, 
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert 'rare_class_idx' in target, 'rare_class_idx not in dict'
        
        x = target['video']
        boxes = target['bbox']
        
        y_offset = 0
        
        box_offset_min = int(torch.min(boxes[target['rare_class_idx']][[0, 2]]).item())
        box_offset_max = int(torch.max(boxes[target['rare_class_idx']][[0, 2]]).item())
        
        x_offset = 0
        width = x.shape[3]
        if width > self._size:
            left = max(0, box_offset_max - self._size)
            right = min(width - self._size, box_offset_min)
            
            if left <= right:
                x_offset = int(np.random.randint(left, right + 1))  # +1 because high is exclusive
            else:
                # Handle case where bbox cannot fit (e.g., resize self._size or skip)
                raise ValueError(
                    f"Bounding box cannot fit in crop: "
                    f"bbox width {box_offset_max - box_offset_min} > crop size {self._size}"
                )
        
        # Calculate original box areas
        original_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        cropped_x, cropped_boxes = transforms_impl.functional.crop_with_boxes(
            x, self._size, boxes, x_offset=x_offset, y_offset=y_offset
        )
        
        # Calculate new box areas
        new_areas = (cropped_boxes[:, 2] - cropped_boxes[:, 0]) * (cropped_boxes[:, 3] - cropped_boxes[:, 1])
        
        # Filter boxes where area is reduced by <= 75%
        # Also handle cases where original_area is 0 (though this shouldn't happen with valid boxes)
        valid_indices = (new_areas / (original_areas + 1e-6)) > 0.75
        
        target['video'] = cropped_x
        target['target'] = target['target'][valid_indices]
        target['bbox'] = cropped_boxes[valid_indices]
        
        return target


class RandomShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: str = "bilinear",
        backend: str = "pytorch",
    ):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return transforms_impl.functional.short_side_scale(
            x, size, self._interpolation, self._backend
        )


class UniformCropVideo(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.uniform_crop``.
    """

    def __init__(
        self, size: int, video_key: str = "video", aug_index_key: str = "aug_index"
    ):
        super().__init__()
        self._size = size
        self._video_key = video_key
        self._aug_index_key = aug_index_key

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (Dict[str, torch.Tensor]): video clip dict.
        """
        x[self._video_key] = transforms_impl.functional.uniform_crop(
            x[self._video_key], self._size, x[self._aug_index_key]
        )
        return x


class Normalize(torchvision.transforms.Normalize):
    """
    Normalize the (CTHW) video clip by mean subtraction and division by standard deviation

    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        vid = x.permute(1, 0, 2, 3)  # C T H W to T C H W
        vid = super().forward(vid)
        vid = vid.permute(1, 0, 2, 3)  # T C H W to C T H W
        return vid


class ConvertFloatToUint8(torch.nn.Module):
    """
    Converts a video from dtype float32 to dtype uint8.
    """

    def __init__(self):
        super().__init__()
        self.convert_func = torchvision.transforms.ConvertImageDtype(torch.uint8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert (
            x.dtype == torch.float or x.dtype == torch.half
        ), "image must have dtype torch.uint8"
        return self.convert_func(x)


class ConvertUint8ToFloat(torch.nn.Module):
    """
    Converts a video from dtype uint8 to dtype float32.
    """

    def __init__(self):
        super().__init__()
        self.convert_func = torchvision.transforms.ConvertImageDtype(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        assert x.dtype == torch.uint8, "image must have dtype torch.uint8"
        return self.convert_func(x)


class MoveChannelRear(torch.nn.Module):
    """
    A Scriptable version to perform C X Y Z -> X Y Z C.
    """

    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        x = x.permute([1, 2, 3, 0])
        return x


class MoveChannelFront(torch.nn.Module):
    """
    A Scriptable version to perform X Y Z C -> C X Y Z.
    """

    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        x = x.permute([3, 0, 1, 2])
        return x


class RandomResizedCrop(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``transforms_impl.functional.random_resized_crop``.
    """

    def __init__(
        self,
        target_height: int,
        target_width: int,
        scale: Tuple[float, float],
        aspect_ratio: Tuple[float, float],
        shift: bool = False,
        log_uniform_ratio: bool = True,
        interpolation: str = "bilinear",
        num_tries: int = 10,
    ) -> None:
        super().__init__()
        self._target_height = target_height
        self._target_width = target_width
        self._scale = scale
        self._aspect_ratio = aspect_ratio
        self._shift = shift
        self._log_uniform_ratio = log_uniform_ratio
        self._interpolation = interpolation
        self._num_tries = num_tries

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input video tensor with shape (C, T, H, W).
        """
        return transforms_impl.functional.random_resized_crop(
            x,
            self._target_height,
            self._target_width,
            self._scale,
            self._aspect_ratio,
            self._shift,
            self._log_uniform_ratio,
            self._interpolation,
            self._num_tries,
        )


class Permute(torch.nn.Module):
    """
    Permutes the dimensions of a video.
    """

    def __init__(self, dims: Tuple[int]):
        """
        Args:
            dims (Tuple[int]): The desired ordering of dimensions.
        """
        assert (
            (d in dims) for d in range(len(dims))
        ), "dims must contain every dimension (0, 1, 2, ...)"

        super().__init__()
        self._dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        return x.permute(*self._dims)


class OpSampler(torch.nn.Module):
    """
    Given a list of transforms with weights, OpSampler applies weighted sampling to
    select n transforms, which are then applied sequentially to the input.
    """

    def __init__(
        self,
        transforms_list: List[Callable],
        transforms_prob: Optional[List[float]] = None,
        num_sample_op: int = 1,
        randomly_sample_depth: bool = False,
        replacement: bool = False,
    ):
        """
        Args:
            transforms_list (List[Callable]): A list of tuples of all available transforms
                to sample from.
            transforms_prob (Optional[List[float]]): The probabilities associated with
                each transform in transforms_list. If not provided, the sampler assumes a
                uniform distribution over all transforms. They do not need to sum up to one
                but weights need to be positive.
            num_sample_op (int): Number of transforms to sample and apply to input.
            randomly_sample_depth (bool): If randomly_sample_depth is True, then uniformly
                sample the number of transforms to apply, between 1 and num_sample_op.
            replacement (bool): If replacement is True, transforms are drawn with replacement.
        """
        super().__init__()
        assert len(transforms_list) > 0, "Argument transforms_list cannot be empty."
        assert num_sample_op > 0, "Need to sample at least one transform."
        assert (
            num_sample_op <= len(transforms_list)
        ), "Argument num_sample_op cannot be greater than number of available transforms."

        if transforms_prob is not None:
            assert (
                len(transforms_list) == len(transforms_prob)
            ), "Argument transforms_prob needs to have the same length as transforms_list."

            assert (
                min(transforms_prob) > 0
            ), "Argument transforms_prob needs to be greater than 0."

        self.transforms_list = transforms_list
        self.transforms_prob = torch.FloatTensor(
            transforms_prob
            if transforms_prob is not None
            else [1] * len(transforms_list)
        )
        self.num_sample_op = num_sample_op
        self.randomly_sample_depth = randomly_sample_depth
        self.replacement = replacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        depth = (
            torch.randint(1, self.num_sample_op + 1, (1,)).item()
            if self.randomly_sample_depth
            else self.num_sample_op
        )
        index_list = torch.multinomial(
            self.transforms_prob, depth, replacement=self.replacement
        )

        for index in index_list:
            x = self.transforms_list[index](x)

        return x