# Modified from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/structures/bounding_box.py
import torch
import numpy as np

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def delete_field(self, field):
        return self.extra_fields.pop(field, None)

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor) and (hasattr(v, "resize")):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and hasattr(v, "resize"):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and (hasattr(v, 'transpose')):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def extend(self, scale):
        """
        Return a extended bounding box copy of this bounding box.
        All other fields should be keep unchanged.
        :param scale: By what extent the bounding boxes will be extended.
        :return: A extended copy.
        """
        if len(scale)<2:
            x_scale = y_scale = scale[0]
        else:
            x_scale = scale[0]
            y_scale = scale[1]
        TO_REMOVE = 1
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        boxw, boxh = xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE
        padw, padh = float(x_scale) * boxw / 2, float(y_scale) * boxh / 2
        extended_xmin = xmin - padw
        extended_ymin = ymin - padh
        extended_xmax = xmax + padw
        extended_ymax = ymax + padh
        extended_box = torch.cat(
            (extended_xmin, extended_ymin, extended_xmax, extended_ymax), dim=-1
        )
        bbox = BoxList(extended_box, self.size, mode='xyxy')
        bbox.clip_to_image()
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def random_aug(self, jitter_x_out, jitter_x_in, jitter_y_out, jitter_y_in):
        TO_REMOVE = 1
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        device = xmin.device

        def torch_uniform(rows, a=0.0, b=1.0):
            return torch.rand(rows, 1, dtype=torch.float32, device=device) * (b - a) + a

        boxw, boxh = xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE

        num_boxes = len(self)

        jitter_xmin = xmin + boxw * torch_uniform(num_boxes, -jitter_x_out, jitter_x_in)
        jitter_ymin = ymin + boxh * torch_uniform(num_boxes, -jitter_y_out, jitter_y_in)
        jitter_xmax = xmax + boxw * torch_uniform(num_boxes, -jitter_x_in, jitter_x_out)
        jitter_ymax = ymax + boxh * torch_uniform(num_boxes, -jitter_y_in, jitter_y_out)
        jitter_xmin.clamp_(min=0, max=self.size[0] - TO_REMOVE - 1)
        jitter_ymin.clamp_(min=0, max=self.size[1] - TO_REMOVE - 1)
        jitter_xmax = torch.max(torch.clamp(jitter_xmax, max=self.size[0] - TO_REMOVE), jitter_xmin + 1)
        jitter_ymax = torch.max(torch.clamp(jitter_ymax, max=self.size[1] - TO_REMOVE), jitter_ymin + 1)

        aug_box = torch.cat(
            (jitter_xmin, jitter_ymin, jitter_xmax, jitter_ymax), dim=-1
        )
        bbox = BoxList(aug_box, self.size, mode='xyxy')
        bbox.clip_to_image(remove_empty=False)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def top_k(self, k):
        # categories, bbox, scores
        if "scores" in self.extra_fields:
            scores = self.extra_fields["scores"]
            length = len(scores)
            start = max(length - k, 0)
            idx = torch.argsort(scores)[start:]
            bbox = BoxList(self.bbox[[idx]], self.size, self.mode)
            for k, v in self.extra_fields.items():
                if isinstance(v, torch.Tensor):
                    bbox.add_field(k, v[idx])
                else:
                    bbox.add_field(k, v)
        else:
            bbox = BoxList(self.bbox[:k], self.size, self.mode)
            for k, v in self.extra_fields.items():
                if isinstance(v, torch.Tensor):
                    bbox.add_field(k, v[:k])
                else:
                    bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item].reshape(-1, 4), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor):
                bbox.add_field(k, v[item])
            else:
                bbox.add_field(k, v)
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
    
    def keep_boxes(self, keep_indices):
        """
        Keep only the boxes at the specified indices and remove all others.
        
        Args:
            keep_indices (Tensor, list, or array): Indices of boxes to keep
        """
        if isinstance(keep_indices, (list, np.ndarray)):
            keep_indices = torch.as_tensor(keep_indices, device=self.bbox.device)
        
        bbox = self.bbox[keep_indices]
        new_boxlist = BoxList(bbox, self.size, self.mode)
        
        # Copy over all fields
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor):
                new_boxlist.add_field(k, v[keep_indices])
            else:
                new_boxlist.add_field(k, v)
        
        return new_boxlist

    def enlarge(self, percent: float) -> 'BoxList':
        """
        Symmetrically enlarge all bounding boxes by a given percentage while keeping their centers fixed.
        The expansion is applied equally in all directions (left, right, top, bottom).
        
        Args:
            percent (float): Percentage to expand the boxes (e.g., 5.0 for 5% expansion)
            
        Returns:
            BoxList: A new BoxList with enlarged boxes
        
        Example:
            >>> boxes = BoxList([[10, 10, 30, 30]], (100, 100))
            >>> enlarged = boxes.enlarge(20.0)  # 20% expansion
            >>> enlarged.bbox
            tensor([[ 8.,  8., 32., 32.]])  # Original center (20,20) remains unchanged
        """
        if percent < 0:
            raise ValueError("Percent must be non-negative")

        # Convert to xyxy format for processing
        boxes_xyxy = self.convert("xyxy").bbox
        img_width, img_height = self.size
        
        # Calculate current dimensions
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        
        # Calculate symmetric expansion amounts
        expand_w = widths * (percent / 200)  # /200 because we expand half in each direction
        expand_h = heights * (percent / 200)
        
        # Calculate new coordinates with clamping
        new_coords = torch.stack([
            torch.clamp(boxes_xyxy[:, 0] - expand_w, min=0),          # x1
            torch.clamp(boxes_xyxy[:, 1] - expand_h, min=0),          # y1
            torch.clamp(boxes_xyxy[:, 2] + expand_w, max=img_width),  # x2
            torch.clamp(boxes_xyxy[:, 3] + expand_h, max=img_height)  # y2
        ], dim=1)
        
        # Create new BoxList
        result = BoxList(new_coords, self.size, mode="xyxy")
        
        # Preserve all extra fields
        for field, value in self.extra_fields.items():
            result.add_field(field, value)
        
        return result.convert(self.mode)

if __name__ == "__main__":
    bbox = BoxList([[5, 5, 9, 9], [2, 2, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
    
    enlarge_bbox = bbox.enlarge(10.0)
    print(enlarge_bbox)
    print(enlarge_bbox.bbox)
