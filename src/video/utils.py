# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import torch


def thwc_to_cthw(data: torch.Tensor) -> torch.Tensor:
    """
    Permute tensor from (time, height, weight, channel) to
    (channel, height, width, time).
    """
    return data.permute(3, 0, 1, 2)


def secs_to_pts(
    time_in_seconds: float,
    time_base: float,
    start_pts: int,
    round_mode: str = "floor",
) -> int:
    """
    Converts a time (in seconds) to the given time base and start_pts offset
    presentation time. Round_mode specifies the mode of rounding when converting time.

    Returns:
        pts (int): The time in the given time base.
    """
    if time_in_seconds == math.inf:
        return math.inf

    assert round_mode in ["floor", "ceil"], f"round_mode={round_mode} is not supported!"

    if round_mode == "floor":
        return math.floor(time_in_seconds / time_base) + start_pts
    else:
        return math.ceil(time_in_seconds / time_base) + start_pts


def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base
