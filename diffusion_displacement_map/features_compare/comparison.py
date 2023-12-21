import dataclasses
import itertools
from typing import Tuple

import torch
import torch.nn.functional as F

from .mapping import Mapping
from .utils import *


@dataclasses.dataclass
class FeatureComparisonWindow:
    a_full: torch.Tensor
    window_a: Tuple[int, int, int, int]
    repro_a: Mapping
    b_full: torch.Tensor
    b_indices: torch.Tensor
    repro_b: Mapping


@dataclasses.dataclass
class FeatureMappingPair:
    feat1: torch.Tensor
    feat1_repro: Mapping
    feat2: torch.Tensor
    feat2_repro: Mapping


def compute_similarity(
    inputs: FeatureMappingPair,
    window_a: Tuple[int, int, int, int],
    b_indices: torch.Tensor,
) -> torch.Tensor:
    y, dy, x, dx = window_a
    a = inputs.feat1[:, :, y : y + dy, x : x + dx]
    b = torch_gather_2d(inputs.feat2, b_indices.to(inputs.feat2.device))

    similarity = cosine_similarity_vector_1d(a.flatten(2), b.flatten(2))
    similarity += (
        torch_gather_2d(inputs.feat1_repro.biases.to(similarity.device), b_indices)
        .to(similarity.device)
        .view(similarity.shape[0], -1)
    )
    similarity += (
        inputs.feat2_repro.biases[:, :, y : y + dy, x : x + dx]
        .to(similarity.device)
        .view(1, -1)
    )
    return similarity


def improve(
    inputs: FeatureMappingPair,
    window_a: Tuple[int, int, int, int],
    b_indices: torch.Tensor,
):
    similarity = compute_similarity(
        inputs,
        window_a,
        b_indices,
    )

    best_candidates = similarity.max(dim=0)

    candidates = torch.gather(
        b_indices.flatten(2),
        dim=0,
        index=best_candidates.indices.to(b_indices).view(1, 1, -1).expand(1, 2, -1),
    ).to("cpu")

    scores = best_candidates.values.view(1, 1, -1).to("cpu")
    cha = inputs.feat1_repro.improve_window_with_indices(
        window_a, scores, candidates.flatten(2)
    )
    chb = inputs.feat2_repro.improve_scatter(candidates.flatten(2), scores, window_a)
    return cha + chb


def compare_features_coarse(
    inputs: FeatureMappingPair,
    parent_a: Mapping,
    split: int,
    radius: int,
):
    if parent_a.indices.shape[2] > inputs.feat1_repro.indices.shape[2]:
        return 0

    total = 0
    for t1, t2 in iterate_range(inputs.feat1.shape[2], split):
        assert t2 >= t1

        factor = inputs.feat1_repro.indices.shape[2] / parent_a.indices.shape[2]
        indices = F.interpolate(
            parent_a.indices.float() * factor,
            size=inputs.feat1_repro.indices.shape[2:],
            mode="nearest",
        )[:, :, t1:t2].long()
        indices += torch.empty_like(indices).random_(-radius, radius + 1)

        indices[:, 0, :, :].clamp_(min=0, max=inputs.feat2.shape[2] - 1)
        indices[:, 1, :, :].clamp_(min=0, max=inputs.feat2.shape[3] - 1)

        total += improve(
            inputs,
            (t1, t2 - t1, 0, inputs.feat1.shape[3]),
            indices,
        )
    return total


def compare_features_identity(
    inputs: FeatureMappingPair,
    split: int,
):
    # from icecream import ic
    # ic(inputs.feat1.shape, inputs.feat2.shape)
    for t1, t2 in iterate_range(inputs.feat1.shape[2], split):
        # ic(t1, t2)
        assert t2 >= t1

        indices = inputs.feat1_repro.indices[:, :, t1:t2]
        ############################################
        # update score in this window
        y, dy, x, dx = (t1, t2 - t1, 0, inputs.feat1.shape[3])
        similarity = compute_similarity(
            inputs,
            (t1, t2 - t1, 0, inputs.feat1.shape[3]),
            indices,
        )
        assert similarity.shape[0] == 1
        inputs.feat1_repro.scores[:, :, y : y + dy, x : x + dx] = similarity.view(
            1, 1, dy, dx
        )


def compare_features_inverse(
    inputs: FeatureMappingPair,
    split: int,
):
    total = 0
    for t1, t2 in iterate_range(inputs.feat1.shape[2], split):
        assert t2 >= t1

        indices = inputs.feat1_repro.indices[:, :, t1:t2]
        total += improve(
            inputs,
            (t1, t2 - t1, 0, inputs.feat1.shape[3]),
            indices,
        )
    return total


def compare_features_random(
    inputs: FeatureMappingPair, radius: int = -1, split: int = 1, times: int = 4
):
    """Generate random coordinates within a radius for each pixel, then compare the
    features to see if the current selection can be improved.
    """

    total = 0
    for t1, t2 in iterate_range(inputs.feat1.shape[2], split):
        assert t2 >= t1

        if radius == -1:
            # Generate random grid size (h, w) with indices in range of B.
            h, w = t2 - t1, inputs.feat1.shape[3]
            indices = torch.empty(
                (times, 2, h, w), dtype=torch.int64, device=inputs.feat1.device
            )
            inputs.feat1_repro.randgrid(
                indices, offset=(0, 0), range=inputs.feat2.shape[2:]
            )
        else:
            indices = inputs.feat1_repro.indices[:, :, t1:t2].clone()
            indices = indices + torch.empty_like(indices).random_(-radius, radius + 1)

            indices[:, 0, :, :].clamp_(min=0, max=inputs.feat2.shape[2] - 1)
            indices[:, 1, :, :].clamp_(min=0, max=inputs.feat2.shape[3] - 1)

        total += improve(
            inputs,
            (t1, t2 - t1, 0, inputs.feat1.shape[3]),
            indices,
        )
    return total


def compare_features_nearby(inputs: FeatureMappingPair, radius: int, split: int = 1):
    """Generate nearby coordinates for each pixel to see if offseting the neighboring
    pixel would provide better results.
    """
    assert isinstance(radius, int)
    padding = radius

    # Compare all the neighbours from the original position.
    original = inputs.feat1_repro.indices.clone()

    if any(
        padding >= dim for dim in (*inputs.feat1.shape[2:], *inputs.feat2.shape[2:])
    ):
        from icecream import ic

        print(
            f"> skipping. padding {padding} >= some shape {inputs.feat1.shape}, {inputs.feat2.shape}"
        )
        return 0

    padded_original = F.pad(
        original.to(dtype=torch.float32).expand(4, -1, -1, -1),
        pad=(padding, padding, padding, padding),
        mode="circular",
        # mode="reflect",
    ).long()

    total = 0
    for t1, t2 in iterate_range(inputs.feat1.shape[2], split):
        h, w = (t2 - t1), inputs.feat1.shape[3]

        x = original.new_tensor([0, 0, -radius, +radius]).view(4, 1, 1)
        y = original.new_tensor([-radius, +radius, 0, 0]).view(4, 1, 1)

        # Create a lookup map with offset coordinates from each coordinate.
        lookup = original.new_empty((4, 2, h, w))
        lookup[:, 0, :, :] = torch.arange(
            t1, t1 + lookup.shape[2], dtype=torch.long
        ).view((1, -1, 1))
        lookup[:, 1, :, :] = torch.arange(0, lookup.shape[3], dtype=torch.long).view(
            (1, 1, -1)
        )
        lookup[:, 0, :, :] += y + padding
        lookup[:, 1, :, :] += x + padding

        # Compute new padded buffer with the current best coordinates.
        indices = padded_original.clone()
        indices[:, 0, :, :] -= y
        indices[:, 1, :, :] -= x

        # Lookup the neighbor coordinates and clamp if the calculation overflows.
        candidates = torch_gather_2d(indices, lookup)

        # Handle `out_of_bounds` by clamping. Could be randomized?
        candidates[:, 0, :, :].clamp_(min=0, max=inputs.feat2.shape[2] - 1)
        candidates[:, 1, :, :].clamp_(min=0, max=inputs.feat2.shape[3] - 1)

        # Update the target window, and the scattered source pixels.
        total += improve(
            inputs,
            (t1, t2 - t1, 0, w),
            candidates,
        )

    return total
