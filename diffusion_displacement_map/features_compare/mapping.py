from typing import Tuple

import torch
import torch.nn.functional as F

from .utils import torch_gather_2d, torch_scatter_2d


class Mapping:
    def __init__(self, size, device="cpu"):
        b, _, h, w = size
        self.device = torch.device(device)
        self.indices = torch.empty((b, 2, h, w), dtype=torch.int64, device=device)
        self.scores = torch.full((b, 1, h, w), float("-inf"), device=device)
        self.biases = None
        self.target_size = None

    def clone(self):
        b, _, h, w = self.indices.shape
        clone = Mapping((b, -1, h, w), self.device)
        clone.indices[:] = self.indices
        clone.scores[:] = self.scores
        clone.biases = self.biases.copy()
        return clone

    def setup_biases(self, target_size):
        b, _, h, w = target_size
        self.biases = torch.full((b, 1, h, w), 0.0, device=self.device)

    def rescale(self, target_size):
        factor = torch.tensor(target_size, dtype=torch.float) / torch.tensor(
            self.target_size[2:], dtype=torch.float
        )
        self.indices = (
            self.indices.float().mul(factor.to(self.device).view(1, 2, 1, 1)).long()
        )
        self.indices[:, 0].clamp_(0, target_size[0] - 1)
        self.indices[:, 1].clamp_(0, target_size[1] - 1)
        self.target_size = self.target_size[:2] + target_size

        self.setup_biases(self.scores.shape[:2] + target_size)

    def resize(self, size):
        self.indices = F.interpolate(
            self.indices.float(), size=size, mode="nearest"
        ).long()
        self.scores = F.interpolate(self.scores, size=size, mode="nearest")

    def improve(self, candidate_scores, candidate_indices):
        candidate_indices = candidate_indices.view(self.indices.shape)
        candidate_scores = candidate_scores.view(self.scores.shape) + torch_gather_2d(
            self.biases, candidate_indices
        )

        cond = candidate_scores > self.scores
        self.indices[:] = torch.where(cond, candidate_indices, self.indices)
        self.scores[:] = torch.where(cond, candidate_scores, self.scores)

    def improve_window(self, this_window, other_window, candidates):
        assert candidates.indices.shape[0] == 1

        sy, dy, sx, dx = other_window
        grid = torch.empty((1, 2, dy, dx), dtype=torch.int64, device=self.device)
        self.meshgrid(grid, offset=(sy, sx), range=(dy, dx))

        chanidx = torch.arange(0, 2, dtype=torch.long, device=self.device).view(-1, 1)
        chanidx = chanidx * (dx * dy)
        indices_2d = torch.index_select(
            grid.flatten(),
            dim=0,
            index=(chanidx + candidates.indices.to(self.device).view(1, -1)).flatten(),
        )

        return self.improve_window_with_indices(
            this_window, candidates.values.to(self.device), indices_2d.view(1, 2, -1)
        )

    def improve_window_with_indices(
        self,
        this_window: Tuple[int, int, int, int],
        scores: torch.Tensor,
        indices: torch.Tensor,
    ):
        start_y, size_y, start_x, size_x = this_window
        assert indices.ndim == 3 and indices.shape[2] == size_y * size_x

        candidate_scores = scores.view(1, 1, size_y, size_x)
        candidate_indices = indices.view(1, 2, size_y, size_x)

        slice_y, slice_x = (
            slice(start_y, start_y + size_y),
            slice(start_x, start_x + size_x),
        )

        cond = candidate_scores > self.scores[:, :, slice_y, slice_x]
        cond_expanded = cond.expand(-1, 2, -1, -1)

        self.indices[:, :, slice_y, slice_x][cond_expanded] = candidate_indices[
            cond_expanded
        ]
        self.scores[:, :, slice_y, slice_x][cond] = candidate_scores[cond]

        return (cond != 0).sum().item()

    def improve_scatter(self, this_indices, scores, other_window):
        sy, dy, sx, dx = other_window
        grid = torch.empty((1, 2, dy, dx), dtype=torch.int64, device=self.device)
        self.meshgrid(grid, offset=(sy, sx), range=(dy, dx))

        this_scores = (
            torch_gather_2d(self.scores, this_indices.view(1, 2, dy, dx))
            + self.biases[:, :, sy : sy + dy, sx : sx + dx]
        )
        cond = scores.flatten(2) > this_scores.flatten(2)

        better_indices = this_indices.flatten(2)[cond.expand(1, 2, -1)].view(1, 2, -1)
        if better_indices.shape[2] == 0:
            return 0

        better_scores = scores.flatten(2)[cond].view(1, 1, -1)
        window_indices = grid.flatten(2)[cond.expand(1, 2, -1)].view(1, 2, -1)

        torch_scatter_2d(self.scores, better_indices, better_scores)
        torch_scatter_2d(self.indices, better_indices, window_indices)

        return better_indices.shape[2]

    def from_random(self, target_size):
        assert target_size[0] == 1, "Only 1 feature map supported."
        self.target_size = target_size
        self.randgrid(self.indices, offset=(0, 0), range=target_size[2:])
        self.setup_biases(target_size)
        return self

    def randgrid(self, output, offset, range):
        torch.randint(
            low=offset[0],
            high=offset[0] + range[0],
            size=output[:, 0, :, :].shape,
            out=output[:, 0, :, :],
        )
        torch.randint(
            low=offset[1],
            high=offset[1] + range[1],
            size=output[:, 1, :, :].shape,
            out=output[:, 1, :, :],
        )

    def meshgrid(self, output, offset, range):
        b, _, h, w = output.shape
        output[:, 0, :, :] = (
            torch.arange(h, dtype=torch.float32)
            .mul(range[0] / h)
            .add(offset[0])
            .view((1, h, 1))
            .expand((b, h, 1))
            .long()
        )
        output[:, 1, :, :] = (
            torch.arange(w, dtype=torch.float32)
            .mul(range[1] / w)
            .add(offset[1])
            .view((1, 1, w))
            .expand((b, 1, w))
            .long()
        )

    def from_linear(self, target_size):
        assert target_size[0] == 1, "Only 1 feature map supported."
        self.target_size = target_size
        self.meshgrid(self.indices, offset=(0, 0), range=target_size[2:])
        self.setup_biases(target_size)
        return self
