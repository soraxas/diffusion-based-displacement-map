import os
import math
import itertools
import collections
from typing import List, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import tqdm

from .critics import GramMatrixCritic, PatchCritic, Critic
from .solvers import (
    SolverSGD,
    SolverLBFGS,
    MultiCriticObjective,
    SequentialCriticObjective,
    Optimiser,
)
from .io import *

Result = collections.namedtuple(
    "Result", ["images", "octave", "scale", "iteration", "loss", "rate", "retries"]
)


class Application:
    def __init__(self, device=None, precision=None):
        # Determine which device use based on what's available.
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # The floating point format is 32-bit by default, 16-bit supported.
        self.precision = getattr(torch, precision or "float32")

    def create_pbar(self, msg):
        self.log = tqdm.tqdm(total=100, desc=msg)

    def run(
        self,
        progress: tqdm.tqdm,
        seed_img: torch.Tensor,
        encoder,
        critics: Iterable[Critic],
        lr=0.1,
        quality=1,
    ):
        for objective_class, solver_class in itertools.product(
            [MultiCriticObjective, SequentialCriticObjective],
            [SolverLBFGS, SolverSGD],
            # [SolverSGD],
        ):
            if solver_class == SolverLBFGS and seed_img.dtype == torch.float16:
                continue

            try:
                critics = list(itertools.chain.from_iterable(critics))
                image = seed_img.to(self.device)
                if image.shape[1] in (2, 4):
                    alpha = image[:, -1].unsqueeze(-1)
                else:
                    alpha = None
                image = image[:, 0:3].detach().requires_grad_(True)

                obj: Critic = objective_class(encoder, critics, alpha=alpha)
                opt: Optimiser = solver_class(obj, image, lr=lr)

                for i, loss, converge, lr, retries, scores in self._iterate(
                    opt, quality=quality
                ):
                    # Constrain the image to the valid color range.
                    image.data.clamp_(0.0, 1.0)

                    # Update the progress bar with the result!
                    p = min(max(converge * 100.0, 0.0), 100.0)
                    # progress.total = int(converge)
                    progress.set_postfix(
                        loss=loss,
                        iter=i,
                        # coverage=p,
                        score=np.array([s.item() for s in scores]),
                    )
                    progress.update(int(p) - progress.n)

                    # Return back to the user...
                    yield loss, image, lr, retries
                progress.close()
                return
            except RuntimeError as e:
                if "CUDA out of memory." not in str(e):
                    raise

                import gc

                gc.collect
                torch.cuda.empty_cache()

        raise RuntimeError("CUDA out of memory.")

    def _iterate(self, opt, quality):
        threshold = math.pow(0.1, 1 + math.log(1 + quality))
        converge = 0.0
        previous, plateau = float("+inf"), 0

        for i in itertools.count():
            # Perform one step of the optimization.
            loss, scores = opt.step()

            # Progress metric loosely based on convergence and time.
            current = (previous - loss) / loss
            c = math.exp(-max(current - threshold, 0.0) / (math.log(2 + i) * 0.05))
            converge = converge * 0.8 + 0.2 * c

            # Return this iteration to the caller...
            yield i, loss, converge, opt.lr, opt.retries, scores

            # See if we can terminate the optimization early.
            if i > 3 and current <= threshold:
                plateau += 1
                if plateau > 2:
                    break
            else:
                plateau = 0

            previous = min(loss, previous)

    def process_octave(self, result_img, encoder, critics, octave, scale, quality):
        # Each octave we start a new optimization process.
        result_img = result_img.to(dtype=self.precision)

        # The first iteration contains the rescaled image with noise.
        yield Result(result_img, octave, scale, 0, float("+inf"), 1.0, 0)

        for iteration, (loss, result_img, lr, retries) in enumerate(
            self.run(
                self.log,
                result_img,
                encoder=encoder,
                critics=critics,
                lr=1.0,
                quality=quality,
            ),
            start=1,
        ):
            yield Result(result_img, octave, scale, iteration, loss, lr, retries)

        # The last iteration is repeated to indicate completion.
        yield Result(result_img, octave, scale, -iteration, loss, lr, retries)
