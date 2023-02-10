import os
import math
import itertools
import collections
from typing import List, Iterable, Set

import numpy as np
import torch
import torch.nn.functional as F

import tqdm
from creativeai.image.encoders.base import Encoder

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
    def __init__(
        self, encoder: Encoder, layers: Set[str], mode, device=None, precision="float32"
    ):
        # Determine which device use based on what's available.
        self.layers = layers
        self.mode = mode
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder = encoder.to(self.device)

        # The floating point format is 32-bit by default, 16-bit supported.
        self.precision = getattr(torch, precision)

    def create_pbar(self, msg):
        self.log = tqdm.tqdm(total=100, desc=msg)

    def run(
        self,
        progress: tqdm.tqdm,
        seed_img: torch.Tensor,
        critics: Iterable[Critic],
        lr=0.1,
        quality=1,
    ):
        for objective_class, solver_class in itertools.product(
            [
                MultiCriticObjective,
                SequentialCriticObjective,
            ],
            [SolverLBFGS, SolverSGD],
            # [SolverSGD],
        ):
            if solver_class == SolverLBFGS and seed_img.dtype == torch.float16:
                continue

            try:
                critics = list(itertools.chain.from_iterable(critics))
                image = seed_img.to(self.device)
                alpha = None
                # contains alpha channel
                if image.shape[1] in (2, 4):
                    alpha = image[:, -1].unsqueeze(-1)

                image = image[:, 0:3].detach().requires_grad_(True)
                #
                # _ori_image = image.detach().clone()

                obj: Critic = objective_class(self.encoder, critics, alpha=alpha)
                opt: Optimiser = solver_class(obj, image, lr=lr)

                for i, loss, converge, lr, retries, scores in self._iterate(
                    opt, quality=quality
                ):

                    # # only update the requested region, if we are re-painting part of the image.
                    # _x = int(0.25 * image.shape[2])
                    # _y = int(0.25 * image.shape[3])

                    # _ori_image = load_tensor_from_file("/home/tin/Downloads/arseniy-senchi-smirnov-rocktexture-portfolio.jpg", mode='L')

                    # _ori_image = F.interpolate(_ori_image, size=image.shape[2:], mode="bicubic", align_corners=False).clamp(
                    #     0.0, 1.0
                    # )
                    # _ori_image = _ori_image.to(image)

                    # _ori_image = torch.randn_like(_ori_image) * .01

                    # mask = torch.zeros_like(_ori_image, dtype=bool)
                    # mask[:] = False
                    # mask[:, :, _x:3*_x, :] = True
                    # print(_x, 3*_x, image.shape)
                    # image[~mask] += _ori_image[~mask]

                    # Constrain the image to the valid color range.
                    image.data.clamp_(0.0, 1.0)

                    # print(image.shape)
                    # exit()

                    # Update the progress bar with the result!
                    p = min(max(converge * 100.0, 0.0), 100.0)
                    # progress.total = int(converge)
                    progress.set_postfix(
                        loss=loss,
                        iter=i,
                        # coverage=p,
                        score=np.array(
                            [
                                s.item() if isinstance(s, torch.Tensor) else s
                                for s in scores
                            ]
                        ),
                    )
                    progress.update(int(p) - progress.n)

                    # Return back to the user...
                    yield loss, image, lr, retries
                progress.close()
                return
            except RuntimeError as e:
                raise
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

    def process_octave(self, result_img, critics, octave, scale, quality):
        # Each octave we start a new optimization process.
        result_img = result_img  # .to(dtype=self.precision)

        # The first iteration contains the rescaled image with noise.
        yield Result(result_img, octave, scale, 0, float("+inf"), 1.0, 0)

        for iteration, (loss, result_img, lr, retries) in enumerate(
            self.run(
                self.log,
                result_img,
                critics=critics,
                lr=1.0,
                quality=quality,
            ),
            start=1,
        ):
            yield Result(result_img, octave, scale, iteration, loss, lr, retries)

        # The last iteration is repeated to indicate completion.
        yield Result(result_img, octave, scale, -iteration, loss, lr, retries)
