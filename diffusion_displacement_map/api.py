import os
import math

import soraxas_toolbox
import torch
import torch.nn.functional as F
from creativeai.image.encoders import models
from creativeai.image.encoders.base import Encoder
from icecream import ic

from .commands import Command
from .critics import GramMatrixCritic, PatchCritic, HistogramCritic
from .app import Application, Result
from .io import *


@torch.no_grad()
def process_iterations(
    cmd: Command,
    size: tuple = None,
    octaves: int = None,
    variations: int = 1,
    quality: float = 2,
    model: str = "VGG11",
    layers: str = None,
    mode: str = None,
    device: str = None,
    precision: str = None,
):
    """Synthesize a new texture and return a PyTorch tensor at each iteration."""

    # Configure the default options dynamically, unless overriden.
    factor = math.sqrt((size[0] * size[1]) / (32**2))
    octaves = octaves or getattr(cmd, "octaves", int(math.log(factor, 2) + 1.0))

    ic(model, factor, octaves)

    # Setup the application to use throughout the synthesis.

    # Encoder used by all the critics at every octave.
    encoder: Encoder = getattr(models, model)(
        pretrained=True, pool_type=torch.nn.AvgPool2d
    )
    # encoder = encoder.to(device=app.device, dtype=app.precision)

    app = Application(encoder, layers, mode, device, precision)

    # Coarse-to-fine rendering, number of octaves specified by user.
    seed = None

    progressive_scales = [2**s for s in range(octaves - 1, -1, -1)]
    ic(progressive_scales)
    for octave, scale in enumerate(progressive_scales):

        app.create_pbar(f"OCTAVE {octave + 1}/{len(progressive_scales)}")

        # app.log.debug("<- scale:", f"1/{scale}")

        # app.progress = app.log.create_progress_bar(100)

        result_size = (variations, 3, size[1] // scale, size[0] // scale)
        # app.log.debug("<- seed:", tuple(result_size[2:]))

        ic(result_size)
        # result_size = (1,3,160,160)
        seed = cmd.prepare_seed_tensor(app, result_size, previous=seed)

        ic(result_size, seed.shape)

        for dtype in [
            torch.float32,
            torch.float16,
        ]:
            # if app.precision != dtype:
            #     app.precision = dtype
            #     app.encoder = app.encoder.to(dtype=dtype)
            #     if seed is not None:
            #         seed = seed.to(app.device)

            try:
                critics = cmd.prepare_critics(app, scale)
                print("=== seed ===")

                soraxas_toolbox.image.display(seed)

                throttled_display = soraxas_toolbox.ThrottledExecution(
                    throttled_threshold=5
                )
                for result in app.process_octave(
                    seed,
                    critics,
                    octave,
                    scale,
                    quality=quality,
                ):
                    if throttled_display:
                        soraxas_toolbox.image.display(result.images, pbar=app.log)

                    yield result
                # import time
                # time.sleep(100)
                seed = result.images
                del result
                break

            except RuntimeError as e:
                raise
                if "CUDA out of memory." not in str(e):
                    raise

                import gc

                gc.collect
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()


@torch.no_grad()
def process_single_command(cmd, output: str = None, **config: dict):
    for result in process_iterations(cmd, **config):
        if result.iteration >= 0:
            continue

        result = Result(
            result.images,
            result.octave,
            result.scale,
            -result.iteration,
            result.loss,
            result.rate,
            result.retries,
        )

        result = cmd.finalize_octave(result)
        images = save_tensor_to_images(result.images)
        filenames = []

        for i, image in enumerate(images):
            # Save the files for each octave to disk.
            filename = output.format(
                octave=result.octave,
                variation=f"_{i}" if len(images) > 1 else "",
                command=cmd.__class__.__name__.lower(),
            )

            soraxas_toolbox.image.display(image)
            soraxas_toolbox.image.display(image.resize(size=config["size"], resample=0))

            image.resize(size=config["size"], resample=0).save(
                filename, lossless=True, compress=6
            )

            print("\n=> output:", filename)
            filenames.append(filename)

    return result, filenames
