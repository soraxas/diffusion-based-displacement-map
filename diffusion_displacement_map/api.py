import os
import math
import pathlib

import soraxas_toolbox
import torch
import torch.nn.functional as F
from creativeai.image.encoders import models
from creativeai.image.encoders.base import Encoder
from icecream import ic

from .arg import Args
from .commands import Command
from .critics import GramMatrixCritic, PatchCritic, HistogramCritic
from .app import Application, Result
from .io import *
from .seamless_modules import WraparoundVGG11


@torch.no_grad()
def process_iterations(
    cmd: Command,
    args: Args,
):
    """Synthesize a new texture and return a PyTorch tensor at each iteration."""

    # Configure the default options dynamically, unless overriden.
    factor = math.sqrt((args.output_size[0] * args.output_size[1]) / (32**2))
    octaves = args.octaves or getattr(cmd, "octaves", int(math.log(factor, 2) + 1.0))

    ic(args.model, factor, octaves)

    # Setup the application to use throughout the synthesis.

    # Encoder used by all the critics at every octave.
    encoder: Encoder = getattr(models, args.model)(
        pretrained=True, pool_type=torch.nn.AvgPool2d
    )
    encoder: Encoder = WraparoundVGG11(
        pretrained=True,
        # input_type='L'
    )

    # encoder = encoder.to(device=app.device, dtype=app.precision)

    app = Application(
        encoder,
        layers=args.layers,
        mode=args.mode,
        device=args.device,
        precision=args.precision,
    )

    # Coarse-to-fine rendering, number of octaves specified by user.
    seed = None

    progressive_scales = [2**s for s in range(octaves - 1, -1, -1)]
    ic(progressive_scales)
    for octave, scale in enumerate(progressive_scales):

        app.create_pbar(f"OCTAVE {octave + 1}/{len(progressive_scales)}")

        # app.log.debug("<- scale:", f"1/{scale}")

        # app.progress = app.log.create_progress_bar(100)

        result_size = (
            args.variations,
            3 if args.img_mode == "RGB" else 1,
            args.output_size[1] // scale,
            args.output_size[0] // scale,
        )
        # app.log.debug("<- seed:", tuple(result_size[2:]))

        ic(result_size)
        # result_size = (1,3,160,160)
        seed = cmd.prepare_seed_tensor(app, result_size, previous=seed)

        # ic(result_size, seed.shape)

        for dtype in [
            torch.float32,
            torch.float16,
        ]:
            if app.precision != dtype:
                app.precision = dtype
                app.encoder = app.encoder.to(dtype=dtype)
                if seed is not None:
                    seed = seed.to(app.device)

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
                    quality=args.quality,
                ):
                    if args.should_stop():
                        raise StopIteration()
                    args.report(
                        image=result.images,
                        octave=octave,
                        total_octave=len(progressive_scales),
                        n=app.log.n,
                        total_n=app.log.total,
                    )
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
def process_single_command(cmd: Command, args: Args):
    for result in process_iterations(cmd, args):
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

        assert len(images) > 0
        for i, image in enumerate(images):
            # Save the files for each octave to disk.
            filename = args.output.format(
                octave=result.octave,
                variation=f"_{i}" if len(images) > 1 else "",
                command=cmd.__class__.__name__.lower(),
            )

            soraxas_toolbox.image.display(image)
            soraxas_toolbox.image.display(
                image.resize(size=args.output_size, resample=0)
            )

            filename = pathlib.Path(filename)
            # create any folder that doesn't exist yet
            filename.parent.mkdir(exist_ok=True, parents=True)

            image.resize(size=args.output_size, resample=0).save(
                filename, lossless=True, compress=6
            )

            print("\n=> output:", filename)
            filenames.append(filename)

    return result, filenames
