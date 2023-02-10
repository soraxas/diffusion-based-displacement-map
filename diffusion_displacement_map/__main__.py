#!/usr/bin/env python3
r"""

Usage:
    texturize remix SOURCE... [options] --size=WxH
    texturize enhance TARGET [with] SOURCE [options] --zoom=ZOOM
    texturize expand TARGET [with] SOURCE [options] --size=WxH
    texturize mashup SOURCE TARGET [options] --size=WxH
    texturize remake TARGET [like] SOURCE [options] --weights=WEIGHTS
    texturize repair TARGET [with] SOURCE [options]
    texturize --help

Examples:
    texturize remix samples/grass.webp --size=1440x960 --output=result.png
    texturize remix samples/gravel.png --quality=1
    texturize remix samples/sand.tiff  --output=tmp/{source}-{octave}.webp
    texturize remix samples/brick.jpg  --device=cpu

Options:
    SOURCE                  Path to source image to use as texture.
    -s WxH, --size=WxH      Output resolution as WIDTHxHEIGHT. [default: 640x480]
    -o FILE, --output=FILE  Filename for saving the result, includes format variables.
                            [default: {command}_{source}{variation}.png]

    --weights=WEIGHTS       Comma-separated list of blend weights. [default: 1.0]
    --zoom=ZOOM             Integer zoom factor for enhancing. [default: 2]

    --variations=V          Number of images to generate at same time. [default: 1]
    --seed=SEED             Configure the random number generation.
    --mode=MODE             Either "patch" or "gram" to manually specify critics.
    --octaves=O             Number of octaves to process. Defaults to 5 for 512x512, or
                            4 for 256x256 equivalent pixel count.
    --quality=Q             Quality for optimization, higher is better. [default: 4]

    --model=MODEL           Name of the convolution network to use. [default: VGG11]
    --layers=LAYERS         Comma-separated list of layers.
    --device=DEVICE         Hardware to use, either "cpu" or "cuda".
    --precision=PRECISION   Floating-point format to use, "float16" or "float32".
    --quiet                 Suppress any messages going to stdout.
    --verbose               Display more information on stdout.
    -h, --help              Show this message.
"""
import argparse
from typing import Tuple, Type, Optional, List

import soraxas_toolbox.image
import click
from tap import Tap
from soraxas_toolbox.torch import TorchNetworkPrinter

from diffusion_displacement_map.arg import Args

p = TorchNetworkPrinter(
    # auto_cleanup=False
)

import os
import glob
import itertools

import docopt
from schema import Schema, Use, And, Or

import torch

from . import api, io, commands


# from .logger import ansi, ConsoleLog


def validate(config):
    # Determine the shape of output tensor (H, W) from specified resolution.
    def split_size(size: str):
        return tuple(map(int, size.split("x")))

    def split_strings(text: str):
        return text.split(",")

    def split_floats(text: str):
        return tuple(map(float, text.split(",")))

    sch = Schema(
        {
            "SOURCE": [str],
            "TARGET": Or(None, str),
            "size": And(Use(split_size), tuple),
            "output": str,
            "weights": Use(split_floats),
            "zoom": Use(int),
            "variations": Use(int),
            "seed": Or(None, Use(int)),
            "mode": Or(None, "patch", "gram", "hist"),
            "octaves": Or(None, Use(int)),
            "quality": Use(float),
            "model": Or("VGG11", "VGG13", "VGG16", "VGG19"),
            "layers": Or(None, Use(split_strings)),
            "device": Or(None, "cpu", "cuda"),
            "precision": Or(None, "float16", "float32"),
            "help": Use(bool),
            "quiet": Use(bool),
            "verbose": Use(bool),
        },
        ignore_extra_keys=True,
    )
    return sch.validate({k.replace("--", ""): v for k, v in config.items()})


# def main():
#     # Parse the command-line options based on the script's documentation.
#     config = docopt.docopt(__doc__, help=False)
#     all_commands = [cmd.lower() for cmd in commands.__all__] + ["--help"]
#     command = [cmd for cmd in all_commands if config[cmd]][0]
#
#     # Ensure the user-specified values are correct, separate command-specific arguments.
#     config = validate(config)
#     sources, target, output, seed = [
#         config.pop(k) for k in ("SOURCE", "TARGET", "output", "seed")
#     ]
#     weights, zoom = [config.pop(k) for k in ("weights", "zoom")]
#
#     # Setup the output logging and display the logo!
#     if config.pop("help") is True:
#         return
#
#     # Scan all the files based on the patterns specified.
#     files = itertools.chain.from_iterable(glob.glob(s) for s in sources)
#     for filename in files:
#         # If there's a random seed, use the same for all images.
#         if seed is not None:
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)
#
#         # Load the images necessary.
#         img_mode = "RGB"
#         img_mode = "L"
#         source_img = io.load_image_from_file(filename, mode=img_mode)
#         target_img = io.load_image_from_file(target, mode=img_mode) if target else None
#         factor = 0.3
#         # factor = 1
#         if factor != 1:
#             new_size = tuple(int(s * factor) for s in source_img.size)
#             # new_size = 512,512
#             # ic(source_img.size, new_size)
#
#             source_img = source_img.resize(new_size)
#             soraxas_toolbox.image.display(source_img)
#
#             if target_img:
#                 target_img = target_img.resize(new_size)
#                 soraxas_toolbox.image.display(target_img)
#
#         # config['size'] = 5120,5120
#
#         # Setup the command specified by user.
#         if command == "remix":
#             cmd = commands.Remix(source_img)
#         if command == "enhance":
#             cmd = commands.Enhance(target_img, source_img, zoom=zoom)
#             config["octaves"] = cmd.octaves
#             # Calculate the size based on the specified zoom.
#             config["size"] = (target_img.size[0] * zoom, target_img.size[1] * zoom)
#         if command == "expand":
#             # Calculate the factor based on the specified size.
#             factor = (
#                 target_img.size[0] / config["size"][0],
#                 target_img.size[1] / config["size"][1],
#             )
#             cmd = commands.Expand(target_img, source_img, factor=factor)
#         if command == "remake":
#             cmd = commands.Remake(target_img, source_img, weights=weights)
#             config["octaves"] = 1
#             config["size"] = target_img.size
#         if command == "mashup":
#             cmd = commands.Mashup([source_img, target_img])
#         if command == "repair":
#             cmd = commands.Repair(target_img, source_img)
#             config["octaves"] = 3
#             config["size"] = target_img.size
#
#         # Process the files one by one, each may have multiple variations.
#         try:
#             config["output"] = output
#             config["output"] = config["output"].replace(
#                 "{source}", os.path.splitext(os.path.basename(filename))[0]
#             )
#             if target:
#                 config["output"] = config["output"].replace(
#                     "{target}", os.path.splitext(os.path.basename(target))[0]
#                 )
#
#             config.pop("quiet"), config.pop("verbose")
#             result, filenames = api.process_single_command(cmd, **config)
#         except KeyboardInterrupt:
#             raise
#             print("\nCTRL+C detected, interrupting...")
#             break


def build_args(input_args: Optional[List[str]] = None) -> Tuple[commands.Command, Args]:
    args = Args().parse_args(input_args)

    # If there's a random seed, use the same for all images.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Load the images necessary.
    source_img = io.load_image_from_file(args.filename, mode=args.img_mode)
    target_img = (
        io.load_image_from_file(args.target, mode=args.img_mode)
        if args.target
        else None
    )

    if args.input_size is not None and args.resize_factor != 1:
        raise ValueError("--input-size and --resize-factor "
                         "options are mutually exclusive!")

    new_size = args.input_size

    if args.resize_factor != 1:
        new_size = tuple(int(s * args.resize_factor) for s in source_img.size)

    if new_size is not None:
        print(f"> resizing source from {source_img.size} to {new_size}")

        source_img = source_img.resize(new_size)
        soraxas_toolbox.image.display(source_img)

        if target_img:
            target_img = target_img.resize(new_size)
            soraxas_toolbox.image.display(target_img)

    # Setup the command specified by user.
    if args.command == "remix":
        cmd = commands.Remix(source_img)
    elif args.command == "enhance":
        cmd = commands.Enhance(target_img, source_img, zoom=zoom)
        config["octaves"] = cmd.octaves
        # Calculate the size based on the specified zoom.
        config["size"] = (target_img.size[0] * zoom, target_img.size[1] * zoom)
    elif args.command == "expand":
        # Calculate the factor based on the specified size.
        factor = (
            target_img.size[0] / config["size"][0],
            target_img.size[1] / config["size"][1],
        )
        cmd = commands.Expand(target_img, source_img, factor=factor)
    elif args.command == "remake":
        cmd = commands.Remake(target_img, source_img, weights=weights)
        config["octaves"] = 1
        config["size"] = target_img.size
    elif args.command == "mashup":
        cmd = commands.Mashup([source_img, target_img])
    elif args.command == "repair":
        cmd = commands.Repair(target_img, source_img)
        config["octaves"] = 3
        config["size"] = target_img.size
    else:
        raise ValueError()

    if args.output_size is None:
        args.output_size = source_img.size

    args.output = args.output.replace(
        "{source}", os.path.splitext(os.path.basename(args.filename))[0]
    )
    return cmd, args


def main():
    cmd, args = build_args()
    result, filenames = api.process_single_command(cmd, args)

    if __name__ == "__main__":
        main()
