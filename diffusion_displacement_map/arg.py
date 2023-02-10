import argparse
from typing import Tuple, List, Callable

from tap import Tap

TYPE_SIZE = Tuple[int, int]


def size_type(value: str) -> TYPE_SIZE:
    splitted = value.split("x")

    if len(splitted) != 2:
        raise argparse.ArgumentTypeError(
            f"{value} must be a string separated by one x, e.g., 1920x1080"
        )

    try:
        return int(splitted[0]), int(splitted[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value} must be a string separated by one x, e.g., 1920x1080"
        )


class Args(Tap):
    command: str
    filename: str
    target: str = None

    report = lambda *_, **__: True

    seed: int = None  # Configure the random number generation

    weights: List[float] = [1.0]  # Comma-separated list of blend weights
    zoom: int = 2  # Integer zoom factor for enhancing
    octaves: int = 0
    """Number of octaves to process. Defaults to 5 for 512x512, or 
    4 for 256x256 equivalent pixel count."""

    variations: int = 1  # Number of images to generate at same time
    quality: float = 4  # Quality for optimization, higher is better

    model: str = "VGG11"  # Name of the convolution network to use

    layers = None
    mode = None
    device = None
    precision = "float32"

    output: str = "output/{command}_{source}{variation}.png"
    input_size: TYPE_SIZE
    output_size: TYPE_SIZE
    img_mode: str
    resize_factor: float = 1.0

    should_stop: Callable[[], bool] = lambda *args: False

    def configure(self):
        self.add_argument("-s", "--output_size", default=None, type=size_type)
        self.add_argument("--input_size", default=None, type=size_type)
        self.add_argument("--img_mode", choices=["RGB", "L"], default="L")
        self.add_argument("-f", "--resize_factor")
        self.add_argument(
            "command",
            choices=["enhance", "expand", "remake", "mashup", "repair", "remix"],
        )
        self.add_argument("filename")
