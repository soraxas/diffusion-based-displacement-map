import abc
import itertools
from typing import Set, Dict, Callable

import numpy as np
import soraxas_toolbox.image
import torch
import torch.nn.functional as F
from icecream import ic

from .patch import PatchBuilder
from .match import FeatureMatcher


class Critic(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def evaluate(self, *args):
        ...

    @abc.abstractmethod
    def get_layers(self, *args) -> Set[str]:
        ...

    @abc.abstractmethod
    def from_features(self, *args):
        ...


class GramMatrixCritic(Critic):
    """A `Critic` evaluates the features of an image to determine how it scores.

    This critic computes a 2D histogram of feature cross-correlations for the specified
    layer (e.g. "1_1") or layer pair (e.g. "1_1:2_1"), and compares it to the target
    gram matrix.
    """

    def __init__(self, layer, offset: float = -1.0):
        super().__init__()
        self.pair = tuple(layer.split(":"))
        if len(self.pair) == 1:
            self.pair = (self.pair[0], self.pair[0])
        self.offset = offset
        self.gram = None

    def evaluate(self, features):
        current = self._prepare_gram(features)
        result = F.mse_loss(current, self.gram.expand_as(current), reduction="none")
        yield 1e4 * result.flatten(1).mean(dim=1)

    def from_features(self, source_features: Dict[str, torch.Tensor]):
        def norm(xs):
            if not isinstance(xs, (tuple, list)):
                xs = (xs,)
            ms = [torch.mean(x, dim=(2, 3), keepdim=True) for x in xs]
            return (sum(ms) / len(ms)).clamp(min=1.0)

        self.means = (
            norm(source_features[self.pair[0]]),
            norm(source_features[self.pair[1]]),
        )
        self.gram = self._prepare_gram(source_features)

    def get_layers(self) -> Set[str]:
        return set(self.pair)

    def _gram_matrix(self, column, row):
        (b, ch, h, w) = column.size()
        f_c = column.view(b, ch, w * h)
        (b, ch, h, w) = row.size()
        f_r = row.view(b, ch, w * h)

        gram = (f_c / w).bmm((f_r / h).transpose(1, 2)) / ch
        assert not torch.isnan(gram).any()

        return gram

    def _prepare_gram(self, features):
        result = 0.0
        for l, u in zip(features[self.pair[0]], features[self.pair[1]]):
            lower = l / self.means[0] + self.offset
            upper = u / self.means[1] + self.offset
            gram = self._gram_matrix(
                lower, F.interpolate(upper, size=lower.shape[2:], mode="nearest")
            )
            result += gram
        return result / len(features[self.pair[0]])


class HistogramCritic(Critic):
    """
    This critic uses the Sliced Wasserstein Distance of the features to approximate the
    distance between n-dimensional histogram.

    See https://arxiv.org/abs/2006.07229 for details.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def get_layers(self) -> Set[str]:
        return {self.layer}

    def from_features(self, source_features: Dict[str, torch.Tensor]):
        self.features = source_features[self.layer]

    def random_directions(self, count, device):
        directions = torch.empty((count, count, 1, 1), device=device).uniform_(
            -1.0, +1.0
        )
        return directions / torch.norm(directions, dim=1, keepdim=True)

    def sorted_projection(self, directions, features):
        proj_t = torch.sum(features * directions, dim=1).flatten(1)
        return torch.sort(proj_t, dim=1).values

    def evaluate(self, features):
        f = features[self.layer]
        assert f.shape[0] == 1

        directions = self.random_directions(f.shape[1], f.device)

        with torch.no_grad():
            source = self.sorted_projection(directions, self.features)
        current = self.sorted_projection(directions, f)

        yield F.mse_loss(current, source)


class PatchCritic(Critic):
    LAST = None

    def __init__(self, layer, variety=0.2):
        super().__init__()
        self.layer: Set[str] = layer
        self.patches: torch.Tensor = None
        self.builder = PatchBuilder(patch_size=2)
        self.matcher = FeatureMatcher(device="cpu", variety=variety)
        # self.matcher = FeatureMatcher(device="cuda", variety=variety)
        self.split_hints = {}

    def get_layers(self) -> Set[str]:
        return {self.layer}

    def from_features(self, source_features: Dict[str, torch.Tensor]):
        self.patches = torch.nn.Parameter(self._prepare(source_features).detach())
        self.matcher.update_sources(self.patches)
        self.iteration = 0

    def _prepare(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(features[self.layer], (tuple, list)):
            sources = [self.builder.extract(f) for f in features[self.layer]]
            chunk_size = min(s.shape[2] for s in sources)
            chunks = [torch.split(s, chunk_size, dim=2) for s in sources]
            return torch.cat(list(itertools.chain.from_iterable(chunks)), dim=3)
        else:
            return self.builder.extract(features[self.layer])

    def auto_split(
        self,
        function: Callable,
        min_dimension: int,
        max_dimension: int,
        *arguments,
        **keywords,
    ):
        key = (self.matcher.target.shape, function)

        for i in self.split_hints.get(key, range(16)):
            try:

                # number of split must be < min dimension
                num_splits = min(2**i, min_dimension - 1)

                rough_elements_per_split = (max_dimension / num_splits) ** 2
                # ic(min_dimension, max_dimension)
                # ic(rough_elements_per_split, rough_elements_per_split * num_splits,
                #    # size_per_split
                #    )

                if rough_elements_per_split > 2e5:
                    # if rough_elements_per_split > 60:
                    each_split_size = 3
                    _num_splits = min_dimension // each_split_size
                    ic(num_splits, min_dimension, _num_splits, rough_elements_per_split)
                    num_splits = _num_splits

                result = function(*arguments, split=num_splits, **keywords)
                self.split_hints[key] = list(range(i, 16))
                return result
            except RuntimeError as e:
                print(e)
                raise
                if "CUDA out of memory." not in str(e):
                    raise
                # raise

        assert False, f"Unable to fit {function} execution into CUDA memory."

    def evaluate(self, features):

        # import icecream
        ic.disable()

        import soraxas_toolbox.globals

        soraxas_toolbox.globals.create_if_not_exists(
            "throttle", lambda: soraxas_toolbox.ThrottledExecution(60)
        )

        self.iteration += 1

        target = self._prepare(features)
        self.matcher.update_target(target)

        matched_target = self._update(target)

        # ic(matched_target.shape, target.shape)

        yield 0.5 * F.mse_loss(target, matched_target)
        # del matched_target

        matched_source = self.matcher.reconstruct_source()

        # if soraxas_toolbox.globals.ns["throttle"]:
        #
        #     def view(_target):
        #
        #         x = _target.detach().cpu().numpy()
        #         x = x.reshape(_target.shape[1], -1)
        #         # labels = np.arange(_target.shape[2] * _target.shape[3])
        #         # labels = np.repeat(labels.reshape(1, -1), _target.shape[1], axis=0).ravel()
        #         labels = np.arange(_target.shape[1])
        #         # grid = np.mgrid[:_target.shape[2], :_target.shape[3]]
        #         # torch
        #         # print(grid)
        #         # print(grid.shape)
        #         soraxas_toolbox.image.view_high_dimensional_embeddings(x, label=labels)
        #
        #     view(matched_source)
        #     view(matched_target)
        #
        # ic(matched_source.shape, self.patches.shape)
        yield 0.5 * F.mse_loss(matched_source, self.patches)
        # exit()
        del matched_source

    @torch.no_grad()
    def _update(self, target):
        # from icecream import ic
        # try:
        #     self.__class__.i.add(self)
        # except:
        #     self.__class__.i = set()
        #     self.__class__.i.add(self)
        # ic(self.__class__.i, len(self.__class__.i), self.split_hints.keys())

        _min_dimension = min(
            *self.matcher.sources.shape[2:],
            *self.matcher.target.shape[2:],
        )
        _max_dimension = max(
            *self.matcher.sources.shape[2:],
            *self.matcher.target.shape[2:],
        )
        ic(self.matcher.sources.shape, self.matcher.target.shape)

        if self.iteration == 1:
            self.auto_split(
                self.matcher.compare_features_identity,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
            )
            self.matcher.update_biases()

        if target.flatten(1).shape[1] < 1_048_576:
            self.auto_split(
                self.matcher.compare_features_matrix,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
            )
        else:
            self.auto_split(
                self.matcher.compare_features_identity,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
            )
            self.auto_split(
                self.matcher.compare_features_inverse,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
            )
            self.auto_split(
                self.matcher.compare_features_random,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
                radius=[16, 8, 4, -1][self.iteration % 4],
            )
            self.auto_split(
                self.matcher.compare_features_nearby,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
                radius=[4, 2, 1][self.iteration % 3],
            )
            self.auto_split(
                self.matcher.compare_features_coarse,
                min_dimension=_min_dimension,
                max_dimension=_max_dimension,
                parent=PatchCritic.LAST,
            )

        PatchCritic.LAST = self.matcher
        self.matcher.update_biases()
        matched_target = self.matcher.reconstruct_target()

        return matched_target
