import abc
import itertools
from typing import Set, Dict

import torch
import torch.nn.functional as F

from .patch import PatchBuilder
from .match import FeatureMatcher


class Critic(abc.ABC):
    def on_start(self, *args):
        pass

    def on_finish(self, *args):
        pass

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
        self.pair = tuple(layer.split(":"))
        if len(self.pair) == 1:
            self.pair = (self.pair[0], self.pair[0])
        self.offset = offset
        self.gram = None

    def evaluate(self, features):
        current = self._prepare_gram(features)
        result = F.mse_loss(current, self.gram.expand_as(current), reduction="none")
        yield 1e4 * result.flatten(1).mean(dim=1)

    def from_features(self, features: Dict[str, torch.Tensor]):
        def norm(xs):
            if not isinstance(xs, (tuple, list)):
                xs = (xs,)
            ms = [torch.mean(x, dim=(2, 3), keepdim=True) for x in xs]
            return (sum(ms) / len(ms)).clamp(min=1.0)

        self.means = (norm(features[self.pair[0]]), norm(features[self.pair[1]]))
        self.gram = self._prepare_gram(features)

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
        self.layer = layer

    def get_layers(self) -> Set[str]:
        return {self.layer}

    def from_features(self, features: Dict[str, torch.Tensor]):
        self.features = features[self.layer]

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
        self.layer = layer
        self.patches: torch.Tensor = None
        self.device = None
        self.builder = PatchBuilder(patch_size=2)
        self.matcher = FeatureMatcher(device="cpu", variety=variety)
        # self.matcher = FeatureMatcher(device="cuda", variety=variety)
        self.split_hints = {}

    def get_layers(self) -> Set[str]:
        return {self.layer}

    def on_start(self):
        self.patches = self.patches.to(self.device)
        self.matcher.update_sources(self.patches)

    def on_finish(self):
        self.matcher.sources = None
        self.patches = self.patches.cpu()

    def from_features(self, features: Dict[str, torch.Tensor]):
        patches = self._prepare(features).detach()
        self.device = patches.device
        self.patches = patches.cpu()
        self.iteration = 0

    def _prepare(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(features[self.layer], (tuple, list)):
            sources = [self.builder.extract(f) for f in features[self.layer]]
            chunk_size = min(s.shape[2] for s in sources)
            chunks = [torch.split(s, chunk_size, dim=2) for s in sources]
            return torch.cat(list(itertools.chain.from_iterable(chunks)), dim=3)
        else:
            return self.builder.extract(features[self.layer])

    def auto_split(self, function, *arguments, **keywords):
        key = (self.matcher.target.shape, function)
        for i in self.split_hints.get(key, range(16)):
            try:
                result = function(*arguments, split=2**i, **keywords)
                self.split_hints[key] = list(range(i, 16))
                return result
            except RuntimeError as e:
                print(e)
                if "CUDA out of memory." not in str(e):
                    raise

        assert False, f"Unable to fit {function} execution into CUDA memory."

    def evaluate(self, features):
        self.iteration += 1

        target = self._prepare(features)
        self.matcher.update_target(target)

        matched_target = self._update(target)
        yield 0.5 * F.mse_loss(target, matched_target)
        del matched_target

        matched_source = self.matcher.reconstruct_source()
        yield 0.5 * F.mse_loss(matched_source, self.patches)
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
        if self.iteration == 1:
            self.auto_split(self.matcher.compare_features_identity)
            self.matcher.update_biases()

        if target.flatten(1).shape[1] < 1_048_576:
            self.auto_split(self.matcher.compare_features_matrix)
        else:
            self.auto_split(self.matcher.compare_features_identity)
            self.auto_split(self.matcher.compare_features_inverse)
            self.auto_split(
                self.matcher.compare_features_random,
                radius=[16, 8, 4, -1][self.iteration % 4],
            )
            self.auto_split(
                self.matcher.compare_features_nearby,
                radius=[4, 2, 1][self.iteration % 3],
            )
            self.auto_split(
                self.matcher.compare_features_coarse, parent=PatchCritic.LAST
            )

        PatchCritic.LAST = self.matcher
        self.matcher.update_biases()
        matched_target = self.matcher.reconstruct_target()

        return matched_target
