import itertools

import torch
import torch.nn.functional as F

from .features_compare.comparison import (
    compute_similarity,
    compare_features_coarse,
    improve,
    FeatureComparisonWindow,
    compare_features_identity,
    compare_features_inverse,
    FeatureMappingPair,
    compare_features_random,
    compare_features_nearby,
)
from .features_compare.mapping import Mapping
from .features_compare.utils import iterate_range


def torch_flatten_2d(a):
    a = a.permute(1, 0, 2, 3)
    return a.reshape(a.shape[:1] + (-1,))


def lol(array, indices):
    assert indices.shape[1] == 2

    # batch_size = 16
    # c, h, w = 256, 16, 16
    # nb_points = 150
    # img_feat = torch.randn(batch_size, c, h, w)
    # x = torch.empty(batch_size, nb_points, dtype=torch.long).random_(h)
    # y = torch.empty(batch_size, nb_points, dtype=torch.long).random_(w)
    # x = x[:, None, :, None].expand(-1, c, -1, w)
    # y = y[:, None, None, :].expand(-1, c, nb_points, -1)

    B, C, h, w = array.shape

    img_feat = array

    x = indices[:, 0, ...].reshape(B, -1)
    y = indices[:, 1, ...].reshape(B, -1)

    print(array.shape)
    print(x.shape, y.shape)

    n_pts = x.shape[-1]

    x = x[:, None, :, None].expand(-1, C, -1, w)
    y = y[:, None, None, :].expand(-1, C, n_pts, -1)

    print(B, C, h, w)
    print(x.shape, y.shape, img_feat.shape)

    points = torch.gather(torch.gather(img_feat, 2, x), 3, y)
    print(points.shape)
    points = points.reshape(B, C, h, w)
    print(points.shape)

    return points

    assert indices.shape[0] == 1, (indices.shape, indices[:, 0, 0, 0])

    from icecream import ic

    array[
        :,
        :,
    ].permute(1, 2)

    x_val = array.gather(
        1, indices[:, 0, ...].unsqueeze(1).expand(-1, array.shape[1], -1, -1)
    )
    y_val = array.gather(
        1, indices[:, 1, ...].unsqueeze(1).expand(-1, array.shape[1], -1, -1)
    )
    ic(array.shape, indices[:, 0, ...].shape)
    ic(x_val.shape, y_val.shape)
    exit()

    return array[:, :, indices[0, 0, ...], indices[0, 1, ...]]


# @profile
def gather_nd_torch(params, indices, batch_dim=1):
    """A PyTorch porting of tensorflow.gather_nd
    This implementation can handle leading batch dimensions in params, see below for detailed explanation.

    The majority of this implementation is from Michael Jungo @ https://stackoverflow.com/a/61810047/6670143
    I just ported it compatible to leading batch dimension.

    Args:
      params: a tensor of dimension [b1, ..., bn, g1, ..., gm, c].
      indices: a tensor of dimension [b1, ..., bn, x, m]
      batch_dim: indicate how many batch dimension you have, in the above example, batch_dim = n.

    Returns:
      gathered: a tensor of dimension [b1, ..., bn, x, c].

    Example:
    >>> batch_size = 5
    >>> inputs = torch.randn(batch_size, batch_size, batch_size, 4, 4, 4, 32)
    >>> pos = torch.randint(4, (batch_size, batch_size, batch_size, 12, 3))
    >>> gathered = gather_nd_torch(inputs, pos, batch_dim=3)
    >>> gathered.shape
    torch.Size([5, 5, 5, 12, 32])

    >>> inputs_tf = tf.convert_to_tensor(inputs.numpy())
    >>> pos_tf = tf.convert_to_tensor(pos.numpy())
    >>> gathered_tf = tf.gather_nd(inputs_tf, pos_tf, batch_dims=3)
    >>> gathered_tf.shape
    TensorShape([5, 5, 5, 12, 32])

    >>> gathered_tf = torch.from_numpy(gathered_tf.numpy())
    >>> torch.equal(gathered_tf, gathered)
    True
    """
    import math

    # sanity_check

    for _d in range(batch_dim):
        assert params.shape[_d] == indices.shape[_d]
    for _m in range(indices.shape[-1]):
        assert params.shape[batch_dim + _m]  ## assert that it exists.
    assert (
        len(params.shape) == batch_dim + indices.shape[-1] + 1
    ), f"{params.shape, batch_dim, indices.shape}"
    assert len(indices.shape) == batch_dim + 1 + 1, f"{len(indices.shape), batch_dim}"

    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = math.prod(batch_dims)  # b1 * ... * bn

    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size(-2)  # x
    n_pos = indices.size(-1)  # m

    # reshape leading batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)

    # build gather indices
    # gather for each of the data point in this "batch"
    batch_enumeration = torch.arange(batch_size).unsqueeze(1)
    gather_dims = [indices[:, :, i] for i in range(len(grid_dims))]
    # gather_dims.insert(0, batch_enumeration)
    gathered = params[[batch_enumeration] + gather_dims]

    # reshape back to the shape with leading batch dims
    gathered = gathered.reshape(*batch_dims, n_indices, c_dim)
    return gathered


# HA = dict()


def torch_gather_2d(array, indices):
    # global HA
    # def tuple_rec(a):
    #     if isinstance(a, list):
    #         return tuple(tuple_rec(_a) for _a in a)
    #     return a
    # i = (tuple_rec(array), tuple_rec(indices.tolist()))
    # # print(i)
    # if i in HA:
    #     HA[i] += 1
    #     print(list(sorted(HA.values())))
    # else:
    #     HA[i] = 1

    if array.shape[0] < indices.shape[0]:
        # broadcast batch dimension
        assert array.shape[0] == 1
        array = array.expand(indices.shape[0], -1, -1, -1)
    assert array.shape[0] == indices.shape[0]

    dim_b, dim_c = array.shape[0:2]
    _ori_shape = indices.shape[2:]
    _arr = array.permute(0, 2, 3, 1)
    _ind = indices.permute(0, 2, 3, 1).reshape(indices.shape[0], -1, 2)

    from icecream import ic

    # ic(dim_b, dim_c, _ori_shape, array.shape, indices.shape)
    out = (
        gather_nd_torch(_arr, _ind).permute(0, 2, 1).reshape(dim_b, dim_c, *_ori_shape)
    )

    return out

    """Extract the content of an array using the 2D coordinates provided."""
    batch = torch.arange(
        0, array.shape[0], dtype=torch.long, device=indices.device
    ).view(-1, 1, 1)

    idx = batch * (array.shape[2] * array.shape[3]) + (
        indices[:, 0, :, :] * array.shape[3] + indices[:, 1, :, :]
    )
    flat_array = torch_flatten_2d(array)
    x = torch.index_select(flat_array, 1, idx.view(-1))

    result = x.view(array.shape[1:2] + indices.shape[:1] + indices.shape[2:])

    return result.permute(1, 0, 2, 3)


def torch_scatter_2d(output, indices, values):
    _, c, h, w = output.shape

    assert output.shape[0] == 1
    assert output.shape[1] == values.shape[1]

    chanidx = torch.arange(0, c, dtype=torch.long, device=indices.device).view(-1, 1, 1)

    idx = chanidx * (h * w) + (indices[:, 0] * w + indices[:, 1])
    output.flatten().scatter_(0, idx.flatten(), values.to(dtype=output.dtype).flatten())


def cosine_similarity_matrix_1d(source, target, eps=None):
    eps = eps or (1e-3 if source.dtype == torch.float16 else 1e-8)
    source = source / (torch.norm(source, dim=1, keepdim=True) + eps)
    target = target / (torch.norm(target, dim=1, keepdim=True) + eps)

    result = torch.bmm(source.permute(0, 2, 1), target)
    return torch.clamp(result, max=1.0 / eps)


def cosine_similarity_vector_1d(source, target, eps=None):
    eps = eps or (1e-3 if source.dtype == torch.float16 else 1e-8)
    source = source / (torch.norm(source, dim=1, keepdim=True) + eps)
    target = target / (torch.norm(target, dim=1, keepdim=True) + eps)

    source = source.expand_as(target)

    result = torch.sum(source * target, dim=1)
    return torch.clamp(result, max=1.0 / eps)


class FeatureMatcher:
    """Implementation of feature matching between two feature maps in 2D arrays, using
    normalized cross-correlation of features as similarity metric.
    """

    def __init__(self, device="cpu", variety=0.0):
        self.device = torch.device(device)
        self.variety = variety

        self.target = None
        self.sources = None
        self.repro_target: Mapping = None
        self.repro_sources: Mapping = None

    @property
    def source_and_target_pair(self) -> FeatureMappingPair:
        return FeatureMappingPair(
            self.sources, self.repro_sources, self.target, self.repro_target
        )

    @property
    def target_and_source_pair(self) -> FeatureMappingPair:
        return FeatureMappingPair(
            self.target,
            self.repro_target,
            self.sources,
            self.repro_sources,
        )

    def clone(self):
        clone = FeatureMatcher(device=self.device)
        clone.sources = self.sources
        clone.target = self.target

        clone.repro_target = self.repro_target.clone()
        clone.repro_sources = self.repro_sources.clone()
        return clone

    def update_target(self, target):
        assert len(target.shape) == 4
        assert target.shape[0] == 1

        self.target = target

        if self.repro_target is None:
            self.repro_target = Mapping(self.target.shape, self.device)
            self.repro_target.from_random(self.sources.shape)
            self.repro_sources.from_random(self.target.shape)

        self.repro_target.scores.fill_(float("-inf"))
        self.repro_sources.scores.fill_(float("-inf"))

        if target.shape[2:] != self.repro_target.indices.shape[2:]:
            self.repro_sources.rescale(target.shape[2:])
            self.repro_target.resize(target.shape[2:])

    def update_sources(self, sources: torch.Tensor):
        assert len(sources.shape) == 4
        assert sources.shape[0] == 1

        self.sources = sources

        if self.repro_sources is None:
            self.repro_sources = Mapping(self.sources.shape, self.device)

        if sources.shape[2:] != self.repro_sources.indices.shape[2:]:
            self.repro_target.rescale(sources.shape[2:])
            self.repro_sources.resize(sources.shape[2:])

    def update_biases(self):
        sources_value = (
            self.repro_sources.scores
            - torch_gather_2d(self.repro_sources.biases, self.repro_sources.indices)
            - self.repro_target.biases
        )
        target_value = (
            self.repro_target.scores
            - torch_gather_2d(self.repro_target.biases, self.repro_target.indices)
            - self.repro_sources.biases
        )

        k = self.variety
        self.repro_target.biases[:] = -k * (sources_value - sources_value.mean())
        self.repro_sources.biases[:] = -k * (target_value - target_value.mean())

        self.repro_target.scores.fill_(float("-inf"))
        self.repro_sources.scores.fill_(float("-inf"))

    def reconstruct_target(self):
        return torch_gather_2d(
            self.sources, self.repro_target.indices.to(self.sources.device)
        )

    def reconstruct_source(self):
        return torch_gather_2d(
            self.target, self.repro_sources.indices.to(self.sources.device)
        )

    def compare_features_coarse(self, parent, radius=2, split=1):
        if parent is None:
            return 0

        ts = compare_features_coarse(
            self.target_and_source_pair,
            parent.repro_target,
            split=split,
            radius=radius,
        )
        st = compare_features_coarse(
            self.source_and_target_pair,
            parent.repro_sources,
            split=split,
            radius=radius,
        )
        return ts + st

    def compare_features_matrix(self, split=1):
        assert self.sources.shape[0] == 1, "Only 1 source supported."

        for (t1, t2), (s1, s2) in itertools.product(
            iterate_range(self.target.shape[2], split),
            iterate_range(self.sources.shape[2], split),
        ):
            assert t2 != t1 and s2 != s1

            target_window = self.target[:, :, t1:t2].flatten(2)
            source_window = self.sources[:, :, s1:s2].flatten(2)

            similarity = cosine_similarity_matrix_1d(target_window, source_window)
            similarity += (
                self.repro_target.biases[:, :, s1:s2]
                .to(similarity.device)
                .reshape(1, 1, -1)
            )
            similarity += (
                self.repro_sources.biases[:, :, t1:t2]
                .to(similarity.device)
                .reshape(1, -1, 1)
            )

            best_source = torch.max(similarity, dim=2)
            self.repro_target.improve_window(
                (t1, t2 - t1, 0, self.target.shape[3]),
                (s1, s2 - s1, 0, self.sources.shape[3]),
                best_source,
            )

            best_target = torch.max(similarity, dim=1)
            self.repro_sources.improve_window(
                (s1, s2 - s1, 0, self.sources.shape[3]),
                (t1, t2 - t1, 0, self.target.shape[3]),
                best_target,
            )

    def compare_features_random(self, radius=-1, split=1, times=4):
        """Generate random coordinates within a radius for each pixel, then compare the
        features to see if the current selection can be improved.
        """

        ts = compare_features_random(self.target_and_source_pair, radius, split, times)
        st = compare_features_random(self.source_and_target_pair, radius, split, times)
        return ts + st

    def compare_features_identity(self, split=1):
        compare_features_identity(
            self.target_and_source_pair,
            split=split,
        )
        compare_features_identity(
            self.source_and_target_pair,
            split=split,
        )

    def compare_features_inverse(self, split=1):
        ts = compare_features_inverse(
            self.target_and_source_pair,
            split=split,
        )
        st = compare_features_inverse(
            self.source_and_target_pair,
            split=split,
        )
        return ts + st

    def compare_features_nearby(self, radius: int, split: int = 1):
        ts = compare_features_nearby(
            self.target_and_source_pair,
            radius=radius,
            split=split,
        )
        st = compare_features_nearby(
            self.source_and_target_pair,
            radius=radius,
            split=split,
        )
        return ts + st
