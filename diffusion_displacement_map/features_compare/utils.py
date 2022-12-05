import torch


def iterate_range(size, split=2):
    assert split <= size, f"{size}, {split}"
    for start, stop in zip(range(0, split), range(1, split + 1)):
        yield (
            max(0, (size * start) // split),
            min(size, (size * stop) // split),
        )


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


def torch_flatten_2d(a):
    a = a.permute(1, 0, 2, 3)
    return a.reshape(a.shape[:1] + (-1,))


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
