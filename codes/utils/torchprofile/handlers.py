from .utils import math

__all__ = ['handlers']


def addmm(node):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p


def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    return n * m


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return b * n * m * p


def matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return n
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return math.prod(b) * n * m
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return math.prod(b) * n * m
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return math.prod(b) * n * m * p


def mul(node):
    os = node.outputs[0].shape
    return math.prod(os)


def convolution(node):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        oc, ic, *ks = node.inputs[1].shape
    else:
        ic, oc, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    return math.prod(os) * ic * math.prod(ks)


def norm(node):
    if node.operator in ['aten::batch_norm', 'aten::instance_norm']:
        affine = node.inputs[1].shape is not None
    elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.operator)

    os = node.outputs[0].shape
    return math.prod(os) if affine else 0


def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    return math.prod(os)


def leaky_relu(node):
    os = node.outputs[0].shape
    return math.prod(os)


def upsample_bilinear2d(node):
    os = node.outputs[0].shape
    return math.prod(os) * 4


# ------------------------- add by myself --------------------
def square(node):
    os = node.outputs[0].shape
    return math.prod(os)

def frobenius_norm(node):
    os = node.inputs[0].shape
    return math.prod(os)
    

def einsum(node):
    if len(node.inputs[1].shape[0]) == 3 and len(node.inputs[1].shape[1]) == 2:
        n, c, t = node.inputs[1].shape[0]
        return n * c * t
    elif len(node.inputs[1].shape[0]) == 3 and len(node.inputs[1].shape[1]) == 3:
        n, c, t = node.inputs[1].shape[0]
        l = node.inputs[1].shape[1][0]
        return n * c * t * l
    elif len(node.inputs[1].shape[0]) == 4:
        n, c, t, l = node.inputs[1].shape[0]
        return n * c * t * l
    else:
        os = node.outputs[0].shape
        return math.prod(os)

def lstm(node):
    batch_size, num_steps, input_size = node.inputs[0].shape
    hidden_size = node.outputs[0].shape[-1]
    
    total_ops = 0

    # i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
    # f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
    # o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
    # g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
    state_ops = (input_size + hidden_size) * hidden_size + hidden_size
    total_ops += state_ops * 4

    # c' = f * c + i * g \\
    # hadamard hadamard add
    total_ops += hidden_size * 3

    # h' = o * \tanh(c') \\
    total_ops += hidden_size
    
    total_ops *= num_steps
    total_ops *= batch_size

    return total_ops

# ----------------------------------------------------


handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    (('aten::linear', 'aten::matmul'), matmul),
    (('aten::mul', 'aten::mul_'), mul),
    ('aten::_convolution', convolution),
    (('aten::batch_norm', 'aten::instance_norm', 'aten::layer_norm',
      'aten::group_norm'), norm),
    (('aten::adaptive_avg_pool1d', 'aten::adaptive_avg_pool2d',
      'aten::adaptive_avg_pool3d', 'aten::avg_pool1d', 'aten::avg_pool2d',
      'aten::avg_pool3d', 'aten::mean'), avg_pool_or_mean),
    ('aten::leaky_relu', leaky_relu),
    ('aten::upsample_bilinear2d', upsample_bilinear2d),
    (('aten::adaptive_max_pool1d', 'aten::adaptive_max_pool2d',
      'aten::adaptive_max_pool3d', 'aten::add', 'aten::add_',
      'aten::alpha_dropout', 'aten::cat', 'aten::chunk', 'aten::clamp',
      'aten::clone', 'aten::constant_pad_nd', 'aten::contiguous',
      'aten::detach', 'aten::div', 'aten::div_', 'aten::dropout',
      'aten::dropout_', 'aten::embedding', 'aten::eq', 'aten::feature_dropout',
      'aten::flatten', 'aten::floor', 'aten::floor_divide', 'aten::gt',
      'aten::hardtanh_', 'aten::index', 'aten::int',  'aten::log_softmax',
      'aten::lt', 'aten::max_pool1d', 'aten::max_pool1d_with_indices',
      'aten::max_pool2d', 'aten::max_pool2d_with_indices', 'aten::max_pool3d',
      'aten::max_pool3d_with_indices', 'aten::max_unpool1d',
      'aten::max_unpool2d', 'aten::max_unpool3d', 'aten::ne',
      'aten::reflection_pad1d', 'aten::reflection_pad2d',
      'aten::reflection_pad3d', 'aten::relu', 'aten::relu_',
      'aten::replication_pad1d', 'aten::replication_pad2d',
      'aten::replication_pad3d', 'aten::rsub', 'aten::select', 'aten::sigmoid',
      'aten::size', 'aten::slice', 'aten::softmax', 'aten::softshrink',
      'aten::squeeze', 'aten::stack', 'aten::sub', 'aten::sum', 'aten::t',
      'aten::tanh', 'aten::threshold', 'aten::to', 'aten::transpose',
      'aten::upsample_nearest2d', 'aten::view', 'aten::zeros',
      'prim::constant', 'prim::listconstruct', 'prim::listunpack',
      'prim::numtotensor', 'prim::tupleconstruct',
      # NOTE: add my myself
      'aten::unsqueeze', 'aten::norm', 'aten::clamp_min', 'aten::expand_as', 'aten::numpy_t',
      'aten::repeat', 'aten::ones_like', 'aten::where', 'aten::unsqueeze_', 'aten::min',
      'aten::maximum', 'aten::arange', 'aten::max'), None),
    (('aten::square', 'aten::sqrt', 'aten::pow', 'aten::log1p'), square),
    (('aten::frobenius_norm', 'aten::prod'), frobenius_norm),
    ('aten::einsum', einsum),
    ('aten::lstm', lstm)
)
