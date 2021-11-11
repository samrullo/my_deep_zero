import os
import telnetlib
import weakref
import numpy as np

import dezero.cuda
import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter
from dezero.utils import pair
from dezero import Variable


# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None, param_name="W"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name=param_name)
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.h2u = Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new


class TimeLSTM(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = LSTM(hidden_size)

    def forward(self, xs):
        N, T, D = xs.shape
        h_list = []
        for t in range(T):
            h = self.lstm(xs[:, t, :])
            h_list.append(F.unsqueeze(h, axis=1))
        return F.concat(*h_list, axis=1)


class TimeEmbed(Layer):
    def __init__(self, in_size, out_size):
        super(TimeEmbed, self).__init__()
        self.out_size = out_size
        self.embed = EmbedID(in_size, out_size)

    def forward(self, xs):
        N, T = xs.shape
        hs_list = []
        for t in range(T):
            xs_emb = self.embed(xs[:, t])
            hs_list.append(F.unsqueeze(xs_emb, axis=1))
        return F.concat(*hs_list, axis=1)


class TimeAffine(Layer):
    def __init__(self, out_size):
        super(TimeAffine, self).__init__()
        self.out_size = out_size
        self.linear = Linear(out_size)

    def forward(self, xs):
        N, T, H = xs.shape
        out_list = []
        for t in range(T):
            y = self.linear(xs[:, t, :])
            out_list.append(F.unsqueeze(y, axis=1))
        return F.concat(*out_list, axis=1)


class RNNEncoder(Layer):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.time_embed = TimeEmbed(V, D)
        self.time_lstm = TimeLSTM(H)

    def forward(self, xs):
        xs_embedded = self.time_embed(xs)
        hs = self.time_lstm(xs_embedded)
        h = F.pick_item(hs, -1)
        return h


class RNNDecoder(Layer):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.time_embed = TimeEmbed(V, D)
        self.time_lstm = TimeLSTM(H)
        self.time_affine = TimeAffine(V)

    def forward(self, xs, h):
        self.time_lstm.lstm.h = h
        xs_embedded = self.time_embed(xs)
        hs = self.time_lstm(xs_embedded)
        out = self.time_affine(hs)
        return out

    def generate(self, h, start_id, sample_size):
        sampled = []
        sampled_id = start_id
        self.time_lstm.lstm.h = h
        for _ in range(sample_size):
            x = np.array([[sampled_id]])
            out = self.time_embed(x)
            hs = self.time_lstm(out)
            score = self.time_affine(hs)
            sampled_id = np.argmax(score.data.flatten())
            sampled.append(sampled_id)
        return sampled


class RNNPeekyDecoder(Layer):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.time_embed = TimeEmbed(V, D)
        self.time_lstm = TimeLSTM(H)
        self.time_affine = TimeAffine(V)

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape
        self.time_lstm.lstm.h = h
        xs_embedded = self.time_embed(xs)
        h1 = F.unsqueeze(h, 1)
        h2 = F.broadcast_to(h1, (N, T, H))
        xs_emb_h = F.concat(xs_embedded, h2, axis=2)
        hs = self.time_lstm(xs_emb_h)
        hs_h = F.concat(hs, h2, axis=2)
        out = self.time_affine(hs_h)
        return out

    def generate(self, h, start_id, sample_size):
        sampled = []
        sampled_id = start_id
        self.time_lstm.lstm.h = h
        for _ in range(sample_size):
            x = np.array([[sampled_id]])
            out = self.time_embed(x)
            h1 = F.unsqueeze(h, 1)
            out_h = F.concat(out, h1, axis=2)
            hs = self.time_lstm(out_h)
            hs_h = F.concat(hs, h1, axis=2)
            score = self.time_affine(hs_h)
            sampled_id = np.argmax(score.data.flatten())
            sampled.append(sampled_id)
        return sampled


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, vocab_size, wordvec_size):
        super().__init__()
        self.W = Parameter(np.random.randn(vocab_size, wordvec_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)


class LayerNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        N = x.shape[0]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros((N, 1), dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones((N, 1), dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones((N, 1), dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros((N, 1), dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.layer_norm(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)


class SelfAttentionSimple(Layer):
    def __init__(self, wordvec_size=512, key_size=64):
        super().__init__()
        self.key_size = key_size
        self.attention_query = Linear(key_size, nobias=True)
        self.attention_key = Linear(key_size, nobias=True)
        self.attention_val = Linear(key_size, nobias=True)

    def forward(self, x):
        q = self.attention_query(x)
        k = self.attention_key(x)
        v = self.attention_val(x)
        z = F.softmax((F.sum(F.linear(q, k.T), axis=1)).reshape(-1, 1) / np.sqrt(self.key_size)) * v
        return z


class SelfAttention(Layer):
    def __init__(self, key_size=64):
        super().__init__()
        self.key_size = key_size
        self.multi_head_count = 8

        self.attention_query_layers = []
        self.attention_key_layers = []
        self.attention_val_layers = []
        for i in range(self.multi_head_count):
            setattr(self, f"attention_query_layer{i + 1}", Linear(key_size, nobias=True, param_name=f"Wq{i + 1}"))
            self.attention_query_layers.append(getattr(self, f"attention_query_layer{i + 1}"))

            setattr(self, f"attention_key_layer{i + 1}", Linear(key_size, nobias=True, param_name=f"Wk{i + 1}"))
            self.attention_key_layers.append(getattr(self, f"attention_key_layer{i + 1}"))

            setattr(self, f"attention_val_layer{i + 1}", Linear(key_size, nobias=True, param_name=f"Wv{i + 1}"))
            self.attention_val_layers.append(getattr(self, f"attention_val_layer{i + 1}"))

        # self.attention_query_layers = [Linear(key_size, nobias=True, param_name=f"Wq{i + 1}") for i in range(8)]
        # self.attention_key_layers = [Linear(key_size, nobias=True, param_name=f"Wk{i + 1}") for i in range(8)]
        # self.attention_val_layers = [Linear(key_size, nobias=True, param_name=f"Wv{i + 1}") for i in range(8)]
        # self.out_linear_layer = Linear(key_size, param_name="Wo")
        self.out_linear_layer = None

    def _init_out_linear_layer(self, out_size):
        self.out_linear_layer = Linear(out_size, param_name="Wo")

    def forward(self, x):
        if not self.out_linear_layer:
            self._init_out_linear_layer(x.shape[1])
        z_list = []
        for query_layer, key_layer, val_layer in zip(self.attention_query_layers, self.attention_key_layers,
                                                     self.attention_val_layers):
            q = query_layer(x)
            k = key_layer(x)
            v = val_layer(x)
            z = F.softmax((F.sum(F.linear(q, k.T), axis=1)).reshape(-1, 1) / np.sqrt(self.key_size)) * v
            z_list.append(z)
        # we want to horizontally stack z_list
        z = F.hstack(*z_list)
        out = self.out_linear_layer(z)
        return out


class Encoder(Layer):
    def __init__(self, key_size=64, out_size=512):
        super().__init__()
        self.key_size = key_size
        self.self_attention_layer = SelfAttention(self.key_size)
        self.fc = Linear(out_size, param_name="encoder_fc_W")

    def forward(self, x):
        h = self.self_attention_layer(x)
        h = LayerNorm()(x + h)
        y = F.relu(self.fc(h))
        y = LayerNorm()(h + y)
        return y


class PositionalEncoder(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        xp = dezero.cuda.get_array_module(x)
        PE_mat = np.zeros(x.shape)
        time_size, wordvec_size = x.shape
        for pos in range(time_size):
            for wv in range(wordvec_size):
                if wv % 2 == 0:
                    pos_encode = xp.sin(pos / 10000 ** (2 * wv / wordvec_size))
                else:
                    pos_encode = xp.cos(pos / 10000 ** (2 * wv / wordvec_size))
                PE_mat[pos, wv] = pos_encode
        return PE_mat
