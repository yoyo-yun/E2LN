import torch
import numbers
import torch.nn as nn
import torch.nn.init as init
from typing import Union, List
from torch import Tensor, Size
from torch.nn.parameter import Parameter


class SelfAttention(nn.Module):
    def __init__(self, input_dim, config):
        super(SelfAttention, self).__init__()
        self.pre_pooling_linear = nn.Linear(input_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, x, mask=None):
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear(x))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)
        weights = self.dropout(weights)
        return torch.mul(x, weights.unsqueeze(2)).sum(dim=1)
        # return weights


class Bilinear(nn.Module):
    def __init__(self, input1_dim, input2_dim):
        super(Bilinear, self).__init__()
        self.bilinear_weights = nn.Parameter(torch.rand(input1_dim, input2_dim))
        self.bilinear_weights.data.uniform_(-0.25, 0.25)

    def forward(self, input_1, input2):
        x = torch.matmul(input_1, self.bilinear_weights)
        if len(x.size()) != len(input2.size()):
            x = x.unsqueeze(1)
        return torch.tanh(torch.mul(x, input2))


class BilinearAttention(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(BilinearAttention, self).__init__()
        self.bilinear = Bilinear(input1_dim, input2_dim)
        self.att = SelfAttention(input2_dim, config)

    def forward(self, input_1, input_2, mask=None):
        # input_1: usr or prd representation, input_2: text representation
        b_x = self.bilinear(input_1, input_2)
        att = self.att(b_x, mask=mask)
        output = torch.mul(input_2, att.unsqueeze(2)).sum(dim=1) + torch.mul(b_x, att.unsqueeze(2)).sum(dim=1)
        return output


class UoPAttention(nn.Module):
    def __init__(self, input1_dim, input2_dim, config):
        super(UoPAttention, self).__init__()
        self.pre_pooling_linear_attr = nn.Linear(input1_dim, config.pre_pooling_dim)
        self.pre_pooling_linear_text = nn.Linear(input2_dim, config.pre_pooling_dim)
        self.pooling_linear = nn.Linear(config.pre_pooling_dim, 1)

    def forward(self, input_1, input_2, mask=None):
        weights = self.pooling_linear(torch.tanh(self.pre_pooling_linear_attr(input_1).unsqueeze(1) + self.pre_pooling_linear_text(input_2))).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        weights = nn.Softmax(dim=-1)(weights)

        return torch.mul(input_2, weights.unsqueeze(2)).sum(dim=1)


_shape_t = Union[int, List[int], Size]


class LN_based_attention(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True, pooling=False):
        super(LN_based_attention, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.pooling = pooling
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, mask=None):
        value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.pooling:
            input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)
            score = torch.matmul(input_, self.weight) # (bs, seq_length)
            if mask is not None:
                score = score.masked_fill(mask == 0, -1e9)
            weigted_score = torch.softmax(score, dim=1)
            output = torch.einsum("bld,bl->bd", value, weigted_score)
            return output
        else:
            return value

class LN_based_attention_(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LN_based_attention_, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, mask=None):
        value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)
        score = torch.matmul(input_, self.weight) # (bs, seq_length)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weigted_score = torch.softmax(score, dim=1)
        output = torch.einsum("bld,bl->bd", value, weigted_score)
        return value, output

class LN_based_attention_UP(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True, up: bool = True):
        super(LN_based_attention_UP, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.up = up
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, up, mask=None):
        if self.up:
            value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps).add(up.unsqueeze(1))
        else:
            value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)  # (bs, seq_length, dim)
        score = torch.einsum("bld, d->bl", input_, self.weight)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weighted_score = torch.softmax(score, dim=1)
        output = torch.einsum("bld,bl->bd", value, weighted_score)
        return output

class LN_based_attention_UP_NSC(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LN_based_attention_UP_NSC, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.att_w = nn.Linear(*normalized_shape, *normalized_shape)
        self.att_v = nn.Linear(*normalized_shape, 1)
        self.weight_ = Parameter(torch.Tensor(*normalized_shape))
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.ones_(self.weight_)
            init.zeros_(self.bias)

    def forward(self, input, up_weight, up_bias, mask=None):
        input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps) # (bs, seq_length, dim)
        value = input_.mul(up_weight.unsqueeze(1)).add(up_bias.unsqueeze(1))
        # (bs, seq_length, dim) * (bs, 1, dim) -> (bs, seq_length)
        score = torch.einsum("bld, bd->bl", input_, up_weight)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weigted_score = torch.softmax(score, dim=1)
        output = torch.einsum("bld,bl->bd", value, weigted_score)
        return output


class LN_based_attention_UP_Linear(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LN_based_attention_UP_Linear, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.out_linear = nn.Linear(normalized_shape[0]*2, *normalized_shape)
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, up_weight, up_bias, mask=None):
        # input
        input_common = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)
        value_common = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        score_common = torch.einsum("bld, d->bl", input_common, self.weight)

        # value
        value_special = input_common.mul(1+up_weight.unsqueeze(1)).add(up_bias.unsqueeze(1))
        score_special = torch.einsum("bld, bd->bl", input_common, 1+up_weight)

        # mask
        if mask is not None:
            score_common = score_common.masked_fill(mask == 0, -1e9)
            score_special = score_special.masked_fill(mask == 0, -1e9)
        weighted_score_common = torch.softmax(score_common, dim=-1)
        weighted_score_special = torch.softmax(score_special, dim=-1)

        # integration
        output_common = torch.einsum("bld,bl->bd", value_common, weighted_score_common)
        output_special = torch.einsum("bld,bl->bd", value_special, weighted_score_special)
        output = torch.tanh(self.out_linear(torch.cat((output_common, output_special), dim=-1)))
        return output


class LN_in_Transformer(nn.Module):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True, up: bool = True):
        super(LN_in_Transformer, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.up = up
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, up_scale=None, up_bias=None, mask=None, window_size=None):
        # if self.up and any((up_scale, up_bias)) is not None:
        if self.up:
            value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
            if up_scale is not None and up_bias is not None:
                value = value.mul(1+up_scale).add(up_bias)
            return value, None
        else:
            if window_size is None:
                return torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps), None
            else:
                size = input.size()
                value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps) # (bs, seq_length, dim)
                input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)  # (bs, seq_length, dim)
                score = torch.einsum("bld, d->bl", input_, self.weight)  # (bs, seq_length)
                if mask is not None:
                    if mask[0][0] == 0.:  # mask in transformer
                        score = score + mask
                    else:
                        score = score.masked_fill(mask == 0, -1e9)
                chunk_score = score.reshape(size[0], size[1] // window_size, window_size)
                weighted_score = torch.softmax(chunk_score, dim=2)  # (bs, chunk, window)
                chunk_value = value.reshape(size[0], size[1] // window_size, window_size, size[2]) # (bs, chunk, window, dim)
                memory_output = torch.einsum("bcw, bcwd->bcd", weighted_score, chunk_value)

                return value, memory_output  # (bs, seq, dim), (bs, chunk, dim)


# # for pooling methods
# class LN_in_Transformer(nn.Module):
#     def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True, up: bool = True):
#         super(LN_in_Transformer, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         self.normalized_shape = tuple(normalized_shape)
#         self.eps = eps
#         self.up = up
#         self.elementwise_affine = elementwise_affine
#         if self.elementwise_affine:
#             self.weight = Parameter(torch.Tensor(*normalized_shape))
#             self.bias = Parameter(torch.Tensor(*normalized_shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#         # for la
#         self.pre_pooling_linear = nn.Linear(normalized_shape[-1], 50)
#         self.pooling_linear = nn.Linear(50, 1)
#
#     def reset_parameters(self) -> None:
#         if self.elementwise_affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)
#
#     def forward(self, input, up_scale=None, up_bias=None, mask=None, window_size=None):
#         # if self.up and any((up_scale, up_bias)) is not None:
#         if self.up:
#             value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
#             if up_scale is not None and up_bias is not None:
#                 value = value.mul(1+up_scale).add(up_bias)
#             return value, None
#         else:
#             if window_size is None:
#                 return torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps), None
#             else:
#                 # lnap
#                 # size = input.size()
#                 # value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps) # (bs, seq_length, dim)
#                 # input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)  # (bs, seq_length, dim)
#                 # score = torch.einsum("bld, d->bl", input_, self.weight)  # (bs, seq_length)
#                 # if mask is not None:
#                 #     if mask[0][0] == 0.:  # mask in transformer
#                 #         score = score + mask
#                 #     else:
#                 #         score = score.masked_fill(mask == 0, -1e9)
#                 # chunk_score = score.reshape(size[0], size[1] // window_size, window_size)
#                 # weighted_score = torch.softmax(chunk_score, dim=2)  # (bs, chunk, window)
#                 # chunk_value = value.reshape(size[0], size[1] // window_size, window_size, size[2]) # (bs, chunk, window, dim)
#                 # memory_output = torch.einsum("bcw, bcwd->bcd", weighted_score, chunk_value)
#                 #
#                 # return value, memory_output  # (bs, seq, dim), (bs, chunk, dim)
#
#                 # max
#                 # size = input.size()
#                 # value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias,
#                 #                          self.eps)  # (bs, seq_length, dim)
#                 # chunk_value = value.reshape(size[0], size[1] // window_size, window_size,
#                 #                             size[2])  # (bs, chunk, window, dim)
#                 # memory_output, _ = torch.max(chunk_value, dim=2)
#                 # return value, memory_output
#
#                 # avg
#                 # size = input.size()
#                 # value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias,
#                 #                          self.eps)  # (bs, seq_length, dim)
#                 # chunk_value = value.reshape(size[0], size[1] // window_size, window_size,
#                 #                             size[2])  # (bs, chunk, window, dim)
#                 # memory_output = torch.mean(chunk_value, dim=2)
#                 # return value, memory_output
#
#                 # la
#                 size = input.size()
#                 value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps) # (bs, seq_length, dim)
#                 score = self.pooling_linear(torch.tanh(self.pre_pooling_linear(input))).squeeze(dim=2) # (bs, seq_length)
#                 if mask is not None:
#                     if mask[0][0] == 0.:  # mask in transformer
#                         score = score + mask
#                     else:
#                         score = score.masked_fill(mask == 0, -1e9)
#                 chunk_score = score.reshape(size[0], size[1] // window_size, window_size)
#                 weighted_score = torch.softmax(chunk_score, dim=2)  # (bs, chunk, window)
#                 chunk_value = value.reshape(size[0], size[1] // window_size, window_size, size[2]) # (bs, chunk, window, dim)
#                 memory_output = torch.einsum("bcw, bcwd->bcd", weighted_score, chunk_value)
#                 return value, memory_output  # (bs, seq, dim), (bs, chunk, dim)

# for places of ln
# class LN_in_Transformer(nn.Module):
#     def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True, up: bool = True):
#         super(LN_in_Transformer, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         self.normalized_shape = tuple(normalized_shape)
#         self.eps = eps
#         self.up = up
#         self.elementwise_affine = elementwise_affine
#         if self.elementwise_affine:
#             self.weight = Parameter(torch.Tensor(*normalized_shape))
#             self.bias = Parameter(torch.Tensor(*normalized_shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#         # for la
#         self.pre_pooling_linear = nn.Linear(normalized_shape[-1], 50)
#         self.pooling_linear = nn.Linear(50, 1)
#
#     def reset_parameters(self) -> None:
#         if self.elementwise_affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)
#
#     def forward(self, input, up_scale=None, up_bias=None, mask=None, window_size=None):
#         value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
#         memory_output = None
#
#         if (up_scale is not None and up_bias is not None) and window_size is None:
#             value = value.mul(1 + up_scale).add(up_bias)
#             return value, memory_output
#
#         if (up_scale is None and up_bias is None) and window_size is None:
#             return value, memory_output
#
#         if (up_scale is not None and up_bias is not None) and window_size is not None:
#             size = input.size()
#             value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias,
#                                      self.eps).mul(1 + up_scale).add(up_bias)  # (bs, seq_length, dim)
#             input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)  # (bs, seq_length, dim)
#             weight = self.weight.unsqueeze(0).mul(1 + up_scale).squeeze(1)
#             score = torch.einsum("bld, bd->bl", input_, weight)  # (bs, seq_length)
#             if mask is not None:
#                 if mask[0][0] == 0.:  # mask in transformer
#                     score = score + mask
#                 else:
#                     score = score.masked_fill(mask == 0, -1e9)
#             chunk_score = score.reshape(size[0], size[1] // window_size, window_size)
#             weighted_score = torch.softmax(chunk_score, dim=2)  # (bs, chunk, window)
#             chunk_value = value.reshape(size[0], size[1] // window_size, window_size,
#                                         size[2])  # (bs, chunk, window, dim)
#             memory_output = torch.einsum("bcw, bcwd->bcd", weighted_score, chunk_value)
#
#             return value, memory_output  # (bs, seq, dim), (bs, chunk, dim)
#
#         if (up_scale is None and up_bias is None) and window_size is not None:
#             size = input.size()
#             value = torch.layer_norm(input, self.normalized_shape, self.weight, self.bias,
#                                      self.eps)  # (bs, seq_length, dim)
#             input_ = torch.layer_norm(input, self.normalized_shape, None, None, self.eps)  # (bs, seq_length, dim)
#             score = torch.einsum("bld, d->bl", input_, self.weight)  # (bs, seq_length)
#             if mask is not None:
#                 if mask[0][0] == 0.:  # mask in transformer
#                     score = score + mask
#                 else:
#                     score = score.masked_fill(mask == 0, -1e9)
#             chunk_score = score.reshape(size[0], size[1] // window_size, window_size)
#             weighted_score = torch.softmax(chunk_score, dim=2)  # (bs, chunk, window)
#             chunk_value = value.reshape(size[0], size[1] // window_size, window_size,
#                                         size[2])  # (bs, chunk, window, dim)
#             memory_output = torch.einsum("bcw, bcwd->bcd", weighted_score, chunk_value)
#
#             return value, memory_output  # (bs, seq, dim), (bs, chunk, dim)


if __name__ == '__main__':
    # print("====testing bilinear modual...")
    # bilinear = Bilinear(100, 32)
    # a = torch.randn(64, 100)  # (bs, dim1)
    # b = torch.randn(64, 15, 32)  # (bs, seq, dim2)
    # output = bilinear(a,b)
    # print(output.shape)
    #
    # c = torch.randn(64, 15, 100) # (bs, seq, dim1)
    # d = torch.randn(64, 15, 32)  # (bs, seq, dim2)
    # output1 = bilinear(c,d)
    # print(output1.shape)
    #
    # e = torch.randn(64, 100) # (bs, dim1)
    # f = torch.randn(64, 32)  # (bs, dim2)
    # output2 = bilinear(e,f)
    # print(output2.shape)
    # print("done!")
    # print()
    #
    # print("====testing bilinear attention modual...")
    # from easydict import EasyDict as edict
    # config = edict()
    # config.pre_pooling_dim = 50
    # bilinear_att = BilinearAttention(100, 32, config)
    # a = torch.randn(64, 100)  # (bs, dim1)
    # b = torch.randn(64, 15, 32)  # (bs, seq, dim2)
    # output_att = bilinear_att(a,b)
    # print(output_att.shape)
    # print("done!")
    # print()

    print("====testing LN based attention modual...")
    from easydict import EasyDict as edict
    config = edict()
    config.pre_pooling_dim = 50
    lnb_att = LN_based_attention([100])
    a = torch.randn(64, 128, 100)  # (bs, dim1)
    output_att = lnb_att(a)
    print(output_att.shape)
    print("done!")
    print()

