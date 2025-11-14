import math
import torch
import torch.nn as nn
from typing import Optional, Any

#try:
#    import xformers
#    import xformers.ops

#    XFORMERS_IS_AVAILBLE = True
#except:
XFORMERS_IS_AVAILBLE = False


def normalization(norm_type, out_channels, group_norm_num=None):
    assert norm_type.lower() in ["groupnorm", "batchnorm2d"], "Invalid norm_type."

    if norm_type.lower() == "groupnorm":
        assert group_norm_num is not None, "Group norm num invalid."
        return nn.GroupNorm(group_norm_num, out_channels)

    elif norm_type.lower() == "batchnorm2d":
        return nn.BatchNorm2d(out_channels)

    else:
        raise ValueError("Unexpected error in normalization selection.")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, use_xformer=False):
        super().__init__()
        self.n_heads = n_heads
        self.xformer = XFORMERS_IS_AVAILBLE and use_xformer
        if self.xformer:
            print("Setting up Multiheaded attention with XFORMERS library")
        self.attention_op: Optional[Any] = None

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if self.xformer:
            a = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
        else:
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = torch.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)
