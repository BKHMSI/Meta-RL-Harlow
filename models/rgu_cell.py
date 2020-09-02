from typing import (
    Tuple,
    List,
    Optional,
    Dict,
    Callable,
    Union,
    cast,
)
from collections import namedtuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch as T
from torch import nn
from torch.nn import functional as F

from torch import Tensor

import pdb

__all__ = [
    'RGUnit',
    'ReciprocallyGated_Cell',
    'ReciprocallyGated_Cell_Builder',
]

GateSpans = namedtuple('GateSpans', ['gh', 'gc'])

ACTIVATIONS = {
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'hard_tanh': nn.Hardtanh(),
    'relu': nn.ReLU(),
}

class ReciprocallyGated_Cell(nn.Module):
    '''
    Adapted from:
    https://papers.nips.cc/paper/7775-task-driven-convolutional-recurrent-models-of-the-visual-system
    arxiv:1807.00053
    with modifications.
    '''
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            + ', '.join(
                [
                    f'in: {self.Dx}',
                    f'hid: {self.Dh}',
                    f'rdo: {self.recurrent_dropout_p}',
                    f'vdo: {self.vertical_dropout_p}',
                ]
            )
            +')'
        )

    def __init__(
            self,
            input_size: int,
            args,
    ):
        super().__init__()
        self._args = args
        self.Dx = input_size
        self.Dh = args.hidden_size
        # self.recurrent_kernel   = nn.Linear(self.Dh, self.Dh * 2)
        # self.cell_memory_kernel = nn.Linear(self.Dh, self.Dh * 2)
        # self.input_kernel       = nn.Linear(self.Dx, self.Dh * 2)

        self.recurrent_kernel = nn.Conv1d(
            in_channels=self.Dh, 
            out_channels=2,
            kernel_size=3,
            stride=1
        )

        self.cell_memory_kernel = nn.Conv1d(
            in_channels=self.Dh, 
            out_channels=2,
            kernel_size=3,
            stride=1
        )

        self.input_kernel = nn.Conv1d(
            in_channels=self.Dh, 
            out_channels=2,
            kernel_size=3,
            stride=1
        )


        self.recurrent_dropout_p = args.recurrent_dropout or 0.0
        self.vertical_dropout_p  = args.vertical_dropout or 0.0

        self.recurrent_dropout = nn.Dropout(self.recurrent_dropout_p)
        self.vertical_dropout  = nn.Dropout(self.vertical_dropout_p)

        self.fun_gate = ACTIVATIONS[args.gate_activation]
        self.fun_main = ACTIVATIONS[args.activation]

        self.reset_parameters_()

    # @T.jit.ignore
    def get_recurrent_weights(self):
        # type: () -> Tuple[GateSpans, GateSpans]
        W = self.recurrent_kernel.weight.chunk(2, 0)
        b = self.recurrent_kernel.bias.chunk(2, 0)
        W = GateSpans(W[0], W[1])
        b = GateSpans(b[0], b[1])
        return W, b

    # @T.jit.ignore
    def get_cell_memory_weights(self):
        # type: () -> Tuple[GateSpans, GateSpans]
        W = self.cell_memory_kernel.weight.chunk(2, 0)
        b = self.cell_memory_kernel.bias.chunk(2, 0)
        W = GateSpans(W[0], W[1])
        b = GateSpans(b[0], b[1])
        return W, b

    @T.jit.ignore
    def reset_parameters_(self):
        rw, rb = self.get_recurrent_weights()
        iw, ib = self.get_cell_memory_weights()

        nn.init.zeros_(self.cell_memory_kernel.bias)
        nn.init.constant_(self.recurrent_kernel.bias, 0.5)

        for W in rw:
            nn.init.orthogonal_(W)
        for W in iw:
            nn.init.xavier_uniform_(W)

    @T.jit.export
    def get_init_state(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = input.shape[1]
        zeros = T.zeros(batch_size, self.Dh, device=input.device)
        return (zeros, zeros)

    def apply_input_kernel(self, xt: Tensor) -> List[Tensor]:
        #^ xt : [b h]
        xto = self.vertical_dropout(xt)
        out = self.input_kernel(xto).chunk(2, 1)
        #^ out : [b h]
        return out

    def apply_recurrent_kernel(self, h_tm1: Tensor) -> List[Tensor]:
        #^ h_tm1 : [b h]
        hto = self.recurrent_dropout(h_tm1)
        out = self.recurrent_kernel(hto).chunk(2, 1)
        #^ out : [b h]
        return out

    def apply_cell_memory_kernel(self, c_tm1: Tensor) -> List[Tensor]:
        out = self.cell_memory_kernel(c_tm1).chunk(2, 1)
        return out

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        #^ input : [b i]
        #^ state.h : [b h]

        h_tm1, c_tm1 = state

        Cc, Ch = self.apply_cell_memory_kernel(c_tm1)
        Hc, Hh = self.apply_recurrent_kernel(h_tm1)
        Xc, Xh = self.apply_input_kernel(input)


        gh = (1 - self.fun_gate(Ch)) * Xh + (1 - self.fun_gate(Hh)) * h_tm1
        gc = (1 - self.fun_gate(Hc)) * Xc + (1 - self.fun_gate(Cc)) * c_tm1
        
        ht = self.fun_main(gh)
        ct = self.fun_main(gc)

        return ht, (ht, ct)

    @T.jit.export
    def loop(self, inputs, state_t0, mask=None):
        # type: (List[Tensor], Tuple[Tensor, Tensor], Optional[List[Tensor]]) -> List[Tensor]
        '''
        This loops over t (time) steps
        '''
        #^ inputs      : t * [b i]
        #^ state_t0[i] : [b s]
        #^ out         : [t b h]
        state = state_t0
        outs = []
        for xt in inputs:
            ht, state = self(xt, state)
            outs.append(ht)

        return outs, state