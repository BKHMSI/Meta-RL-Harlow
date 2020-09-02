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

import numpy as np

import torch as T
from torch import nn
from torch.nn import functional as F

from torch import Tensor
from models.rgu_cell import ReciprocallyGated_Cell

@dataclass
class ReciprocallyGated_Cell_Builder:
    # input_size: int
    hidden_size: int
    vertical_dropout            : float = 0.0
    recurrent_dropout           : float = 0.0
    input_kernel_initialization : str   = 'xavier_uniform'
    gate_activation             : str   = 'sigmoid'
    activation                  : str   = 'tanh'

    def make(self, input_size: int):
        return ReciprocallyGated_Cell(input_size, self)

class RGU_Layer(nn.Module):
    def __init__(
            self,
            cell,
            direction='forward',
            batch_first=False,
    ):
        super().__init__()
        if isinstance(batch_first, bool):
            batch_first = (batch_first, batch_first)
        self.batch_first = batch_first
        self.direction = direction
        self.cell_ = cell

    @T.jit.ignore
    def forward(self, input, state_t0, return_state=None):
        if self.batch_first[0]:
        #^ input : [b t i]
            input = input.transpose(1, 0)
        #^ input : [t b i]
        inputs = input.unbind(0)

        if state_t0 is None:
            state_t0 = self.cell_.get_init_state(input)
    
        sequence, state = self.cell_.loop(inputs, state_t0)
       
        #^ sequence : t * [b h]
        sequence = T.stack(sequence)
        #^ sequence : [t b h]

        if self.batch_first[1]:
            sequence = sequence.transpose(1, 0)
        #^ sequence : [b t h]  

        return sequence, state
   
class RGUnit(nn.Module):
    def __init__(
            self,
            input_size    : int,
            num_layers    : int,
            batch_first   : bool = False,
            scripted      : bool = True,
            *args, **kargs,
    ):
        super().__init__()
        self._cell_builder = ReciprocallyGated_Cell_Builder(*args, **kargs)
    
        Dh = self._cell_builder.hidden_size
        def make(isize: int):
            cell = self._cell_builder.make(isize)
            return RGU_Layer(cell, isize, batch_first=batch_first)

        rnns = [
            make(input_size),
            *[
                make(Dh)
                for _ in range(num_layers - 1)
            ],
        ]

        self.rnn = nn.Sequential(*rnns)

        self.input_size = input_size
        self.hidden_size = self._cell_builder.hidden_size
        self.num_layers = num_layers

    def __repr__(self):
        return (
            f'${self.__class__.__name__}'
            + '('
            + f'in={self.input_size}, '
            + f'hid={self.hidden_size}, '
            + f'layers={self.num_layers}, '
            + f'bi={self.bidirectional}'
            + '; '
            + str(self._cell_builder)
        )

    def forward(self, inputs, state_t0=None):
        for rnn in self.rnn:
            inputs, state = rnn(inputs, state_t0) 
        return inputs, state 

    def reset_parameters(self):
        for rnn in self.rnn:
            rnn.cell_.reset_parameters_()

if __name__ == "__main__":
    
    rgu = RGUnit(
        input_size=128,
        hidden_size=256,
        num_layers=1,
        batch_first=False
    )

    x_t = T.rand(1, 16, 128)
    state_0 = (T.zeros(1, 256), T.zeros(1, 256))

    h_t, state_t = rgu(x_t, state_0)
    import pdb; pdb.set_trace()