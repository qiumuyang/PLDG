import copy

import torch
import torch.nn as nn


class EMA(nn.Module):

    def __init__(self, model, decay=0.99):
        super().__init__()
        self.decay = decay

        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_param, model_param in zip(self.ema_model.parameters(),
                                          model.parameters()):
            ema_param.copy_(ema_param * self.decay + model_param *
                            (1 - self.decay))

        for ema_buffer, model_buffer in zip(self.ema_model.buffers(),
                                            model.buffers()):
            ema_buffer.copy_(model_buffer)

    def forward(self, x):
        return self.ema_model(x)
