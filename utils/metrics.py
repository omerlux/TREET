import numpy as np
import torch
import torch.nn as nn


class DV_Loss(nn.Module):
    def __init__(self, clip_th, alpha_dv_reg=0, logger=print):
        super(DV_Loss, self).__init__()
        self.clip_th = clip_th
        self.alpha_dv_reg = alpha_dv_reg
        self.C = 0
        self.logger = logger
        self.logger('| DV_Loss: clip_th = ±{}'.format(clip_th))
        self.logger('| DV_Loss: alpha_dv_reg = {}'.format(alpha_dv_reg))

    def forward(self, outputs: [torch.Tensor, torch.Tensor]) -> torch.Tensor:
        t_pred, t_tilde_pred = outputs
        t_tilde_pred = torch.clip(t_tilde_pred, -self.clip_th, self.clip_th)

        t_mean = torch.mean(t_pred)
        t_log_mean_exp = torch.log(torch.mean(torch.exp(t_tilde_pred)))
        if self.alpha_dv_reg:
            reg = self.alpha_dv_reg * (t_log_mean_exp - self.C) ** 2
        else:
            reg = 0
        loss = t_mean - t_log_mean_exp - reg
        return -1 * loss        # for maximization
