import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = self._acquire_device()
        self.models = self._build_model()
        for model in self.models.values():
            model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('| Use GPU')
        else:
            device = torch.device('cpu')
            self.logger.info('| Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def reset_states(self, mode='train'):
        for model in self.models.values():
            if mode == 'train':
                model.train()
            elif mode == 'eval':
                model.eval()
            if hasattr(model, 'erase_states'):
                model.erase_states()
        if hasattr(self, 'channel'):
            self.channel.erase_states()
