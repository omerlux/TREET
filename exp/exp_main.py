from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LSTM , Decoder_Model
from utils.tools import EarlyStopping, adjust_learning_rate, visual, to_devices
from utils.metrics import DV_Loss

import os
import time
import torch
import warnings
import numpy as np
import wandb
from torch.nn import MSELoss
from torch import optim

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args, logger):
        super(Exp_Main, self).__init__(args, logger)

    def _build_model(self):
        model_dict = {
            'LSTM': LSTM,
            'Decoder_Model': Decoder_Model,
        }
        self.args.x_in = False
        model_y = model_dict[self.args.model].Model(self.args).float()
        self.args.x_in = True
        model_xy = model_dict[self.args.model].Model(self.args).float()

        models = to_devices({'y': model_y, 'xy': model_xy}, self.args)

        total_params = sum(x.data.nelement() for x in models['y'].parameters())
        self.logger.info('| >>> Model Y total parameters: {:,}'.format(total_params))
        total_params = sum(x.data.nelement() for x in models['xy'].parameters())
        self.logger.info('| >>> Model XY total parameters: {:,}'.format(total_params))

        models_y_args = [(k, v) for k, v in sorted(models['y'].__dict__.items()) if isinstance(v, (int, float, str, list, dict))
                         if k[0] != '_']
        models_xy_args = [(k, v) for k, v in sorted(models['xy'].__dict__.items()) if isinstance(v, (int, float, str, list, dict))
                          if k[0] != '_']
        self.logger.info("| >>> {} Model Y's Arguments: {}".format(self.args.model, models_y_args))
        self.logger.info("| >>> {} Model XY's Arguments: {}".format(self.args.model, models_xy_args))

        return models

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.logger)
        if hasattr(data_set, 'capacity_gt'):
            self.logger.info('| >>> Max Capacity: {:.5f}'.format(data_set.capacity_gt))
            if self.args.wandb:
                wandb.log({"capacity_gt": data_set.capacity_gt})
        return data_set, data_loader

    def _select_optimizer(self, optim_name):
        models_optim = {}
        for key in self.models.keys():
            if optim_name == 'rmsprop':
                models_optim[key] = optim.RMSprop(self.models[key].parameters(), lr=self.args.learning_rate)
            else:
                models_optim[key] = optim.Adam(self.models[key].parameters(), lr=self.args.learning_rate)
        return models_optim

    def _select_criterion(self, loss):
        loss_function = {
            'dv': DV_Loss(self.args.exp_clipping, self.args.alpha_dv_reg, logger=self.logger.info),
            'mse': MSELoss(),
        }
        criterion = loss_function[loss]
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        metrics = {'y': [], 'xy': []}
        self.reset_states(mode='eval')

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                if self.args.process_info['memory_cut']:
                    self.reset_states()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_y_tilde = draw_y(batch_y, vali_data.min_max)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = {'y': self.models['y'](batch_y, y_tilde=vali_data.min_max)[:2],
                               'xy': self.models['xy'](batch_y, y_tilde=vali_data.min_max, x=batch_x)[:2]}
                else:
                    outputs = {'y': self.models['y'](batch_y, y_tilde=vali_data.min_max),
                               'xy': self.models['xy'](batch_y, y_tilde=vali_data.min_max, x=batch_x)}

                loss = {'y': criterion(outputs['y'][:2]),
                        'xy': criterion(outputs['xy'][:2])}

                metrics['y'].append(-loss['y'].item())
                metrics['xy'].append(-loss['xy'].item())

        metrics = {k: np.array(metrics[k]).mean() for k in metrics.keys()}
        for k in self.models.keys():
            self.models[k].train()
        return metrics

    def train(self, setting):
        self.logger.info("| Loading data '{}'".format(self.args.data))
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save, setting, self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        models_optim = self._select_optimizer(self.args.optimizer)
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.logger.info('-' * 80)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            metrics_train = {'y': [], 'xy': []}

            self.reset_states(mode='train')

            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if self.args.process_info['memory_cut']:
                    self.reset_states()

                iter_count += 1
                for model_optim in models_optim.values():
                    model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_y_tilde = draw_y(batch_y, train_data.min_max)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = {'y': self.models['y'](batch_y, y_tilde=train_data.min_max)[:2],
                               'xy': self.models['xy'](batch_y, y_tilde=train_data.min_max, x=batch_x)[:2]}
                else:
                    outputs = {'y': self.models['y'](batch_y, y_tilde=train_data.min_max),
                               'xy': self.models['xy'](batch_y, y_tilde=train_data.min_max, x=batch_x)}

                loss = {'y': criterion(outputs['y'][:2]),
                        'xy': criterion(outputs['xy'][:2])}

                metrics_train['y'].append(-loss['y'].item())
                metrics_train['xy'].append(-loss['xy'].item())

                if (i + 1) % self.args.log_interval == 0:
                    metrics_train_y = np.array(metrics_train['y'][-self.args.log_interval:]).mean(axis=0)
                    metrics_train_xy = np.array(metrics_train['xy'][-self.args.log_interval:]).mean(axis=0)
                    self.logger.info(
                        "|\t\t iters: {}, epoch: {} | DV loss: {:.5f}, Y net loss: {:.5f}, XY net loss: {:.5f}, "
                        .format(i + 1, epoch + 1, metrics_train_xy - metrics_train_y, metrics_train_y,
                                metrics_train_xy,))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)        # change train_steps to 10000
                    self.logger.info('|\t\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    if self.args.wandb:
                        wandb_dictionary = {"train_dv_loss": metrics_train_xy - metrics_train_y,
                                            "train_y_net_loss": metrics_train_y,
                                            "train_xy_net_loss": metrics_train_xy}
                        if hasattr(train_data, 'capacity_gt'):
                            wandb_dictionary['train_dv_diff'] = np.abs((metrics_train_xy - metrics_train_y) - train_data.capacity_gt)
                            wandb_dictionary['train_dv_diff_rate'] = np.abs(1 - (metrics_train_xy - metrics_train_y) / train_data.capacity_gt)
                        wandb.log(wandb_dictionary)

                if self.args.use_amp:
                    loss = loss['xy'] + loss['y']
                    scaler.scale(loss).backward()
                    # scaler.scale(loss['xy']).backward()
                    # scaler.scale(loss['y']).backward()
                    scaler.step(models_optim['y'])
                    scaler.step(models_optim['xy'])
                    scaler.update()
                else:
                    loss = loss['xy'] + loss['y']
                    loss.backward()
                    # loss['xy'].backward()
                    # loss['y'].backward()
                    models_optim['y'].step()
                    models_optim['xy'].step()

            self.logger.info("| Epoch: {}, cost time: {:4.3f}, Y learning rate: {:2.2e}, XY learning rate: {:2.2e}"
                             .format(epoch + 1, time.time() - epoch_time, models_optim['y'].param_groups[0]['lr'],
                                     models_optim['xy'].param_groups[0]['lr']))
            metrics_train = {k: np.array(metrics_train[k]).mean() for k in metrics_train.keys()}
            metrics_valid = self.vali(vali_data, vali_loader, criterion)
            # metrics_test = self.vali(test_data, test_loader, criterion)

            if self.args.wandb:
                wandb_dictionary = {"valid_dv_loss": metrics_valid['xy'] - metrics_valid['y'],
                                    "valid_y_net_loss": metrics_valid['y'],
                                    "valid_xy_net_loss": metrics_valid['xy']}
                if hasattr(vali_data, 'capacity_gt'):
                    wandb_dictionary['valid_dv_diff'] = np.abs((metrics_valid['xy'] - metrics_valid['y']) - vali_data.capacity_gt)
                    wandb_dictionary['valid_dv_diff_rate'] = np.abs(1 - (metrics_valid['xy'] - metrics_valid['y']) / vali_data.capacity_gt)
                wandb.log(wandb_dictionary)

            self.logger.info("| Epoch: {}, Steps: {} ".format(epoch + 1, train_steps))
            self.logger.info("| Train\t- DV Loss: {:.5f}, Y Net Loss: {:.5f}, XY Net Loss: {:.5f}, DV-GT Diff: {:.5f}, DV-GT Diff Rate: {:.5f}"
                             .format(metrics_train['xy'] - metrics_train['y'], metrics_train['y'], metrics_train['xy'],
                                     np.abs((metrics_train['xy'] - metrics_train['y']) - train_data.capacity_gt) if hasattr(train_data, 'capacity_gt') else float(np.NAN),
                                     np.abs(1 - (metrics_train['xy'] - metrics_train['y']) / train_data.capacity_gt) if hasattr(train_data, 'capacity_gt') else float(np.NAN)))
            self.logger.info("| Valid\t- DV Loss: {:.5f}, Y Net Loss: {:.5f}, XY Net Loss: {:.5f}, DV-GT Diff: {:.5f}, DV-GT Diff Rate: {:.5f}"
                             .format(metrics_valid['xy'] - metrics_valid['y'], metrics_valid['y'], metrics_valid['xy'],
                                     np.abs((metrics_valid['xy'] - metrics_valid['y']) - vali_data.capacity_gt) if hasattr(vali_data, 'capacity_gt') else float(np.NAN),
                                     np.abs(1 - (metrics_valid['xy'] - metrics_valid['y']) / vali_data.capacity_gt) if hasattr(vali_data, 'capacity_gt') else float(np.NAN)))
            early_stopping(metrics_valid, self.models, path)      # [0] = mse
            if early_stopping.early_stop:
                self.logger.info("| Early stopping!!")
                break

            adjust_learning_rate(models_optim['y'], epoch + 1, self.args.learning_rate, self.args, self.logger)
            adjust_learning_rate(models_optim['xy'], epoch + 1, self.args.learning_rate, self.args, self.logger)

            self.logger.info('-' * 80)

        for key in self.models.keys():
            self.models[key].load_state_dict(torch.load(os.path.join(path, key + '_checkpoint.pth')))

        return to_devices(self.models, self.args)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info('| loading model')
            for key in self.models.keys():
                self.models[key].load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, key + '_checkpoint.pth')))
        preds = []
        trues = []
        folder_path = os.path.join(self.args.save, setting, 'test_results')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.reset_states(mode='eval')

        criterion = self._select_criterion(self.args.loss)
        metrics = {'y': [], 'xy': []}
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                if self.args.process_info['memory_cut']:
                    self.reset_states()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # batch_y_tilde = draw_y(batch_y, test_dat.min_max)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = {'y': self.models['y'](batch_y, y_tilde=test_data.min_max)[:2],
                               'xy': self.models['xy'](batch_y, y_tilde=test_data.min_max, x=batch_x)[:2]}
                else:
                    outputs = {'y': self.models['y'](batch_y, y_tilde=test_data.min_max),
                               'xy': self.models['xy'](batch_y, y_tilde=test_data.min_max, x=batch_x)}

                loss = {'y': criterion(outputs['y'][:2]),
                        'xy': criterion(outputs['xy'][:2])}

                metrics['y'].append(-loss['y'].item())
                metrics['xy'].append(-loss['xy'].item())

        metrics['y'] = np.array(metrics['y']).mean()
        metrics['xy'] = np.array(metrics['xy']).mean()
        if self.args.wandb:
            wandb_dictionary = {"test_dv_loss": metrics['xy'] - metrics['y'],
                                "test_y_net_loss": metrics['y'],
                                "test_xy_net_loss": metrics['xy']}
            if hasattr(test_data, 'capacity_gt'):
                wandb_dictionary['test_dv_diff'] = np.abs((metrics['xy'] - metrics['y']) - test_data.capacity_gt)
                wandb_dictionary['test_dv_diff_rate'] = np.abs(1 - (metrics['xy'] - metrics['y']) / test_data.capacity_gt)
            wandb.log(wandb_dictionary)

        self.logger.info("| Test\t- DV Loss: {:.5f}, Y Net Loss: {:.5f}, XY Net Loss: {:.5f}, DV-GT Diff: {:.5f}, DV-GT Diff Rate: {:.5f}"
                         .format(metrics['xy'] - metrics['y'], metrics['y'], metrics['xy'],
                                 np.abs((metrics['xy'] - metrics['y']) - test_data.capacity_gt) if hasattr(test_data, 'capacity_gt') else float(np.NAN),
                                 np.abs(1 - (metrics['xy'] - metrics['y']) / test_data.capacity_gt) if hasattr(test_data, 'capacity_gt') else float(np.NAN)))

        return

