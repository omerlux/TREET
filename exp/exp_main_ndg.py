from data_provider.channel import AWGNChannel, GMA1Channel, GAR1Channel, GMA100Channel
from exp.exp_basic import Exp_Basic
from models import LSTM, Decoder_Model, NDG
from utils.tools import EarlyStopping, adjust_learning_rate, visual, to_devices, wandb_hist_log, plot_histogram_with_gaussian_fit, save_mean_attn
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
        channel_dict = {
            'AWGN': AWGNChannel,
            'GMA1': GMA1Channel,
            'GAR1': GAR1Channel,
            'GMA100': GMA100Channel,
        }
        self.args.x_in = False
        model_y = model_dict[self.args.model].Model(self.args).float()
        self.args.x_in = True
        model_xy = model_dict[self.args.model].Model(self.args).float()
        ndg = NDG.Model(self.args).float().to(self.device)

        models = to_devices({'y': model_y, 'xy': model_xy, 'ndg': ndg}, self.args)

        total_params = sum(x.data.nelement() for x in models['y'].parameters())
        self.logger.info('| >>> Model Y total parameters: {:,}'.format(total_params))
        total_params = sum(x.data.nelement() for x in models['xy'].parameters())
        self.logger.info('| >>> Model XY total parameters: {:,}'.format(total_params))
        total_params = sum(x.data.nelement() for x in models['ndg'].parameters())
        self.logger.info('| >>> Model NDG total parameters: {:,}'.format(total_params))

        models_y_args = [(k, v) for k, v in sorted(models['y'].__dict__.items()) if isinstance(v, (int, float, str, list, dict))
                         if k[0] != '_']
        models_xy_args = [(k, v) for k, v in sorted(models['xy'].__dict__.items()) if isinstance(v, (int, float, str, list, dict))
                          if k[0] != '_']
        model_ndg_args = [(k, v) for k, v in sorted(models['ndg'].__dict__.items()) if isinstance(v, (int, float, str, list, dict))
                          if k[0] != '_']
        self.logger.info("| >>> {} Model Y's Arguments: {}".format(self.args.model, models_y_args))
        self.logger.info("| >>> {} Model XY's Arguments: {}".format(self.args.model, models_xy_args))
        self.logger.info("| >>> NDG Model Arguments: {}".format(model_ndg_args))

        self.logger.info('| Creating channel with parameters: {}'.format(self.args.channel_ndg_info))
        self.channel = channel_dict[self.args.channel_ndg_info['type']](self.args, self.logger)
        if hasattr(self.channel, 'capacity_gt'):
            self.logger.info('| >>> Capacity GT: {:.5f}'.format(self.channel.capacity_gt))
            if self.args.wandb:
                pass
                wandb.log({"capacity_gt": self.channel.capacity_gt})

        self.train_samples = int(self.args.channel_ndg_info['n_samples'] * 0.8 / self.args.batch_size)
        self.vali_samples = int(self.args.channel_ndg_info['n_samples'] * 0.1 / self.args.batch_size)
        self.test_samples = int(self.args.channel_ndg_info['n_samples'] * 0.1 / self.args.batch_size)

        self.args.ndg_learning_rate = self.args.channel_ndg_info['ndg']['learning_rate']
        self.ndg_start_training = self.args.channel_ndg_info['ndg']['start_train']
        self.alternate_rate = self.args.channel_ndg_info['ndg']['alternate_rate']
        assert self.alternate_rate > 0, f'Alternate rate must be grater than 0!'
        self.logger.info(f'| Alternating training every {self.alternate_rate} epochs, from epoch {self.ndg_start_training}')

        return models

    def _select_optimizer(self, optim_name):
        models_optim = {}
        for key in self.models.keys():
            lr = self.args.ndg_learning_rate if key == 'ndg' else self.args.learning_rate
            if optim_name == 'rmsprop':
                models_optim[key] = optim.RMSprop(self.models[key].parameters(), lr=lr)
            else:
                models_optim[key] = optim.Adam(self.models[key].parameters(), lr=lr)
        return models_optim

    def _select_criterion(self, loss):
        loss_function = {
            'dv': DV_Loss(self.args.exp_clipping, self.args.alpha_dv_reg, logger=self.logger.info),
            'mse': MSELoss(),
        }
        criterion = loss_function[loss]
        return criterion

    def _call_estimation_nn(self, batch_x, batch_y, iter=None):
        min_max = (batch_y.min().detach().cpu().item(), batch_y.max().detach().cpu().item())

        if self.args.output_attention and self.args.model != 'LSTM':
            out_y = self.models['y'](batch_y, y_tilde=min_max)
            attns_y = out_y[2:]
            out_xy = self.models['xy'](batch_y, y_tilde=min_max, x=batch_x)
            attns_xy = out_xy[2:]
            if iter is not None and iter % 5 == 0:
                # taking the mean over batch and heads. in addition, taking the last pred_len time steps
                save_mean_attn(attns_y[0][0].mean(axis=[0, 1])[-self.args.pred_len:], 'y', base_folder=self.args.save)
                save_mean_attn(attns_xy[0][0].mean(axis=[0, 1])[-self.args.pred_len:], 'xy', base_folder=self.args.save)
            outputs = {'y': out_y[:2],
                       'xy': out_xy[:2]}
        else:
            outputs = {'y': self.models['y'](batch_y, y_tilde=min_max),
                       'xy': self.models['xy'](batch_y, y_tilde=min_max, x=batch_x)}
        return outputs

    def _call_ndg(self, batch_size=None):
        # generating inputs from noise with NDG
        if batch_size is None:
            batch_size = self.args.batch_size
        batch_x, batch_y = self.models['ndg'](self.channel, batch_size)
        return batch_x, batch_y

    def vali(self, criterion, test=False):
        metrics = {'y': [], 'xy': []}

        self.reset_states(mode='eval')

        with torch.no_grad():
            for i in range(self.test_samples if test else self.vali_samples):
                if self.args.channel_ndg_info['memory_cut']:
                    self.reset_states()

                # Generating {x, y} pairs from NDG and channel
                batch_x, batch_y = self._call_ndg()

                # Generating the outputs of the DV representation networks
                outputs = self._call_estimation_nn(batch_x, batch_y, iter=i)

                loss = {'y': criterion(outputs['y'][:2]),
                        'xy': criterion(outputs['xy'][:2])}

                metrics['y'].append(-loss['y'].item())
                metrics['xy'].append(-loss['xy'].item())

        metrics = {k: np.array(metrics[k]).mean() for k in metrics.keys()}
        for k in self.models.keys():
            self.models[k].train()
        return metrics

    def train(self, setting):
        path = os.path.join(self.args.save, setting, self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = self.train_samples
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        models_optim = self._select_optimizer(self.args.optimizer)
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        train_ndg = False       # in order to alternate train ndg or dine
        epochs_ndg = 0
        epochs_dine = 0
        self.logger.info('-' * 80)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            metrics_train = {'y': [], 'xy': []}
            self.reset_states()

            # alternate training between DINE and NDG
            if epoch >= self.ndg_start_training:
                if (epoch - self.ndg_start_training) % self.alternate_rate == 0:
                    train_ndg = True
                else:
                    train_ndg = False
                if epochs_ndg == 0:
                    self.logger.info(f'| Start alternate training, NDG alternate rate every {self.alternate_rate} epochs')
                self.logger.info(f'| Switch training to {"NDG" if train_ndg else "TREET"}...')


            epoch_time = time.time()
            for i in range(train_steps):
                if self.args.channel_ndg_info['memory_cut']:
                    self.reset_states()

                iter_count += 1

                for model_optim in models_optim.values():
                    model_optim.zero_grad()

                # Generating {x, y} pairs from NDG and channel
                batch_x, batch_y = self._call_ndg()

                # Generating the outputs of the DV representation networks
                outputs = self._call_estimation_nn(batch_x, batch_y)

                # Calculating the loss by DV representation
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
                        if hasattr(self.channel, 'capacity_gt'):
                            wandb_dictionary['train_dv_diff'] = np.abs((metrics_train_xy - metrics_train_y) - self.channel.capacity_gt)
                            wandb_dictionary['train_dv_diff_rate'] = np.abs(1 - (metrics_train_xy - metrics_train_y) / self.channel.capacity_gt)
                        if self.models['ndg'].memory_cut:
                            with torch.no_grad():
                                ndg_hist_x, ndg_hist_y = wandb_hist_log(self)
                                wandb_dictionary['train_ndg_histogram_x'] = ndg_hist_x
                                wandb_dictionary['train_ndg_histogram_y'] = ndg_hist_y
                                self.models['ndg'].erase_states()
                        wandb.log(wandb_dictionary)

                if self.args.use_amp:
                    if train_ndg:
                        loss = loss['xy'] - loss['y']
                        scaler.scale(loss).backward()
                        scaler.step(models_optim['ndg'])
                    else:
                        loss = loss['xy'] + loss['y']
                        scaler.scale(loss).backward()
                        scaler.step(models_optim['y'])
                        scaler.step(models_optim['xy'])
                    scaler.update()
                else:
                    if train_ndg:
                        # maximizing the TE(X->Y) = DKL_xy - DKL_y
                        loss = loss['xy'] - loss['y']
                        # = min{(-dxy) - (-dy)} = -max{(dxy - dy)} = -max{TE}
                        loss.backward()
                        models_optim['ndg'].step()
                    else:
                        # maximizing the output for DV representation for XY and Y
                        loss = loss['xy'] + loss['y']
                        # = min{(-dxy) + (-dy)} = - max{(dxy + dy)} = -max{dxy} - max{dy}
                        loss.backward()
                        models_optim['y'].step()
                        models_optim['xy'].step()

            self.logger.info("| Epoch: {}, cost time: {:4.3f}, Y learning rate: {:2.2e}, XY learning rate: {:2.2e}, NDG learning rate: {:2.2e}"
                             .format(epoch + 1, time.time() - epoch_time, models_optim['y'].param_groups[0]['lr'],
                                     models_optim['xy'].param_groups[0]['lr'], models_optim['ndg'].param_groups[0]['lr']))
            metrics_train = {k: np.array(metrics_train[k]).mean() for k in metrics_train.keys()}
            metrics_valid = self.vali(criterion)

            xs = batch_x.detach().cpu().numpy()
            ys = batch_y.detach().cpu().numpy()
            for j in range(self.args.x_dim):
                plot_histogram_with_gaussian_fit(xs[..., j].reshape(-1, 1), 60, os.path.join(path, f'epoch{epoch:3d}_batch_x{j}.png'), self.args.wandb)
                plot_histogram_with_gaussian_fit(ys[..., j].reshape(-1, 1), 60, os.path.join(path, f'epoch{epoch:3d}_batch_y{j}.png'), self.args.wandb)
            plot_histogram_with_gaussian_fit(xs.reshape(-1, 1), 60, os.path.join(path, f'epoch{epoch:3d}_batch_x.png'), self.args.wandb)
            plot_histogram_with_gaussian_fit(ys.reshape(-1, 1), 60, os.path.join(path, f'epoch{epoch:3d}_batch_y.png'), self.args.wandb)

            if self.args.wandb:
                wandb_dictionary = {"valid_dv_loss": metrics_valid['xy'] - metrics_valid['y'],
                                    "valid_y_net_loss": metrics_valid['y'],
                                    "valid_xy_net_loss": metrics_valid['xy']}
                if hasattr(self.channel, 'capacity_gt'):
                    wandb_dictionary['valid_dv_diff'] = np.abs((metrics_valid['xy'] - metrics_valid['y']) - self.channel.capacity_gt)
                    wandb_dictionary['valid_dv_diff_rate'] = np.abs(1 - (metrics_valid['xy'] - metrics_valid['y']) / self.channel.capacity_gt)
                with torch.no_grad():
                    ndg_hist_x, ndg_hist_y = wandb_hist_log(self)
                    wandb_dictionary['valid_ndg_histogram_x'] = ndg_hist_x
                    wandb_dictionary['valid_ndg_histogram_y'] = ndg_hist_y
                wandb.log(wandb_dictionary)

            self.logger.info("| Epoch: {}, Steps: {} ".format(epoch + 1, train_steps))
            self.logger.info("| Train\t- DV Loss: {:.5f}, Y Net Loss: {:.5f}, XY Net Loss: {:.5f}, DV-GT Diff: {:.5f}, DV-GT Diff Rate: {:.5f}"
                             .format(metrics_train['xy'] - metrics_train['y'], metrics_train['y'], metrics_train['xy'],
                                     np.abs((metrics_train['xy'] - metrics_train['y']) - self.channel.capacity_gt) if hasattr(self.channel, 'capacity_gt') else 'NA',
                                     np.abs(1 - (metrics_train['xy'] - metrics_train['y']) / self.channel.capacity_gt) if hasattr(self.channel, 'capacity_gt') else 'NA'))
            self.logger.info("| Valid\t- DV Loss: {:.5f}, Y Net Loss: {:.5f}, XY Net Loss: {:.5f}, DV-GT Diff: {:.5f}, DV-GT Diff Rate: {:.5f}"
                             .format(metrics_valid['xy'] - metrics_valid['y'], metrics_valid['y'], metrics_valid['xy'],
                                     np.abs((metrics_valid['xy'] - metrics_valid['y']) - self.channel.capacity_gt) if hasattr(self.channel, 'capacity_gt') else 'NA',
                                     np.abs(1 - (metrics_valid['xy'] - metrics_valid['y']) / self.channel.capacity_gt) if hasattr(self.channel, 'capacity_gt') else 'NA'))
            if epochs_ndg > 0:
                early_stopping(metrics_valid, self.models, path)
            if early_stopping.early_stop:
                self.logger.info("| Early stopping!!")
                break

            if train_ndg:
                adjust_learning_rate(models_optim['ndg'], epochs_ndg + 1, self.args.ndg_learning_rate, self.args, self.logger)
                epochs_ndg += 1
            else:
                adjust_learning_rate(models_optim['y'], epochs_dine + 1, self.args.learning_rate, self.args, self.logger)
                adjust_learning_rate(models_optim['xy'], epochs_dine + 1, self.args.learning_rate, self.args, self.logger)
                epochs_dine += 1
            self.logger.info('-' * 80)

        for key in self.models.keys():
            self.models[key].load_state_dict(torch.load(os.path.join(path, key + '_checkpoint.pth')))

        return to_devices(self.models, self.args)

    def test(self, setting, test=0):
        if test:
            self.logger.info('| loading model')
            for key in self.models.keys():
                self.models[key].load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, key + '_checkpoint.pth')))

        xs = []
        ys = []
        preds = []
        trues = []
        folder_path = os.path.join(self.args.save, setting, 'test_results')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.reset_states(mode='eval')

        criterion = self._select_criterion(self.args.loss)
        metrics = {'y': [], 'xy': []}
        with torch.no_grad():
            for i in range(self.test_samples):
                if self.args.channel_ndg_info['memory_cut']:
                    self.reset_states()

                # Generating {x, y} pairs from NDG and channel
                batch_x, batch_y = self._call_ndg()
                xs.append(batch_x.detach().cpu().numpy())
                ys.append(batch_y.detach().cpu().numpy())

                # Generating the outputs of the DV representation networks
                outputs = self._call_estimation_nn(batch_x, batch_y)

                loss = {'y': criterion(outputs['y'][:2]),
                        'xy': criterion(outputs['xy'][:2])}

                metrics['y'].append(-loss['y'].item())
                metrics['xy'].append(-loss['xy'].item())

        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        for j in range(self.args.x_dim):
            plot_histogram_with_gaussian_fit(xs[..., j].reshape(-1, 1), 60, os.path.join(folder_path, f'batch_x{j}.png'), self.args.wandb)
            plot_histogram_with_gaussian_fit(ys[..., j].reshape(-1, 1), 60, os.path.join(folder_path, f'batch_y{j}.png'), self.args.wandb)
        plot_histogram_with_gaussian_fit(xs.reshape(-1, 1), 60, os.path.join(folder_path, f'batch_x.png'), self.args.wandb)
        plot_histogram_with_gaussian_fit(ys.reshape(-1, 1), 60, os.path.join(folder_path, f'batch_y.png'), self.args.wandb)

        metrics['y'] = np.array(metrics['y']).mean()
        metrics['xy'] = np.array(metrics['xy']).mean()
        if self.args.wandb:
            wandb_dictionary = {"test_dv_loss": metrics['xy'] - metrics['y'],
                                "test_y_net_loss": metrics['y'],
                                "test_xy_net_loss": metrics['xy']}
            if hasattr(self.channel, 'capacity_gt'):
                wandb_dictionary['test_dv_diff'] = np.abs((metrics['xy'] - metrics['y']) - self.channel.capacity_gt)
                wandb_dictionary['test_dv_diff_rate'] = np.abs(1 - (metrics['xy'] - metrics['y']) / self.channel.capacity_gt)
            with torch.no_grad():
                ndg_hist_x, ndg_hist_y = wandb_hist_log(self)
                wandb_dictionary['test_ndg_histogram_x'] = ndg_hist_x
                wandb_dictionary['test_ndg_histogram_y'] = ndg_hist_y
            wandb.log(wandb_dictionary)

        self.logger.info("| Test\t- DV Loss: {:.5f}, Y Net Loss: {:.5f}, XY Net Loss: {:.5f}, DV-GT Diff: {:.5f}, DV-GT Diff Rate: {:.5f}"
                         .format(metrics['xy'] - metrics['y'], metrics['y'], metrics['xy'],
                                 np.abs((metrics['xy'] - metrics['y']) - self.channel.capacity_gt) if hasattr(self.channel, 'capacity_gt') else 'NA',
                                 np.abs(1 - (metrics['xy'] - metrics['y']) / self.channel.capacity_gt) if hasattr(self.channel, 'capacity_gt') else 'NA'))

        return

