import os
import sys
import copy
import time
import shutil
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import wandb
import json5
import logging
from scipy.stats import norm

plt.switch_backend('agg')


def seed_init(fix_seed):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


def adjust_learning_rate(optimizer, epoch, learning_rate, args, logger):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if 'type1' in args.lradj:
        if len(args.lradj.split('_')) > 1:
            rate = float(args.lradj.split('_')[1])
            lr_adjust = {epoch: learning_rate * (rate ** ((epoch - 1) // 1))}
        else:
            lr_adjust = {epoch: learning_rate * (0.8 ** ((epoch - 1) // 1))}

    elif args.lradj == 'type2':
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {5: learning_rate * 0.5,
                     10: learning_rate * 0.25}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        logger.info('| * Updating learning rate to {:2.2e}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.05, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.logger = logger
        self.past_scores = {'y': [], 'xy': []}
        self.stable_value = {'y': None, 'xy': None}

    def __call__(self, val_loss, models, path):
        self.past_scores['y'].append(val_loss['y'])
        self.past_scores['xy'].append(val_loss['xy'])

        if len(self.past_scores['y']) < self.patience or len(self.past_scores['xy']) < self.patience:
            self.stable_value['y'] = sum(self.past_scores['y']) / len(self.past_scores['y'])
            self.stable_value['xy'] = sum(self.past_scores['xy']) / len(self.past_scores['xy'])
            # self.save_checkpoint(models, path)
        else:
            self.stable_value['y'] = sum(self.past_scores['y'][-self.patience:]) / self.patience
            self.stable_value['xy'] = sum(self.past_scores['xy'][-self.patience:]) / self.patience
            if abs(val_loss['y'] - self.stable_value['y']) <= self.delta and \
                    abs(val_loss['xy'] - self.stable_value['xy']) <= self.delta:
                self.counter += 1
                self.logger.info(f"| * EarlyStopping counter: {self.counter} out of {self.patience}. (stable y: {self.stable_value['y']:.5f}, stable xy: {self.stable_value['xy']:.5f})")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                # self.save_checkpoint(models, path)
                self.counter = 0
        self.save_checkpoint(models, path)

    def save_checkpoint(self, models, path):
        if self.verbose:
            y = 0 if self.stable_value['y'] is None else round(self.stable_value['y'], 3)
            xy = 0 if self.stable_value['xy'] is None else round(self.stable_value['xy'], 3)
            self.logger.info('| * Validation average loss updated (y: {:.6f}, xy: {:.6f}).  Saving model ...'.format(y, xy))
        for key in models.keys():
            torch.save(copy.deepcopy(models[key]).cpu().state_dict(), path + '/' + key + '_checkpoint.pth')


class EarlyStoppingMax:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_max = -np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, models, path):
        score = val_loss['y'] + val_loss['xy']
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, models, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'| * EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, models, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, models, path):
        if self.verbose:
            self.logger.info('| * Validation loss increased ({:.6f} --> {:.6f}).  Saving model ...'
                             .format(self.val_loss_max, val_loss['y'] + val_loss['xy']))
        for key in models.keys():
            torch.save(copy.deepcopy(models[key]).cpu().state_dict(), path + '/' + key + '_checkpoint.pth')
        self.val_loss_max = val_loss['y'] + val_loss['xy']


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# Function to load and process the configuration file
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            # if isinstance(value, dict):
            #     setattr(self, key, Config(value))
            # else:
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


def load_config(config_file):
    with open(config_file) as file:
        data = json5.load(file)
    config = Config(data)
    config.config_file = config_file
    return config


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.grid()
    plt.savefig(name, bbox_inches='tight')


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path.split('/')[0]):
        os.mkdir(path.split('/')[0])
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def to_devices(models, args):
    if args.use_gpu and len(args.device_ids) > 1:
        for key in models.keys():
            models[key] = torch.nn.DataParallel(models[key], device_ids=args.device_ids)
    return models


def wandb_hist_log(exp):
    exp.reset_states(mode='eval')
    batch_x, batch_y = exp._call_ndg(10000)
    batch_x = batch_x[:, -1].detach().cpu().numpy().flatten()
    batch_y = batch_y[:, -1].detach().cpu().numpy().flatten()
    ndg_hist_x = wandb.Histogram(batch_x)  # last timestep first value of the vector
    ndg_hist_y = wandb.Histogram(batch_y)
    return ndg_hist_x, ndg_hist_y


def set_parameters(args):
    gettrace = getattr(sys, 'gettrace', lambda: None)
    args.debug = gettrace() is not None
    if hasattr(args, 'exp_clipping'):
        args.exp_clipping = float(args.exp_clipping)
    if args.use_ndg:
        args.channel_ndg_info = eval(args.channel_ndg_info) if type(args.channel_ndg_info) == str else args.channel_ndg_info
        args.data = args.channel_ndg_info['type']
        main_script = 'exp/exp_main_ndg.py'
        from exp.exp_main_ndg import Exp_Main
    else:
        args.process_info = eval(args.process_info) if type(args.process_info) == str else args.process_info
        args.data = args.process_info['type']
        main_script = 'exp/exp_main.py'
        from exp.exp_main import Exp_Main

    seed_init(args.seed)
    if args.wandb:
        # group_name = wandb.util.generate_id()
        group_name = "_".join([f"{key}-{getattr(args, key)}" for key in
                               ["model", "data", "use_ndg", "y_dim", "x_dim", "c_out", "pred_len", "label_len",
                                "n_draws"]]) \
            if args.model_id == 'EXP' else args.model_id
        tags = ["Debug" if args.debug else "Exp", args.data, args.model]
        if hasattr(args, 'tags'):
            tags.extend(eval(args.tags) if type(args.tags) == str else args.tags)
        wandb.init(project="dineformer", entity="omerlux", config=args, dir='./',
                   group=group_name, tags=tags)
        config = wandb.config

    # setting a folder to current run
    if args.save == 'TMP':
        if args.data == 'custom':
            args.save = args.root_path.split('/')[1][:4].upper() + '-' + args.model
        else:
            if args.use_ndg:
                args.save = 'NDG-' + args.data + '-' + args.model
            else:
                args.save = args.data + '-' + args.model
        if args.debug:
            args.save += '-DEBUG'
        else:
            args.save += '-EXP'

        if args.model_id != 'EXP':
            args.save += '-' + args.model_id

        args.save = 'saves/{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

    if 'Decoder_Model' in args.model:
        model_script = 'models/Decoder_Model.py'
    elif 'LSTM' in args.model:
        model_script = 'models/LSTM.py'
        if not args.use_ndg:
            args.process_info['type'] = 'RNN_' + args.process_info['type']
    create_exp_dir(args.save, scripts_to_save=[model_script, main_script])

    # setting logger
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        logging.info('| GPU devices: cuda:{}'.format(device_ids))
        args.device_ids = [int(id_) for id_ in
                           device_ids]  # list(range(len(device_ids)))  # [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return (args, Exp_Main, group_name) if args.wandb else (args, Exp_Main)


def plot_histogram_with_gaussian_fit(data, bins, filename, log_wandb=False):
    fig, ax = plt.subplots()  # Create a new figure and axis
    ax.hist(data, bins=bins, density=True, alpha=0.6, color='g')

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the PDF.
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    title = f"Fit results: mu = %.2f, std = %.2f" % (mu, std)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

    if log_wandb:
        wandb.log({filename.split('.')[0].split('/')[-1]: wandb.Image(fig)})

    plt.savefig(filename)
    plt.show()
    plt.close()


def save_mean_attn(mean_attn: torch.tensor, name: str, base_folder=''):
    base_folder = os.path.join(base_folder, 'attns')
    # check if attns folder exists:
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    # get all file names start with name in attns/
    files = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f)) and f.startswith(name + '_')]
    i = len(files)
    # save the mean attn
    logging.info(f'Saving mean attn to attns/{name}_{i:0000d}.npy')
    np.save(f'{base_folder}/{name}_{i:04d}.npy', mean_attn.detach().cpu().numpy())




