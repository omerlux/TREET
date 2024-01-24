import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import logging
import argparse
from utils.tools import set_parameters, seed_init, load_config
import wandb

parser = argparse.ArgumentParser(description='TREET: TRansfer Entropy Estimation via Transformers')

# config file load if exists
parser.add_argument("--config_file", type=str, default="", help="Path to the JSON configuration file.")

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--wandb', action='store_true', help='activating wandb updates')
parser.add_argument('--save', type=str, default='TMP', help='name of main directory')
parser.add_argument('--model_id', type=str, default='EXP', help='model id')
parser.add_argument('--model', type=str, default='Transformer_Encoder',
                    help='model name, options: [Transformer_Encoder, LSTM, Autoformer, Informer, Transformer]')
parser.add_argument('--seed', type=int, default=2021, help='seed number')

# data loader
# parser.add_argument('--data', type=str, default='AWGN', help='dataset type')
parser.add_argument('--use_ndg', action='store_true',
                    help='use NDG instead of past created dataset.')
parser.add_argument('--process_info', type=str,
                    default="{'type': 'AWGN', 'n_samples': 900000, 'stride_cancel': True, 'sigma_x': 1, 'sigma_noise': 1}",
                    help='process information.')
parser.add_argument('--channel_ndg_info', type=str, default="{}",
                    help="channel and ndg information. ONLY when use_ndg is true")
parser.add_argument('--checkpoints', type=str, default='checkpoints', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=0, help='input sequence length')
parser.add_argument('--label_len', type=int, default=10,
                    help='start token length. pre-prediction sequence length for the encoder')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# model define
parser.add_argument('--y_dim', type=int, default=1, help='y input size - exogenous values')
parser.add_argument('--x_dim', type=int, default=1, help='x input size - endogenous values')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
parser.add_argument('--ff_layers', type=int, default=1, help='num of feed forward layers')
parser.add_argument('--time_layers', type=int, default=1, help='num of time layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor (c hyper-parameter)')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--embed', type=str, default='fixed',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='elu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--log_interval', type=int, default=100, help='training log print interval')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='dv', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer name, options: [adam, rmsprop]')
parser.add_argument('--n_draws', type=int, default=1, help='number of draws for DV potential calculation')
parser.add_argument('--exp_clipping', type=float, default='inf', help='exponential clipping for DV potential calculation')
parser.add_argument('--alpha_dv_reg', type=float, default=0., help='alpha for DV regularization on C constant')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu - if mentioned in args, no gpu')
parser.add_argument('--devices', type=str, default="2", help='device ids of multile gpus')

args = parser.parse_args()
if args.config_file:
    # Load the configuration - running over the config file
    args = load_config(os.path.join('config', args.config_file))
if args.wandb:
    args, Exp_Main, group_name = set_parameters(args)
else:
    args, Exp_Main = set_parameters(args)

logging.info('| >>> Args in experiment:')
logging.info('| {}'.format(args))

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        logging.info('=' * 80)
        # setting record of experiments
        setting = '{}_{}_{}_ndg-{}_ydim{}_xdim{}_ll{}_pl{}_sd{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.use_ndg if args.use_ndg == False else args.channel_ndg_info['ndg']['model'],
            args.y_dim,
            args.x_dim,
            args.label_len,
            args.pred_len,
            args.seed,
            ii
        )

        exp = Exp(args, logging)  # set experiments
        logging.info('=========== Start training : {} ==========='.format(setting))
        exp.train(setting)

        logging.info('=========== Testing : {} ==========='.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()

        if args.wandb and ii < args.itr - 1:
            wandb.finish()
            args.seed += 1
            seed_init(args.seed)
            wandb.init(project="project_name", entity="user_name", config=args,
                       group=group_name, reinit=True,
                       tags=["Debug" if args.debug else "Exp", args.data.split('/')[-1], args.model])

else:
    ii = 0
    setting = '{}_{}_{}_ndg-{}_ydim{}_xdim{}_ll{}_pl{}_sd{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.use_ndg if args.use_ndg == False else args.channel_ndg_info['ndg']['model'],
        args.y_dim,
        args.x_dim,
        args.label_len,
        args.pred_len,
        args.seed,
        ii
    )

    exp = Exp(args)  # set experiments
    logging.info('=========== Testing : {} ==========='.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
