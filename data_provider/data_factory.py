import numpy as np
from data_provider.data_loader import Dataset_Transformer_AWGN, Dataset_RNN_AWGN, Dataset_RNN_Apnea, Dataset_Transformer_Apnea, Dataset_RNN_TEintro42, Dataset_Transformer_TEintro42
from torch.utils.data import DataLoader

data_dict = {
    'AWGN': Dataset_Transformer_AWGN,
    'RNN_AWGN': Dataset_RNN_AWGN,
    'Apnea': Dataset_Transformer_Apnea,
    'RNN_Apnea': Dataset_RNN_Apnea,
}


def data_provider(args, flag, logger):
    Data = data_dict[args.process_info['type']]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True if args.model != 'LSTM' or args.process_info.get('memory_cut') else False
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        batch_size=args.batch_size,
        dim=args.y_dim,
        process_info=args.process_info,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    if args.data == 'AWGN':
        if args.x_dim != args.y_dim:
            del data_set.capacity_gt        # delete the ground truth capacity if dimensions are different
        logger.info('| {:<5} length {:,}, process dimension {}, SNR {} dB'.format(flag, len(data_set),
                    data_set.dim, round(10 * np.log10(data_set.p_std / data_set.n_std), 2)))
    else:
        logger.info('| {:<5} length {:,}'.format(flag, len(data_set)))
    return data_set, data_loader
