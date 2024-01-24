import os
import numpy as np
import pandas as pd
import mat73
import pickle
import hickle as hkl
import hdf5storage
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from timeout_decorator import timeout
import warnings

warnings.filterwarnings('ignore')


class Dataset_RNN(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        if self.data_stride:
            batch_series_len = len(self) // self.batch_size     # how many samples are in a batch
            batch_series_idx = index % self.batch_size          # index of batch sequence

            new_index = batch_series_idx * batch_series_len + \
                        (index // self.batch_size) % batch_series_len

            s_begin = new_index * (self.pred_len + self.label_len)
        else:
            s_begin = index

        r_begin = s_begin
        r_end = s_begin + self.label_len + self.pred_len

        seq_x = self.data_x[r_begin:r_end]      # same input for encoder and decoder
        seq_y = self.data_y[r_begin:r_end]
        if hasattr(self, 'last_x_zero'):
            seq_x[-1].zero_()  # for transfer entropy definition without x_t
        if hasattr(self, 'x_length'):
            if self.x_length < self.label_len:
                seq_x[:-self.x_length].zero_()
        return seq_x, seq_y

    def __len__(self):
        if self.data_stride:
            return len(self.data_x) // (self.label_len + self.pred_len)  # for data stride = self.pred_len
        else:
            return len(self.data_x) - self.label_len - self.pred_len + 1


class Dataset_Transformer(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        if self.data_stride:
            s_begin = index * self.pred_len  # data stride is self.pred_len
        else:
            s_begin = index

        r_begin = s_begin
        r_end = s_begin + self.label_len + self.pred_len

        seq_x = self.data_x[r_begin:r_end]      # same input for encoder and decoder
        seq_y = self.data_y[r_begin:r_end]

        if hasattr(self, 'last_x_zero'):
            seq_x[-1].zero_()  # for transfer entropy definition without x_t
        if hasattr(self, 'x_length'):
            if self.x_length < self.label_len:
                seq_x[:-self.x_length].zero_()
        return seq_x, seq_y

    def __len__(self):
        if self.data_stride:
            return (len(self.data_x) - self.label_len) // self.pred_len     # for data stride = self.pred_len
        else:
            return len(self.data_x) - self.label_len - self.pred_len + 1


class Dataset_AWGN(Dataset):
    def __init__(self, flag='train', size=None, timeenc=0, batch_size=32, dim=1, process_info={}):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.batch_size = batch_size
        self.dim = dim
        self.n_samples = process_info['n_samples']
        self.data_stride = not process_info['memory_cut']
        self.p_std = process_info['sigma_x']
        self.n_std = process_info['sigma_noise']

        self.timeenc = timeenc
        self.__read_data__()

    @staticmethod
    def create_process(P, N, num, dim):
        xn = np.zeros((num, dim))
        zn = np.zeros((num, dim))
        for i in range(dim):
            xn[:, i] = np.random.normal(0, np.sqrt(P / dim), num)
            zn[:, i] = np.random.normal(0, np.sqrt(N / dim), num)
        yn = np.add(xn, zn)
        features = torch.tensor(xn)
        labels = torch.tensor(yn)
        return {'features': features, 'labels': labels}

    def __read_data__(self):
        self.scaler = StandardScaler()
        data = self.create_process(P=self.p_std, N=self.n_std, num=self.n_samples, dim=self.dim)
        self.capacity_gt = self.dim * 0.5 * np.log(1 + self.p_std / self.n_std)
        self.min_max = (data['features'].min(), data['features'].max())

        num_train = int(self.n_samples * 0.8)
        num_test = int(self.n_samples * 0.1)
        num_vali = self.n_samples - num_train - num_test
        border1s = [0, num_train - self.label_len, self.n_samples - num_test - self.label_len]
        border2s = [num_train, num_train + num_vali, self.n_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = data['features'][border1:border2]
        self.data_y = data['labels'][border1:border2]
        self.data_stamp = None

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Apnea(Dataset):
    def __init__(self, flag='train', size=None, timeenc=0, batch_size=32, dim=1, process_info={}):

        # info
        if size == None:
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # assert self.label_len == 0, "Only support label_len=0"
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.batch_size = batch_size
        assert dim == 1, "Only support dim=1"
        self.dim = dim
        self.data_stride = not process_info['memory_cut']
        self.x_type = process_info['x']
        self.x_length = process_info['x_length']
        self.last_x_zero = True     # zeroing the last x input when returning a batch
        self.scale = True
        self.__read_data__()

    def __read_data__(self):
        assert os.path.isdir('./datasets/apnea'), "No Apnea data found in datasets/apnea."
        heart_rates = []
        breath_rates = []
        oxygen_rates = []
        for file in [f for f in os.listdir('./datasets/apnea') if 'txt' in f]:
            df = pd.read_csv(os.path.join('datasets/apnea', file), sep=' ', header=None, names=['heart', 'breath', 'oxygen'], index_col=False)
            heart_rates.append(df['heart'].values)
            breath_rates.append(df['breath'].values)
            oxygen_rates.append(df['oxygen'].values)
        heart_rates = np.array(heart_rates)
        breath_rates = np.array(breath_rates)
        oxygen_rates = np.array(oxygen_rates)
        # data_raw = np.stack([heart_rate, breath_rate, oxygen_rate], axis=1) # no oxygen now...
        data_raw1 = np.stack([heart_rates[0], breath_rates[0]], axis=1)
        data_raw2 = np.stack([heart_rates[1], breath_rates[1]], axis=1)
        feature_dict = {'heart': 0, 'breath': 1} #, 'oxygen': 2}
        self.x_feature = feature_dict[self.x_type]
        indices = list(range(data_raw1.shape[1]))
        indices.remove(self.x_feature)
        indices.insert(0, self.x_feature)
        data1 = data_raw1[:, indices]
        data2 = data_raw2[:, indices]
        self.n_samples1 = len(data1)
        self.n_samples2 = len(data2)
        if self.scale:
            self.scaler1 = StandardScaler(with_std=True)
            self.scaler2 = StandardScaler(with_std=True)
            data1 = self.scaler1.fit_transform(data1)
            data2 = self.scaler2.fit_transform(data2)
        self.min_max = [min(data1[:, 1:].min(), data2[:, 1:].min()),
                        max(data2[:, 1:].max(), data2[:, 1:].max())]
        # num_train = int(self.n_samples1 * 0.9 + self.n_samples2 * 0.9)
        # border1s = [0, num_train - self.label_len, 0]
        # border2s = [num_train, self.n_samples1 + self.n_samples2, self.n_samples1 + self.n_samples2]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        self.data_x = torch.tensor(np.concatenate([data1[:, :1], data2[:, :1]], axis=0))
        self.data_y = torch.tensor(np.concatenate([data1[:, 1:], data2[:, 1:]], axis=0))

        self.data_stamp = None

    def inverse_transform(self, data):
        assert self.scale, "No scaler found."
        assert data.shape[1] == self.data_x.shape[1] + self.data_y.shape[1], "Data shape not match. Concate X (first) with Y."
        return self.scaler.inverse_transform(data)


class Dataset_Transformer_AWGN(Dataset_Transformer, Dataset_AWGN):
    def __init__(self, *args, **kwargs):
        Dataset_Transformer.__init__(self)
        Dataset_AWGN.__init__(self, *args, **kwargs)


class Dataset_RNN_AWGN(Dataset_RNN, Dataset_AWGN):
    def __init__(self, *args, **kwargs):
        Dataset_RNN.__init__(self)
        Dataset_AWGN.__init__(self, *args, **kwargs)


class Dataset_Transformer_Apnea(Dataset_Transformer, Dataset_Apnea):
    def __init__(self, *args, **kwargs):
        Dataset_Transformer.__init__(self)
        Dataset_Apnea.__init__(self, *args, **kwargs)


class Dataset_RNN_Apnea(Dataset_RNN, Dataset_Apnea):
    def __init__(self, *args, **kwargs):
        Dataset_RNN.__init__(self)
        Dataset_Apnea.__init__(self, *args, **kwargs)


class Dataset_Transformer_TEintro42(Dataset_Transformer, Dataset_TEintro42):
    def __init__(self, *args, **kwargs):
        Dataset_Transformer.__init__(self)
        Dataset_TEintro42.__init__(self, *args, **kwargs)


class Dataset_RNN_TEintro42(Dataset_RNN, Dataset_TEintro42):
    def __init__(self, *args, **kwargs):
        Dataset_RNN.__init__(self)
        Dataset_TEintro42.__init__(self, *args, **kwargs)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
