import os
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# column names of CMAPSS Dataset
# CMAPSS数据集列名
columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
           's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

feature_columns = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                   's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21',
                   'cycle_norm']
feature_columns_reduce = ['setting1', 'setting2','s2', 's3', 's4',  's6', 's7', 's8',
                   's9', 's11', 's12', 's13', 's14', 's15', 's17',  's20', 's21',
                   'cycle_norm']


class CMAPSSDataset():
    def __init__(self, fd_number, batch_size, sequence_length):
        super(CMAPSSDataset).__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_data = None
        self.test_data = None
        self.train_data_encoding = None
        self.test_data_encoding = None

        # \s+ 匹配一个或多个空格
        data = pd.read_csv("./data/CMAPSSData/train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        data.columns = columns


        # 计算该数据集包含的engine数目
        self.engine_size = data['id'].unique().max()

        # 计算每一行的剩余cycle
        rul = pd.DataFrame(data.groupby('id')['cycle'].max()).reset_index()#算出总共的cycle
        rul.columns = ['id', 'max']
        data = data.merge(rul, on=['id'], how='left')
        data['RUL'] = data['max'] - data['cycle']
        data['RUL'][data['RUL']>130] = 130
        # data['RUL'] = data['RUL']/130

        # data.drop(['cycle', 'setting1', 'setting2', 'setting3'], axis=1, inplace=True)
        data.drop(['max'], axis=1, inplace=True)

        # 将id之外的列正规化
  
        # self.min_max = MinMaxScaler((0,1))
        # X_train_minmax = min_max_scaler.fit_transform(X_train)
        data['cycle_norm'] = data['cycle']
        cols_normalize = data.columns.difference(['id', 'cycle', 'RUL'])
        self.std = StandardScaler()
        # self.std = MinMaxScaler((0,1))
        self.std.fit(data[cols_normalize])
        # norm_data = pd.DataFrame(self.std.fit_transform(data[cols_normalize]),
        #                          columns=cols_normalize, index=data.index)
        norm_data= pd.DataFrame(self.std.transform(data[cols_normalize]),
                                 columns=cols_normalize, index=data.index)
        join_data = data[data.columns.difference(cols_normalize)].join(norm_data)
        self.train_data = join_data.reindex(columns=data.columns)


        test_data = pd.read_csv("./data/CMAPSSData/test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        test_data.columns = columns
        truth_data = pd.read_csv("./data/CMAPSSData/RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
        truth_data.columns = ['truth']
        truth_data['id'] = truth_data.index + 1

        test_rul = pd.DataFrame(test_data.groupby('id')['cycle'].max()).reset_index()
        test_rul.columns = ['id', 'elapsed']
        test_rul = test_rul.merge(truth_data, on=['id'], how='left')
        test_rul['max'] = test_rul['elapsed'] + test_rul['truth']

        test_data = test_data.merge(test_rul, on=['id'], how='left')
        test_data['RUL'] = test_data['max'] - test_data['cycle']
        test_data['RUL'][test_data['RUL'] > 130] = 130
        # test_data['RUL'] = test_data['RUL']/130

        test_data.drop(['max'], axis=1, inplace=True)

        test_data['cycle_norm'] = test_data['cycle']
        norm_test_data = pd.DataFrame(self.std.transform(test_data[cols_normalize]),
                                      columns=cols_normalize, index=test_data.index)
        join_test_data = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_data)
        self.test_data = join_test_data.reindex(columns=test_data.columns)
        self.reduce_unimportant()
        self.columns = self.train_data.columns
    # In this function we reduce the unrelevant features which are of the same value
    def reduce_unimportant(self):
        remaining_col = self.train_data.columns.difference(['s1', 's10', 's16','s18','s19','s5','setting3'])
        self.train_data = pd.DataFrame(self.train_data[remaining_col])
        self.test_data = pd.DataFrame(self.test_data[remaining_col])


    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_feature_slice(self, input_data):
        # Reshape the data to (samples, time steps, features)
        def reshapeFeatures(input, columns, sequence_length):
            data = input[columns].values
            num_elements = data.shape[0]
            # print(num_elements)
            for start, stop in zip(range(0, num_elements - sequence_length), range(sequence_length, num_elements)):
                yield (data[start:stop, :])

        feature_list = [list(reshapeFeatures(input_data[input_data['id'] == i], feature_columns_reduce, self.sequence_length))
                        for i in range(1, self.engine_size + 1) if
                        len(input_data[input_data['id'] == i]) > self.sequence_length]
        #feature_list[i]中的是第i个发动机的length长度的数据

        feature_array = np.concatenate(list(feature_list), axis=0).astype(np.float32)
        #所有的合在一起
        length = len(feature_array) // self.batch_size
        #
        return feature_array[:length * self.batch_size]

    def get_one_piece(self, number,input_data ):
        def reshapeFeatures(input, columns, sequence_length):
            data = input[columns].values
            num_elements = data.shape[0]
            for start, stop in zip(range(0, num_elements - sequence_length), range(sequence_length, num_elements)):
                yield (data[start:stop, :])

        feature_list = [list(reshapeFeatures(input_data[input_data['id'] == i], feature_columns_reduce, self.sequence_length))
                        for i in range(1, self.engine_size + 1) if
                        len(input_data[input_data['id'] == i]) > self.sequence_length]
        def reshapeLabels(input, sequence_length, columns=['RUL']):
            data = input[columns].values
            num_elements = data.shape[0]
            return (data[sequence_length:num_elements, :]) #每次滑动一位，如果取小序列的最后一点为rul，刚好为sequence_legth: 最后
        label_list = [reshapeLabels(input_data[input_data['id'] == i], self.sequence_length)
                      for i in range(1, self.engine_size + 1)]

        return feature_list[number], label_list[number]


    def get_engine_id(self, input_data):
        def reshapeLabels(input, sequence_length, columns=['id']):
            data = input[columns].values
            num_elements = data.shape[0]
            return (data[sequence_length:num_elements, :])

        label_list = [reshapeLabels(input_data[input_data['id'] == i], self.sequence_length)
                      for i in range(1, self.engine_size + 1)]
        label_array = np.concatenate(label_list).astype(np.int8)
        length = len(label_array) // self.batch_size
        return label_array[:length * self.batch_size]

    def get_label_slice(self, input_data):
        def reshapeLabels(input, sequence_length, columns=['RUL']):
            data = input[columns].values
            num_elements = data.shape[0]
            return (data[sequence_length:num_elements, :]) #每次滑动一位，如果取小序列的最后一点为rul，刚好为sequence_legth: 最后
        label_list = [reshapeLabels(input_data[input_data['id'] == i], self.sequence_length)
                      for i in range(1, self.engine_size + 1)]
        label_array = np.concatenate(label_list).astype(np.float32)
        length = len(label_array) // self.batch_size
        return label_array[:length * self.batch_size]

    # 每个engine只取最后一个sequence_length（如果该engine的数据条目数大于sequence_length的话，否则就舍弃）
    # 用于最后的evaluation
    def get_last_data_slice(self, input_data):
        num_engine = input_data['id'].unique().max()
        test_feature_list = [input_data[input_data['id'] == i][feature_columns].values[-self.sequence_length:]
                             for i in range(1, num_engine + 1) if
                             len(input_data[input_data['id'] == i]) >= self.sequence_length]
        test_feature_array = np.asarray(test_feature_list).astype(np.float32)
        length_test = len(test_feature_array) // self.batch_size

        test_label_list = [input_data[input_data['id'] == i]['RUL'].values[-1:]
                           for i in range(1, num_engine + 1) if
                           len(input_data[input_data['id'] == i]) >= self.sequence_length]
        test_label_array = np.asarray(test_label_list).astype(np.float32)
        length_label = len(test_label_array) // self.batch_size

        return test_feature_array[:length_test * self.batch_size], test_label_array[:length_label * self.batch_size]

    #
    def set_test_data_encoding(self, test_data_encoding):
        self.test_data_encoding = test_data_encoding

    def set_train_data_encoding(self, train_data_encoding):
        self.train_data_encoding = train_data_encoding


if __name__ == "__main__":
    datasets = CMAPSSDataset(fd_number='1', batch_size=10, sequence_length=50)


