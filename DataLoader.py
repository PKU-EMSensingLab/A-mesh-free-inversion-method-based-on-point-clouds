import numpy as np
import h5py
import math

def TrainLoader(data_root, batch_size, n, field_dim, pc_dim, tot_train):
    f = h5py.File(data_root+'/PcsTrainDataset.h5', 'r')
    index = 0
    while index < math.ceil(tot_train/batch_size):
        if (index+1) * batch_size >= tot_train:
            Esct_data = np.zeros((tot_train-index*batch_size, field_dim))
            epsr_data = np.zeros((tot_train-index*batch_size, n, pc_dim))
            for i in range(index*batch_size, tot_train):
                Esct = f[str(i)]['Esct'][:field_dim]
                Esct = np.reshape(Esct, (1, -1))
                Esct_data[i-index*batch_size, :] = Esct / 2.0
                pcs = f[str(i)]['pcs_epsr'][:, :]
                index_n = np.random.choice(pcs.shape[0], n, replace=False)
                epsr_data[i-index*batch_size, :, :] = pcs[index_n]
            index = math.ceil(tot_train/batch_size)
        else:
            Esct_data = np.zeros((batch_size, field_dim))
            epsr_data = np.zeros((batch_size, n, pc_dim))
            for i in range(index*batch_size, (index+1)*batch_size):
                Esct = f[str(i)]['Esct'][:field_dim]
                Esct = np.reshape(Esct, (1, -1))
                Esct_data[i-index*batch_size, :] = Esct / 2.0
                pcs = f[str(i)]['pcs_epsr'][:, :]
                index_n = np.random.choice(pcs.shape[0], n, replace=False)
                epsr_data[i-index*batch_size, :, :] = pcs[index_n]
            index = index + 1
        yield Esct_data, epsr_data
    f.close()


def DevLoader(data_root, batch_size, n, field_dim, pc_dim, tot_dev):
    f = h5py.File(data_root+'/PcsDevDataset.h5', 'r')
    index = 0
    while index < math.ceil(tot_dev/batch_size):
        if (index+1) * batch_size >= tot_dev:
            Esct_data = np.zeros((tot_dev-index*batch_size, field_dim))
            epsr_data = np.zeros((tot_dev-index*batch_size, n, pc_dim))
            for i in range(index*batch_size, tot_dev):
                Esct = f[str(i)]['Esct'][:field_dim]
                Esct = np.reshape(Esct, (1, -1))
                Esct_data[i-index*batch_size, :] = Esct / 2.0
                pcs = f[str(i)]['pcs_epsr'][:, :]
                index_n = np.random.choice(pcs.shape[0], n, replace=False)
                epsr_data[i-index*batch_size, :, :] = pcs[index_n]
            index = math.ceil(tot_dev/batch_size)
        else:
            Esct_data = np.zeros((batch_size, field_dim))
            epsr_data = np.zeros((batch_size, n, pc_dim))
            for i in range(index*batch_size, (index+1)*batch_size):
                Esct = f[str(i)]['Esct'][:field_dim]
                Esct = np.reshape(Esct, (1, -1))
                Esct_data[i-index*batch_size, :] = Esct / 2.0 
                pcs = f[str(i)]['pcs_epsr'][:, :]
                index_n = np.random.choice(pcs.shape[0], n, replace=False)
                epsr_data[i-index*batch_size, :, :] = pcs[index_n]
            index = index + 1
        yield Esct_data, epsr_data
    f.close()


def DevReader(data_root, batch_size, field_dim, tot_dev):
    f = h5py.File(data_root+'/PcsDevDataset.h5', 'r')
    Esct_data = []
    epsr_data = []
    point_num = []
    index_list = np.random.choice(tot_dev, batch_size, replace=False)
    for index in index_list:
        Esct = f[str(index)]['Esct'][:field_dim]
        Esct = np.reshape(Esct, (1, -1))
        Esct_data.append(Esct / 2.0)
        pcs = f[str(index)]['pcs_epsr'][:, :]
        pcs = np.expand_dims(pcs, axis=0)
        epsr_data.append(pcs)
        point_num.append(pcs.shape[1])
    f.close()
    return Esct_data, epsr_data, point_num


def TestReader(data_root, batch_size, field_dim, tot_test):
    f = h5py.File(data_root+'/PcsTestDataset.h5', 'r')
    Esct_data = []
    epsr_data = []
    point_num = []
    index_list = np.random.choice(tot_test, batch_size, replace=False)
    for index in index_list:
        Esct = f[str(index)]['Esct'][:field_dim]
        Esct = np.reshape(Esct, (1, -1))
        Esct_data.append(Esct / 2.0)
        pcs = f[str(index)]['pcs_epsr'][:, :]
        pcs = np.expand_dims(pcs, axis=0)
        epsr_data.append(pcs)
        point_num.append(pcs.shape[1])
    f.close()
    return Esct_data, epsr_data, point_num