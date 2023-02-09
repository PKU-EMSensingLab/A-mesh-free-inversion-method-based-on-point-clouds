import os
import torch
from DataLoader import *
from config import cfg
from nice import NICE
from utils import *
from torch.autograd import Variable
import numpy as np


def save_inverse_sample(cfg, model, total_test):

    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    Esct_data, true_epsr_data, _ = TestReader(cfg['DATA_ROOT'], 1, cfg['FIELD_DIM'], total_test)
    
    with torch.no_grad():
        point_num=10000
        square = square_generator(1, point_num)
        np.savetxt('./Result/square.txt', square[0])
        square = np.transpose(square, (0, 2, 1))
        square = Variable(torch.Tensor(square)).cuda()
        Esct = Variable(torch.Tensor(Esct_data[0])).cuda()

        pred_pc, _ = model.forward(Esct, square, point_num)
        pred_pc = pred_pc[:, :cfg['PC_DIM'], :]
        pred_pc = pred_pc.permute(0, 2, 1)
        pred_pc = pred_pc.cpu().detach().numpy()
        true_epsr = true_epsr_data[0]

    plot_name = os.path.join('./Result', 'true_epsr')
    plot_scatter(plot_name, true_epsr[0])
    plot_name = os.path.join('./Result', 'pred_epsr')
    plot_scatter(plot_name, pred_pc[0])


def plot_scatter(filename, pcds, size=0.2, cmap=plt.cm.jet, xlim=(cfg['DOI_MIN'], cfg['DOI_MAX']), ylim=(cfg['DOI_MIN'], cfg['DOI_MAX']), zlim=(cfg['DOI_MIN'], cfg['DOI_MAX'])):

    fig = plt.figure(figsize=(1*3, 1*3)) # W,H
    ax = fig.add_subplot(1, 1, 1, projection = '3d')
    ax.scatter(pcds[:, 2], pcds[:, 0], pcds[:, 1], c=pcds[:, 3], s=size, marker='.', cmap=cmap, vmin=0, vmax=1.0)
    ax.set_axis_off()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.invert_zaxis()
    ax.view_init(15, 35)

    fig.savefig(filename+'.png')
    plt.close(fig)

    np.savetxt(filename+'.txt', pcds)

def save_forward_sample(cfg, model, total_test):

    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    Esct_data, true_epsr_data, point_nums = TestReader(cfg['DATA_ROOT'], 1, cfg['FIELD_DIM'], total_test)

    with torch.no_grad():
        point_num = point_nums[0]
        pc = np.transpose(true_epsr_data[0], (0, 2, 1))
        pc = Variable(torch.Tensor(pc)).cuda()
        pred_Esct = model.sample(pc, 1, point_num)
        pred_Esct = pred_Esct.cpu().detach().numpy()
        true_Esct = Esct_data[0]

    plot_name = os.path.join('./Result', 'Esct')
    plot_lines(plot_name, pred_Esct[0], true_Esct[0])


def plot_lines(filename, preds, trues):

    fig = plt.figure(figsize=(1*3, 1*3)) # W,H
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(trues,color='r')
    ax.plot(preds,color='b')
    ax.set_axis_off()

    fig.savefig(filename+'.png')
    plt.close(fig)

    np.savetxt(filename+'_true.txt', trues)
    np.savetxt(filename+'_pred.txt', preds)

def save_inverse_samples_multi_rows(cfg, model, total_test, grid_x, grid_y):

    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    Esct_data, true_epsr_data, _ = TestReader(cfg['DATA_ROOT'], grid_x*grid_y, cfg['FIELD_DIM'], total_test)
    
    pred_list = []
    true_list = []
    title_list = []
    for i in range(grid_x):
        pred=[]
        true=[]
        title=[]
        for j in range(grid_y):
            with torch.no_grad():
                point_num = 10000
                square = square_generator(1, point_num)
                square = np.transpose(square, (0, 2, 1))
                square = Variable(torch.Tensor(square)).cuda()
                Esct = Variable(torch.Tensor(Esct_data[i*grid_y+j])).cuda()

                pred_pc, _ = model.forward(Esct, square, point_num)
                pred_pc = pred_pc[:, :cfg['PC_DIM'], :]
                pred_pc = pred_pc.permute(0, 2, 1)
                pred_pc = pred_pc.cpu().detach().numpy()
                true_epsr = true_epsr_data[i*grid_y+j]

            pred.append(pred_pc[0,...])
            title.append("S_%d" % (i * grid_y + j))
            true.append(true_epsr[0,...])
            
        pred_list.append(pred)
        title_list.append(title)
        true_list.append(true)

    plot_name = os.path.join('./Result', 'pred_epsr')
    plot_scatter_multi_rows(plot_name, pred_list, title_list)
    plot_name = os.path.join('./Result', 'true_epsr')
    plot_scatter_multi_rows(plot_name, true_list, title_list)


def plot_scatter_multi_rows(filename, pcds, titles, suptitle='', size=0.2, cmap=plt.cm.jet, xlim=(cfg['DOI_MIN'], cfg['DOI_MAX']), ylim=(cfg['DOI_MIN'], cfg['DOI_MAX']), zlim=(cfg['DOI_MIN'], cfg['DOI_MAX'])):

    fig = plt.figure(figsize=(len(pcds[0])*3, len(pcds)*3)) # W,H
    for i in range(len(pcds)):
        for j, pcd in enumerate(pcds[i]):
            ax = fig.add_subplot(len(pcds), len(pcds[i]), i * len(pcds[i]) + j + 1, projection = '3d')
            ax.scatter(pcd[:, 2], pcd[:, 0], pcd[:, 1], c=pcd[:, 3], s=size, marker='.', cmap=cmap, vmin=0, vmax=1.0)
            ax.set_title(titles[i][j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.invert_zaxis()
            ax.view_init(15, 35)

            np.savetxt(filename+'_%d.txt'%(i*len(pcds[i])+j), pcd)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename+'.png')
    plt.close(fig)


def save_forward_samples_multi_rows(cfg, model, total_test, grid_x, grid_y):

    if not os.path.exists('./Result'):
        os.makedirs('./Result')

    Esct_data, true_epsr_data, point_nums = TestReader(cfg['DATA_ROOT'], grid_x*grid_y, cfg['FIELD_DIM'], total_test)

    pred_list = []
    true_list = []
    epsr_list = []
    title_list = []
    for i in range(grid_x):
        pred=[]
        true=[]
        epsr=[]
        title=[]
        for j in range(grid_y):
            with torch.no_grad():
                point_num = point_nums[i*grid_y+j]
                pc = np.transpose(true_epsr_data[i*grid_y+j], (0, 2, 1))
                pc = Variable(torch.Tensor(pc)).cuda()
                pred_Esct = model.sample(pc, 1, point_num)
                pred_Esct = pred_Esct.cpu().detach().numpy()*2.0
                true_Esct = Esct_data[i*grid_y+j]*2.0
            pred.append(pred_Esct[0,...])
            title.append("S_%d" % (i * grid_y + j))
            true.append(true_Esct[0,...])
            epsr.append(true_epsr_data[i*grid_y+j][0,...])

        pred_list.append(pred)
        title_list.append(title)
        true_list.append(true)
        epsr_list.append(epsr)

    plot_name = os.path.join('./Result', 'Esct')
    plot_lines_multi_rows(plot_name, pred_list, true_list, title_list)


def plot_lines_multi_rows(filename, preds, trues, titles, suptitle=''):

    fig = plt.figure(figsize=(len(trues[0])*3, len(trues)*3)) # W,H
    for i in range(len(trues)):
        for j in range(len(trues[i])):
            ax = fig.add_subplot(len(trues), len(trues[i]), i * len(trues[i]) + j + 1)
            ax.plot(trues[i][j],color='r')
            ax.plot(preds[i][j],color='b')
            ax.set_title(titles[i][j])
            ax.set_axis_off()

            np.savetxt(filename+'_true_%d.txt'%(i*len(trues[i])+j), trues[i][j])
            np.savetxt(filename+'_pred_%d.txt'%(i*len(trues[i])+j), preds[i][j])

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename+'.png')
    plt.close(fig)



model = NICE(cfg)
model.cuda()

save_path = os.path.join(cfg['MODEL_SAVE_PATH'], 'model_dev_best.pt')
model_dict = torch.load(save_path)
model_dict = model.load_state_dict(torch.load(save_path))

total_test = 500

# save_inverse_sample(cfg, model, total_test)
# save_forward_sample(cfg, model, total_test)
save_inverse_samples_multi_rows(cfg, model, total_test, 8, 8)
save_forward_samples_multi_rows(cfg, model, total_test, 8, 8)