import torch
import os
from DataLoader import *
from torch.autograd import Variable
import matplotlib.pylab  as plt
from config import cfg

class AverageValueMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def square_generator(bs,n):

    square = cfg['DOI_MIN'] + (cfg['DOI_MAX']-cfg['DOI_MIN']) * np.random.random((bs, n, 3))
    square = np.concatenate([square, np.zeros((bs, n, 1))], axis=-1)

    return square


def chamfer_loss(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    xy = torch.bmm(x, y.transpose(2, 1))
    dtype = torch.cuda.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(xy.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(xy)
    P = (rx.transpose(2, 1) + ry - 2 * xy)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.mean(mins)

    return loss_1 + loss_2


def draw_inverse_sample(cfg, model, step, total_dev):

    if not os.path.exists(cfg['FIG_ROOT']):
        os.makedirs(cfg['FIG_ROOT'])

    grid_x = 8
    grid_y = 8

    Esct_data, true_epsr_data, point_nums = DevReader(cfg['DATA_ROOT'], grid_x*grid_y, cfg['FIELD_DIM'], total_dev)
    
    pred_list = []
    true_list = []
    title_list = []
    for i in range(grid_x):
        pred=[]
        true=[]
        title=[]
        for j in range(grid_y):
            with torch.no_grad():
                point_num=point_nums[i*grid_y+j]
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

    plot_name = os.path.join(cfg['FIG_ROOT'], 'pred_epsr_'+str(step) + ".png")
    plot_scatter_multi_rows(plot_name, pred_list, title_list)
    plot_name = os.path.join(cfg['FIG_ROOT'], 'true_epsr_'+str(step) + ".png")
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

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def draw_forward_sample(cfg, model, step, total_dev):

    if not os.path.exists(cfg['FIG_ROOT']):
        os.makedirs(cfg['FIG_ROOT'])

    grid_x = 8
    grid_y = 8

    Esct_data, true_epsr_data, point_nums = DevReader(cfg['DATA_ROOT'], grid_x*grid_y, cfg['FIELD_DIM'], total_dev)

    pred_list = []
    true_list = []
    title_list = []
    for i in range(grid_x):
        pred=[]
        true=[]
        title=[]
        for j in range(grid_y):
            with torch.no_grad():
                point_num=point_nums[i*grid_y+j]
                pc = np.transpose(true_epsr_data[i*grid_y+j], (0, 2, 1))
                pc = Variable(torch.Tensor(pc)).cuda()
                pred_Esct = model.sample(pc, 1, point_num)
                pred_Esct = pred_Esct.cpu().detach().numpy()
                true_Esct = Esct_data[i*grid_y+j]
            pred.append(pred_Esct[0,...])
            title.append("S_%d" % (i * grid_y + j))
            true.append(true_Esct[0,...])

        pred_list.append(pred)
        title_list.append(title)
        true_list.append(true)

    plot_name = os.path.join(cfg['FIG_ROOT'], 'Esct_'+str(step) + ".png")
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

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)