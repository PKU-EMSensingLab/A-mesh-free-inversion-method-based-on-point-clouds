import os
import torch
import torch.optim as optim
from tqdm import tqdm
from DataLoader import *
from config import cfg
from nice import NICE
import math
from utils import *
import time
from torch.autograd import Variable
import torch.nn as nn

if not os.path.exists(cfg['MODEL_SAVE_PATH']):
    os.makedirs(cfg['MODEL_SAVE_PATH'])

model = NICE(cfg)
model.cuda()

if cfg['LOAD']:
    save_path = os.path.join(cfg['MODEL_SAVE_PATH'], 'model_dev_best.pt')
    model_dict = torch.load(save_path)
    model_dict = model.load_state_dict(torch.load(save_path))

print('# Parameters:', sum(param.numel() for param in model.parameters()))

alpha = cfg['WEIGHT']
beta1 = 0.5
beta2 = 0.99

opt = optim.Adam(model.parameters(), lr=cfg['LEARNING_RATE'], betas=(beta1, beta2))

start_time = time.time()
train_loss = AverageValueMeter()
train_forward_loss = AverageValueMeter()
train_inverse_loss = AverageValueMeter()
val_loss = AverageValueMeter()
val_forward_loss = AverageValueMeter()
val_inverse_loss = AverageValueMeter()

total_train = cfg['TRAIN_NUM']
total_dev = cfg['DEV_NUM']

best_train_loss = np.inf
best_val_loss = np.inf

for epoch in range(1, cfg['EPOCHS']):
    print('Epoch: ',epoch)
    model.train()
    train_loss.reset()
    train_forward_loss.reset()
    train_inverse_loss.reset()
    for Esct_data, epsr_data in tqdm(TrainLoader(cfg['DATA_ROOT'], cfg['BATCH_SIZE'], cfg['POINT_NUM'], cfg['FIELD_DIM'], cfg['PC_DIM'], total_train), total=math.ceil(total_train/cfg['BATCH_SIZE'])):
        opt.zero_grad()

        # inverse: Esct -> epsr
        square = square_generator(cfg['BATCH_SIZE'], cfg['POINT_NUM'])
        square = np.transpose(square, (0, 2, 1))
        square = Variable(torch.Tensor(square)).cuda()
        Esct = Variable(torch.Tensor(Esct_data)).cuda()

        pred_pc, _ = model.forward(Esct, square, cfg['POINT_NUM'])
        pred_pc = pred_pc[:, :cfg['PC_DIM'], :]
        pred_pc = pred_pc.permute(0, 2, 1)
        real_pc = Variable(torch.Tensor(epsr_data)).cuda()

        loss_inverse = chamfer_loss(pred_pc, real_pc)

        train_inverse_loss.update(loss_inverse.item())

        # forward: epsr -> Esct
        pc = np.transpose(epsr_data, (0, 2, 1))
        pc = Variable(torch.Tensor(pc)).cuda()
        pred_Esct = model.sample(pc, cfg['BATCH_SIZE'], cfg['POINT_NUM'])
        real_Esct = Variable(torch.Tensor(Esct_data)).cuda()

        loss_forward = nn.functional.mse_loss(pred_Esct, real_Esct)

        train_forward_loss.update(loss_forward.item())

        loss = loss_inverse + alpha * loss_forward

        loss.backward()
        opt.step()

        train_loss.update(loss.item())

    model.eval()
    val_loss.reset()
    val_forward_loss.reset()
    val_inverse_loss.reset()
    with torch.no_grad():
        for Esct_data, epsr_data in tqdm(DevLoader(cfg['DATA_ROOT'], cfg['BATCH_SIZE'], cfg['POINT_NUM'], cfg['FIELD_DIM'], cfg['PC_DIM'], total_dev),total=math.ceil(total_dev/cfg['BATCH_SIZE'])):
            
            # inverse: Esct -> epsr
            square = square_generator(cfg['BATCH_SIZE'], cfg['POINT_NUM'])
            square = np.transpose(square, (0, 2, 1))
            square = Variable(torch.Tensor(square)).cuda()
            Esct = Variable(torch.Tensor(Esct_data)).cuda()

            pred_pc, _ = model.forward(Esct, square, cfg['POINT_NUM'])
            pred_pc = pred_pc[:, :cfg['PC_DIM'], :]
            pred_pc = pred_pc.permute(0, 2, 1)
            real_pc = Variable(torch.Tensor(epsr_data)).cuda()

            vloss_inverse = chamfer_loss(pred_pc, real_pc)

            val_inverse_loss.update(vloss_inverse.item())

            # forward: epsr -> Esct
            pc = np.transpose(epsr_data, (0, 2, 1))
            pc = Variable(torch.Tensor(pc)).cuda()
            pred_Esct = model.sample(pc, cfg['BATCH_SIZE'], cfg['POINT_NUM'])
            real_Esct = Variable(torch.Tensor(Esct_data)).cuda()

            vloss_forward = nn.functional.mse_loss(pred_Esct, real_Esct)

            val_forward_loss.update(vloss_forward.item())

            vloss = vloss_inverse + alpha * vloss_forward

            val_loss.update(vloss.item())

    time_tick = time.time() - start_time
    print('time: %2dm %2ds'%(time_tick/60, time_tick%60))
    print('train_loss: %.8f val_loss: %.8f'%(train_loss.avg, val_loss.avg))
    print('train_forward_loss: %.8f val_forward_loss: %.8f'%(train_forward_loss.avg, val_forward_loss.avg))
    print('train_inverse_loss: %.8f val_inverse_loss: %.8f'%(train_inverse_loss.avg, val_inverse_loss.avg))
    print('learning_rate: %f'% (opt.param_groups[0]['lr']))

    if epoch % cfg['SNAPSHOT'] == 0:
        draw_inverse_sample(cfg, model, epoch, total_dev)
        draw_forward_sample(cfg, model, epoch, total_dev)
        text = 'Figures have saved!!!\nEpoch is %d'%(epoch)
    
    if best_train_loss > train_loss.avg:
        save_path = os.path.join(cfg['MODEL_SAVE_PATH'], 'model_train_best.pt')
        torch.save(model.state_dict(), save_path)
        print('train: ',best_train_loss,'==>', train_loss.avg)
        best_train_loss = train_loss.avg

    if best_val_loss > val_loss.avg:
        save_path = os.path.join(cfg['MODEL_SAVE_PATH'], 'model_dev_best.pt')
        torch.save(model.state_dict(), save_path)
        print('dev: ', best_val_loss,'==>', val_loss.avg)
        best_val_loss = val_loss.avg

save_path = os.path.join(cfg['MODEL_SAVE_PATH'], 'model_final.pt')
torch.save(model.state_dict(), save_path)