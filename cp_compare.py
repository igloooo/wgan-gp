import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from gan_definition import *
import os
from matplotlib import pyplot as plt
import pickle
from time import time
import numpy as np


class DataLogger:
    def __init__(self):
        self.dict = dict()

    def add_scalar(self, keys, scalar):
        # add a scalar
        if type(keys) in (list, tuple):
            # multiple keys
            key = '/'.join(keys)
        else:
            key = keys

        if key in self.dict:
            self.dict[key].append(scalar)
        else:
            self.dict[key] = []

    def get_scalars(self, keys):
        # return a list of scalars under the key
        if type(keys) in (list, tuple):
            # multiple keys
            key = '/'.join(keys)
        else:
            key = keys
        return self.dict[key]



# set random seed
torch.manual_seed(1)
# hyper-parameters, experiment name and all the paths, dirs
exp_name = 'norm'  ##########
diff_only = False  ########
print(exp_name, 'diff_only=', diff_only)
cp_indices = list(range(999,33000,1000))
gt_path = 'tmp/mnist/checkpoints/iteration42999.pth'
dump_folder = 'simu_mnist42999_ncritic10'

plots_dir = '/'.join(['tmp', dump_folder, 'plots'])
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
logger_dir = '/'.join(['tmp', dump_folder, 'log'])
if not os.path.exists(logger_dir):
    os.makedirs(logger_dir)
#tfboard_path = 'tmp/{}/log/exp1'.format(dump_folder)
# # initialize summary writer
# if not os.path.exists(tfboard_path):
#     os.makedirs(tfboard_path)
# writer = SummaryWriter(tfboard_path)



# initialize logger
logger = DataLogger()
# find device
device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')  
# create two generators
netG_gt = Generator().to(device)
netG_cp = Generator().to(device)

# load gt parameters
gt_state = torch.load(gt_path, map_location=device)['p_g']
netG_gt.load_state_dict(gt_state)

for cp_index in cp_indices:
    #time1 = time()
    # load checkpoint
    cp_path = '/'.join(['tmp', dump_folder, 'checkpoints','iteration{}.pth'.format(cp_index)])
    cp_state = torch.load(cp_path, map_location=device)['p_g']
    netG_cp.load_state_dict(cp_state)
    # compare parameters layerwise
    for k,v in netG_gt.state_dict().items():
        if 'bias' in k:
            continue
        v_ = netG_cp.state_dict()[k]
        # reshape conv kernel into 2d matrix
        if len(v.size()) == 4:
            v = v.permute((0,2,3,1))
            v_ = v_.permute((0,2,3,1))
            v = v.reshape((-1, v.shape[-1])) # (input_dim, output_dim)
            v_ = v_.reshape((-1, v_.shape[-1]))
        #======modify the folloiwng part when trying a different experiment======
        normalized_v = v / torch.norm(v, dim=0, keepdim=True) # normalize the rows
        normalized_v_ = v_ / torch.norm(v_, dim=0, keepdim=True)
        sim_mat = torch.matmul(normalized_v_.T, normalized_v)
        sim = torch.mean(torch.max(sim_mat, dim=1)[0])#torch.mean(torch.diag(sim_mat))
        logger.add_scalar(k, sim)

        # cos_sim = torch.sum(v*v_) / (torch.norm(v)*torch.norm(v_))
        # logger.add_scalar(k, cos_sim)

        # rel_diff = torch.norm(v-v_) / torch.norm(v_)
        # logger.add_scalar(k, rel_diff)

        # try:
        #     t_learned = np.linalg.svd(v, compute_uv=False)[0] #torch.sum(v)/torch.numel(v)
        #     t_gt = np.linalg.svd(v_, compute_uv=False)[0] #torch.sum(v_)/torch.numel(v_)
        # except IndexError:
        #     t_learned = 0
        #     g_gt = 0

        t_learned = torch.norm(v)
        t_gt = torch.norm(v_)
        logger.add_scalar([k, 'learned'], t_learned)
        logger.add_scalar([k, 'gt'], t_gt)
        #======================================================================
    logger.add_scalar('indices', cp_index)
    #print(time() - time1)

        #writer.add_scalars(k, {'learned': t_learned}, cp_index)
        #writer.add_scalars(k, {'gt': t_gt}, cp_index)


#writer.close()
with open('/'.join([logger_dir, exp_name+'.pkl']), 'wb') as f:
    pickle.dump(logger, f)

layer_names = [k for k in netG_gt.state_dict()]

if not diff_only:
    for k in layer_names:
        if 'bias' not in k:
        # only plot the weights
            short_k = k.split('.')[0]
            plt.plot(logger.get_scalars('indices'), logger.get_scalars([k, 'gt']), label='ground-truth')
            plt.plot(logger.get_scalars('indices'), logger.get_scalars([k, 'learned']), label='learned')
            plt.legend()
            plt.title(exp_name+'_'+short_k)
            plt.savefig('/'.join([plots_dir, exp_name+'_'+short_k+'.png']))
            plt.close()

    for k in layer_names:
        if 'bias' not in k:
        # only plot the weights
            short_k = k.split('.')[0]
            learned_scalars = logger.get_scalars([k, 'learned'])
            gt_scalars = logger.get_scalars([k, 'gt'])
            rel_diff = [(learned_scalars[i] - gt_scalars[i])/gt_scalars[i] for i in range(len(gt_scalars))]
            plt.plot(
                logger.get_scalars('indices'), 
                rel_diff,
                label=short_k)
    plt.legend()
    plt.title(exp_name+' (learned - gt) / gt')
    plt.savefig('/'.join([plots_dir, exp_name + '_rel_diff.png']))
    plt.close()
else:
    for k in layer_names:
        if 'bias' not in k:
            # only plot the weights
            short_k = k.split('.')[0]
            diff = logger.get_scalars(k)
            plt.plot(
                logger.get_scalars('indices'), 
                diff,
                label=short_k)
    plt.legend()
    plt.title(exp_name)
    plt.savefig('/'.join([plots_dir, exp_name + '.png']))
    plt.close()
