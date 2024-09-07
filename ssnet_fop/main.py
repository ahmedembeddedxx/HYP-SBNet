from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
from scipy import random
from sklearn import preprocessing
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from retrieval_model import FOP, HyperbolicLoss  # Updated import

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average


def read_data(FLAGS):
    
    print('Split Type: %s'%(FLAGS.split_type))
    
    if FLAGS.split_type == 'voice_only':
        print('Reading Voice Train')
        train_file_voice = '../data/voice/voiceTrain.csv'
        train_data = pd.read_csv(train_file_voice, header=None)
        train_label = train_data[512]
        le = preprocessing.LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label)
        train_data = np.asarray(train_data)
        train_data = train_data[:, :-1]
        
        return train_data, train_label
        
    elif FLAGS.split_type == 'face_only':
        print('Reading Face Train')
        train_file_face = '../data/face/facenetfaceTrain.csv'
        train_data = pd.read_csv(train_file_face, header=None)
        train_label = train_data[512]
        le = preprocessing.LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label)
        train_data = np.asarray(train_data)
        train_data = train_data[:, :-1]
        
        return train_data, train_label
    
    train_data = []
    train_label = []
    
    train_file_face = '../data/face/facenetfaceTrain.csv'
    train_file_voice = '../data/voice/voiceTrain.csv'
    
    print('Reading Train Faces')
    img_train = pd.read_csv(train_file_face, header=None)
    train_tmp = img_train[512]
    img_train = np.asarray(img_train)
    img_train = img_train[:, :-1]
    
    train_tmp = np.asarray(train_tmp)
    train_tmp = train_tmp.reshape((train_tmp.shape[0], 1))
    print('Reading Train Voices')
    voice_train = pd.read_csv(train_file_voice, header=None)
    voice_train = np.asarray(voice_train)
    voice_train = voice_train[:, :-1]
    
    combined = list(zip(img_train, voice_train, train_tmp))
    random.shuffle(combined)
    img_train, voice_train, train_tmp = zip(*combined)
    
    if FLAGS.split_type == 'random':
        train_data = np.vstack((img_train, voice_train))
        train_label = np.vstack((train_tmp, train_tmp))
        combined = list(zip(train_data, train_label))
        random.shuffle(combined)
        train_data, train_label = zip(*combined)
        train_data = np.asarray(train_data).astype(np.float)
        train_label = np.asarray(train_label)
    
    elif FLAGS.split_type == 'vfvf':
        for i in range(len(voice_train)):
            train_data.append(voice_train[i])
            train_data.append(img_train[i])
            train_label.append(train_tmp[i])
            train_label.append(train_tmp[i])
            
    elif FLAGS.split_type == 'fvfv':
        for i in range(len(voice_train)):
            train_data.append(img_train[i])
            train_data.append(voice_train[i])
            train_label.append(train_tmp[i])
            train_label.append(train_tmp[i])        
    
    elif FLAGS.split_type == 'hefhev':
        train_data = np.vstack((img_train, voice_train))
        train_label = np.vstack((train_tmp, train_tmp))
        
    elif FLAGS.split_type == 'hevhef':
        train_data = np.vstack((voice_train, img_train))
        train_label = np.vstack((train_tmp, train_tmp))
    
    else:
        print('Invalid Split Type')
    
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    
    print("Train file length", len(img_train))
    print('Shuffling\n')
    
    train_data = np.asarray(train_data).astype(np.float)
    train_label = np.asarray(train_label)
    
    return train_data, train_label

def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(train_data, train_label):
    n_class = 901
    model = FOP(FLAGS, train_data.shape[1], n_class)
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss().cuda()
    hyperbolic_loss = HyperbolicLoss(FLAGS).cuda()  # Use Hyperbolic Loss
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        hyperbolic_loss.cuda()
        cudnn.benchmark = True
    
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    for alpha in FLAGS.alpha_list:
        eer_list = []
        epoch=1
        num_of_batches = (len(train_label) // FLAGS.batch_size)
        loss_plot = []
        auc_list = []
        loss_per_epoch = 0
        s_fac_per_epoch = 0
        d_fac_per_epoch = 0
        txt_dir = 'output'
        save_dir = 'fc2_%s_%s_alpha_%0.2f'%(FLAGS.split_type, FLAGS.save_dir, alpha)
        txt = '%s/ce_hyperbolic_%03d_%0.2f.txt'%(txt_dir, FLAGS.max_num_epoch, alpha)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        
        with open(txt,'w+') as f:
            f.write('EPOCH\tLOSS\tEER\tAUC\tS_FAC\tD_FAC\n')
        
        save_best = 'best_%s'%(save_dir)
        
        if not os.path.exists(save_best):
            os.mkdir(save_best)
        with open(txt,'a+') as f:
            while (epoch < FLAGS.max_num_epoch):
                print('%s\tEpoch %03d'%(FLAGS.split_type, epoch))
                for idx in tqdm(range(num_of_batches)):
                    train_batch, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, train_data)
                    loss_tmp, loss_hyperbolic, loss_soft, s_fac, d_fac = train(train_batch, 
                                                                 batch_labels, 
                                                                 model, optimizer, ce_loss, hyperbolic_loss, alpha)
                    loss_per_epoch+=loss_tmp
                    s_fac_per_epoch+=s_fac
                    d_fac_per_epoch+=d_fac
                
                loss_per_epoch/=num_of_batches
                s_fac_per_epoch/=num_of_batches
                d_fac_per_epoch/=num_of_batches
                
                loss_plot.append(loss_per_epoch)
                if FLAGS.split_type == 'voice_only' or FLAGS.split_type == 'face_only':
                    eer, auc = onlineTestSingleModality.test(FLAGS, model, test_feat)
                else:
                    eer, auc = online_evaluation.test(FLAGS, model, test_feat)
                eer_list.append(eer)
                auc_list.append(auc)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d_%0.3f.pth.tar'%(epoch, eer*100))

                print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f, Min_EER: %0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, alpha, min(eer_list)))
                
                if eer <= min(eer_list):
                    min_eer = eer
                    max_auc = auc
                    save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, save_best, 'checkpoint.pth.tar')
            
                f.write('%04d\t%0.4f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc, s_fac_per_epoch, d_fac_per_epoch))
                loss_per_epoch = 0
                s_fac_per_epoch = 0
                d_fac_per_epoch = 0
                epoch += 1
        
        return loss_plot, min_eer, max_auc

def train(train_batch, labels, model, optimizer, ce_loss, hyperbolic_loss, alpha):
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    hyperbolic_losses = RunningAverage()

    model.train()
    train_batch = torch.from_numpy(train_batch).float()
    labels = torch.from_numpy(labels).long()
    if FLAGS.cuda:
        train_batch = train_batch.cuda()
        labels = labels.cuda()

    logits, embeddings = model(train_batch)
    
    loss = alpha * hyperbolic_loss(embeddings, labels)[0] + (1 - alpha)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        average_loss.update(loss.item())
        soft_losses.update(ce_loss(logits, labels).item())
        hyperbolic_losses.update(hyperbolic_loss(embeddings, labels)[0].item())
        
    return average_loss.avg, hyperbolic_losses.avg, soft_losses.avg, hyperbolic_losses.avg, soft_losses.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperbolic Loss for Face Verification')
    parser.add_argument('--split_type', default='voice_only', type=str)
    parser.add_argument('--save_dir', default='my_experiment', type=str)
    parser.add_argument('--dim_embed', default=256, type=int)
    parser.add_argument('--max_num_epoch', default=100, type=int)
    parser.add_argument('--alpha_list', default=[0.5], type=list)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', default=True, type=bool)
    FLAGS = parser.parse_args()
    
    train_data, train_label = read_data(FLAGS)
    main(train_data, train_label)
