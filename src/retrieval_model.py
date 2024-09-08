import torch
import torch.nn as nn
import torch.nn.functional as F
import hyptorch.nn as hypnn  # Ensure this is properly installed

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))

class HyperbolicLoss(nn.Module):
    def __init__(self, c, reduction='mean'):
        super(HyperbolicLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, s_fac, labels):
        # Compute loss between the hyperbolic embeddings (s_fac, d_fac) and the true labels
        dist = torch.cdist(s_fac, s_fac, p=2)  # Euclidean distance
        target = labels.float()
        loss = F.mse_loss(dist, target, reduction=self.reduction)

        return loss

        # # Compute loss between the hyperbolic embeddings (s_fac, d_fac) and the true labels
        # dist = torch.cdist(s_fac, d_fac, p=2)  # Euclidean distance
        # target = target.float()
        # loss = F.mse_loss(dist, target, reduction=self.reduction)  # Mean squared error for hyperbolic distance

        # return loss


'''
Embedding Extraction Module
'''        

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()
        self.fc2 = make_fc_1d(embedding_dim, embedding_dim)
        self.fc3 = make_fc_1d(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.functional.normalize(x) 
        return x


'''
Main Module
'''

class FOP(nn.Module):
    def __init__(self, args, feat_dim, n_class):
        super(FOP, self).__init__()
        
        self.embed_branch = EmbedBranch(feat_dim, args.dim_embed)

        self.c = 0.33
        self.ball_dim = 20
        self.ln = nn.Linear(args.dim_embed, self.ball_dim)
        self.tp = hypnn.ToPoincare(c=self.c, train_x=False, train_c =False, ball_dim=self.ball_dim)
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.ball_dim, n_classes=n_class, c=self.c)

        self.logits_layer = nn.Linear(args.dim_embed, n_class)

        if args.cuda:
            self.cuda()

    def forward(self, feats):
        feats = self.embed_branch(feats)

        feats = self.ln(feats)
        feats = self.tp(feats)
        logits = self.mlr(feats, c=self.tp.c)
        
        return feats, logits
    
    def train_forward(self, feats):
        comb = self(feats)
        return comb
