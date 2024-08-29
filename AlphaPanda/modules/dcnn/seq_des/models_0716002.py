import torch
import torch.nn as nn
import AlphaPanda.modules.dcnn.seq_des.util.data as data
import AlphaPanda.modules.dcnn.common.atoms


def init_ortho_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight)
        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.orthogonal_(module.weight)


class seqPred(nn.Module):
    def __init__(self, nic, nf=64, momentum=0.01):
        super(seqPred, self).__init__()
        self.nic = nic
        self.model = nn.Sequential(
            # 20 -- 10
            nn.Conv3d(nic, nf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf, nf, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf, nf, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 10 -- 5
            nn.Conv3d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 2, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 2, nf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 2, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            # 5 -- 1
            nn.Conv3d(nf * 2, nf * 4, 5, 1, 0, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm3d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
       #huyue 
        # res pred
        )
        # res pred
        self.out = nn.Sequential(
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, nf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm1d(nf * 4, momentum=momentum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(nf * 4, len(AlphaPanda.modules.dcnn.common.atoms.label_res_dict.keys()), 3, 1, 1, bias=False),
        )


    def forward(self, input):
        bs = input.size()[0]
        feat = self.model(input).view(bs, -1, 1)
        res_pred = self.out(feat).view(bs, -1)

        return res_pred
