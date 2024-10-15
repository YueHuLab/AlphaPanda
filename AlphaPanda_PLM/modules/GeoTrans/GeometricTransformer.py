from builtins import print
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import math
from schnetpack import Properties
#from Utils import AtomDistances
from diffab.modules.GeoTrans.Utils import AtomDistances #huyue
from torch.nn import LayerNorm
###
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module
from torch.overrides import handle_torch_function, has_torch_function

import schnetpack
#########################################
class GeoTransformer(nn.Module):
    #def __init__(self, nhead=1, num_encoder_layers=3, d_model=128, dropout=0, property_stats = None,atomref=None):
    def __init__(self, nhead=1, num_encoder_layers=3, d_model=128, dropout=0): #huyue
        super(GeoTransformer, self).__init__()
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.d_ff = self.d_model*4
        self.dropout = dropout

        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout=dropout)
        ff = torch.nn.Sequential(
            *[GEGLU(d_model, self.d_ff, bias=False),
                nn.Dropout(p=dropout),
                nn.Linear(self.d_ff, d_model, bias=False)])
        self.Encoder = Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout), num_encoder_layers)
            
        for name, p  in self.named_parameters():
            if p.dim() > 1 and not any([x in name for x in ['molecule', 'transform_dist']]):
                nn.init.xavier_uniform_(p)
        self.AtomEmb = torch.nn.Embedding(50, d_model, padding_idx=0)
        self.distances = AtomDistances()
        self.pos_encoder = Pos_encoding(d_model,dims=50)
        #self.mean = property_stats[0] if property_stats is not None else 0
        #self.std = property_stats[1] if property_stats is not None else 1
        #huyue
        #self.atomref = nn.Embedding.from_pretrained(
        #    torch.from_numpy(atomref.astype(np.float32))) if atomref is not None else lambda x: 0
        
        # self.pred_net = nn.Linear(d_model,1)
        #huyue 
        #self.pred_net = nn.Sequential(
        #    schnetpack.nn.MLP(d_model, 1, None, 1, schnetpack.nn.activations.shifted_softplus))
    def forward(self, inputs):
        charges = inputs['PropertiesZ']
        atom_mask = inputs['Propertiesatom_mask'].bool()

        pairwise_distances = self.distances(
            inputs['PropertiesR'],
            inputs['Propertiesneighbors'],
            neighbor_mask=atom_mask,
            inverse_flag=True)
            #inverse_flag=False) #huyue 0525
        coord_repr = self.Encoder(self.AtomEmb(charges) + self.pos_encoder(pairwise_distances,charges), atom_mask, src_distances=pairwise_distances)
        #print(' pairwise_distance \n')
        #print(pairwise_distances)
        #print(' pairwise_distance \n')
        #print(' self atomEmb \n')
        #print(self.AtomEmb(charges))
        #print(' \n')
        #print(' self pos_encoder \n')
        #print(self.pos_encoder(pairwise_distances,charges))
        #print(' \n')
        #print(' coord_repr \n')
        #print(coord_repr)
        #print('  \n')
        #print(' atom_mask \n')
        #print(atom_mask)
        #print(' \n')
        #pred = (self.pred_net(repr))* self.std + self.mean +self.atomref(charges)
        #return torch.sum(pred*inputs[Properties.atom_mask][..., None], 1)
        #huyue
        return coord_repr 

    def extra_repr(self):
        return 'nhead={}, num_encoder_layers={}, d_model={}, d_ff={}, dropout={}'.format(
            self.nhead, self.num_encoder_layers, self.d_model, self.d_ff, self.dropout)


#########################################
class Pos_encoding(nn.Module):
    def __init__(self,d_model,dims=50):
        super(Pos_encoding, self).__init__()
        self.net = torch.nn.Sequential(*[nn.Linear(1, dims,
                                                   bias=True), torch.nn.GELU(), nn.Linear(dims, 1,
                                                                                          bias=True),
                                         torch.nn.GELU()])
        self.embed = nn.Linear(1, d_model,
                               bias=True)

    def forward(self, x, mask):
        x = self.net(x.unsqueeze(-1))
        x = x*(torch.nn.functional.pad(mask.unsqueeze(-2),pad=(0,x.shape[-2]-mask.shape[-1]),mode='constant',value=0).unsqueeze(-1)>0)
        x = x*(mask.unsqueeze(-1).unsqueeze(-1)>0)
        x = torch.sum(x, -2)
        return self.embed(x)

class distance_NN(nn.Module):
    def __init__(self, hidden_size=10, num_layers=0):
        super(distance_NN, self).__init__()

        non_lin_fun = torch.nn.ReLU
        layers = [torch.nn.Linear(1, hidden_size), non_lin_fun()]
        for kk in range(num_layers):
            layers += [torch.nn.Linear(hidden_size, hidden_size),
                       non_lin_fun()]
        layers += [torch.nn.Linear(hidden_size, 1)]
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, D_mat):
        return self.seq(D_mat.unsqueeze(-1)).squeeze(-1)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, bias=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

        self.transform_dist = torch.nn.Identity()
        self.transform_dist = distance_NN(50,2)

        self.attention_fun = self.attention_mul_after_softmax

    def forward(self, query, key, value, mask, src_distances=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,
                                                                 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention_fun(query, key, value, heads=self.h,
                                  mask=mask,src_distances=src_distances)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


    def attention_mul_after_softmax(self, query, key, value, heads,
                                    mask=None,src_distances=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        src_distances = self.transform_dist(src_distances)
        #scores = torch.matmul(query,key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.matmul(query,key.transpose(-2, -1)) /( math.sqrt(d_k) + float(1e-8) ) #huyue

        #scores.diagonal(dim1=-2, dim2=-1)[:] = float('-inf')
        scores.diagonal(dim1=-2, dim2=-1)[:] = float(-1e+8)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0,
                                        #float('-inf')).masked_fill(
                                        float(-1e+8)).masked_fill( #huyue
                mask.unsqueeze(-2) == 0, float(-1e+8)) #huyue
                #mask.unsqueeze(-2) == 0, float('-inf'))
            src_distances = src_distances.masked_fill(
                mask.squeeze().unsqueeze(-1) == 0,
                0).masked_fill(
                mask.squeeze().unsqueeze(-2) == 0,
                0)

        p_attn = F.softmax(scores, dim=-1)
        #TODO put the **2 before the repeat
        p_attn = p_attn * src_distances.unsqueeze(1).repeat(1,
                                                            heads,
                                                            1,
                                                            1) ** 2
        if mask is not None:
            p_attn = p_attn.masked_fill(mask.unsqueeze(-1) == 0, 0)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

#########################################
#########################################
#########################################

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, src_distances):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, src_distances)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask, src_distances):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask,
                                                         src_distances))
        return self.sublayer[1](x, self.feed_forward)

#########################################
#########################################
def geglu_fun(input, weight, weight2, bias=None, bias2=None):
    # tens_ops = (input, weight)
    # if not torch.jit.is_scripting():
    #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
    #         return handle_torch_function(linear, tens_ops, input, weight, bias=bias)
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    ret = output
    if bias2 is not None:
        ret = F.gelu(ret) * (input.matmul(weight2.t()) + bias2)
    else:
        ret = F.gelu(ret) * input.matmul(weight2.t())
    return ret


class GEGLU(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(GEGLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight2 = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias2 = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            #bound = 1 / math.sqrt(fan_in)
            bound = 1 / (math.sqrt(fan_in) + float(1e-8) ) #huyue
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.bias2, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return geglu_fun(input, self.weight, self.weight2, self.bias, self.bias2)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

#########################################

if __name__ == '__main__':
    pass
