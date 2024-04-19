import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from AlphaPanda.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from AlphaPanda.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from AlphaPanda.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


from .getBackAnlge import get_batch_N_CA_C_align_back, get_batch_N_CA_align_back 
#huyue
#from AlphaPanda.modules.alphafold2_pytorch.alphafold2 import Evoformer
#from AlphaPanda.modules.alphafold2_pytorch.alphafold2 import PairwiseAttentionBlock
#from AlphaPanda.modules.alphafold2_pytorch.alphafold2 import FeedForward
from einops import rearrange, reduce
#from AlphaPanda.modules.GeoTrans.GeometricTransformer import GeoTransformer #huyue
from AlphaPanda.modules.dcnn.seq_des.util.voxelize import voxelize #huyue
import AlphaPanda.modules.dcnn.seq_des.util.canonicalize as canonicalize
import AlphaPanda.modules.dcnn.seq_des.models as TreeCNNmodels
import AlphaPanda.modules.dcnn.common.atoms
from AlphaPanda.modules.common.geometry import reconstruct_backbone_partially
#huyue
#from AlphaPanda.modules.common.so3 import so3vec_to_rotation
from AlphaPanda.modules.common.geometry import reconstruct_backbone
from AlphaPanda.utils.protein.constants import BBHeavyAtom
#huyue 0519
#torch.set_printoptions(profile="full")

def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_dim=res_feat_dim
        self.nf=64
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        #huyue add Evofomer pairwise block.
        #self.pairwise_attention= PairwiseAttentionBlock( dim = dim, dim_head = dim_head, heads = heads, seq_len = max_seq_len)
       # self.evoformer=Evoformer(
        #              dim = res_feat_dim,
         ##           depth = 2,
           #     seq_len = 2048,
            #        heads = 2,
             #       dim_head = 64,
              #      attn_dropout = 0.,
               #     ff_dropout = 0.
                #    )
# may the configs could have parameters
    #    self.pairwiseAttention = PairwiseAttentionBlock(
     #   dim = pair_feat_dim,
      #  dim_head = 64,
       # heads = 8, # huyue 
        #seq_len = 2048
        #)
        
        #self.pair_ff=FeedForward(dim = pair_feat_dim, dropout =  0.)

        #huyue Geo tranformer
        self.d_model=256
        #self.geo_trans= GeoTransformer(nhead=256, num_encoder_layers=10, d_model=512) #huyue .to(device)
        #self.geo_trans_0= GeoTransformer(nhead=32, num_encoder_layers=5, d_model=64) #huyue .to(device)
        #huyue
        #self.geo_trans_1= GeoTransformer(nhead=32, num_encoder_layers=5, d_model=256) #huyue .to(device)
        #self.geo_trans_2= GeoTransformer(nhead=32, num_encoder_layers=5, d_model=128) #huyue .to(device)
        #self.geo_trans_3= GeoTransformer(nhead=32, num_encoder_layers=5, d_model=128) #huyue .to(device)

        #end
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        #self.eps_crd_net = nn.Sequential(
        
        #    nn.Conv1d(self.nf * 4 + res_feat_dim+3, self.nf * 4, 3, 1, 1, bias=False),
        #    nn.BatchNorm1d(self.nf * 4, momentum=0.01),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Dropout(0.1),
        #    nn.Conv1d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
        #    nn.BatchNorm1d(self.nf * 4, momentum=0.01),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Dropout(0.1),
        #    nn.Conv1d(self.nf * 4, 3, 3, 1, 1, bias=False)
        #)

        #self.eps_rot_net = nn.Sequential(
        #    nn.Conv1d(self.nf * 4 + res_feat_dim+3, self.nf * 4, 3, 1, 1, bias=False),
        #    nn.BatchNorm1d(self.nf * 4, momentum=0.01),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Dropout(0.1),
        #    nn.Conv1d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
        #    nn.BatchNorm1d(self.nf * 4, momentum=0.01),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Dropout(0.1),
        #    nn.Conv1d(self.nf * 4, 3, 3, 1, 1, bias=False)
        #)

        #self.eps_seq_net = nn.Sequential(
        #    nn.Conv1d(self.nf * 4 + res_feat_dim+3, self.nf * 4, 3, 1, 1, bias=False),
        #    nn.BatchNorm1d(self.nf * 4, momentum=0.01),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Dropout(0.1),
        #    nn.Conv1d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
        #    nn.BatchNorm1d(self.nf * 4, momentum=0.01),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    nn.Dropout(0.1),
        #    nn.Conv1d(self.nf * 4, 20, 3, 1, 1, bias=False),nn.Softmax(dim=-1)
        #)
        self.eps_crd_net = nn.Sequential(
            nn.Linear(self.nf * 4 + res_feat_dim+3, res_feat_dim * 2 ), nn.ReLU(),
            nn.Linear(res_feat_dim * 2 , res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(self.nf * 4 + res_feat_dim+3, res_feat_dim * 2), nn.ReLU(),
            nn.Linear(res_feat_dim * 2 , res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(self.nf * 4 + res_feat_dim+3, res_feat_dim * 2), nn.ReLU(),
            #nn.Conv1d(self.nf * 4 + res_feat_dim+3, self.nf * 4, 3, 1, 1, bias=False),
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )



        #self.TreeCNN=TreeCNNmodels.seqPred(nic=len(AlphaPanda.modules.dcnn.common.atoms.atoms) + 1 + 21, nf=64, momentum=0.01)
        #self.TreeCNN=TreeCNNmodels.seqPred(nic=13, nf=self.nf, momentum=0.01)
        #self.TreeCNN=TreeCNNmodels.seqPred(nic=35, nf=self.nf, momentum=0.01) # huyue 14+1+20
        #self.TreeCNN=TreeCNNmodels.seqPred(nic=14, nf=self.nf, momentum=0.01) # huyue 14+1+20
        self.TreeCNN=TreeCNNmodels.seqPred(nic=35, nf=self.nf, momentum=0.01) # huyue 14+1+20
        self.TreeCNN.apply(TreeCNNmodels.init_ortho_weights)

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, batch):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        #huyue
        #res_feat_dummy=rearrange(res_feat, 'b n d -> b () n d')
        #pair_feat_mask=rearrange(mask_res, 'b i -> b i ()') * rearrange(mask_res, 'b j -> b () j') if exists(mask_res) else None
        #pair_feat_mask=rearrange(mask_res, 'b i -> b i ()') * rearrange(mask_res, 'b j -> b () j') 
        #pair_feat,res_feat_dummy = self.evoformer(pair_feat,res_feat_dummy,pair_feat_mask,None)  # huyue
       # res_feat=rearrange(res_feat_dummy, 'b () n d -> b n d') # reduce
        #print("mask res in dpm\n")
        #print(mask_res)
        #print("\n")
        #print("mask generate in dpm\n")
        #print(mask_generate)
        #print("\n")
        mask_pair= torch.logical_and(mask_res,~mask_generate)
        #print("mask pair in dpm\n")
        #print(mask_pair)
        #print("\n")
        pair_feat_mask=rearrange(mask_pair, 'b i -> b i ()') * rearrange(mask_pair, 'b j -> b () j') 
        #pair_feat_mask=torch.full_like(pair_feat_mask,True) 
        #print("mask pair feat mask in dpm\n")
        #print(pair_feat_mask)
        #print("\n")
        #print("pair feat before in dpm\n")
        #print(pair_feat)
        #print("\n")
        #pair_feat = self.pairwiseAttention(pair_feat,pair_feat_mask,None,None)  # huyue
        #pair_feat = self.pairwiseAttention(pair_feat,pair_feat_mask,None,None)  # huyue 230509
        #print("pair feat after in dpm\n")
        #print(pair_feat)
        #print("\n")
        #pair_feat = self.pair_ff(pair_feat)+pair_feat 230509


        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        #p_t geometric tranformer by huyue
        #mask_geo_00= torch.logical_or(mask_res,mask_generate)
        #mask_geo= torch.full_like(mask_geo_00,False) # huyue 0519
        mask_geo= torch.logical_or(mask_res,mask_generate)
        #print("mask_res \n")
        #print(mask_res)
        #print("\n")
        #print("mask_generate \n")
        #print(mask_generate)
        #print("\n")
     #   print("mask_geo_00 \n")
    #    print(mask_geo_00)
        #print("\n")
      #  print("mask_geo_00 \n")
       # print(mask_geo_00)
        #print("\n")
        #test_Z=torch.full_like(mask_geo,6)
        #print("test.Z \n")
        #print(test_Z)
        #print("\n")
        b_Z,L_Z=mask_geo.size()
        #print("b  \n")
        #print(b_Z)
        #print("\n")
        #print("L  \n")
        #print(L_Z)
        #print("\n")
        #print("ZZ  size\n")
        #print(test_Z.size())
        #print("\n")
        bb_Z=torch.arange(0,L_Z,step = 1)
        #print("bb Z\n")
        #print(bb_Z)
        #print("\n")
            #DD=ZZ[]
        test_neighbors=torch.Tensor.repeat(bb_Z,b_Z,L_Z,1)
        #print("test.neighbors \n")
        #print(test_neighbors)
        #print("\n")
        #print("test.neighbors  size\n")
        #print(test_neighbors.size())
        #print("\n")
        test_Z_00=torch.full((b_Z,L_Z),7)
        
        test_Z_0=test_Z_00.masked_fill(mask=~mask_res,value=0)
        #print("test_Z_0 ~mask res\n")
        #print(test_Z_0)
        #print("\n")

        test_Z_11=torch.full((b_Z,L_Z),6)
        test_Z_22=torch.full((b_Z,L_Z),6)
        test_Z_33=torch.full((b_Z,L_Z),8)
        
        test_Z_1=test_Z_11.masked_fill(mask=~mask_res,value=0)
        test_Z_2=test_Z_22.masked_fill(mask=~mask_res,value=0)
        test_Z_3=test_Z_33.masked_fill(mask=~mask_res,value=0)
        #print("test.Z \n")
        #print(test_Z_0)
        #print("\n")
        #inputs_geo[Properties.atom_mask]=mask_geo
        #inputs_geo[Properties.Z]=test_Z
        #inputs_geo[Properties.R]=p_t
        #inputs_geo[Properties.neighbors]=test_neighbors
        #inputs_geo = {
        #            'Propertiesatom_mask': [],
        #            'PropertiesZ': [],
        #            'PropertiesR': [],
        #            'Propertiesneighbors': [],
        #            }
        #inputs_geo['Propertiesatom_mask']=mask_geo
        #inputs_geo['PropertiesZ']=test_Z
        #inputs_geo['PropertiesR']=p_t
        #inputs_geo['Propertiesneighbors']=test_neighbors




        #repr_geo=self.geo_trans(inputs_geo)
        #huyue
        mask_atoms = batch['mask_heavyatom']
        
        
        #print("batch mask heavy atoms\n")
        #print(batch['mask_heavyatom'])
        #print("\n")
        #print("batch mask heavy atoms size\n")
        #print(batch['mask_heavyatom'].size())
        #print("\n")
        #print("batch pos heavy atoms\n")
        #print(batch['pos_heavyatom'])
        #print("\n")
        #x_coord=batch['pos_heavyatom']
        #maybe problems# huyue
        x_coord, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx=batch['pos_heavyatom'],
                R_new=R, 
                t_new= p_t, 
                aa=s_t,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms=batch['mask_heavyatom'],
                mask_recons=batch['generate_flag'])
        #   chain_nb = batch['chain_nb'],res_nb = batch['res_nb'],mask = mask_res

        #print("batch mask res CA\n")

        #print(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA])
        #print("\n")
        #print("batch mask genetate flag\n")

        #print(batch['generate_flag'])
        #print("\n")
        #print("batch res_nb\n")
       # print(batch['res_nb'])
        #print("\n")
        #print("batch res_nb size\n")
        #print(batch['res_nb'].size())
        #print("\n")
        #print("batch aa\n")
        #print(batch['aa'])
        #print("\n")
        #print("batch aa size\n")
        #print(batch['aa'].size())
        #print("\n")
        #print(" \n")

        batchS,residueS,atomS,coordS=batch['pos_heavyatom'].size()
        
        #masked
        #mask_atoms=mask_atoms.reshape(batchS*residueS*atomS,1)

        bb_SS=torch.arange(0,batchS*residueS,step = 1) # 0 base 1 base？
        #bb_SS=bb_SS.reshape(batchS,residueS)
        bb_SS=bb_SS.reshape(-1,1)
        residue_idx=torch.Tensor.repeat(bb_SS,1,atomS)
        #residue_idx=residue_idx.transpose(0,1)
        residue_idx=residue_idx.reshape(batchS,residueS,atomS,1)
        #print("residue_idx \n")
        #print(residue_idx)
        #print("\n")
        #print("residue_idx size\n")
        #print(residue_idx.size())
        #print("\n")

        resType=batch['aa']
        resType=resType.repeat(atomS,1,1,1)
        resType=resType.transpose(0,2).transpose(1,3)
        #print("residue_type \n")
        #print(resType)
        #print("\n")
        #print("residue_type size\n")
        #print(resType.size())
        #print("\n")
        #print("\n")
        BB_i=torch.tensor([1,1,1,1,1,0,0,0,0,0,0,0,0,0,1])
        BB_i=BB_i.reshape(-1,1)
        BB_i=BB_i.repeat(batchS,residueS,1,1)
        #print("BB_i \n")
        #print(BB_i)
        #print("\n")
        #print("BB_i size \n")
        #print(BB_i.size())
        #print("\n")
        atomType=torch.arange(0,15,step = 1) # 0 base 1 base？
        atomType=atomType.reshape(-1,1)
        atomType=atomType.repeat(batchS,residueS,1,1)
        #print("atomType \n")
        #print(atomType)
        #print("\n")
        #print("atomType size \n")
        #print(atomType.size())
        #print("\n")
        x_data=torch.cat((residue_idx,BB_i,atomType,resType),3)
        #print("x_data \n")
        #print(x_data)
        #print("\n")
        #print("x_data size \n")
        #print(x_data.size())
        #print("\n")
        #print("x_coord size\n")
        #print(x_coord.size())
        #print("\n")
        x_coord=x_coord.reshape(batchS*residueS*atomS,1,3) #huyue 0716
        x_data=x_data.reshape(batchS*residueS*atomS,1,4)
        
        #debug huyue
        #x_coord=x_coord.reshape(1,batchS*residueS*atomS,3)
        #x_data=x_data.reshape(1,batchS*residueS*atomS,4)
        #x_data=x_data.reshape(3)
        x_coord=x_coord.numpy()
        x_data=x_data.numpy()
        #print("x_coord before \n")
        #print(x_coord)
        #print("\n")
        residue_bb_index=torch.tensor([0,1,2,4])
        residue_bb_index=residue_bb_index.reshape(-1,1)
        residue_bb_index=residue_bb_index.repeat(batchS,residueS,1,1)
        residue_bb_index=residue_bb_index.squeeze(dim=3)
        residue_bb_index=residue_bb_index.reshape(batchS*residueS,4)
        
        residue_add=torch.arange(0,15*residueS*batchS,step = 15) # 0 base 1 base？
        residue_add=residue_add.reshape(-1,1)
        residue_add=torch.Tensor.repeat(residue_add,1,4)
        #print("residue add\n")
        #print(residue_add)
        #print("\n")
        #print("residue add size\n")
        #print(residue_add.size())
        #print("\n")
        #residue_add=residue_add.squeeze(dim=3)
        residue_bb_index=residue_add+residue_bb_index
        
        #debug huyue 
        #residue_bb_index=residue_bb_index.reshape(1,batchS*residueS,4)
        
        #print("residue_bb_index \n")
        #print(residue_bb_index)
        #print("\n")
        #print("residue_bb_index size\n")
        #print(residue_bb_index.size())
        #print("\n")
        residue_bb_index=residue_bb_index.numpy()
        #x_coord,x_data=canonicalize.batch_canonicalize_coords(x_coord,x_data,residue_bb_index)
        idx_CB=  residue_bb_index[:, 3]
        idx_CA=  residue_bb_index[:, 1]
        x_CA_coord=x_coord[idx_CA]
        #print("x_coord 1/4 CB\n")
        #print(x_coord[idx_CB])
        #print("\n")
        x_coord,x_data,center_coord, fixed_CB_coord,x_idxN_coord, x_idxC_coord, x_idxCA_coord,x_CB_coord,vector_x_back,vector_z_normal_back,can_res_num,can_atom_num=canonicalize.batch_canonicalize_coords(x_coord,x_data,residue_bb_index) #huyue 0823
        #print("center coord \n")
        #print(center_coord)
        #print("\n")
        #print("fix_CB_coord \n")
        #print(fixed_CB_coord)
        #print("\n")
        #print("x_idxN_coord \n")
        #print(x_idxN_coord)
        #print("\n")
        #print("x_idxC_coord \n")
        #print(x_idxC_coord)
        #print("\n")
        #print("x_idxCA_coord \n")
        #print(x_idxCA_coord)
        #print("\n")
        #print("x_CB_coord \n")
        #print(x_CB_coord)
        #print("\n")
        #print("x_coord middle \n")
        #print(x_coord)
        #print("\n")
        #print("x_coord canon shape \n")
        #print(x_coord.shape)
        #print("\n")
        #print("x_data cannon shape \n")
        #print(x_data.shape)
        #print("\n")
        #Cback_z,Cback_Bz=canonicalize.get_batch_N_CA_C_align_back(vector_z_normal_back,r=can_res_num,n=can_atom_num)
        #Cback_x,Cback_Bx=canonicalize.get_batch_N_CA_align_back(vector_x_back,r=can_res_num,n=can_atom_num)
        #back_z,back_Bz=get_batch_N_CA_C_align_back(vector_z_normal_back,r=can_res_num,n=can_atom_num)
        #back_x,back_Bx=get_batch_N_CA_align_back(vector_x_back,r=can_res_num,n=can_atom_num)
        back_z,back_Bz=get_batch_N_CA_C_align_back(torch.as_tensor(vector_z_normal_back),r=can_res_num,n=can_atom_num)
        back_x,back_Bx=get_batch_N_CA_align_back(torch.as_tensor(vector_x_back),r=can_res_num,n=can_atom_num)
        #idx_CB=  residue_bb_index[:, 3]
        #voxeOut=voxelize(x_coord,x_data,bb_only=1)
        #print("x_coord back \n")
        #print(back_x(back_z(torch.from_numpy(x_CB_coord).type(torch.FloatTensor).reshape(L,1,3)).reshape(L,1,3)))
        #print("\n")
        #print("x_coord back last \n")
        #print(Cback_x(Cback_z(x_CB_coord)))
        #print("\n")
        #print("x_coord back add\n")
        #print(back_x(back_z(torch.from_numpy(x_CB_coord).type(torch.FloatTensor).reshape(L,1,3)).reshape(L,1,3)).reshape(L,1,3)+torch.from_numpy(x_CA_coord).reshape(L,1,3))
        #print("\n")
        #print("x_coord back last add\n")
        #print(torch.from_numpy(Cback_x(Cback_z(x_CB_coord))).reshape(L,1,3)+torch.from_numpy(x_CA_coord).reshape(L,1,3))
        #print("\n")
        #voxeOut=voxelize(x_coord,x_data,bb_only=1) # huyue 15
        voxeOut=voxelize(x_coord,x_data,bb_only=0) # huyue 35
        #print("voxeOut \n")
        #print(voxeOut.sum(axis=0))
        #print("\n")
        #print("voxeOut \n")
        #print(voxeOut.sum(axis=1))
        #print("\n")
        #print("voxeOut 2\n")
        #print(voxeOut.sum(axis=2))
        #print("\n")
        #print("voxeOut 3\n")
        #print(voxeOut.sum(axis=3))
        #print("\n")
        #print("voxeOut 4\n")
        #print(voxeOut.sum(axis=4))
        #print("\n")
        #print("voxeOut map sum\n")
        #print(sum(map(sum,map(sum,map(sum,map(sum,voxeOut))))))
        #print("\n")
        #print("voxeOut size \n")
        #print("voxeOut size \n")
        #print(voxeOut.shape)
        #print("\n")

        voxeOut=torch.Tensor(voxeOut)


        TreeOut=self.TreeCNN(voxeOut)
        #print("TreeOut size \n")
        #print(TreeOut.size())
        #print("\n")
        #print("TreeOut  \n")
        #print(TreeOut)
       # print("\n")


        mask_res = mask_atoms[:, :, BBHeavyAtom.CA]
        backbone_coord=reconstruct_backbone(R,p_t,s_t,chain_nb = batch['chain_nb'],res_nb = batch['res_nb'],mask = mask_res)
        #N
        #print("mask_atoms \n")
        #print(mask_atoms)
        #print("\n")
        #print("mask_res atoms \n")
        #print(mask_res)
        #print("\n")
       
        #mask_p_t_00=torch.Tensor.repeat(mask_res,1,1,3)
        #mask_p_t=mask_p_t_00.permute(0,2,1)
        mask_p_t=mask_res[:, :, None].expand_as(p_t)
        #print("maks_p_t size \n")
        #print(mask_p_t.size())
        #print("\n")
        #print("voxeOut size \n")
        #print(voxeOut.shape)
        #print("\n")

        mask_res = mask_atoms[:, :, BBHeavyAtom.CA]
        backbone_coord=reconstruct_backbone(R,p_t,s_t,chain_nb = batch['chain_nb'],res_nb = batch['res_nb'],mask = mask_res)
        #N
        #print("mask_atoms \n")
        #print(mask_atoms)
        #print("\n")
        #print("mask_res atoms \n")
        #print(mask_res)
        #print("\n")
       
        #mask_p_t_00=torch.Tensor.repeat(mask_res,1,1,3)
        #mask_p_t=mask_p_t_00.permute(0,2,1)
        mask_p_t=mask_res[:, :, None].expand_as(p_t)
        #print("maks_p_t size \n")
        #print(mask_p_t.size())
        #print("\n")
        #print("voxeOut size \n")
        #print(voxeOut.shape)
        #print("\n")

        mask_res = mask_atoms[:, :, BBHeavyAtom.CA]
        backbone_coord=reconstruct_backbone(R,p_t,s_t,chain_nb = batch['chain_nb'],res_nb = batch['res_nb'],mask = mask_res)
        #N
        #print("mask_atoms \n")
        #print(mask_atoms)
        #print("\n")
        #print("mask_res atoms \n")
        #print(mask_res)
        #print("\n")
       
        #mask_p_t_00=torch.Tensor.repeat(mask_res,1,1,3)
        #mask_p_t=mask_p_t_00.permute(0,2,1)
        mask_p_t=mask_res[:, :, None].expand_as(p_t)
        #print("maks_p_t size \n")
        #print(mask_p_t.size())
        #print("\n")
        #print("maks_p_t \n")
        #print(mask_p_t)
        #print("\n")
        
        p_t_00=backbone_coord[:, :, 0]
        p_t_0=p_t_00.masked_fill(mask=~mask_p_t,value=0)
        #print("p_t_0 size \n")
        #print(p_t_0.size())
        #print("\n")
        #print("p_t_0 \n")
        #print(p_t_0)
        #print("\n")
        #print("p_t for compare \n")
        #print(p_t)
        #print("\n")
        #print("test_Z_0 \n")
        #print(test_Z_0)
        #print("\n")
        inputs_geo_0 = {
                    'Propertiesatom_mask': [],
                    'PropertiesZ': [],
                    'PropertiesR': [],
                    'Propertiesneighbors': [],
                    }
        inputs_geo_0['Propertiesatom_mask']=mask_geo
        inputs_geo_0['PropertiesZ']=test_Z_0
        inputs_geo_0['PropertiesR']=p_t_0
        inputs_geo_0['Propertiesneighbors']=test_neighbors
        #repr_geo_0=self.geo_trans_0(inputs_geo_0)
        #huyue

        #C alpha
        p_t_11=backbone_coord[:, :, 1]
        p_t_1=p_t_11.masked_fill(mask=~mask_p_t,value=0)
        inputs_geo_1 = {
                    'Propertiesatom_mask': [],
                    'PropertiesZ': [],
                    'PropertiesR': [],
                    'Propertiesneighbors': [],
                    }
        inputs_geo_1['Propertiesatom_mask']=mask_geo
        inputs_geo_1['PropertiesZ']=test_Z_1
        inputs_geo_1['PropertiesR']=p_t_1
        inputs_geo_1['Propertiesneighbors']=test_neighbors
        #repr_geo_1=self.geo_trans_1(inputs_geo_1)

        #C

        #p_t_2=backbone_coord[:, :, 2]
        p_t_22=backbone_coord[:, :, 2]
        p_t_2=p_t_22.masked_fill(mask=~mask_p_t,value=0)
        inputs_geo_2 = {
                    'Propertiesatom_mask': [],
                    'PropertiesZ': [],
                    'PropertiesR': [],
                    'Propertiesneighbors': [],
                    }
        inputs_geo_2['Propertiesatom_mask']=mask_geo
        inputs_geo_2['PropertiesZ']=test_Z_2
        inputs_geo_2['PropertiesR']=p_t_2
        inputs_geo_2['Propertiesneighbors']=test_neighbors
        #repr_geo_2=self.geo_trans_2(inputs_geo_2)

        #O
        #p_t_3=backbone_coord[:, :, 3]
        p_t_33=backbone_coord[:, :, 3]
        p_t_3=p_t_33.masked_fill(mask=~mask_p_t,value=0)
        inputs_geo_3 = {
                    'Propertiesatom_mask': [],
                    'PropertiesZ': [],
                    'PropertiesR': [],
                    'Propertiesneighbors': [],
                    }
        inputs_geo_3['Propertiesatom_mask']=mask_geo
        inputs_geo_3['PropertiesZ']=test_Z_3
        inputs_geo_3['PropertiesR']=p_t_3
        inputs_geo_3['Propertiesneighbors']=test_neighbors
        #repr_geo_3=self.geo_trans_3(inputs_geo_3)
        
        #print("p_t \n")
        #print(p_t)
        #print("\n")
        #print("p_t_0 \n")
        #print(p_t_0)
        #print("\n")
        #print("p_t_1 \n")
        #print(p_t_1)
        #print("\n")
        #print("p_t_2 \n")
        #print(p_t_2)
        #print("\n")
        #print("p_t_3 \n")
        #print(p_t_3)
        #print("\n")



        #############
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        #in_feat = torch.cat([res_feat, t_embed], dim=-1)
        #huyue 0511
        #in_feat = torch.cat([res_feat,repr_geo_0, repr_geo_1, repr_geo_2, repr_geo_3, t_embed], dim=-1)
        #in_feat = torch.cat([res_feat,repr_geo_0, t_embed], dim=-1) # 0519
        #in_feat = torch.cat([repr_geo_1, t_embed], dim=-1)
       
        #huyue 0716
        res_feat2=res_feat.reshape(L,self.res_feat_dim)
        t_embed2=t_embed.reshape(L,3)
        in_feat2 = torch.cat([TreeOut,res_feat2[..., None], t_embed2[..., None]], 1 )
        
        TreeOut3=TreeOut.reshape(N,L,self.nf * 4)
        t_embed3=t_embed.reshape(N,L,3)
        res_feat3=res_feat.reshape(N,L,self.res_feat_dim)

        in_feat3 = torch.cat([TreeOut3,res_feat3, t_embed3], dim=-1 )
        #print("TreeOut size\n")
        #print(TreeOut.size())
        #print("\n")
        #print("TreeOut size\n")
        #print(TreeOut.size())
        #print("\n")
        #in_feat2 = torch.cat([TreeOut, t_embed2[..., None]], 1 )
        #huyue
        # Position changes
        #eps_crd = self.eps_crd_net(in_feat2)    # (N, L, 3)

        #eps_crd=eps_crd.reshape(N,L,3)
        #eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        #eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))
        
                #huyue0823
                        # Position changes
        eps_crd = self.eps_crd_net(in_feat3)    # (N, L, 3)
        #eps_crd = torch.from_numpy(back_x(back_z(eps_crd.reshape(L,1,3).detach().numpy()))) #huyue
        #eps_crd = torch.from_numpy(back_x(back_z(eps_crd.reshape(L,1,3).numpy()))) #huyue
        #eps_crd1 = back_z(eps_crd.reshape(L,1,3)) #huyue
        #eps_crd = back_x(back_z(eps_crd.reshape(L,1,3)).reshape(L,1,3).type(torch.FloatTensor)) #huyue
        #test_eps_crd=apply_rotation_to_vector(R, eps_crd.reshape(N,L,3)) 
        #print('rotation apply crd\n')
        #print(test_eps_crd)
        #print('\n')
        eps_crd = back_x(back_z(eps_crd.reshape(L,1,3)).reshape(L,1,3)) #huyue

        eps_crd=eps_crd.reshape(N,L,3)
        #print('Axis-anlge crd\n')
        #print(eps_crd)
        #print('\n')
        


        #eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos=eps_crd # no rotation #huyue
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))


        #if not torch.isfinite(eps_pos):
        #    print('eps_pos is NaN or Inf detected.')
        ##   print('eps_pos \n')
        #    print(eps_pos)
        #    print(' \n')
        #    print('in_feat \n')
        #    print(in_feat)
        #    print(' \n')
        #print('eps_pos is NaN or Inf detected.')
        #print('eps_pos \n')
        #print(eps_pos)
        #print(' \n')
        #if not torch.isfinite(in_feat):
        #    print('in_feat is Nan or InF\n')
        #    print(in_feat)
        #    print(' \n')
        #print('in_feat \n')
        #print(in_feat)
        #print(' \n')

        # New orientation
        #eps_rot = self.eps_rot_net(in_feat2)    # (N, L, 3)
        #eps_rot=eps_rot.reshape(N,L,3) #huyue
        #U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        #R_next = R @ U
        #v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        #v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)
       



       # if not torch.isfinite(R_next):
       #orientation 0823

        eps_rot = self.eps_rot_net(in_feat3)    # (N, L, 3)
        #eps_rot = torch.from_numpy(back_x(back_z(eps_rot.reshape(L,1,3).detach().numpy()))) #huyue
        #eps_rot = back_x(back_z(eps_rot.reshape(L,1,3).type(torch.DoubleTensor)).reshape(L,1,3)) #huyue
        #eps_rot = back_x(back_z(eps_rot.reshape(L,1,3)).reshape(L,1,3).type(torch.FloatTensor)) #huyue
        U = quaternion_1ijk_to_rotation_matrix(eps_rot.reshape(N,L,3)) # (N, L, 3, 3)
        R_next = R @ U
        #print('test R  next rotation \n') 
        #print(test_R_next) 
        #print(' \n') 
        #print('test v  next rotation \n') 
        #print(rotation_to_so3vec(test_R_next)) 
        #print(' \n') 
        #eps_rot = back_x(back_z(eps_rot.reshape(L,1,3)).reshape(L,1,3)) #huyue
        #eps_rot=eps_rot.reshape(N,L,3) #huyue
        #U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
                                               #R_next = R @ U
        #R_next = R @ U ### ????
        #R_next = U # no rotation #huyue
        #print('R  next Angle axis \n') 
        #print(R_next) 
        #print(' \n') 
        #print('test v  next Angle axis \n') 
        #print(rotation_to_so3vec(R_next)) 
        #print(' \n') 
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)
        #    print('R_next is NaN or Inf detected.')
         #   print('R_next \n')
          #  print(R_next)
           # print(' \n')
        #print('R_next is NaN or Inf detected.')
        #print('R_next \n')
        #print(R_next)
        #print(' \n')

        # New sequence categorical distributions
        #c_denoised = self.eps_seq_net(in_feat2)  # Already softmax-ed, (N, L, 20)
        c_denoised = self.eps_seq_net(in_feat3)  # Already softmax-ed, (N, L, 20)
        c_denoised = c_denoised.reshape(N,L,20)
        #debug huyue
        #if not torch.isfinite(c_denoised):
        #    print('c denosied is NaN or Inf detected.')
        #    print('C denoised \n')
        #    print(c_denoised)
        #    print(' \n')
        #print('c denosied is NaN or Inf detected.')
        #print('C denoised \n')
        #print(c_denoised)
        #print(' \n')

        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, batch, denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res, batch
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot
        
        #debug huyue
        if not torch.isfinite(loss_rot):
            print('loss rot is NaN or Inf detected.')
            print('R_pred \n')
            print(R_pred)
            print(' \n')
            print('R_0 \n')
            print(R_0)
            print(' \n')
            #raise KeyboardInterrupt()



        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos
        #debug huyue
        if not torch.isfinite(loss_pos):
            print('loss pos is NaN or Inf detected.')
            print('eps_p_pred \n')
            print(eps_p_pred)
            print(' \n')
            print('eps_p \n')
            print(eps_p)
            print(' \n')
            #raise KeyboardInterrupt()

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq
        #huyue change loss_dict
        #debug huyue
        if not torch.isfinite(loss_seq):
            print('loss seq is NaN or Inf detected.')
            print('s_noisy \n')
            print(s_noisy)
            print(' \n')
            print('s_0 \n')
            print(s_0)
            print(' \n')
            #raise KeyboardInterrupt()

        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, batch, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, batch
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, batch, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, batch
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
