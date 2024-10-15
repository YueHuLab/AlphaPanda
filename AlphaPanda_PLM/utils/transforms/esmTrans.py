import torch

from ._base import _mask_select_data, register_transform,esm_model
from ..protein import constants

from diffab.utils.utils_trx import *
import diffab.utils.residue_constants
from diffab.utils.protein.constants import BBHeavyAtom, AA
from einops import rearrange


@register_transform('esm_trans')
class EsmTrans(object):

    def __init__(self):
        super().__init__()
        self.esm_model = esm_model


    def __call__(self, data):        

        window=500
        #shift=500 
        shift=300 
        #huyue
        ###huyue
        #esm huyue
        #huyue
        #huyue
        #huyue
        #huyue
        #huyue
        #huyue
        #huyue
        #huyue
        #print("data AA \n")
        #torch.set_printoptions(threshold=np.inf)
        #print(data["aa"])
        #print("\n")
        #print("data AA shape \n")
        #print(data["aa"].shape)
        #print("\n")
        #print("flatten  \n")
        #torch.set_printoptions(threshold=np.inf)
        #print(data["aa"].flatten())
        #print("dim \n")
        #print(data["aa"].dim())
        #print("\n")
        #print("size \n")
        #print(data["aa"].size())
        #print("\n")
        #print("size 0\n")
        #print(data["aa"].size(0))
        #print("\n")
        #print("esm AA\n")
        data_in=data["aa"].reshape(1,data["aa"].size(0))
        #print("data_in \n")
        #print(data_in)
        #print("\n")
        #print("\n")
        #print("\n")
        #print("\n")		
		
        L = data["aa"].size(0)
        idx_pdb = torch.arange(L).long().view(1, L)
        # esm-1b can only handle sequences with <1024 AAs
        # run esm-1b by crops if length > 1000
        if L > 1000:
            esm_out = {
                'attentions': torch.zeros((L, L, 660)),
                'representations': torch.zeros((L, 1280)),
            }
            count_1d = torch.zeros((L))
            count_2d = torch.zeros((L, L))
            #
            grids = np.arange(0, L - window + shift, shift)
            ngrids = grids.shape[0]
            print("ngrid:     ", ngrids)
            print("grids:     ", grids)
            print("windows:   ", window)

            for i in range(ngrids):
                for j in range(i, ngrids):
                    start_1 = grids[i]
                    end_1 = min(grids[i] + window, L)
                    start_2 = grids[j]
                    end_2 = min(grids[j] + window, L)
                    sel = np.zeros((L)).astype(np.bool)
                    sel[start_1:end_1] = True
                    sel[start_2:end_2] = True

                    input_seq = data_in[:, sel]
                    input_seq = torch.from_numpy(mymsa_to_esmmsa(input_seq, input_type='fasta')).long()
                    input_idx = idx_pdb[:, sel]

                    print("running crop: %d-%d/%d-%d" % (start_1, end_1, start_2, end_2), input_seq.shape)
                    #with torch.no_grad():
                       # results= self.esm_model(seq_esm,repr_layers=[3], need_head_weights=True, return_contacts=False)
                    with torch.no_grad():
                        #results = esm_model(input_seq,repr_layers=[3], need_head_weights=True, return_contacts=False)
                        results = esm_model(input_seq,repr_layers=[33], need_head_weights=True, return_contacts=False)
                    attentions_crop=results["attentions"]
                    representations_crop=results["representations"][33]
                    #representations_crop=results["representations"][3]
                       # attentions_crop, representations_crop = esm_model(input_seq)[:2]
                    #empty_cache()

                    weight = 1
                    sub_idx = input_idx[0].cpu()
                    sub_idx_2d = np.ix_(sub_idx, sub_idx)
                    count_1d[sub_idx] += weight
                    count_2d[sub_idx_2d] += weight

                    esm_out['representations'][sub_idx] += weight * representations_crop.squeeze(0)[1:-1]
                    #attentions_crop = attentions_crop.squeeze(0)[:,:,:, 1:-1, 1:-1]
                    attentions_crop = attentions_crop[:,:,:, 1:-1, 1:-1]
                    #attentions_crop = rearrange(attentions_crop, 'l h m n -> m n (l h)')
                    attentions_crop = torch.squeeze(rearrange(attentions_crop, 'b l h m n -> b m n (l h)'),dim=0)
                    attentions_crop *= weight
                    esm_out['attentions'][sub_idx_2d] += attentions_crop
                    del representations_crop, attentions_crop
                    #empty_cache()

            esm_out['representations'] /= count_1d[:, None]
            esm_out['attentions'] /= count_2d[:, :, None]
        else:
            seq_esm = torch.from_numpy(mymsa_to_esmmsa(data_in, input_type='fasta')).long()
            with torch.no_grad():
                results = esm_model(seq_esm,repr_layers=[33], need_head_weights=True, return_contacts=False)
                #results = esm_model(seq_esm,repr_layers=[3], need_head_weights=True, return_contacts=False)
            attentions=results["attentions"]
            representations=results["representations"][33]
            #representations=results["representations"][3]
            #empty_cache()
            #huyue
            print("esm 222222 attentation size \n")
            print(attentions.shape)
            print("esm 222222 attentation size \n")
            ###
            esm_out = {
                    'attentions': torch.squeeze(rearrange(attentions[:,:,:, 1:-1, 1:-1], 'b l h m n -> b m n (l h)'),dim=0),
                'representations': representations.squeeze(0)[1:-1],
            }
        data['attentions']=esm_out['attentions']
        data['representations']=esm_out['representations']
        print("data attentions shape\n")
        print(data["attentions"].shape)
        print("\n")
        print("data presentations shape\n")
        print(data['representations'].shape)
        print("\n")
        return data    
