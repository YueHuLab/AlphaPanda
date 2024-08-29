import torch
import torch.nn as nn

from AlphaPanda.modules.common.geometry import construct_3d_basis
from AlphaPanda.modules.common.so3 import rotation_to_so3vec
from AlphaPanda.modules.encoders.residue import ResidueEmbedding
from AlphaPanda.modules.encoders.pair import PairEmbedding
from AlphaPanda.modules.diffusion.dpm_full import FullDPM
from AlphaPanda.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )

    def encode(self, batch , remove_structure, remove_sequence):
        """
        input: EsmToken_res_feat (N,L,1280)
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
           # esm_res_feat = batch['representations'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            #esm_pair_feat=batch['attentions'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            #batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            #batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
##huyue
        #print("res_feat shape \n")
        #print(res_feat.shape)
        #print("\n")
        #print("pair_feat shape \n")
        #print(pair_feat.shape)
        #print("\n")
        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, batch,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
        ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        #print("res_feat shape \n")
        #print(res_feat.shape)
        #print("\n")
        #print("pair_feat shape \n")
        #print(pair_feat.shape)
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, batch, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res, batch, **optimize_opt)
        return traj
