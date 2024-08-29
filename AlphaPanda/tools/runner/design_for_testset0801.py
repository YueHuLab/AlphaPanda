import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from AlphaPanda.datasets import get_dataset
from AlphaPanda.models import get_model
from AlphaPanda.modules.common.geometry import reconstruct_backbone_partially
from AlphaPanda.modules.common.so3 import so3vec_to_rotation
from AlphaPanda.utils.inference import RemoveNative
from AlphaPanda.utils.protein.writers import save_pdb
from AlphaPanda.utils.train import recursive_to
from AlphaPanda.utils.misc import *
from AlphaPanda.utils.data import *
from AlphaPanda.utils.transforms import *
from AlphaPanda.utils.inference import *


#huyue
import torch
import esm
from esm import Alphabet
cpu_num=5
torch.set_num_threads(cpu_num)
from AlphaPanda.datasets.sabdab import AA_tensor_to_sequence
from AlphaPanda.utils.utils_trx import *
import residue_constants
from einops import rearrange
from AlphaPanda.utils.protein.constants import BBHeavyAtom, AA


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            #huyue
            str_hy=structure_factory()
            print("structure heavy \n")
            print(str_hy["heavy"])
            print("\n")
            print("structure light\n")
            print("\n")
            print(str_hy["light"])
            print("\n")
            print("structure antige\n")
            print(str_hy["antigen"])
            print("\n")
            print("\n")
            print("data_var in create \n")
            print(data_var["aa"])
            print("\n")
            print("\n")
            print("data_var in create aa.shape \n")
            print(data_var["aa"].shape)
            print("\n")
            #huyue
            residue_first, residue_last = get_residue_first_last(data_var)
            print("\n")
            print("residue first \n")
            print(residue_first)
            print("\n")
            print("\n")
            print("residue last \n")
            print(residue_last)
            print("\n")
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset
    dataset = get_dataset(config.dataset.test)
    print("length dateset \n")
    print(len(dataset))
    print("length dateset end\n")
    get_structure = lambda: dataset[args.index]

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']
    tag_postfix = '_%s' % args.tag if args.tag else ''
    log_dir = get_new_log_dir(os.path.join(args.out_root, config_name + tag_postfix), prefix='%04d_%s' % (args.index, structure_['id']))
    logger = get_logger('sample', log_dir)
    logger.info('Data ID: %s' % structure_['id'])
    data_native = MergeChains()(structure_)
    save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    logger.info(str(lsd))
    #huyue
    print("load esm model \n")
    esm_model,alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    print("load esm model end \n")

    # Make data variants
    data_variants = create_data_variants(
        config = config,
        structure_factory = get_structure,
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': args.index,
        'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]
    #inference_tfm = [ ] #huyue
    if 'abopt' not in config.mode:  # Don't remove native CDR in optimization mode
        inference_tfm.append(RemoveNative(
            remove_structure = config.sampling.sample_structure,
            remove_sequence = config.sampling.sample_sequence,
        ))
    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")

        save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
        print("\n")
        print("variant in before \n")
        print(variant['data']["aa"])
        print("\n")
        print("\n")
        print("variant before aa.shape \n")
        print(variant['data']["aa"].shape)
        print("\n")
    
        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )
        print("\n")
        print("data_cropped in afterr \n")
        print(data_cropped["aa"])
        print("\n")
        print("\n")
        print("data cropped after aa.shape \n")
        print(data_cropped["aa"].shape)
        print("\n")
        data_list_repeat = [ data_cropped ] * config.sampling.num_samples
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        count = 0
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            esm_model.eval() #huyue
            batch = recursive_to(batch, args.device)
            #################################################################










            #huyue
            print("batch AA \n")
            #torch.set_printoptions(threshold=np.inf)
            print(batch["aa"])
            print("\n")
            print("batch icode \n")
            #torch.set_printoptions(threshold=np.inf)
            print(batch["icode"])
            print("\n")
            print("\n")
            print("batch anchor flag \n")
            #torch.set_printoptions(threshold=np.inf)
            print(batch["anchor_flag"])
            print("\n")
            print("\n")
            print("batch fragment_type \n")
            #torch.set_printoptions(threshold=np.inf)
            print(batch["fragment_type"])
            print("\n")
            print("batch chain id \n")
            #torch.set_printoptions(threshold=np.inf)
            print(batch["chain_id"])
            print("\n")
            print("\n")
            #print("batch heavy \n")
            #torch.set_printoptions(threshold=np.inf)
           # print(batch["heavy"])
            print("\n")
           # print(batch["heavy"].shape)
            print("\n")
            #print("batch light \n")
            #torch.set_printoptions(threshold=np.inf)
           # print(batch["light"])
            print("\n")
           # print(batch["light"].shape)
            print("\n")
            #print("batch antigen \n")
            #torch.set_printoptions(threshold=np.inf)
            #print(batch["antigen"])
            print("\n")
            #print(batch["antigen"].shape)
            print("\n")
            print("batch AA shape \n")
            print(batch["aa"].shape)
            print("\n")
            print("flatten  \n")
            #torch.set_printoptions(threshold=np.inf)
            print(batch["aa"].flatten())
            print("\n")
            print("esm AA\n")
            #seq_esm = torch.from_numpy(mymsa_to_esmmsa(batch["aa"], input_type='msa')).long()
            seq_esm = torch.from_numpy(mymsa_to_esmmsa(batch["aa"], input_type='fasta')).long()
            print(seq_esm)
            print("\n")
            print("original AA\n")
            original_seq = esmmsa_to_mymsa(seq_esm)
            print(original_seq)
            print("original seq shape\n")
            print(original_seq.shape)
            print("\n")
            #seq_esm = torch.from_numpy(mymsa_to_esmmsa(batch["aa"], input_type='fasta')).long().to(device)
            with torch.no_grad():
                results= esm_model(seq_esm,repr_layers=[3], need_head_weights=True, return_contacts=False)
            #empty_cache()
            #esm_out = {
             #         'attentions': Rearrange('l h m n -> m n (l h)')(attentions.squeeze(0)[..., 1:-1, 1:-1]),
              #        'representations': representations.squeeze(0)[1:-1],
               #        }
            #torch.set_printoptions(threshold=np.inf)
            print("\n")
            print("seq_esm shape \n")
            print(seq_esm.shape)

            

            token_represtations=results['representations'][3]
            token_attentions=results["attentions"]
            #cdr_flag=batch["cdr_flag"]
            #print(AA_tensor_to_sequence(batch["aa"][cdr_flag]))
            print("\n")
            print(" representations \n")
            print(results['representations'])
            print(" token representations shape \n")
            print(token_represtations.shape)
            print("\n")
            print("\n")
           # print(" attentions \n")
           # print(results["attentions"])
           # print(" token attentions shape \n")
           # print(token_attentions.shape)
            print("\n")
            print(" token pair feat representations shape \n")
            #EsmToken_pair_feat=rearrange('b l h m n -> b m n (l h)')(token_attentions[..., ..., ..., 1:-1, 1:-1])
            #EsmToken_pair_feat=rearrange(token_attentions[..., ..., ..., 1:-1, 1:-1],'b l h m n -> b m n (l h)')
            token_attentions=token_attentions[:, :, :, 1:-1, 1:-1]
            EsmToken_pair_feat=rearrange(token_attentions,'b l h m n -> b m n (l h)')
            print(EsmToken_pair_feat.shape)
            print("\n")
            print(" token res feat representations shape \n")
            #token_res_feat=token_represtations[:, 1:-1] #fasta
            #EsmToken_res_feat=token_represtations[:, 1:]# msa 
            EsmToken_res_feat=token_represtations[:, 1:-1]# fasta 
            print(EsmToken_res_feat.shape)
            print("\n")
            print("mask heavy atoms \n")
            print(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA])
            print("\n")
            print("mask generate flag \n")
            print(batch['generate_flag'])
            print("\n")
            print("context mask \n")
            context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA],
                ~batch['generate_flag'] )    # Context means ``not generated''
            print(context_mask)
            print("\n")
            print("mask res batch mask ")
            print(batch['mask'])
            print("\n")
            print("\n")
            print("mask generate batch generate_flag ")
            print(batch['generate_flag'])
            print("\n")


    




            


            if 'abopt' in config.mode:
                # Antibody optimization starting from native
                traj_batch = model.optimize(batch, EsmToken_res_feat,  EsmToken_pair_feat, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
            else:
                # De novo design
                traj_batch = model.sample(batch,  EsmToken_res_feat,  EsmToken_pair_feat, sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })

            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_tmpl = variant['data']
                aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                pos_ha  = (
                    apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )

                save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                save_pdb({
                    'chain_nb': data_tmpl['chain_nb'],
                    'chain_id': data_tmpl['chain_id'],
                    'resseq': data_tmpl['resseq'],
                    'icode': data_tmpl['icode'],
                    # Generated
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha,
                }, path=save_path)
                # save_pdb({
                #     'chain_nb': data_cropped['chain_nb'],
                #     'chain_id': data_cropped['chain_id'],
                #     'resseq': data_cropped['resseq'],
                #     'icode': data_cropped['icode'],
                #     # Generated
                #     'aa': aa_new[i],
                #     'mask_heavyatom': mask_atom_new[i],
                #     'pos_heavyatom': pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(),
                # }, path=os.path.join(log_dir, variant['tag'], '%04d_patch.pdb' % (count, )))
                count += 1

        logger.info('Finished.\n')


if __name__ == '__main__':
    main()
