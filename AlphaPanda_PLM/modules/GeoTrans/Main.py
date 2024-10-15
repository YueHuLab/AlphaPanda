import torch
from torch.optim import Adam
import argparse
from schnetpack.datasets import *
from GeometricTransformer import GeoTransformer
import random
import logging
from datetime import datetime
from schnetpack import train_test_split
from augmentation_collate_fun import _collate_aseatoms_Transformer, _collate_aseatoms_Transformer_Augment
import time
from schnetpack import Properties
#####################
#####################
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)
#####################
#####################
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    property_pred = [eval(f"QM9.{args.property}")]
    #####################

    logging.info(f'Property to predict: {property_pred}')
    qm9_path = "qm9.db"
    if not os.path.exists(qm9_path):
        dataset = QM9(qm9_path, remove_uncharacterized=True, download=True)
    dataset = QM9(qm9_path, remove_uncharacterized=True,
                load_only=property_pred, download=False)

    train, val, test = train_test_split(data=dataset,num_train=args.train_size, num_val=10000)
    train_loader = spk.data.AtomsLoader(train, batch_size=args.batch_size,
                                        num_workers=args.n_workers,
                                        shuffle=True,
                                        collate_fn=_collate_aseatoms_Transformer)
    val_loader = spk.data.AtomsLoader(val, batch_size=args.batch_size,
                                    collate_fn=_collate_aseatoms_Transformer)
    test_loader = spk.data.AtomsLoader(test, batch_size=args.batch_size,
                                    collate_fn=_collate_aseatoms_Transformer)
    logging.info(
        f'Dataset size: {len(dataset)}: Train size {len(train_loader.dataset)}, Val Size {len(val_loader.dataset)}, Test Size {len(test_loader.dataset)}')

    atomrefs = dataset.get_atomref(property_pred)
    means, stddevs = train_loader.get_statistics(
        property_pred, divide_by_atoms=True, single_atom_ref=atomrefs)
    means, stddevs = means[property_pred[0]].item(), stddevs[property_pred[0]].item()
    logging.info(f'Statistics = {means},{stddevs}')
    train_loader = spk.data.AtomsLoader(train,
                                        batch_size=args.batch_size,
                                        num_workers=args.n_workers,
                                        shuffle=True,
                                        collate_fn=_collate_aseatoms_Transformer_Augment)
    #####################
    model = GeoTransformer(nhead=args.nhead, num_encoder_layers=args.num_encoder_layers, d_model=args.d_model, property_stats = [means, stddevs],atomref=atomrefs[property_pred[0]]).to(device)
    logging.info(model)
    logging.info(f"# Params: {np.sum([np.prod(p.shape) for p in model.parameters()])}")
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8, min_lr=1e-6)
    criterion = torch.nn.L1Loss()
    #####################
    best_val = float('inf')
    for epoch in range(5000):
        loss_epoch = 0.
        num_samples = 0.
        t = time.time()
        for idx_batch, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            train_batch = {k: v.to(device) for k, v in train_batch.items() if k in [Properties.Z,Properties.atom_mask,Properties.R,Properties.neighbors,property_pred[0]]}
            result = model(train_batch)
            loss = criterion(result, train_batch[property_pred[0]])
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * result.shape[0]
            num_samples += result.shape[0]
            if idx_batch%100==0:
                print(f"Epoch {epoch} ({idx_batch}/{len(train_loader)}): Curr Loss = {loss.item():.3e}, Loss={loss_epoch/num_samples:.3e}")
        loss_epoch /= num_samples
        logging.info(f"Epoch {epoch}: LR={optimizer.param_groups[0]['lr']:.3e}, Train Loss = Train MAE = {loss_epoch:.3e}, time={time.time()-t:.1f}sec")
        ###
        loss_val = 0.
        num_samples = 0.
        t = time.time()
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.to(device) for k, v in val_batch.items() if k in [Properties.Z,Properties.atom_mask,Properties.R,Properties.neighbors,property_pred[0]]}
                result = model(val_batch)
                loss = criterion(result, val_batch[property_pred[0]])
                loss_val += loss.item() * result.shape[0]
                num_samples += result.shape[0]
            loss_val /= num_samples
            logging.info(f"Epoch {epoch}: Val Loss = Val MAE = {loss_val:.3e}, time={time.time()-t:.1f}sec")
        scheduler.step(loss_val)
        if best_val > loss_val:
            best_val = loss_val
            torch.save(model, os.path.join(args.path, 'best_model'))
            logging.info('Model saved')
        ###
        loss_val = 0.
        num_samples = 0.
        t = time.time()
        with torch.no_grad():
            for test_batch in test_loader:
                test_batch = {k: v.to(device) for k, v in test_batch.items() if k in [Properties.Z,Properties.atom_mask,Properties.R,Properties.neighbors,property_pred[0]]}
                result = model(test_batch)
                loss = criterion(result, test_batch[property_pred[0]])
                loss_val += loss.item() * result.shape[0]
                num_samples += result.shape[0]
            loss_val /= num_samples
            logging.info(f"Epoch {epoch}: Test Loss = Test MAE = {loss_val:.3e}, time={time.time()-t:.1f}sec")
#####################
#####################
#####################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Geometric Transformer')
    parser.add_argument('--gpus', type=str, default="2")
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--res_dir', type=str, default='runs')
    parser.add_argument('--train_size', type=int, default=110000)
    parser.add_argument('--property', type=str, default='U0',choices=['mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv'])
    # Encoder
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=256)
    parser.add_argument('--num_encoder_layers', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0)
    # Optimizer
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    #####################
    path = os.path.join(f"Results_Geometric_Transformer/{args.property}__{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}")
    os.makedirs(path, exist_ok=True)
    args.path = path
    handlers = [
        logging.FileHandler(os.path.join(path, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {path}")
    logging.info(args)

    main(args)