import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm

from typing import Optional, List, NamedTuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT

from multiprocessing import shared_memory
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from optim import AdamW, PlainRAdam, RAdam

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    n_id: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            n_id=self.n_id.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]

def create_mlp(in_channels, mlp_channels, dropout=0.5):
    mlp_list = []
    for hidden_channels in mlp_channels:
        mlp_list.extend([BatchNorm1d(in_channels),
                         ReLU(inplace=True),
                         Dropout(p=dropout),
                         Linear(in_channels, hidden_channels),
                        ])
        in_channels = hidden_channels
    return Sequential(*mlp_list)

class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int],
                 in_memory: bool = False, author_labels: bool = False,
                 save_eval_probs: bool = False, testing: bool = False,
                 train_set: str = 'train'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.in_memory = in_memory
        self.author_labels = author_labels
        self.save_eval_probs = save_eval_probs
        self.testing = testing
        self.train_set = train_set
        self.dataset = MAG240MDataset(self.data_dir)

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def prepare_data(self):
        dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

        if self.author_labels:
            labels_path = osp.join(dataset.dir, 'processed', 'author', f'author_{self.train_set}_label.npy')
            if not osp.exists(labels_path):
                t = time.perf_counter()
                print(f'Generating author {self.train_set} labels...', end=' ', flush=True)
                author_writes_authors, author_writes_papers = dataset.edge_index('author', 'writes', 'paper')
                author_writes_papers_argsort = np.argsort(author_writes_papers)
                papers_written_papers = author_writes_papers[author_writes_papers_argsort]
                papers_written_authors = author_writes_authors[author_writes_papers_argsort]
                papers, author_counts = np.unique(papers_written_papers, return_counts=True)
                paper_row_start = np.insert(np.cumsum(author_counts), 0, 0)

                paper_labels = dataset.all_paper_label.astype(np.int16)

                author_label_counts = np.zeros((dataset.num_authors, dataset.num_classes), dtype=np.int16)
                for paper in self.get_idx_split('train', append_authors=False):
                    cur_paper_row_start = paper_row_start[paper]
                    paper_authors = papers_written_authors[cur_paper_row_start:cur_paper_row_start+author_counts[paper]]
                    author_label_counts[paper_authors, paper_labels[paper]] += 1
                author_label_probs = np.divide(author_label_counts, author_label_counts.sum(axis=-1, keepdims=True), dtype = np.float16)
                author_labels = author_label_probs.argmax(axis=-1)
                author_label_probs = author_label_probs[np.arange(author_label_probs.shape[0]), author_labels]
                author_labels[author_label_probs != 1.0] = -1
                author_labels[np.isnan(author_label_probs)] = -1
                np.save(labels_path, author_labels)
                print(f'Done! [{time.perf_counter() - t:.2f}s]')
            idx_path = osp.join(dataset.dir, 'processed', 'author', f'author_{self.train_set}_idx.npy')
            if not osp.exists(idx_path):
                t = time.perf_counter()
                print(f'Generating author {self.train_set} idx...', end=' ', flush=True)
                author_labels = np.load(labels_path)
                author_idx = np.arange(author_labels.shape[0])[author_labels != -1] + dataset.num_papers
                np.save(idx_path, author_idx)
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

        if self.save_eval_probs:
            shared_mem_pred_probs = shared_memory.SharedMemory(name='shared_mem_pred_probs', create=True, size=dataset.num_papers * self.num_classes * 4)
            pred_probs = np.ndarray(shape=(dataset.num_papers, self.num_classes), dtype=np.float32, buffer=shared_mem_pred_probs.buf)
            pred_probs[:] = -1.0

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        dataset = MAG240MDataset(self.data_dir)

        self.train_idx = self.get_idx_split('train')
        self.train_idx.share_memory_()
        self.val_idx = self.get_idx_split('valid')
        self.val_idx.share_memory_()
        self.test_idx = self.get_idx_split('test')
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, self.num_features))

        if self.in_memory:
            self.x = np.empty((N, self.num_features), dtype=np.float16)
            self.x[:] = x
            self.x = torch.from_numpy(self.x).share_memory_()
        else:
            self.x = x

        if self.save_eval_probs:
            self.shared_mem_pred_probs = shared_memory.SharedMemory(name='shared_mem_pred_probs')
            self.pred_probs = np.ndarray(shape=(dataset.num_papers, self.num_classes), dtype=np.float32, buffer=self.shared_mem_pred_probs.buf)

        self.y = torch.from_numpy(dataset.all_paper_label)
        if self.author_labels:
            author_labels = torch.from_numpy(np.load(osp.join(self.dataset.dir, 'processed', 'author', f'author_{self.train_set}_label.npy')))
            self.y = torch.cat([self.y, author_labels], dim=0)

        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=4)

    def val_dataloader(self):
        if self.testing:
            idx = self.test_idx
        else:
            idx = self.val_idx
        return NeighborSampler(self.adj_t, node_idx=idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=2)

    def convert_batch(self, batch_size, n_id, adjs):
        if self.in_memory:
            x = self.x[n_id].to(torch.float)
        else:
            x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs], n_id=n_id[:batch_size])

    def get_idx_split(self, s, append_authors=True):
        idx = torch.from_numpy(self.dataset.get_idx_split(s))
        if s == 'train':
            if self.train_set == 'train_val':
                idx = torch.cat([idx, self.get_idx_split('valid')], dim=0)
            elif self.train_set != 'train':
                raise Exception('Unknown train split.')
            if self.author_labels and append_authors:
                author_idx = torch.from_numpy(np.load(osp.join(self.dataset.dir, 'processed', 'author', f'author_{self.train_set}_idx.npy')))
                idx = torch.cat([idx, author_idx], dim=0)
        return idx

class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 optim: str='adam', sched: str='step', 
                 max_epochs: int = 100, lr: float = 0.001,
                 heads: int = 4, dropout: float = 0.5, extra_mlp: bool = False,
                 extra_mlp_hidden_channels: list = [-1],
                 save_eval_probs: bool = False, testing: bool = False,):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout
        self.optim = optim
        self.sched = sched
        self.max_epochs = max_epochs
        self.lr = lr
        self.testing = testing
        self.save_eval_probs = save_eval_probs

        self.extra_mlp = extra_mlp
        if self.extra_mlp:
            for i in range(len(extra_mlp_hidden_channels)):
                if extra_mlp_hidden_channels[i] == -1:
                    extra_mlp_hidden_channels[i] = hidden_channels
            self.extra_mlps = ModuleList()
            for _ in range(num_layers):
                self.extra_mlps.append(
                ModuleList([
                    create_mlp(in_channels=hidden_channels,
                               mlp_channels=extra_mlp_hidden_channels,
                               dropout=self.dropout)
                    for _ in range(num_relations)
                ]))

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels),
        )

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                if subadj_t.nnz() > 0:
                    cur_out = self.convs[i][j]((x, x_target), subadj_t)
                    if self.extra_mlp:
                        cur_out = self.extra_mlps[i][j](cur_out)
                    out += cur_out

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        self.train_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        self.val_acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        if not self.testing:
            self.test_acc(y_hat.softmax(dim=-1), batch.y)
            self.log('test_acc', self.test_acc, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True)
        if self.save_eval_probs:
            self.trainer.datamodule.pred_probs[batch.n_id.cpu()] = y_hat.cpu()

    def configure_optimizers(self):
        print(f'Optimizer: {self.optim}')
        if (self.optim == 'adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif (self.optim == 'radam'):
            optimizer = RAdam(self.parameters(), lr=self.lr)
        elif (self.optim == 'plainradam'):
            optimizer = PlainRAdam(self.parameters(), lr=self.lr)
        elif (self.optim == 'adamw'):
            optimizer = AdamW(self.parameters(), lr=self.lr)
        print(f'Scheduler: {self.sched}')
        if (self.sched == 'step'):
            scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        elif (self.sched == 'cosine'):
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        elif (self.sched == 'cosinewr'):
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.max_epochs/4))
        elif self.sched == 'none':
            return [optimizer]
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--in-memory', action='store_true')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','radam','plainradam','adamw'])
    parser.add_argument('--scheduler', type=str, default='step', choices=['step','cosine','cosinewr', 'none'])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--train_set', type=str, default='train', choices=['train','train_val'])
    parser.add_argument('--extra_mlp', action='store_true')
    parser.add_argument('--extra_mlp_hidden_channels', type=str, default='-1')
    parser.add_argument('--author_labels', action='store_true')
    parser.add_argument('--save_eval_probs', action='store_true')
    parser.add_argument('--eval_size', type=int, default=160)
    parser.add_argument('--eval_size_dynamic', action='store_true')
    parser.add_argument('--eval_batch', type=int, default=16)
    parser.add_argument('--num_eval_runs', type=int, default=1)

    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    args.extra_mlp_hidden_channels = [int(i) for i in args.extra_mlp_hidden_channels.split(',')]
    print(args)
    print(f'PID: {os.getpid()}')
    print(f'Loading data from {ROOT}')

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.in_memory,
                         train_set=args.train_set, author_labels=args.author_labels,
                         testing=args.test)
    ckpt = None
    dirs = glob.glob(f'logs_rgnn/{args.model}/{args.exp_name}/lightning_logs/*')
    if len(dirs) > 0:
        version = max([int(x.split(os.sep)[-1].split('_')[-1]) for x in dirs])
        logdir = f'logs_rgnn/{args.model}/{args.exp_name}/lightning_logs/version_{version}'
        print(f'Loading saved model in {logdir}...')
        ckptdirs = glob.glob(f'{logdir}/checkpoints/*')
        if len(ckptdirs) > 0:
            ckpt = ckptdirs[0]
            model = RGNN.load_from_checkpoint(
                        checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml',
                        save_eval_probs=args.save_eval_probs, testing=args.test,
                        train_set=args.train_set)
            print(f'Restored {ckpt}')
        else:
            print('No model found!')
            if args.evaluate:
                return
            else:
                print('Training from scratch...')

    if not args.evaluate:
        if ckpt is None:
            model = RGNN(model=args.model, in_channels=datamodule.num_features,
                     out_channels=datamodule.num_classes, hidden_channels=args.hidden_channels,
                     num_relations=datamodule.num_relations, num_layers=len(args.sizes),
                     dropout=args.dropout, optim=args.optimizer, sched=args.scheduler, 
                     max_epochs=args.epochs, lr=args.learning_rate,
                     extra_mlp=args.extra_mlp, extra_mlp_hidden_channels=args.extra_mlp_hidden_channels,
                     )
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode = 'max', save_top_k=1)
        trainer = Trainer(gpus=args.device, accelerator=args.accelerator, max_epochs=args.epochs,
                          callbacks=[checkpoint_callback],
                          default_root_dir=f'logs_rgnn/{args.model}/{args.exp_name}',
                          num_sanity_val_steps=0, precision=args.precision,)
        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        print(f'Evaluating saved model in {logdir}...')

        trainer = Trainer(gpus=args.device, accelerator=args.accelerator,
                          resume_from_checkpoint=ckpt, num_sanity_val_steps=0,
                          precision=args.precision,)

        for run in range(args.num_eval_runs):
            datamodule.batch_size = args.eval_batch
            print(f'Batch size: {datamodule.batch_size}')
            if not args.eval_size_dynamic:
                datamodule.sizes = [args.eval_size] * len(args.sizes)  # (Almost) no sampling...
            else:
                datamodule.sizes = [args.eval_size * sz for sz in args.sizes]
            print(f'Neighborhood size: {datamodule.sizes}')
            trainer.test(model=model, datamodule=datamodule)
            if args.save_eval_probs:
                if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                    if args.test:
                        pred_probs =  datamodule.pred_probs[datamodule.test_idx]
                        evaluator = MAG240MEvaluator()
                        res = {'y_pred': pred_probs.argmax(axis=-1)}
                        evaluator.save_test_submission(res, f'results_rgnn/{args.model}/{args.exp_name}')
                        save_path = f'results_rgnn/testing/{args.model}/{args.exp_name}/{args.eval_size}/run_{run}'
                    else:
                        pred_probs =  datamodule.pred_probs[datamodule.val_idx]
                        evaluator = MAG240MEvaluator()
                        res = {'y_pred': pred_probs.argmax(axis=-1), 'y_true': datamodule.y[datamodule.val_idx]}
                        eval = evaluator.eval(res)
                        print("Eval results: ", eval)
                        save_path = f'results_rgnn/valid/{args.model}/{args.exp_name}/{args.eval_size}/run_{run}'
                    os.makedirs(save_path, exist_ok = True)
                    print("Saving pred probabilities to ", save_path)
                    np.save(f'{save_path}/pred_probs.npy', pred_probs)
        if args.save_eval_probs:
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                datamodule.shared_mem_pred_probs.unlink()
            datamodule.shared_mem_pred_probs.close()



if __name__ == '__main__':
    main()

