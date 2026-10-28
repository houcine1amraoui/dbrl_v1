"""Matched temporal / residual-context encoders. No upper hierarchy is created."""
import copy
import hashlib
import time
import numpy as np
import torch
from torch import nn
from src.models.hdeg.dbrl import DBRL
from src.models.hdeg.residual_context_dbrl import ResidualContextDBRL
from src.scientific_validation.behavior_preservation.core import reconstruction_loss


class TemporalOnly(nn.Module):
    # Reuse the exact DBRL temporal computation without registering graph parameters.
    _encode_temporal_behavior = DBRL._encode_temporal_behavior
    _validate_input = DBRL._validate_input

    def __init__(self, graph):
        super().__init__()
        self.input_dim = graph.input_dim
        self.embedding_dim = graph.embedding_dim
        self.temporal_encoder = copy.deepcopy(graph.temporal_encoder)
        self.embedding_projection = copy.deepcopy(graph.embedding_projection)

    def forward(self, x):
        self._validate_input(x)
        return self._encode_temporal_behavior(x)


class ReconstructionArm(nn.Module):
    def __init__(self, encoder, decoder, window, channels):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.window, self.channels = window, channels

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z).reshape(-1, self.window, self.channels)


def state_hash(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def paired_models(seed, channels, window=30):
    torch.manual_seed(seed)
    graph = ResidualContextDBRL(hidden_dim=64, embedding_dim=64, gru_layers=1,
        dropout=0., graph_top_k=15, graph_self_loops=True, graph_symmetric=True,
        graph_heads=1, graph_dropout=0., input_dim=6)
    local = TemporalOnly(graph)
    decoder = nn.Sequential(nn.Flatten(1), nn.Linear(channels*64, 64),
                            nn.GELU(), nn.Linear(64, window*channels))
    arms = dict(temporal=ReconstructionArm(local, copy.deepcopy(decoder), window, channels),
                dbrl=ReconstructionArm(graph, copy.deepcopy(decoder), window, channels))
    for key in ('temporal_encoder', 'embedding_projection'):
        if state_hash(getattr(local, key)) != state_hash(getattr(graph, key)):
            raise ValueError('Common encoder initialization differs: '+key)
    if state_hash(arms['temporal'].decoder) != state_hash(arms['dbrl'].decoder):
        raise ValueError('Decoder initialization differs')
    return arms


def schedule(seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(1024) for _ in range(8)]).reshape(1024, 8)


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def gradients_finite(model):
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise ValueError('Missing or nonfinite parameter gradient')


def backward_gate(model, batch, device):
    before = state_hash(model)
    model.train(); model.zero_grad(set_to_none=True)
    x = torch.from_numpy(batch['X'][:8]).to(device)
    _, pred = model(x)
    loss = reconstruction_loss(pred, torch.from_numpy(batch['values'][:8]).to(device),
                               torch.from_numpy(batch['status'][:8]).to(device))
    if not torch.isfinite(loss): raise ValueError('Nonfinite step-zero loss')
    loss.backward(); gradients_finite(model)
    model.zero_grad(set_to_none=True)
    if before != state_hash(model): raise ValueError('Backward gate changed parameters')
    return float(loss.detach())


def train_arm(model, batch, ids, device, on_step=None):
    model.train()
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999),
                                 eps=1e-8, weight_decay=0)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    sync(device); started = time.perf_counter(); history = []
    for step, selected in enumerate(ids, 1):
        optimizer.zero_grad(set_to_none=True)
        _, pred = model(torch.from_numpy(batch['X'][selected]).to(device))
        loss = reconstruction_loss(pred, torch.from_numpy(batch['values'][selected]).to(device),
                                   torch.from_numpy(batch['status'][selected]).to(device))
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        loss.backward(); gradients_finite(model); optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite updated parameter')
        if on_step: on_step(step)
        if step == 1 or step % 128 == 0:
            history.append(dict(step=step, reconstruction_mse=float(loss.detach())))
    sync(device)
    delta = {}
    for part in ('encoder','decoder'):
        delta[part] = float(sum((v.detach().cpu().double()-initial[k].double()).square().sum()
            for k,v in model.state_dict().items() if k.startswith(part+'.')).sqrt())
    if any(v == 0 for v in delta.values()): raise ValueError('Trainable component did not update')
    return dict(history=history, seconds=time.perf_counter()-started,
        parameter_delta_l2=delta, steps=len(ids),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)


def extract(model, batch, device):
    model.eval(); features=[]; errors=[]
    sync(device); started=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(batch['X']),8):
            z,pred=model(torch.from_numpy(batch['X'][i:i+8]).to(device))
            features.append(z.flatten(1).cpu().numpy())
            values=torch.from_numpy(batch['values'][i:i+8]).to(device)
            valid=torch.from_numpy(batch['status'][i:i+8]).to(device)==0
            e=torch.where(valid,(pred-values).square(),0.).sum((1,2))/valid.sum((1,2))
            errors.append(e.cpu().numpy())
    sync(device)
    return np.concatenate(features), np.concatenate(errors), time.perf_counter()-started
