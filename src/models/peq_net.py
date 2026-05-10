import torch
import torch.nn as nn

import numpy as np
from scipy.special import logit, expit
from copy import deepcopy
from typing import Optional
import logging

from pytorch_lightning import LightningModule

from src.models.utils_dltmle import SinusoidalEncoder, solve_one_dimensional_submodel_regularized
torch.set_default_dtype(torch.double)

logger = logging.getLogger(__name__)


class peq_net(LightningModule):
    '''Class for deep ltmle that uses transformer as Q and G model for simultaneous quasi stochastic gradient descent with targeted loss

    '''
    def __init__(self,
                 dim_static,
                 dim_dynamic,
                 projection_horizon,
                 dim_treatments=1,
                 dim_emb=16,
                 dim_emb_time=8,
                 dim_emb_type=8,
                 hidden_size=32,
                 num_layers=2,
                 nhead=8,
                 dropout=0.0,
                 learning_rate=1e-3,
                 alpha = 1,
                 beta = 0,
                 outcome_type='continuous',  # 'binary', 'binary_irreversible' or 'continuous'
                 q_head='shared', # 'shared' or 'separate'
                 use_adapter = True,
                 policy_embedding_dim = 8,
                 policy_seq_embedding_dim = 16,
                 use_target_network: bool = False,
                 target_polyak_tau: Optional[float] = None,
                 **kwargs):
        '''Initialization of DeepLTMLE with specified hyperparameters
        
        Parameters
        ----------
        dim_static: `integer`
            dimension of static covariates (baseline covariates)
        dim_dynamic: `integer`
            dimension of dynaic covariates (time-dependent covariates)
        projection_horizon: `integer`
            length of projection horizon
        dim_emb: `integer`
            dimension of value embedding to transformer
        dim_emb_time: `integer`
            dimension of time embedding to transformer
        dim_emb_type: `integer`
            dimension of type embedding to transformer
        hidden_size: `integer`
            dimension of hidden state in transformer
        num_layers: `integer`
            number of layers in transformer
        nhead: `integer`
            number of attention heads in transformer
        dropout: `float`
            drop-out rate of transformer
        alpha: `float`
            weight of loss for propensity score
        beta: `float`
            weight of targeting loss
        outcome_type: `str`
            'binary' for binary outcomes (0/1), 'continuous' for continuous outcomes, 'binary_irreversible' for binary outcomes with irreversible outcomes
        q_head: `str`
            'shared' for shared Q head, 'separate' for separate Q heads for each time step
        '''
        super().__init__()

        self.projection_horizon = projection_horizon
        self.tau = projection_horizon + 1
        self.use_adapter = use_adapter
        self.policy_embedding_dim = policy_embedding_dim
        self.policy_seq_embedding_dim = policy_seq_embedding_dim

        # Target network configuration
        self.use_target_network = bool(use_target_network)
        self.target_polyak_tau = target_polyak_tau
        if self.use_target_network:
            assert self.target_polyak_tau is not None, "target_polyak_tau must be set when use_target_network=True"
            assert 0.0 < float(self.target_polyak_tau) <= 1.0, "target_polyak_tau must be in (0, 1]"
        self.target_network = None  # created after module construction
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.outcome_type = outcome_type

        self.dim_input_L = dim_static + dim_dynamic

        # embeddings
        self.emb_W = nn.Linear(dim_static, dim_emb)
        self.emb_L = nn.Linear(dim_dynamic, dim_emb)
        assert dim_treatments == 1, "our debias step only works for binary treatments"
        self.emb_A = nn.Linear(dim_treatments,   dim_emb)
        # TODO: even though we support multi-dimensional treatments, our debias step only works for binary treatments

        # temporal embeddings
        self.emb_time = nn.Sequential(
            SinusoidalEncoder(dim=dim_emb_time),
            nn.Linear(dim_emb_time, dim_emb_time)
        )

        # type embeddings
        self.emb_type = nn.Parameter(torch.randn(3, dim_emb_type), requires_grad=True)

        # transformer encoder
        d_model = dim_emb + dim_emb_time + dim_emb_type
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        self.q_head = q_head

        if q_head == "shared":
            if use_adapter:
                q_input_dim = d_model + self.policy_seq_embedding_dim
            else:
                q_input_dim = d_model
            self.logit_Q = nn.Sequential(nn.LayerNorm(q_input_dim), nn.Linear(q_input_dim, 1))
        else:
            if use_adapter:
                q_input_dim = d_model + self.policy_seq_embedding_dim
            else:
                q_input_dim = d_model
            self.logit_Q = nn.ModuleList([nn.Linear(q_input_dim, 1) for _ in range(self.tau)])

        self.G = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        if use_adapter:
            self.cf_rnn_e2e = nn.RNN(
                input_size=self.policy_embedding_dim,
                hidden_size=self.policy_seq_embedding_dim,
                num_layers=num_layers,
            )

        if beta == 0:
            self.eps = nn.Parameter(torch.zeros(projection_horizon + 1), requires_grad=False)
        else:
            self.eps_scalar = nn.Parameter(torch.zeros(1), requires_grad=True)

        if self.use_target_network:
            self._init_target_network()

    def _init_target_network(self):
        """
        Create a frozen copy of the current network to be used for stable regression targets.
        We keep it as a submodule so it follows device moves with the LightningModule.
        """
        target = deepcopy(self)
        # Ensure the target doesn't recursively create/hold its own target.
        target.use_target_network = False
        target.target_network = None
        # Freeze
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
        self.target_network = target
        # Start synced (one-time hard copy at init; no periodic hard-update mode).
        self.target_network.load_state_dict(self._online_state_dict_for_target(), strict=True)
    
    def _online_state_dict_for_target(self):
        """
        Extract this module's online weights/buffers, excluding any nested `target_network.*` keys.
        """
        return {k: v for k, v in self.state_dict().items() if not k.startswith("target_network.")}

    def _soft_update_target_network(self, tau: float):
        """Polyak update: target = (1-tau)*target + tau*online."""
        if self.target_network is None:
            return
        tau = float(tau)
        online_params = {n: p for n, p in self.named_parameters() if not n.startswith("target_network.")}
        with torch.no_grad():
            for name, tparam in self.target_network.named_parameters():
                oparam = online_params.get(name, None)
                if oparam is None:
                    continue
                tparam.data.lerp_(oparam.data, tau)

    def _get_attention_mask(self, seq_len, device):
        if hasattr(self, '_attention_mask') and hasattr(self, '_attention_mask_seq_len'):
            if self._attention_mask_seq_len == seq_len:
                return self._attention_mask

        # input nodes = [W[0:1], L[0:tau], A[0:tau]]
        # DGP at t: L[t] > A[t]

        # Use boolean masks directly on the target device
        I = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)  # attention < t
        J = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=0)  # attention <= t

        # [0, 1, 1]
        # [0, I, J]
        # [0, I, I]

        mask = torch.ones((seq_len * 2 + 1, seq_len * 2 + 1), dtype=torch.bool, device=device)
        mask[:, 0] = False

        for i in range(2):
            for j in range(2):
                mask[(i*seq_len+1):((i+1)*seq_len+1), (j*seq_len+1):((j+1)*seq_len+1)] = I if i >= j else J
        
        self._attention_mask = mask
        self._attention_mask_seq_len = seq_len

        return self._attention_mask

    def _compute_cf_embeddings(self, policy_embedding, batch_size, tau):
        """
        RNN embeddings over `policy_embedding` time slices (counterfactual policy path).

        Parameters
        ----------
        policy_embedding: tensor of shape (batch_size, time, policy_embedding_dim)
            Time length should be at least tau + 1 (index 0 may be a start token; steps 1..tau are used).
        batch_size: int
        tau: int

        Returns
        -------
        cf_embeddings_raw: tensor of shape (batch_size, tau, policy_seq_embedding_dim)
        """
        device = policy_embedding.device

        cf_embeddings_raw = torch.zeros(batch_size, tau, self.policy_seq_embedding_dim, device=device)

        seq_future = policy_embedding[:, 1 : tau + 1, :]  # (batch_size, tau, input_size)

        rnn_input = torch.flip(seq_future, dims=[1]).transpose(0, 1)  # (tau, batch_size, input_size)
        rnn_output, _ = self.cf_rnn_e2e(rnn_input)

        idxs = torch.arange(tau - 1, 0, -1, device=device) - 1
        cf_embeddings_raw[:, : tau - 1, :] = rnn_output.index_select(0, idxs).transpose(0, 1)

        return cf_embeddings_raw

    def forward(self, W, L, A, Y, a, policy_embedding=None, **kwargs):
        """
        Modified forward method to handle arbitrary counterfactual sequences
        
        Parameters
        ----------
        W: static covariates
        L: dynamic covariates
        A: factual treatment sequence
        Y: outcomes
        a: augmented counterfactual treatment sequence
            (shape: batch_size x tau)
            Can contain arbitrary 0/1 patterns
        policy_embedding: tensor of shape (batch_size, time, policy_embedding_dim), optional
            Required when ``use_adapter`` is True; feeds the policy RNN for Q heads.
        **kwargs:
            Ignored extra batch fields (e.g., policy identifiers for logging).
        """
        
        batch_size, seq_len = L.shape[0], L.shape[1]
        tau = self.projection_horizon + 1
        device = L.device
        dtype = L.dtype
        
        # Handle counterfactual sequence
        cf_seq = a.clone()
        assert cf_seq.shape[1] == tau, f"Counterfactual sequence length {a.shape[1]} must match factual sequence length {tau}" # Ensure a has same length as A?
        a = A.clone()
        a[:, -cf_seq.shape[1]:,:] = cf_seq
        if self.use_adapter:
            cf_embeddings_raw = self._compute_cf_embeddings(policy_embedding, batch_size, tau)
        else:
            cf_embeddings_raw = None

        # embeddings
        # shape (batch_size, tau, dim_emb)
        z_W = self.emb_W(W[:,None,:])
        z_L = self.emb_L(L)
        z_A = self.emb_A(A.double())
        z_a = self.emb_A(a.double())

        # add time embeddings (must be on the same device as the batch)
        # shape (batch_size, tau, dim_emb+dim_emb_time)
        t_w = torch.tensor([-1], device=device, dtype=dtype)
        t = torch.arange(seq_len, device=device, dtype=dtype)
        T_W = self.emb_time(t_w).repeat(batch_size, 1, 1)
        T = self.emb_time(t).repeat(batch_size, 1, 1)

        # add type embeddings
        # shape (batch_size, tau, dim_emb+dim_emb_time+dim_emb_type)
        type_W = self.emb_type[0].repeat(batch_size, 1, 1)
        type_L = self.emb_type[1].repeat(batch_size, seq_len, 1)
        type_A = self.emb_type[2].repeat(batch_size, seq_len, 1)

        z_W = torch.cat([z_W, T_W, type_W], axis=-1)
        z_L = torch.cat([z_L, T,   type_L], axis=-1)
        z_A = torch.cat([z_A, T,   type_A], axis=-1)
        z_a = torch.cat([z_a, T,   type_A], axis=-1)

        # transformer
        mask = self._get_attention_mask(seq_len, device=z_L.device)

        x = torch.cat([z_W, z_L, z_A], axis=1) # shape: (batch_size, 1+2*seq_len, dim_emb+dim_emb_time+dim_emb_type)

        # input:  W[0:1], L[0:seq_len], A[0:seq_len]
        # transformer output: W[0:1], L[0:seq_len], A[0:seq_len] (full sequence)
        # we need to extract: L[seq_len-tau:seq_len], A[seq_len-tau:seq_len] for G and Q
        x = self.transformer(x, mask=mask)
        
        # Extract the forecast window tokens
        start_idx = seq_len - tau
        z_L_forecast = x[:, 1+start_idx:1+seq_len, :]  # L[seq_len-tau:seq_len]
        z_A_forecast = x[:, 1+seq_len+start_idx:1+2*seq_len, :]  # A[seq_len-tau:seq_len]
        
        # Stack L and A forecast tokens: [L_forecast, A_forecast]
        z_forecast = torch.stack([z_L_forecast, z_A_forecast], dim=1)  # (batch_size, 2, tau, features)
        z_G, z_Q = z_forecast[:, 0], z_forecast[:, 1]  # Split back to L and A

        G = self.G(z_G)
        
        # Concatenate cf_embeddings_raw to z_Q features
        if cf_embeddings_raw is not None:
            z_Q_modulated = torch.concat([z_Q, cf_embeddings_raw], dim=2)
        else:
            z_Q_modulated = z_Q
        
        # if self.outcome_type == 'binary' or self.outcome_type == 'binary_irreversible':
        logit_Q_l = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        if self.q_head == 'shared':
            # Use modulated features for consistent predictions
            logit_Q_l[:, 1:] = self.logit_Q(z_Q_modulated)
        else:
            logit_Q_l[:, 1:] = torch.stack([self.logit_Q[i](z_Q_modulated[:, i, :]) for i in range(self.tau)], dim=1)
        
        if hasattr(self,'eps'):
            eps = self.eps.view(1, tau, 1).repeat(batch_size, 1, 1)
        else:
            eps = self.eps_scalar.view(1, 1, 1).repeat(batch_size, tau, 1)
        
        logit_Q_l_star = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_l_star[:, 1:] = logit_Q_l[:, 1:] + eps
        
        Q_l = torch.sigmoid(logit_Q_l)
        Q_l_star = torch.sigmoid(logit_Q_l_star)
        
        # ----------------------------------------------
        # Counterfactual - Causal Consistency Fix
        # For each time t, Q_t should only depend on factual treatments A[0:t-1] and counterfactual treatments a[t:tau]
        # This ensures Q_a[:,t,:] is identical for sequences sharing the same suffix a[t:tau]

        # NOTE: We vectorize by batching the `tau` variants and running the transformer once.

        # Build `tau` mixed-treatment variants in a single batched tensor.
        # For each forecast t_idx, we replace treatments from absolute time (start_idx + t_idx) onwards.
        d_model = z_A.shape[-1]
        total_len = 1 + 2 * seq_len

        # (tau, seq_len) mask: True where we should use counterfactual treatment tokens
        time_idx = torch.arange(seq_len, device=device)
        cf_start = start_idx + torch.arange(tau, device=device)
        replace_mask = (time_idx[None, :] >= cf_start[:, None])  # (tau, seq_len)
        replace_mask = replace_mask[None, :, :, None]  # (1, tau, seq_len, 1)

        # Expand to (batch, tau, seq_len, d_model)
        z_A_rep = z_A[:, None, :, :].expand(batch_size, tau, seq_len, d_model)
        z_a_rep = z_a[:, None, :, :].expand(batch_size, tau, seq_len, d_model)
        z_A_mixed_all = torch.where(replace_mask, z_a_rep, z_A_rep)

        # Build transformer input for all variants: (batch, tau, total_len, d_model) -> (batch*tau, total_len, d_model)
        z_W_rep = z_W[:, None, :, :].expand(batch_size, tau, 1, d_model)
        z_L_rep = z_L[:, None, :, :].expand(batch_size, tau, seq_len, d_model)
        x_cf = torch.cat([z_W_rep, z_L_rep, z_A_mixed_all], dim=2).reshape(batch_size * tau, total_len, d_model)

        # Single transformer pass for all counterfactual variants
        x_cf = self.transformer(x_cf, mask=mask)

        # Extract the A-token at the corresponding t_idx for each variant row.
        # Token position in the concatenated sequence for forecast t_idx:
        #   1 + seq_len + start_idx + t_idx
        pos_by_t = (1 + seq_len + start_idx + torch.arange(tau, device=device))  # (tau,)
        t_idx_all = torch.arange(tau, device=device).repeat(batch_size)  # (batch*tau,)
        pos_all = pos_by_t[t_idx_all]  # (batch*tau,)

        row_idx = torch.arange(batch_size * tau, device=device)
        z_A_t = x_cf[row_idx, pos_all, :].view(batch_size, tau, d_model)  # (batch, tau, d_model)

        # Concatenate adapter (policy RNN) embeddings if provided
        if cf_embeddings_raw is not None:
            z_Q_a_modulated = torch.cat([z_A_t, cf_embeddings_raw], dim=2)
        else:
            z_Q_a_modulated = z_A_t
        
        # if self.outcome_type == 'binary' or self.outcome_type == 'binary_irreversible':
        logit_Q_a = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        if self.q_head == 'shared':
            # Use the same modulated features as factual case for consistency
            logit_Q_a[:, :-1] = self.logit_Q(z_Q_a_modulated).detach() # block back propagation
        else:
            logit_Q_a[:, :-1] = torch.stack([self.logit_Q[i](z_Q_a_modulated[:, i, :]).detach() for i in range(self.tau)], dim=1)
        
        logit_Q_a_star = torch.zeros(batch_size, tau + 1, 1, device=A.device)
        logit_Q_a_star[:, :-1] = logit_Q_a[:, :-1] + eps.detach()
        
        Q_a = torch.sigmoid(logit_Q_a)
        Q_a_star = torch.sigmoid(logit_Q_a_star)
        
        # degeneration of Q
        Q_l, Q_l_star, Q_a, Q_a_star = self._set_deterministic_Q(Q_l, Q_l_star, Q_a, Q_a_star, Y)
 
        # IPW
        # J = (A == a) / (G * A + (1 - G) * (1 - A)).detach()
        A_forecast = A[:,start_idx:,:]
        J = (A_forecast == cf_seq) / (G * A_forecast + (1 - G) * (1 - A_forecast)).detach()
        g = torch.ones(batch_size, tau + 1, 1, device=A.device)
        g[:, 1:] = J.cumprod(dim=1)
        g = torch.clip(g, 0, 100)

        IC = (g * (Q_a_star - Q_l_star)).sum(dim=1)

        return {
            "Q_l": Q_l,
            "Q_l_star": Q_l_star,
            "Q_a": Q_a,
            "Q_a_star": Q_a_star,
            "G": G,
            "g": g,
            "J": J,
            "IC": IC,
        }
    
    def _set_deterministic_Q(self, Q_l, Q_l_star, Q_a, Q_a_star, Y):
        n = Y.shape[0]
        tau = self.projection_horizon + 1

        T0 = torch.zeros((n, tau + 1, 1), device=Y.device) # indicator of t == 0
        T0[:, 0] = 1

        if self.outcome_type == 'binary_irreversible':
            # Original DLTMLE implementation for binary_irreversible outcomes
            R = torch.ones((n, tau + 1, 1), device=Y.device) # survival indicator
            R[:, 2:] = 1 - Y[:, :-1]

            Q_a[:, -1] = Y[:, -1]
            Q_a[:, 1:-1] = torch.where(Y[:, :-1] == 1, 1, Q_a[:, 1:-1]) # Q^a_{t+1} = 1 if Y_t = 1 for t = 0, ..., tau-1    
            
            Q_l = torch.where(R == 1, Q_l, 1) # Q^l_{t+2} = 1 if Y_t = 1 for t = 0, ..., tau-1
            Q_l = torch.where(T0 == 0, Q_l, Q_a[:, 0].mean())

            Q_a_star[:, -1] = Y[:, -1]
            Q_a_star[:, 1:-1] = torch.where(Y[:, :-1] == 1, 1, Q_a_star[:, 1:-1]) # Q^a_{t+1} = 1 if Y_t = 1 for t = 0, ..., tau-1
            
            Q_l_star = torch.where(R == 1, Q_l_star, 1) # Q^l_{t+2} = 1 if Y_t = 1 for t = 0, ..., tau-1
            Q_l_star = torch.where(T0 == 0, Q_l_star, Q_a_star[:, 0].mean())
        
        elif self.outcome_type == 'binary' or self.outcome_type == 'continuous':
            Q_a[:, -1] = Y[:, -1]
            Q_a_star[:, -1] = Y[:, -1]

            Q_l = torch.where(T0 == 0, Q_l, Q_a[:, 0].mean())
            Q_l_star = torch.where(T0 == 0, Q_l_star, Q_a_star[:, 0].mean())
        
        else:
            raise ValueError(f"Invalid outcome type: {self.outcome_type}")

        return Q_l, Q_l_star, Q_a, Q_a_star

    def loss(self, S_hat, S, S_hat_target=None):
        Q_l, Q_l_star, Q_a, Q_a_star, G, g, _, IC = S_hat.values()
        A, Y = S["A"], S["Y"]
        A_forecast = A[:,A.shape[1]-G.shape[1]:,:]

        if S_hat_target is not None:
            Q_a = S_hat_target['Q_a'].detach()
            Q_a_star = S_hat_target['Q_a_star'].detach()
        else:
            Q_a = S_hat['Q_a']
            Q_a_star = S_hat['Q_a_star']

        Q_a = torch.clip(Q_a, 0, 1)
        
        if self.outcome_type == 'binary_irreversible':
            H = nn.BCELoss(reduction='none')
            R = torch.ones_like(Y)
            R[:, 1:] = 1 - Y[:, :-1] # indicator of survival
            R = R[:,A.shape[1]-G.shape[1]:,:]
            sample_Q = (R * H(Q_l[:, 1:], Q_a[:, 1:])).sum(dim=1).squeeze(-1)
            loss_Q_star = (g[:, 1:] * R * H(Q_l_star[:, 1:], Q_a_star[:, 1:])).sum(dim=1).mean()
            loss_Q_last = (R[:,-1] * H(Q_l_star[:, -1], Q_a_star[:, -1])).mean()
        
        elif self.outcome_type == 'binary':
            H = nn.BCELoss(reduction='none')
            R = torch.ones_like(A_forecast)
            sample_Q = (R * H(Q_l[:, 1:], Q_a[:, 1:])).sum(dim=1).squeeze(-1)
            loss_Q_star = (g[:, 1:] * R * H(Q_l_star[:, 1:], Q_a_star[:, 1:])).sum(dim=1).mean()
            loss_Q_last = (R[:,-1] * H(Q_l_star[:, -1], Q_a_star[:, -1])).mean()
            
        elif self.outcome_type == 'continuous':
            H = nn.MSELoss(reduction='none')
            # For continuous outcomes, all time steps are valid (no survival concept)
            R = torch.ones_like(A_forecast)
            
            # mse loss in the logit space
            sample_Q = H(Q_l[:, 1:], Q_a[:, 1:]).sum(dim=1).squeeze(-1)
            loss_Q_star = (g[:, 1:] * H(Q_l_star[:, 1:], Q_a_star[:, 1:])).sum(dim=1).mean()
            loss_Q_last = H(Q_l_star[:, -1], Q_a_star[:, -1]).mean()
        
        else:
            raise ValueError(f"Invalid outcome type: {self.outcome_type}")

        loss_Q = sample_Q.mean()

        # Propensity score loss (always binary for treatments)
        H_binary = nn.BCELoss(reduction='none')
        loss_G = (R * H_binary(G, A_forecast)).mean(dim=1).mean()

        loss = loss_Q + self.alpha * loss_G + self.beta * loss_Q_star

        return {
            "L": loss,
            "Q": loss_Q,
            "G": loss_G,
            "GQ": loss_G + loss_Q,
            "Q_star": loss_Q_star,
            "Q_last": loss_Q_last,
            "PnIC": IC.mean(),
            "PnIC2": (IC ** 2).mean()
        }

    def training_step(self, batch, batch_idx):
        S_hat = self(**batch)
        S_hat_target = None
        if self.use_target_network and (self.target_network is not None):
            with torch.no_grad():
                S_hat_target = self.target_network(**batch)
        loss = self.loss(S_hat, batch, S_hat_target=S_hat_target)

        for k, v in loss.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=(self.logger is not None))

        # Per-policy diagnostics (helps attribute spikes to a specific policy).
        if "policy_idx" in batch:
            Q_l = S_hat["Q_l"]
            G = S_hat["G"]
            A, Y = batch["A"], batch["Y"]
            A_forecast = A[:, A.shape[1] - G.shape[1] :, :]

            # Match loss() behavior for Q targets when target network is enabled.
            if S_hat_target is not None:
                Q_a = S_hat_target["Q_a"].detach()
            else:
                Q_a = S_hat["Q_a"]
            Q_a = torch.clip(Q_a, 0, 1)

            if self.outcome_type == "binary_irreversible":
                Hq = nn.BCELoss(reduction="none")
                R = torch.ones_like(Y)
                R[:, 1:] = 1 - Y[:, :-1]
                R = R[:, A.shape[1] - G.shape[1] :, :]
                sample_Q = (R * Hq(Q_l[:, 1:], Q_a[:, 1:])).sum(dim=1).squeeze(-1)
            elif self.outcome_type == "binary":
                Hq = nn.BCELoss(reduction="none")
                R = torch.ones_like(A_forecast)
                sample_Q = (R * Hq(Q_l[:, 1:], Q_a[:, 1:])).sum(dim=1).squeeze(-1)
            elif self.outcome_type == "continuous":
                Hq = nn.MSELoss(reduction="none")
                R = torch.ones_like(A_forecast)
                sample_Q = Hq(Q_l[:, 1:], Q_a[:, 1:]).sum(dim=1).squeeze(-1)
            else:
                sample_Q = None
                R = torch.ones_like(A_forecast)

            policy_idx = batch["policy_idx"].view(-1).to(dtype=torch.long)
            for pid in torch.unique(policy_idx):
                mask = policy_idx == pid
                if not bool(mask.any()):
                    continue
                pid_int = int(pid.item())
                if sample_Q is not None:
                    self.log(
                        f"train/Q_policy_{pid_int}",
                        sample_Q[mask].mean(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=(self.logger is not None),
                    )
        
        return loss["L"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Called after the optimizer step; good place to update the target network.
        if self.use_target_network and self.target_network is not None:
            self._soft_update_target_network(self.target_polyak_tau)

    def validation_step(self, batch, batch_idx):
        loss = self.loss(self(**batch), batch)

        for k, v in loss.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=(self.logger is not None))
    
    def test_step(self, batch, batch_idx):
        loss = self.loss(self(**batch), batch)

        for k, v in loss.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=(self.logger is not None))
        
        return loss["L"]

    def predict_step(self, batch, batch_idx):
        x = self(**batch)
        x["loss"] = self.loss(x, batch)
        return x

    def configure_optimizers(self):
        params = [
            p
            for n, p in self.named_parameters()
            if not n.startswith("target_network.") and p.requires_grad
        ]
        return torch.optim.Adam(params, lr=float(self.learning_rate))
    
    def solve_canonical_gradient(self, trainer, loader, tau, lambda_reg=5e-3):
        device = next(self.parameters()).device
        eps = torch.zeros(tau + 1, device=device)

        preds = trainer.predict(self, loader)
        if self.outcome_type == "binary_irreversible":
            Y = np.concatenate([x["Y"] for x in loader], axis=0)[:, :, 0]
            r = np.ones((len(loader.dataset), tau))
            r[:, 1:] = 1 - Y[:, :-1]
        elif self.outcome_type == "binary" or self.outcome_type == "continuous":
            r = np.ones((len(loader.dataset), tau))
        else:
            raise ValueError(f"Invalid outcome type: {self.outcome_type}")

        y_hat = np.concatenate([x["Q_l"] for x in preds], axis=0)[:, :, 0]
        y = np.concatenate([x["Q_a"] for x in preds], axis=0)[:, :, 0]
        g = np.concatenate([x["g"] for x in preds], axis=0)[:, :, 0]

        for t in reversed(range(tau)):
            eps[t] = solve_one_dimensional_submodel_regularized(
                y_hat[:, t + 1],
                expit(logit(y[:, t + 1]) + float(eps[t + 1])),
                r[:, t] * g[:, t + 1],
                lmbda=lambda_reg,
            )

        print("eps: ", eps)
        self.eps = torch.nn.Parameter(eps[:-1].detach(), requires_grad=False)

    def get_estimates_from_prediction(self, pred, loader, verbose=True, use_key="Q_a_star", clip_Q_a=False):
        Q_l_star = torch.cat([x["Q_l_star"] for x in pred], axis=0).detach().numpy().squeeze()
        Q_a_extracted = torch.cat([x[use_key] for x in pred], axis=0).detach().numpy().squeeze()
        if clip_Q_a:
            Q_a_extracted = np.clip(Q_a_extracted, 0, 1)
        IC = torch.cat([x["IC"] for x in pred], axis=0).detach().numpy().squeeze()

        PnIC = IC.mean()
        PnIC2 = (IC ** 2).mean()
        EIC = np.abs(PnIC / PnIC2 ** 0.5)

        Y = torch.cat([x["Y"] for x in loader], axis=0).detach().numpy()[:,:,0]
        R = np.ones((Y.shape[0], Y.shape[1] + 1))
        if self.outcome_type == 'binary_irreversible':
            R[:, 2:] = 1 - Y[:, :-1]
        R = R[:,-Q_l_star.shape[1]:]

        est = Q_a_extracted[:, 0].mean()
        se = np.sqrt((IC ** 2).mean() / IC.shape[0])

        if verbose:
            print("est: ", est)
            print("CI: ", est - 1.96 * se, est + 1.96 * se)
            print("se: ", se)

            print("E_n[IC] = {}".format(PnIC))
            print("PnIC/√PnIC2 = {}".format(EIC))

            print("Q_l_star", (R * Q_l_star).sum(axis=0) / R.sum(axis=0))
            print(use_key, (R * Q_a_extracted).sum(axis=0) / R.sum(axis=0))

        return est, se, PnIC, EIC