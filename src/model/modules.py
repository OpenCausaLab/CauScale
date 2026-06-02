import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

class TwoStreamEncoder(nn.Module):
    """
    Two-stream axial transformer encoder for CauScale.
    - Data stream: (B, m, n) observations -> axial attention over samples and variables
    - Graph stream: (B, n, n) precision matrix -> axial attention over nodes
    Each block fuses the data stream into the graph stream via Data2GraphLayer.
    Reduction units halve the data-stream sample dimension every k blocks.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.padding_idx = 0
        self.embed_data = nn.Linear(2, args.embed_dim)  # FLOP: 2*B*m*n*2(input_dim)*d(output_dim)
        self.embed_graph = nn.Linear(1, args.embed_dim)
        self.dropout_module = nn.Dropout(self.args.dropout)

        # Transformer encoder layers
        layers = []
        for _ in range(self.args.transformer_num_layers):
            graph_layer = AxialTransformerLayer(
                embedding_dim=self.args.embed_dim,
                ffn_embedding_dim=self.args.ffn_embed_dim,
                num_attention_heads=self.args.n_heads,
                dropout=self.args.dropout,
                scale_rows=args.scale_graph_rows,
                scale_cols=args.scale_graph_cols,
                attn_shape=args.attn_shape,
                head_dim=args.head_dim,
            )
            data_layer = AxialTransformerLayer(
                embedding_dim=self.args.embed_dim,
                ffn_embedding_dim=self.args.ffn_embed_dim,
                num_attention_heads=self.args.n_heads,
                dropout=self.args.dropout,
                scale_rows=True,
                scale_cols=args.scale_data_cols,
                attn_shape=args.attn_shape,
                head_dim=args.head_dim,
            )
            data2graph_layer = Data2GraphLayer(
                embed_dim=self.args.embed_dim,
                n_heads=self.args.n_heads,
                dropout=self.args.dropout,
                attn_shape=args.attn_shape,
                head_dim=args.head_dim,
            )
            layers.append(DataGraphBlock(graph_layer, data_layer, data2graph_layer))
        self.layers = nn.ModuleList(layers)
        self.reduction_k = 2  # apply reduction unit every k blocks
        self.reduction_r = 2  # reduction factor per unit
        self.reduction_unit = ReductionUnit(self.reduction_r)

        self.data_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim) # FLOP: 8*B*m*n*d
        self.data_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.graph_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.graph_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)

    def forward(self, batch):
        """
            B: batch size, m: number of samples, n: number of nodes
        """
        # initialize tensors
        datas = batch["data"]  # [B, m, n]
        interv = batch["interv"]  # [B, m, n]
        graphs = batch["feats"]   # [B, n, n]
        assert datas.ndim == 3, datas.shape
        padding_mask_data = datas.eq(self.padding_idx)
        padding_mask_graph = graphs.eq(self.padding_idx)
        if not padding_mask_data.any():
            padding_mask_data = None
        if not padding_mask_graph.any():
            padding_mask_graph = None
        datas = torch.stack([datas, interv], dim=-1) # [B, m, n, 2]

        # embedding and layer norm
        datas = self.embed_data(datas) # [B, m, n, 2] -> [B, m, n, d]
        graphs = self.embed_graph(graphs[...,None]) # [B, n, n, 1] -> [B, n, n, d]
        datas = self.data_layer_norm_before(datas)
        datas = self.dropout_module(datas)
        graphs = self.graph_layer_norm_before(graphs)
        graphs = self.dropout_module(graphs)
        if padding_mask_data is not None:
            datas = datas * (1 - padding_mask_data.unsqueeze(-1).type_as(datas))
        if padding_mask_graph is not None:
            graphs = graphs * (1 - padding_mask_graph.unsqueeze(-1).type_as(graphs))

        # into DataGraph Blocks & Reduction Units
        datas = datas.permute(1, 2, 0, 3) # [B, m, n, d] -> [m, n, B, d]
        graphs = graphs.permute(1, 2, 0, 3) # [B, n, n, d] -> [n, n, B, d]
        for layer_idx, layer in enumerate(self.layers):
            datas, graphs = layer(
                datas, graphs, 
                padding_mask_data=padding_mask_data,
                padding_mask_graph=padding_mask_graph,
            )
            if not self.args.disable_reduction_unit and layer_idx >= self.reduction_k and layer_idx % self.reduction_k == 0:
                datas, padding_mask_data = self.reduction_unit(datas, padding_mask_data)

        # output processing
        datas = self.data_layer_norm_after(datas)
        datas = datas.permute(2, 0, 1, 3)  # m x n x B x D -> B x m x n x D
        graphs = self.graph_layer_norm_after(graphs)
        graphs = graphs.permute(2, 0, 1, 3)  # n x n x B x D -> B x n x n x D
        if padding_mask_graph is not None:
            graphs = graphs * (1 - padding_mask_graph.unsqueeze(-1).type_as(graphs))

        return graphs

    @property
    def num_layers(self):
        return self.args.transformer_num_layers


class DataGraphBlock(nn.Module):
    """
    Single DataGraph block: data axial attention → fuse into graph stream → graph axial attention.
    """
    def __init__(self, graph_layer, data_layer, data2graph_layer):
        super().__init__()
        self.graph_layer = graph_layer
        self.data_layer = data_layer
        self.data2graph_layer = data2graph_layer

        self.linear = nn.Linear(self.data_layer.embedding_dim + 1,
                                self.data_layer.embedding_dim)

    def forward(self, datas, graphs, padding_mask_data, padding_mask_graph):
        """
            datas = num_samples, num_nodes, batch_size, embed_dim
            graphs = num_nodes, num_nodes, batch_size, embed_dim
        """
        datas = self.data_layer(datas, padding_mask_data) # [m, n, B, d]
        data_feats = self.data2graph_layer(datas, padding_mask_data) # [m, n, B, d] -> [B, n, n]
        data_feats = data_feats.unsqueeze(-1) # [B, n, n, 1]
        data_feats = data_feats.permute(1, 2, 0, 3) # [n, n, B, 1]
        graphs = torch.cat([graphs, data_feats], dim=-1) # [n, n, B, d] + [n, n, B, 1] -> [n, n, B, d + 1]
        graphs = self.linear(graphs) # [n, n, B, d + 1] -> [n, n, B, d]
        graphs = self.graph_layer(graphs, padding_mask_graph) # [n, n, B, d]
        return datas, graphs


class ReductionUnit(nn.Module):
    """
    Reduction Unit: reduces the data-stream sample dimension by factor r
    via mean-pooling consecutive groups of r samples.
    Applied every k blocks, giving effective sample length m_b = m / r^(floor(b/k)).
    Has no learnable parameters.
    """
    def __init__(self, r=2):
        super().__init__()
        self.r = r  # reduction factor

    def forward(self, datas, padding_mask_data):
        """
        datas: [m, n, B, d]
        padding_mask_data: [B, m, n] or None
        """
        m, n, B, d = datas.shape
        m = (m // self.r) * self.r
        datas = datas[:m]
        if padding_mask_data is not None:
            padding_mask_data = padding_mask_data[:, :m]
            padding_mask_data = padding_mask_data.reshape(B, m//self.r, self.r, n)
            padding_mask_data = padding_mask_data.float().mean(dim=2).bool()
        datas = datas.reshape(m//self.r, self.r, n, B, d).mean(dim=1)
        return datas, padding_mask_data


class AxialTransformerLayer(nn.Module):
    """
    2D axial transformer layer with row and column self-attention.
    Modified from:
    https://github.com/facebookresearch/esm/blob/main/esm/modules.py
    """
    def __init__(
                self, *,
                embedding_dim: int = 128,
                ffn_embedding_dim: int = 512,
                num_attention_heads: int = 16,
                dropout: float = 0.1,
                activation_dropout: float = 0.1,
                scale_rows=False,
                scale_cols=False,
                attn_shape="hnij",
                head_dim=-1,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        row_self_attention = RowSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            use_scaling=scale_rows,
            attn_shape=attn_shape,
            head_dim=head_dim
        )
        column_self_attention = ColumnSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            use_scaling=scale_cols,
            attn_shape=attn_shape,
            head_dim=head_dim
        )

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim, 
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
        )

        self.row_self_attention = self.build_residual(row_self_attention)
        self.column_self_attention = self.build_residual(column_self_attention)
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        x = batch_size, num_rows, num_cols, embedding_dim
        """
        x = self.row_self_attention(
            x, 
            self_attn_padding_mask=self_attn_padding_mask
        )
        x = self.column_self_attention(
            x, 
            self_attn_padding_mask=self_attn_padding_mask
        )
        x = self.feed_forward_layer(x)
        return x

class RowSelfAttention(nn.Module):
    """
    Row-wise self-attention over input [m, n, B, d].
    Modified from:
    https://github.com/facebookresearch/esm/blob/main/esm/axial_attention.py
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        use_scaling=False,
        attn_shape: str = "hnij",
        head_dim: int = -1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads if head_dim == -1 else head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_scaling = use_scaling
        self.attn_shape = attn_shape

        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_heads)
        self.q_proj = nn.Linear(embed_dim, self.head_dim * num_heads)

        self.out_proj = nn.Linear(self.num_heads*self.head_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        if not self.use_scaling:
            return 1.0
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def compute_attention_weights(
        self,
        x,
        scaling: float,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(q)

        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_padding_mask is not None:
            if self.attn_shape=="hnij":
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                    -10000,
                )
            elif self.attn_shape=="rhnij":
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(1).unsqueeze(3),
                    -10000,
                )
            else:
                raise NotImplementedError

        return attn_weights

    def compute_attention_update(
        self, 
        x, 
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, self.num_heads*self.head_dim)
        output = self.out_proj(context)
        return output

    def forward(self, x, self_attn_padding_mask=None):
        """
        x: [m, n, B, d]
        self_attn_padding_mask: [B, m, n]
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(x, scaling, self_attn_padding_mask)
        attn_probs = attn_weights.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        output = self.compute_attention_update(x, attn_probs)
        return output


class ColumnSelfAttention(RowSelfAttention):
    """Column-wise self-attention: transposes to [n, m, B, d] then applies row attention."""

    def forward(self, x, self_attn_padding_mask=None):
        x = x.permute(1, 0, 2, 3)
        if self_attn_padding_mask is not None:
            self_attn_padding_mask = self_attn_padding_mask.transpose(1, 2)
        output = super().forward(x, self_attn_padding_mask)
        return output.permute(1, 0, 2, 3)

class Data2GraphLayer(nn.Module):

    def __init__(self, embed_dim, n_heads, dropout, attn_shape="hnij", head_dim=-1):
        super().__init__()
        self.proj_u = PoolingFFN(embed_dim, embed_dim)
        self.proj_v = PoolingFFN(embed_dim, embed_dim)
        self.axial_layer = AxialTransformerLayer(
                embedding_dim=embed_dim,
                ffn_embedding_dim=embed_dim,
                num_attention_heads=n_heads,
                dropout=dropout,
                scale_rows=True,
                attn_shape=attn_shape,
                head_dim=head_dim,
            )

    def forward(self, x, padding_mask):
        x_info = self.axial_layer(x, padding_mask) # [m, n, B, d]
        x_info = x_info.permute(2, 0, 1, 3) # [m, n, B, d] -> [B, m, n, d]
        u = self.proj_u(x_info) # [B, m, n, d] -> [B, n, d]
        v = self.proj_v(x_info) # [B, m, n, d] -> [B, n, d]
        x_feats = torch.matmul(u, v.transpose(-2, -1)) # [B, n, d] * [B, n, d] -> [B, n, n]
        return x_feats
        

class PoolingFFN(nn.Module):

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        # B x m x n x d -> B x n x d
        x = features.mean(dim=1) # [B, n, d]
        x = self.linear(x) # FLOP: 2*n*d*d
        x = self.activation_fn(x)
        x = self.layer_norm(x) 
        x = self.linear2(x)  # FLOP: 2*n*d*d
        return x

class NormalizedResidualBlock(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = ESM1bLayerNorm(self.embed_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x)) # FLOP: 2*B*m*n*embed_dim*ffn_embed_dim
        x = self.activation_dropout_module(x) # [B, m, n, d]
        x = self.fc2(x) # FLOP: 2*B*m*n*ffn_embed_dim*d
        return x


class TopLayer(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.linear(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.linear2(x)
        return x

