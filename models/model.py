import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from Optim import ScheduledOptim

import os
from transformers import AutoModel, AutoConfig

from einops import rearrange
from .xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .xlstm.blocks.slstm.block import sLSTMBlockConfig
from .xlstm.blocks.slstm.layer import sLSTMLayerConfig
from .xlstm.components.feedforward import FeedForwardConfig

def trans_to_cuda(variable):
    """
    Move tensor to GPU if CUDA is available.
    
    Args:
        variable: Input tensor or PyTorch module.
    
    Returns:
        The input moved to GPU if available, otherwise on CPU.
    """
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    """
    Move tensor to CPU.
    
    Args:
        variable: Input tensor or PyTorch module.
    
    Returns:
        The input moved to CPU.
    """
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def get_previous_user_mask(seq, user_size):
    """
    Create a mask that prevents future information leakage by masking out
    users that haven't been activated yet in the cascade sequence.
    This ensures auto-regressive prediction consistency.
    
    Args:
        seq (torch.Tensor): Input sequence tensor of shape [batch, seq_len].
        user_size (int): Total number of unique users.
    
    Returns:
        torch.Tensor: Mask tensor of shape [batch, seq_len, user_size] where
            masked positions are set to -inf.
    """
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # Force the PAD token (index 0) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()

class GraphATT(nn.Module):
    """
    Graph Attention Network for learning social friendship representations.
    
    This module applies multiple GAT convolution layers to aggregate information
    from the social graph structure.
    """
    def __init__(self, ninp, nout, layers):
        """
        Initialize the GraphATT module.
        
        Args:
            ninp (int): Input feature dimension.
            nout (int): Output feature dimension.
            layers (int): Number of GAT convolution layers.
        """
        super(GraphATT, self).__init__()

        self.ninp = ninp
        self.nout = nout
        self.gnn_list = [GATConv(ninp, nout).cuda() for _ in range(layers)]
        self.init_weights()

    def init_weights(self):
        """Initialize weights with uniform distribution."""
        stdv = 1.0 / math.sqrt(self.nout)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, graph, embedding, layer):
        """
        Forward pass through one GAT convolution layer.
        
        Args:
            graph (Data): PyTorch Geometric Data object containing graph structure.
            embedding (torch.Tensor): Input node embeddings.
            layer (int): Index of the GAT layer to use.
        
        Returns:
            torch.Tensor: Output node embeddings after convolution.
        """
        graph_edge_index = graph.edge_index
        graph_output = self.gnn_list[layer](embedding, graph_edge_index)
        return graph_output

class LayerNorm(nn.Module):
    """
    Layer normalization module following TensorFlow style where epsilon
    is added inside the square root.
    """
    def __init__(self, hidden_size, eps=1e-12):
        """
        Initialize LayerNorm.
        
        Args:
            hidden_size (int): Feature dimension to be normalized.
            eps (float): Small epsilon for numerical stability. Defaults to 1e-12.
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """
        Forward pass of layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Normalized output tensor.
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FilterLayer(nn.Module):
    """
    Frequency domain filtering layer based on FFT.
    This layer applies learnable filtering in the frequency domain to capture
    long-range dependencies efficiently.
    """
    def __init__(self, args):
        """
        Initialize the FilterLayer.
        
        Args:
            args: Argument object containing hyperparameters.
        """
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.filter_seq//2+1, args.embSize, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.filter_dropout_prob)
        self.LayerNorm = LayerNorm(args.embSize, eps=1e-12)

    def forward(self, input_tensor):
        """
        Forward pass applying FFT filtering.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape [batch, seq_len, hidden].
        
        Returns:
            torch.Tensor: Filtered output tensor with residual connection.
        """
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    """
    Intermediate feed-forward module after filtering.
    This applies two linear projections with GELU activation and dropout.
    """
    def __init__(self, args):
        """
        Initialize the Intermediate module.
        
        Args:
            args: Argument object containing hyperparameters.
        """
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.embSize, args.embSize * 4)
        self.intermediate_act_fn = nn.GELU()

        self.dense_2 = nn.Linear(4 * args.embSize, args.embSize)
        self.LayerNorm = LayerNorm(args.embSize, eps=1e-12)
        self.dropout = nn.Dropout(args.filter_dropout_prob)

    def forward(self, input_tensor):
        """
        Forward pass through the feed-forward network.
        
        Args:
            input_tensor (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor with residual connection.
        """
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FilterMLP(nn.Module):
    """
    Combined filtering and MLP module for processing sequential data.
    Stacks multiple FilterLayer and Intermediate blocks.
    """
    def __init__(self, args):
        """
        Initialize the FilterMLP module.
        
        Args:
            args: Argument object containing hyperparameters.
        """
        super(FilterMLP, self).__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)
        self.layers = args.filter_num

    def forward(self, hidden_states):
        """
        Forward pass through multiple filtering and MLP layers.
        
        Args:
            hidden_states (torch.Tensor): Input sequence tensor.
        
        Returns:
            torch.Tensor: Output after all layers.
        """
        for _ in range(self.layers):
            hidden_states = self.filterlayer(hidden_states)
            hidden_states = self.intermediate(hidden_states)
            intermediate_output = hidden_states
        return intermediate_output

class fusion(nn.Module):
    """
    Attention-based fusion module to combine multiple embedding sources.
    Learns adaptive weights to fuse static and dynamic embeddings.
    """
    def __init__(self, input_size, dropout=0.2):
        """
        Initialize the fusion module.
        
        Args:
            input_size (int): Input feature dimension.
            dropout (float): Dropout probability. Defaults to 0.2.
        """
        super(fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize weights with Xavier normal initialization."""
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        """
        Forward pass fusing two embedding sources.
        
        Args:
            hidden (torch.Tensor): First embedding tensor of shape [B, L, H].
            dy_emb (torch.Tensor): Second embedding tensor of shape [B, L, H].
        
        Returns:
            torch.Tensor: Fused output tensor of shape [B, L, H].
        """
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class P_PsLSTM(nn.Module):
    """
    Patch-based sLSTM for processing related memory cascades.
    This module processes patches of retrieved related cascades using xLSTM
    to capture diffusion patterns from episodic memory.
    """
    def __init__(self, configs, patch_num):
        """
        Initialize the P_PsLSTM module.
        
        Args:
            configs: Configuration object containing model hyperparameters.
            patch_num (int): Number of patches (window size) from retrieved cascades.
        """
        super(P_PsLSTM, self).__init__()
        self.p_hidden_dim = configs.embedding_dim
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.emb_size = configs.embSize
        self.num_blocks = configs.num_blocks
        self.patch_num = patch_num

        # Configure xLSTM stack with sLSTM blocks
        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(num_heads=configs.num_heads,
                                       conv1d_kernel_size=configs.conv1d_kernel_size,
                                       dropout=configs.dropout,
                                       bias_init="powerlaw_blockdependent"),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=self.emb_size,
            num_blocks=self.num_blocks,
            embedding_dim=self.p_hidden_dim,
            slstm_at="all"
        )

        self.embedding = nn.Linear(self.emb_size, self.p_hidden_dim)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.projection = nn.Linear(self.p_hidden_dim * self.patch_num, self.emb_size)

    def forward(self, input):
        """
        Forward pass processing related cascade patches.
        
        Args:
            input (torch.Tensor): Input tensor of shape [B, L, E, H] where:
                B is batch size, L is sequence length, E is number of elements (top_k),
                H is embedding dimension.
        
        Returns:
            torch.Tensor: Output tensor of shape [B, L, emb_size] containing
                aggregated memory features.
        """
        B, L, E, H = input.shape

        x = rearrange(input, 'b l e h -> (b l) e h')
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        x = x.flatten(1)

        x = self.projection(x)
        x = rearrange(x, '(b l) c -> b l c', b=B, l=L)

        return x

class LSTMGNN(nn.Module):
    """
    MemIDP: Memory-based Information Diffusion Prediction with Dual Implicit Memory.
    
    This model combines:
    1. **Episodic Memory**: Retrieves and processes historically similar cascades using xLSTM
    2. **Semantic Memory**: Encodes structural information via hypergraphs and GNN
    3. **PLM Fusion**: Uses pre-trained language model for feature fusion
    4. **Dual Memory Integration**: Fuses both memory types for personalized prediction
    
    The model predicts the next user to be influenced in an information diffusion cascade.
    """
    def __init__(self, adjacency, hypergraphs, args, dropout=0.2):
        """
        Initialize the MemIDP model.
        
        Args:
            adjacency: Social graph adjacency from PyTorch Geometric.
            hypergraphs (list): List containing two sparse hypergraph matrices:
                [HG_Source, HG_Item] - source-user hypergraph and item-user hypergraph.
            args: Argument object containing all hyperparameters.
            dropout (float): Dropout probability. Defaults to 0.2.
        """
        super(LSTMGNN, self).__init__()

        # Model dimensions
        self.emb_size = args.embSize
        self.pos_dim = args.posSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.n_channel = len(hypergraphs)+1  # 2 hypergraphs + 1 social graph
        self.seq_length = args.max_lenth-1

        # Graph structures
        self.adjacency = adjacency     # Social friendship graph
        self.H_Source = hypergraphs[0] # Source-user hypergraph (semantic memory)
        self.H_Item = hypergraphs[1]   # Cascade-item hypergraph (semantic memory)

        # Base embeddings
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)

        # Channel-wise self-gating for adaptively modulating features
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        # Cross-channel attention for aggregation
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        # GAT for social graph aggregation
        self.GraphConv = GraphATT(self.emb_size, self.emb_size, self.layers)

        # Patch-based sLSTM for processing retrieved episodic memory
        self.rpslstm = P_PsLSTM(args, self.window_size)
        self.rfilter = FilterMLP(args)

        # Frequency domain filtering + PLM fusion
        self.window_size = args.window_size
        self.topk = args.top_k

        # Pre-trained language model for knowledge-enhanced fusion
        base_dir = os.path.dirname(__file__)
        self.lm_name = getattr(args, "lm_name", "gpt2")
        self.lm_path = os.path.join(base_dir, self.lm_name)

        self.lm_config = AutoConfig.from_pretrained(self.lm_path)
        self.lm = AutoModel.from_pretrained(self.lm_path)
        for param in self.lm.parameters():
            param.requires_grad = False  # Freeze PLM parameters

        print("lm_config:", self.lm.config)
        lm_hidden_size = getattr(self.lm.config, "n_embd", getattr(self.lm.config, "hidden_size", None))
        self.gpt_proj_in = nn.Linear(self.emb_size * 2, lm_hidden_size)
        self.gpt_proj_out = nn.Linear(lm_hidden_size, self.emb_size)

        # Final projection to vocabulary
        self.proj_user = nn.Linear(self.emb_size, self.n_node)

        # Optimizer with learning rate scheduling
        self.optimizerAdamW = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=args.lr)
        self.optimizer = ScheduledOptim(self.optimizerAdamW, self.emb_size, args.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all parameters with uniform distribution."""
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        """
        Channel-wise self-gating mechanism to adaptively modulate features.
        
        Args:
            em (torch.Tensor): Input embedding matrix.
            channel (int): Channel index selecting the gating parameters.
        
        Returns:
            torch.Tensor: Gated output embedding.
        """
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def channel_attention(self, *channel_embeddings):
        """
        Cross-channel attention to aggregate embeddings from different channels.
        Learns adaptive weights for each channel based on the current context.
        
        Args:
            *channel_embeddings: Variable number of channel embedding tensors.
        
        Returns:
            tuple: (mixed_embeddings, score) where:
                - mixed_embeddings: Attention-weighted aggregation of all channels.
                - score: Attention weights for each channel.
        """
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    def structure_embed(self):
        """
        Compute semantic memory embeddings by aggregating information from
        three different graph channels (two hypergraphs + one social graph).
        
        This method performs multi-layer graph convolution and cross-channel
        attention aggregation to produce the final user embeddings.
        
        Returns:
            tuple: (agg_Uemb, high_embs, u_emb_c2, simple_user_emb) where:
                - agg_Uemb: Final aggregated user embeddings (all channels).
                - high_embs: Aggregated embeddings from the two hypergraph channels.
                - u_emb_c2: Embeddings from item hypergraph channel.
                - simple_user_emb: Embeddings from social graph channel.
        """
        # Apply self-gating to each channel
        u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c2 = self.self_gating(self.user_embedding.weight, 1)
        simple_user_emb = self.self_gating(self.user_embedding.weight, 2)

        all_emb_c1 = [u_emb_c1]
        all_emb_c2 = [u_emb_c2]
        all_emb_simple = [simple_user_emb]

        # Multi-layer propagation
        for k in range(self.layers):
            # Channel 1: Source-user hypergraph
            u_emb_c1 = torch.sparse.mm(self.H_Source, u_emb_c1)
            norm_embeddings1 = F.normalize(u_emb_c1, p=2, dim=1)
            all_emb_c1 += [norm_embeddings1]

            # Channel 2: Item-user hypergraph
            u_emb_c2 = torch.sparse.mm(self.H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            # Channel 3: Social graph with GAT convolution
            simple_user_emb = self.GraphConv(self.adjacency, self.user_embedding.weight, k)
            norm_embeddings3 = F.normalize(simple_user_emb, p=2, dim=1)
            all_emb_simple += [norm_embeddings3]

        # Aggregate across layers
        u_emb_c1 = torch.stack(all_emb_c1, dim=1)
        u_emb_c1 = torch.sum(u_emb_c1, dim=1)
        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)

        simple_user_emb = torch.stack(all_emb_simple, dim=1)
        simple_user_emb = torch.sum(simple_user_emb, dim=1)

        # Cross-channel attention aggregation
        high_embs, attention_score = self.channel_attention(u_emb_c1, u_emb_c2)
        agg_Uemb = high_embs + simple_user_emb / 2

        return agg_Uemb, high_embs, u_emb_c2, simple_user_emb

    def draw_embed_visualization(self, user_embed):
        """
        Generate t-SNE visualization of user embeddings and save to PDF.
        
        Args:
            user_embed (torch.Tensor): User embedding matrix to visualize.
        """
        import numpy as np
        from openTSNE import TSNE
        import matplotlib.pyplot as plt

        np.random.seed(2025)
        idx = np.random.choice(self.n_node, 3000, replace=False)
        X = user_embed[idx].detach().cpu().numpy()

        X_2d = TSNE(n_components=2,
                    perplexity=5,
                    learning_rate=800,
                    n_iter=1000,
                    random_state=2025).fit(X)

        plt.figure(figsize=(5, 5))
        plt.scatter(X_2d[:, 0], X_2d[:, 1],
                    s=10, c='#CBAF9B', alpha=1)

        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])

        plt.savefig("RAGPT_tsne_3000_users.pdf",
                    dpi=300,
                    bbox_inches='tight')
        print("Visualization completed, saved to RAGPT_tsne_3000_users.pdf")

    def forward(self, input, label, input_time, label_time, related_cas, related_mask):
        """
        Forward pass for training with ground truth supervision.
        
        Args:
            input (torch.Tensor): Input cascade sequence [batch, seq_len].
            label (torch.Tensor): Target next user sequence [batch, seq_len].
            input_time (torch.Tensor): Timestamps for input sequence [batch, seq_len].
            label_time (torch.Tensor): Timestamps for target sequence [batch, seq_len].
            related_cas (torch.Tensor): Retrieved related cascades from memory [batch, (max_len-1)*top_k*window].
            related_mask (torch.Tensor): Padding mask for related cascades [batch, (max_len-1)*top_k*window].
        
        Returns:
            torch.Tensor: Prediction logits of shape [batch*seq_len, num_users].
        """
        # Get semantic memory embeddings from graph structures
        agg_Uemb, _, _, s_Uemb = self.structure_embed()

        # Process episodic memory: retrieved related cascades
        related_cas_emb = F.embedding(related_cas, s_Uemb)
        related_output = related_cas_emb * related_mask.unsqueeze(2)
        related_output = self.rfilter(related_output)
        related_output = rearrange(related_output, 'b (seq h) d -> b seq h d', seq=self.seq_length*self.topk, h=self.window_size)

        related_output = self.rpslstm(related_output)
        related_output = torch.reshape(related_output, (-1, self.seq_length, self.topk, self.emb_size))
        related_output = torch.sum(related_output, 2)

        # Get current cascade embedding from semantic memory
        cas_seq_emb = F.embedding(input, agg_Uemb)

        # PLM-based fusion of current and memory features
        combined_input = torch.cat([cas_seq_emb, related_output], dim=-1)
        gpt_input_proj = self.gpt_proj_in(combined_input)
        gpt_output = self.lm(inputs_embeds=gpt_input_proj).last_hidden_state
        output = self.gpt_proj_out(gpt_output)

        # Compute logits for next user prediction
        pre_y = torch.matmul(output, torch.transpose(agg_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()

    def model_prediction(self, input, input_time, related_cas, related_mask):
        """
        Model prediction for inference/testing.
        Same as forward pass but doesn't require ground truth.
        
        Args:
            input (torch.Tensor): Input cascade sequence [batch, seq_len].
            input_time (torch.Tensor): Timestamps for input sequence [batch, seq_len].
            related_cas (torch.Tensor): Retrieved related cascades from memory [batch, (max_len-1)*top_k*window].
            related_mask (torch.Tensor): Padding mask for related cascades [batch, (max_len-1)*top_k*window].
        
        Returns:
            torch.Tensor: Prediction logits of shape [batch*seq_len, num_users].
        """
        # Get semantic memory embeddings from graph structures
        agg_Uemb, _, _, s_Uemb = self.structure_embed()

        # Process episodic memory: retrieved related cascades
        related_cas_emb = F.embedding(related_cas, s_Uemb)
        related_output = related_cas_emb * related_mask.unsqueeze(2)
        related_output = self.rfilter(related_output)
        related_output = rearrange(related_output, 'b (seq h) d -> b seq h d', seq=self.seq_length*self.topk, h=self.window_size)

        related_output = self.rpslstm(related_output)
        related_output = torch.reshape(related_output, (-1, self.seq_length, self.topk, self.emb_size))
        related_output = torch.sum(related_output, 2)

        # Get current cascade embedding from semantic memory
        cas_seq_emb = F.embedding(input, agg_Uemb)

        # PLM-based fusion of current and memory features
        combined_input = torch.cat([cas_seq_emb, related_output], dim=-1)
        gpt_input_proj = self.gpt_proj_in(combined_input)
        gpt_output = self.lm(inputs_embeds=gpt_input_proj).last_hidden_state
        output = self.gpt_proj_out(gpt_output)

        # Compute logits for next user prediction
        pre_y = torch.matmul(output, torch.transpose(agg_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()


