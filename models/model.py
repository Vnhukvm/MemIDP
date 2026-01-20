import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from torch_geometric.nn import GATConv
from Optim import ScheduledOptim

import os
from transformers import AutoModel, AutoConfig
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    pass

from einops import rearrange
from .xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .xlstm.blocks.slstm.block import sLSTMBlockConfig
from .xlstm.blocks.slstm.layer import sLSTMLayerConfig
from .xlstm.components.feedforward import FeedForwardConfig

'''To GPU'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

'''Mask previous activated users'''
def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()

'''Learn friendship network'''
class GraphATT(nn.Module):
    def __init__(self, ninp, nout, layers):
        super(GraphATT, self).__init__()

        self.ninp = ninp
        self.nout = nout
        self.gnn_list = [GATConv(ninp, nout).cuda() for _ in range(layers)]
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.nout)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, graph, embedding, layer):
        graph_edge_index = graph.edge_index
        graph_output = self.gnn_list[layer](embedding, graph_edge_index)
        return graph_output

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.filter_seq//2+1, args.embSize, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.filter_dropout_prob)
        self.LayerNorm = LayerNorm(args.embSize, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, 25, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.embSize, args.embSize * 4)
        self.intermediate_act_fn = nn.GELU()

        self.dense_2 = nn.Linear(4 * args.embSize, args.embSize)
        self.LayerNorm = LayerNorm(args.embSize, eps=1e-12)
        self.dropout = nn.Dropout(args.filter_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FilterMLP(nn.Module):
    def __init__(self, args):
        super(FilterMLP, self).__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)
        self.layers = args.filter_num

    def forward(self, hidden_states):
        for _ in range(self.layers):
            hidden_states = self.filterlayer(hidden_states)
            hidden_states = self.intermediate(hidden_states)
            intermediate_output = hidden_states
        return intermediate_output

class fusion(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        # hidden,dy_emb [B, L, H]
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0) # 纵向拼接
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0) # 
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class P_PsLSTM(nn.Module):

    def __init__(self, configs, patch_num):
        super(P_PsLSTM, self).__init__()
        self.p_hidden_dim = configs.embedding_dim  
        self.patch_size = configs.patch_size  
        self.stride = configs.stride  
        self.emb_size = configs.embSize  
        self.num_blocks = configs.num_blocks
        self.patch_num = patch_num 
        
        # xLSTM configuration
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
        self.related_rnn = nn.LSTM(input_size=self.p_hidden_dim, hidden_size=self.p_hidden_dim, batch_first=True)

    def forward(self, input):
        B, L, E, H = input.shape  # [B, L, ele, 64]
        
        x = rearrange(input, 'b l e h -> (b l) e h')  
        x = self.embedding(x)  
        x = self.xlstm_stack(x)  # [(B * L), Num_patches, p_h_dim]
        # x, _ = self.related_rnn(x)
        x = x.flatten(1)  # [(B * L), Num_patches * p_h_dim] 

        x = self.projection(x)  # [(B * L), Embed_size]
        x = rearrange(x, '(b l) c -> b l c', b=B, l=L)  # [B, L, Embed_size]

        return x

class LSTMGNN(nn.Module):
    def __init__(self, adjacency, hypergraphs, args, dropout=0.2):
        super(LSTMGNN, self).__init__()

        # parameters
        self.emb_size = args.embSize
        self.pos_dim = args.posSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.n_channel = len(hypergraphs)+1  ## Hypergraph 2, Social graph
        self.seq_length = args.max_lenth-1

        # graph and hypergraph
        self.adjacency = adjacency     #social graph
        self.H_Source = hypergraphs[0] #source-user hypergraph
        self.H_Item = hypergraphs[1]   #cascade-item hypergraph

        ###### user and position embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)

        ### channel self-gating parameters
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        # graph model
        self.GraphConv = GraphATT(self.emb_size, self.emb_size, self.layers)

        #sequence model
        self.past_rnn = nn.GRU(input_size=self.emb_size + self.pos_dim, hidden_size=self.emb_size, batch_first= True)
        self.future_rnn = nn.GRU(input_size=self.emb_size + self.pos_dim, hidden_size=self.emb_size, batch_first=True, bidirectional=True)
        self.related_rnn = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)

        self.linear = nn.Linear(self.emb_size +self.pos_dim, self.emb_size)

        ### fusion layer
        self.fus = fusion(self.emb_size, dropout)

        self.reset_parameters()

        #### optimizer and loss function
        self.optimizerAdamW = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=args.lr)
        self.optimizer = ScheduledOptim(self.optimizerAdamW, self.emb_size, args.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        
        self.window_size = args.window_size
        self.topk = args.top_k
        
        ### PsLSTM and Filter
        self.rpslstm = P_PsLSTM(args, self.window_size)
        self.rfilter = FilterMLP(args)
        self.proj_user = nn.Linear(self.emb_size, self.n_node)

        base_dir = os.path.dirname(__file__)
        self.lm_name = getattr(args, "lm_name", "gpt2")
        self.lm_path = os.path.join(base_dir, self.lm_name)

        self.lm_config = AutoConfig.from_pretrained(self.lm_path)
        self.lm = AutoModel.from_pretrained(self.lm_path)
        for param in self.lm.parameters():
            param.requires_grad = False

        print("lm_config:", self.lm.config)
        lm_hidden_size = getattr(self.lm.config, "n_embd", getattr(self.lm.config, "hidden_size", None))
        self.gpt_proj_in = nn.Linear(self.emb_size * 2, lm_hidden_size)
        self.gpt_proj_out = nn.Linear(lm_hidden_size, self.emb_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings, score

    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self):

        u_emb_c1 = self.self_gating(self.user_embedding.weight, 0)
        u_emb_c2 = self.self_gating(self.user_embedding.weight, 1)
        simple_user_emb = self.self_gating(self.user_embedding.weight, 2)

        all_emb_c1 = [u_emb_c1]
        all_emb_c2 = [u_emb_c2]
        all_emb_simple = [simple_user_emb]

        for k in range(self.layers):
            # Channel Source
            u_emb_c1 = torch.sparse.mm(self.H_Source, u_emb_c1)
            norm_embeddings1 = F.normalize(u_emb_c1, p=2, dim=1)
            all_emb_c1 += [norm_embeddings1]

            # Channel Item
            u_emb_c2 = torch.sparse.mm(self.H_Item, u_emb_c2)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            # user convolution
            simple_user_emb = self.GraphConv(self.adjacency, self.user_embedding.weight, k)
            norm_embeddings3 = F.normalize(simple_user_emb, p=2, dim=1)
            all_emb_simple += [norm_embeddings3]

        u_emb_c1 = torch.stack(all_emb_c1, dim=1)
        u_emb_c1 = torch.sum(u_emb_c1, dim=1)
        u_emb_c2 = torch.stack(all_emb_c2, dim=1)
        u_emb_c2 = torch.sum(u_emb_c2, dim=1)

        simple_user_emb = torch.stack(all_emb_simple, dim=1)
        simple_user_emb = torch.sum(simple_user_emb, dim=1)

        # aggregating channel-specific embeddings
        high_embs, attention_score = self.channel_attention(u_emb_c1, u_emb_c2)
        agg_Uemb = high_embs + simple_user_emb / 2

        return agg_Uemb, high_embs, u_emb_c2, simple_user_emb

    def draw_embed_visualization(self, user_embed):
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

        # 去掉边框和刻度
        plt.axis('off')          # 关闭坐标轴
        plt.gca().set_position([0, 0, 1, 1])  # 让坐标轴占满整张图，无留白

        plt.savefig("RAGPT_tsne_3000_users.pdf",
                    dpi=300,
                    bbox_inches='tight')     # 去掉四周白边
        print("绘图已完成，已保存为 PDF...")

    def forward(self, input, label, input_time, label_time, related_cas, related_mask):

        '''structure embeddding'''
        agg_Uemb, _, _, s_Uemb = self.structure_embed()
        # self.draw_embed_visualization(agg_Uemb)

        '''related embedding'''
        related_cas_emb = F.embedding(related_cas, s_Uemb)  # [batch, 4975, 64]
        related_output = related_cas_emb * related_mask.unsqueeze(2)  # 扩张后广播相乘
        related_output = self.rfilter(related_output)
        related_output = rearrange(related_output, 'b (seq h) d -> b seq h d', seq=self.seq_length*self.topk, h=self.window_size)
        
        related_output = self.rpslstm(related_output) # [batch, 199*topk, 64]
        related_output = torch.reshape(related_output, (-1, self.seq_length, self.topk, self.emb_size)) 
        related_output = torch.sum(related_output, 2)   # [batch, 199, 64]
        
        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, agg_Uemb)  # [batch, seq_len=199, 64]
        
        '''GPT4Fusion'''
        combined_input = torch.cat([cas_seq_emb, related_output], dim=-1)
        
        # align gpt input dim 
        gpt_input_proj = self.gpt_proj_in(combined_input)  # [batch, 199, 768(gpt's dim)]
        gpt_output = self.lm(inputs_embeds=gpt_input_proj).last_hidden_state 
        output = self.gpt_proj_out(gpt_output) # [batch, 199, 64]
        
        pre_y = torch.matmul(output, torch.transpose(agg_Uemb, 1, 0)) # [batch, seq_len=199, user_size]
        mask = get_previous_user_mask(input, self.n_node) # [batch, seq_len, user_size]
        
        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()

    def model_prediction(self, input, input_time, related_cas, related_mask):

        '''structure embeddding'''
        agg_Uemb, _, _, s_Uemb = self.structure_embed()

        '''related embedding'''
        related_cas_emb = F.embedding(related_cas, s_Uemb)  # [batch, 4975, 64]
        related_output = related_cas_emb * related_mask.unsqueeze(2) 
        related_output = self.rfilter(related_output)
        related_output = rearrange(related_output, 'b (seq h) d -> b seq h d', seq=self.seq_length*self.topk, h=self.window_size)
        
        related_output = self.rpslstm(related_output) # [batch, 199*topk, 64]
        related_output = torch.reshape(related_output, (-1, self.seq_length, self.topk, self.emb_size)) 
        related_output = torch.sum(related_output, 2)   # [batch, 199, 64]
        
        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, agg_Uemb)  # [batch, seq_len=199, 64]
        
        '''GPT4Fusion'''
        combined_input = torch.cat([cas_seq_emb, related_output], dim=-1)
        
        # align gpt input dim 
        gpt_input_proj = self.gpt_proj_in(combined_input)  # [batch, 199, 768(gpt's dim)]
        gpt_output = self.lm(inputs_embeds=gpt_input_proj).last_hidden_state 
        output = self.gpt_proj_out(gpt_output) # [batch, 199, 64]
        
        pre_y = torch.matmul(output, torch.transpose(agg_Uemb, 1, 0)) # [batch, seq_len=199, user_size]
        mask = get_previous_user_mask(input, self.n_node) # [batch, seq_len, user_size]
        
        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()


