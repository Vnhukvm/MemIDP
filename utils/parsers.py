import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-preprocess', type=bool, default=False, help="preprocess dataset")    #when you first run code, you should set it to true.

##### model parameters
parser.add_argument('-data_name', type=str, default='android', choices=['memes', 'twitter', 'douban', 'android', 'memetracker_RotDiff', 'christianity'], help="dataset")
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-max_lenth', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=16) 
parser.add_argument('--embSize', type=int, default=64, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used')
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-dropout', type=float, default=0.3)
parser.add_argument('-mini_seq_len', type=int, default=5)


#####data process
parser.add_argument('-train_rate', type=float, default=0.8)
parser.add_argument('-valid_rate', type=float, default=0.1)

###save model
parser.add_argument('-save_path', default= "./checkpoint/")
parser.add_argument('-result_path', default= "./results/")
parser.add_argument('-patience', type=int, default=6, help="control the step of early-stopping")


parser.add_argument('--channel', type=int, default=7, help='channel') 
parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension')
parser.add_argument('--num_heads', type=int, default=4, help='number of heads')
parser.add_argument('--conv1d_kernel_size', type=int, default=6, help='conv1d kernel size')
parser.add_argument('--group_norm_weight', type=bool, default=True, help='group normalization weight')
parser.add_argument('--num_blocks', type=int, default=1, help='number of blocks')
parser.add_argument('--patch_size', type=int, default=6, help='Size of the patches for the time series')
parser.add_argument('--stride', type=int, default=6, help='Stride for creating patches from the time series')

### retrieval
parser.add_argument("--window_size", type=int, default=3, help='window size')
parser.add_argument("--top_k", type=int, default=9, help='Top-K')

parser.add_argument("--filter_dropout_prob", type=float, default=0.5, help='Filter layer dropout probability')
parser.add_argument("--filter_seq", type=int, default=4975, help='Filter layer sequence length')
parser.add_argument("--filter_num", type=int, default=1, help='Filter layer number')


parser.add_argument("--lm_name", type=str, default='gpt2', choices=['gpt2', 'phi-1_5', 'Llama-3.2-1B', 'gemma-3-1b-pt', 'Qwen2.5-0.5B'], help='LM name')