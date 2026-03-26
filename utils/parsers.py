"""
Command-line argument parser for MemIDP.
Defines all hyperparameters and configuration options for the model.
"""
import argparse

parser = argparse.ArgumentParser(description='MemIDP: Memory-based Information Diffusion Prediction')

parser.add_argument('-preprocess', type=bool, default=False, help="Whether to preprocess the dataset. Set to True on first run.")

# Model architecture parameters
parser.add_argument('-data_name', type=str, default='android', choices=['memes', 'twitter', 'douban', 'android', 'memetracker_RotDiff', 'christianity'], help="Dataset name")
parser.add_argument('-epoch', type=int, default=50, help="Maximum number of training epochs")
parser.add_argument('-max_lenth', type=int, default=200, help="Maximum cascade sequence length after padding")
parser.add_argument('-batch_size', type=int, default=16, help="Training batch size")
parser.add_argument('--embSize', type=int, default=64, help='User embedding dimension')
parser.add_argument('--l2', type=float, default=1e-5, help='L2 regularization weight decay')
parser.add_argument('--lr', type=float, default=2e-3, help='Initial learning rate')
parser.add_argument('--layer', type=int, default=1, help='Number of graph convolution layers')
parser.add_argument('-n_warmup_steps', type=int, default=1000, help='Number of warmup steps for learning rate scheduler')
parser.add_argument('-dropout', type=float, default=0.3, help='Dropout probability')
parser.add_argument('-mini_seq_len', type=int, default=5, help='Minimum sequence length for retrieval')

# Data processing
parser.add_argument('-train_rate', type=float, default=0.8, help='Fraction of data for training')
parser.add_argument('-valid_rate', type=float, default=0.1, help='Fraction of data for validation')

# Checkpoint and results
parser.add_argument('-save_path', default="./checkpoint/", help='Directory to save model checkpoints')
parser.add_argument('-result_path', default="./results/", help='Directory to save result files')
parser.add_argument('-patience', type=int, default=6, help='Patience for early stopping (number of epochs to wait before stopping)')

# xLSTM parameters for sLSTM
parser.add_argument('--channel', type=int, default=7, help='channel')
parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension inside xLSTM')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in sLSTM')
parser.add_argument('--conv1d_kernel_size', type=int, default=6, help='Kernel size for 1D convolution in sLSTM')
parser.add_argument('--group_norm_weight', type=bool, default=True, help='Whether to use group normalization on weights')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of xLSTM blocks in the stack')
parser.add_argument('--patch_size', type=int, default=6, help='Patch size for patching retrieved sequences')
parser.add_argument('--stride', type=int, default=6, help='Stride for creating patches')

# Memory retrieval parameters
parser.add_argument("--window_size", type=int, default=3, help='Window size for slicing retrieved cascades')
parser.add_argument("--top_k", type=int, default=9, help='Number of top related cascades to retrieve')

# FilterMLP parameters
parser.add_argument("--filter_dropout_prob", type=float, default=0.5, help='Dropout probability in filter layer')
parser.add_argument("--filter_seq", type=int, default=4975, help='Maximum sequence length for filtering')
parser.add_argument("--filter_num", type=int, default=1, help='Number of filtering layers')

# Pre-trained language model for fusion
parser.add_argument("--lm_name", type=str, default='gpt2', choices=['gpt2', 'phi-1_5', 'Llama-3.2-1B', 'gemma-3-1b-pt', 'Qwen2.5-0.5B'], help='Name of pre-trained language model')
