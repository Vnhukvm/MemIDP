# MemIDP: Memory-based Information Diffusion Prediction with Dual Implicit Memory

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the paper: **Episodic Experience, Semantic Cognition: Towards Personalized Information Diffusion Prediction with Dual Implicit Memory**.

## Abstract

Information diffusion prediction **IDP** is a core task in viral marketing and personalized recommendation. Existing studies mainly focus on structural and temporal patterns, neglecting diffusion experience and user cognition, which limits personalized understanding of individual users that drive diverse propagation scenarios. To address these limitations, we propose **MemIDP**, a novel memory-augmented agent that equips large language model (LLM) with two complementary implicit memory modules: a trajectory-episodic memory for retaining historical diffusion experiences and a user-semantic memory for capturing user cognitive knowledge. **MemIDP** invokes the dual memory through a memory retrieval strategy and compresses the retrieved information into implicit memory tokens via latent memory converter. These implicit memories are seamlessly integrated during inference, enabling the LLM to preserve both experiential fidelity and semantic coherence in reasoning. Extensive experiments conducted on five real-world datasets demonstrate that **MemIDP** consistently outperforms all state-of-the-art baselines. 

## Project Structure

```
MemIDP/
├── run.py                 # Main training and evaluation script
├── Constants.py           # Special token constants (PAD, EOS)
├── dataLoader.py          # Data loading and preprocessing
├── Optim.py               # Scheduled optimizer with warm-up
├── models/
│   ├── model.py           # Main MemIDP model architecture
│   └── xlstm/             # xLSTM implementation from NXAI
├── utils/
│   ├── parsers.py         # Command-line argument parser
│   ├── Metrics.py         # Evaluation metrics (Hits@K, MAP@K)
│   ├── EarlyStopping.py   # Early stopping implementation
│   └── graphConstruct.py  # Graph and hypergraph construction
├── data/                  # Dataset directory (create this)
├── checkpoint/            # Model checkpoints (auto-created)
└── README.md              # This file
```

## Environment Configuration

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Vnhukvm/MemIDP.git
cd MemIDP
```

2. Create conda environment (recommended):
```bash
conda create -n memidp python=3.11
conda activate memidp
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained GPT-2 weights:
```bash
mkdir -p models/gpt2
# The code will automatically download from HuggingFace on first run
```

## Running the Code

### 1. Preprocessing (First run only)

Preprocess the dataset and split into train/validation/test:

```bash
python run.py -preprocess True -data_name android
```

This will create:
- `data/{dataset_name}/u2idx.pickle` - user to index mapping
- `data/{dataset_name}/idx2u.pickle` - index to user mapping
- `data/{dataset_name}/cascadetrain.txt` - training split
- `data/{dataset_name}/cascadevalid.txt` - validation split
- `data/{dataset_name}/cascadetest.txt` - testing split

### 2. Training

Train the model with default hyperparameters:

```bash
python run.py -data_name android
```

Train with custom hyperparameters:

```bash
python run.py \
  -data_name android \
  -epoch 50 \
  -batch_size 16 \
  -lr 2e-3 \
  --embSize 64 \
  -top_k 9 \
  --window_size 3 \
  --lm_name gpt2 \
  -patience 6
```

### 3. Hyperparameters

Key hyperparameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_name` | Dataset name | `android` |
| `epoch` | Maximum training epochs | `50` |
| `batch_size` | Training batch size | `16` |
| `embSize` | User embedding dimension | `64` |
| `lr` | Initial learning rate | `2e-3` |
| `layer` | Number of graph convolution layers | `1` |
| `dropout` | Dropout probability | `0.3` |
| `l2` | L2 regularization | `1e-5` |
| `top_k` | Number of related cascades to retrieve | `9` |
| `window_size` | Window size for patching | `3` |
| `patch_size` | Patch size for sLSTM | `6` |
| `num_blocks` | Number of xLSTM blocks | `1` |
| `num_heads` | Number of heads in sLSTM | `4` |
| `lm_name` | Pre-trained language model name | `gpt2` |
| `patience` | Early stopping patience | `6` |

### 4. Evaluation

The training script automatically evaluates on the test set after each epoch and reports the best results.

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the authors.
