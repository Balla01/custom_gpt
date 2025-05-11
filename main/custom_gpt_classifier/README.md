Here's a comprehensive README file based on your instructions for **"Custom GPT for Content Classification."** 

---

# Custom GPT for Content Classification

This project enables training and inference of a GPT-based model for classifying textual data into multiple classes.

---

## Installation

Ensure you have **Python 3.11** installed. Then, clone the repository and install the required packages:

```bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt
```

---

## Dataset Preparation

Your dataset should be structured as follows:

- The root folder should contain subfolders, each named after a class label (e.g., `AWB`, `CI`, etc.).
- Each subfolder should contain `.txt` files, with each file representing a sample text.

**Example structure:**

```
dataset/
├── AWB/
│   ├── sample1.txt
│   ├── sample2.txt
│   └── ...
├── CI/
│   ├── sample1.txt
│   ├── sample2.txt
│   └── ...
└── ...
```

In addition, there should be a `.txt` file in the root indicating the total number of samples for each class (as needed).

---

## Configuration

Update the `constants.py` file with your dataset path and parameters:

```python
DATASET_PATH = "/path/to/your/dataset"
numclasses = 5  # Set to total number of classes
```

---

## Training

Use `train_main.py` to train the model. The training configuration must be specified precisely in the `training_config_master` dictionary:

```python
training_config_master = {
    'out_dir': 'path/to/save/model',                  # Directory for saving outputs and checkpoints
    'eval_interval': 1000,                            # How often to evaluate during training (in steps)
    'log_interval': 100,                              # How often to log training info
    'eval_iters': 100,                                # Number of evaluation iterations
    'eval_only': False,                               # If True, only evaluate the model
    'always_save_checkpoint': True,                   # Save checkpoint after every epoch
    'init_from': 'scratch',                           # Initialization: 'scratch' or checkpoint path
    'wandb_log': False,                               # Whether to use Weights & Biases logging
    'wandb_project': 'your-wandb-project',           # WandB project name
    'wandb_run_name': 'run-name',                     # WandB run name
    'dataset': 'your-dataset-name',                   # Dataset identifier
    'gradient_accumulation_steps': 1,                  # Steps for gradient accumulation
    'batch_size': 64,                                 # Batch size per step
    'block_size': 1024,                               # Context length of sequences
    'n_layer': 8,                                     # Number of transformer layers
    'n_head': 8,                                      # Number of attention heads
    'n_embd': 512,                                    # Embedding dimension
    'dropout': 0.1,                                   # Dropout rate
    'bias': True,                                     # Use bias in layers
    'learning_rate': 3e-4,                            # Optimizer learning rate
    'max_iters': 50000,                               # Total training iterations
    'weight_decay': 0.01,                             # Weight decay for optimizer
    'beta1': 0.9,                                     # Adam optimizer beta1
    'beta2': 0.999,                                   # Adam optimizer beta2
    'grad_clip': 1.0,                                 # Gradient clipping threshold
    'decay_lr': True,                                 # Use learning rate decay
    'warmup_iters': 1000,                              # Number of warmup iterations
    'lr_decay_iters': 40000,                           # Iterations over which to decay LR
    'min_lr': 1e-5,                                   # Minimum learning rate after decay
    'backend': 'nccl',                                # Backend (e.g., 'nccl', 'gloo')
    'device': 'cuda',                                 # Device to run training ('cuda' or 'cpu')
    'dtype': 'float16',                               # Data type precision
    'compile': False                                  # Whether to compile model using PyTorch 2.0
}
```

---

## Explanation of Parameters

- **out_dir**: Path directory to save models and logs.
- **eval_interval**: Number of steps between evaluations.
- **log_interval**: Number of steps between logging training info.
- **eval_iters**: Number of inference steps during evaluation.
- **eval_only**: If `True`, runs only evaluation, no training.
- **always_save_checkpoint**: Save model after every epoch.
- **init_from**: Starting point; `'scratch'` or checkpoint path.
- **wandb_log**/**wandb_project**/**wandb_run_name**: For Weights & Biases integration.
- **dataset**: Name or identifier for your dataset.
- **gradient_accumulation_steps**: For larger effective batch sizes.
- **batch_size**: Samples per batch.
- **block_size**: Max sequence length.
- **n_layer, n_head, n_embd**: Model architecture parameters.
- **dropout**: Dropout rate during training.
- **bias**: Whether layers include bias.
- **learning_rate**: Initial optimizer learning rate.
- **max_iters**: Total training steps.
- **weight_decay**: Regularization parameter.
- **beta1, beta2**: Adam optimizer's moment estimates.
- **grad_clip**: Clipping gradients to stabilize training.
- **decay_lr**: Whether to decay learning rate.
- **warmup_iters**: Steps to warm up learning rate.
- **lr_decay_iters**: During how many iterations to decay lr.
- **min_lr**: Minimum learning rate.
- **backend**: Distributed training backend.
- **device**: Hardware device.
- **dtype**: Floating Point precision.
- **compile**: Whether to compile the model (PyTorch 2.0 feature).

---

## Inference

Run inference with `inference.py`. Make sure to update the paths and class mappings as shown:

```python
import os

# Path to the checkpoint
out_dir = '/home/ntlpt19/personal_projects/model_architecture/src/main/nanoGPT-master/out'
ckpt_path = os.path.join(out_dir, 'ckpt_epoch_1.pt')

# Class names
class_names = ["AWB", "CI", "COO", "IC", "PI"]
idx_to_label = {v: k for v, k in enumerate(class_names)}
```

You can pass these configurations to inference script as needed for your setup.

---

**Feel free to customize the paths and parameters according to your environment.**  
Let me know if you'd like me to add anything else!
