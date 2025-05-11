"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
from train_utility import get_dataset, get_costom_batch



# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_costom_batch(split, main_data_set, batch_size)
            # X, Y = get_batch(split)#, main_data_set, batch_size)#get_batch
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from uuid import uuid4


def validate_model(model, val_loader, device, master_process):
    """Perform validation and compute loss, confusion matrix, and F1 score."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for bh_idx, batch_data in enumerate(val_loader):
            fil_pth, X, atten_msk, Y = batch_data['file_path'], batch_data['input_ids'], batch_data['attention_mask'], batch_data['labels']
            X = X.squeeze(1)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if master_process:
        print(f"Validation Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
    
    model.train()
    return avg_loss, cm, f1

def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    device='cuda',
    num_epochs=10,
    batch_size=64,
    eval_interval=1,
    log_interval=100,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    grad_clip=1.0,
    decay_lr=True,
    always_save_checkpoint=False,
    out_dir='checkpoints',
    master_process=True,
    training_config_master={},
):
    """
    Main training loop with epoch-based iteration and validation metrics.

    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run training on
        num_epochs: Number of epochs to train
        batch_size: Number of samples per batch
        eval_interval: How often (in epochs) to evaluate
        log_interval: How often (in iterations) to log training progress
        gradient_accumulation_steps: Number of steps for gradient accumulation
        learning_rate: Initial learning rate
        grad_clip: Gradient clipping threshold
        decay_lr: Whether to decay learning rate
        always_save_checkpoint: If True, save checkpoint regardless of validation loss
        out_dir: Directory to save checkpoints
        ddp_rank: Rank of the current process in DDP
        ddp_world_size: Total number of processes in DDP
        master_process: Whether this is the main process
    """
    # Initialize training parameters
    iter_num = 0
    best_val_loss = float('inf')
    running_mfu = -1.0
    max_iters = num_epochs * len(train_loader)

    # Setup DDP if enabled
    # ddp = ddp_world_size > 1
    # if ddp:
    #     setup_ddp(ddp_rank, ddp_world_size)
    #     model = DDP(model, device_ids=[ddp_rank])
    raw_model = model.module if ddp else model
    model.to(device)

    # Initialize gradient scaler for mixed precision
    # scaler = torch.cuda.amp.GradScaler(enabled=(model.dtype == torch.float16))
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # ctx = torch.cuda.amp.autocast(enabled=(model.dtype == torch.float16))
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        # Iterate over all batches in the training DataLoader
        for batch_idx, batch_data in enumerate(train_loader):
            fil_pth, X, atten_msk, Y = batch_data['file_path'], batch_data['input_ids'], batch_data['attention_mask'], batch_data['labels']
            # X = torch.stack(X)
            X = X.squeeze(1)
            X, Y = X.to(device), Y.to(device)

            # Set learning rate
            # lr = get_lr(iter_num, max_iters, learning_rate) if decay_lr else learning_rate
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Training step with gradient accumulation
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

            # Gradient clipping and optimization step
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * gradient_accumulation_steps

            # Logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                lossf = loss.item() * gradient_accumulation_steps
                if iter_num >= 5:
                    mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"Epoch {epoch+1}/{num_epochs}, Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

            iter_num += 1

        # Log epoch average loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        if master_process:
            print(f"Epoch {epoch+1}/{num_epochs} completed. Average Train Loss: {avg_epoch_loss:.4f}")

        # Validation after each epoch
        if (epoch + 1) % eval_interval == 0 and master_process:
            val_loss, cm, f1 = validate_model(model, val_loader, device, master_process)
            
            # Save checkpoint if validation loss improves or always_save_checkpoint is True
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'config': training_config_master,
                }
                print(f"saving checkpoint to {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                checkpoint_path = os.path.join(out_dir, f'ckpt_epoch_{epoch+1}.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)

    # Cleanup
    if ddp:
        dist.destroy_process_group()

    return best_val_loss

if __name__ == "__main__":
        
    main_data_set = get_dataset()

    out_dir = 'out'
    eval_iters = 20
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' #'gpt2'#'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'gpt2' # 'run' + str(time.time())
    # data
    dataset = 'custom_data'#'shakespeare_char' # 'openwebtext' or 'shakespeare_char' or 'sherlock'
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 512
    # model
    n_layer = 4
    n_head = 4
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 1e-3 # max learning rate
    max_iters = 2000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 100 # how many steps to warm up for
    lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
    min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    print(device_type)
    # poor man's data loader
    dataset = 'custom_data' 
    data_dir = os.path.join('data', dataset)

    
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    vocab_size_ = 30522

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    os.makedirs(out_dir, exist_ok=True)

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = vocab_size_
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery

    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)
    


    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    num_epochs=1
    log_interval=1 #???????
    eval_interval=1 #250
    gradient_accumulation_steps=1

    train_dataset = main_data_set['train']#YourDataset(split='train')
    val_dataset = main_data_set['val']#YourDataset(split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print('READY TO TRAIN')
    training_config_master = {'out_dir': out_dir, 'eval_interval': eval_interval, 
                              'log_interval': log_interval, 'eval_iters': eval_iters, 
                              'eval_only': eval_only, 'always_save_checkpoint': always_save_checkpoint,
                              'init_from': init_from, 'wandb_log': wandb_log, 'wandb_project': wandb_project, 
                              'wandb_run_name': wandb_run_name, 'dataset': dataset, 
                              'gradient_accumulation_steps': gradient_accumulation_steps,
                              'batch_size': batch_size, 'block_size': block_size, 'n_layer': n_layer, 
                              'n_head': n_head, 'n_embd': n_embd, 'dropout': dropout, 'bias': bias, 
                              'learning_rate': learning_rate, 'max_iters': max_iters, 'weight_decay': weight_decay,
                              'beta1': beta1, 'beta2': beta2, 'grad_clip': grad_clip, 
                              'decay_lr': decay_lr, 'warmup_iters': warmup_iters, 
                              'lr_decay_iters': lr_decay_iters, 'min_lr': min_lr, 
                              'backend': backend, 'device': device, 'dtype': dtype, 'compile': compile}
    
    best_val_loss = train_model(model, optimizer, train_loader, val_loader, 
                                device=device, num_epochs=num_epochs, 
                                batch_size=batch_size, eval_interval = eval_interval, 
                                learning_rate = learning_rate, grad_clip=grad_clip,
                                decay_lr=decay_lr, always_save_checkpoint=always_save_checkpoint, 
                                out_dir=out_dir, master_process=master_process, training_config_master= training_config_master)
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print('Successfully trained the model!')