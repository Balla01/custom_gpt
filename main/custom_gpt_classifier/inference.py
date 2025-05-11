import os
import torch
import numpy as np

# Assuming GPT and GPTConfig are defined in your codebase
from model import GPTConfig, GPT

# -----
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory where the checkpoint is saved
out_dir = '/home/ntlpt19/personal_projects/model_architecture/src/main/nanoGPT-master/out'
ckpt_path = os.path.join(out_dir, 'ckpt_epoch_1.pt')
class_names = ["AWB","CI", "COO", "IC", "PI"]
idx_to_label = {"AWB": 0, "CI": 1, "COO": 2, "IC": 3, "PI": 4}


# Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location=device)

# Initialize model configuration
gptconf = GPTConfig(**checkpoint['model_args'])
# Initialize model
model = GPT(gptconf).to(device)
print(device)

# Load model weights
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()
from train_utility import get_dataset, get_costom_batch
main_data_set = get_dataset()
print(main_data_set)
X, Y = get_costom_batch('train', main_data_set, 12) # fetch the very first batch
logits, loss = model(X, Y)
print("Logits shape:", logits.shape)
print("Loss shape:", loss.shape)
print("Logits:", logits)
print("Loss:", loss)
##################################################################################
# Step 1: Apply softmax to get probabilities (optional)
probabilities = torch.softmax(logits, dim=-1)
print("Probabilities:\n", probabilities)
# Step 2: Get predicted class labels (index of max logit)
predicted_labels = torch.argmax(logits, dim=-1)
print("Predicted Labels:\n", predicted_labels)
  # Replace with your classes
predicted_class_names = [class_names[idx] for idx in predicted_labels]
print("Predicted Class Names:", predicted_class_names)
##################################################################################