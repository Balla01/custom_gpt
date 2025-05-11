import os
import json
import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value
from tqdm import tqdm
from transformers import BertTokenizer
import logging
import ast
import torch
from constants import machine_device as device
from constants import DATASET_PATH
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_SEQ_LENGTH = 512
VOCA = "bert-base-uncased"

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

def tokenize_function(examples, label2idx):
    """
    Tokenize the context and map class names to label indices.
    
    Args:
        examples (dict): Dataset examples with 'context' and 'class_name'.
        label2idx (dict): Mapping of class names to indices.
    
    Returns:
        dict: Tokenized inputs with 'input_ids', 'attention_mask', and 'labels'.
    """

    tokenized = tokenizer(
        examples['context'],
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors='pt'
    )
    
    # Map class names to labels with error handling
    labels = []
    # print('????????')
    # print(label2idx)
    # for label in examples['class_name']:
    #     if label not in label2idx:
    #         logger.error(f"Label '{label}' not found in label2idx: {label2idx}")
    #         raise ValueError(f"Label '{label}' not found in label2idx")
    #     labels.append(label2idx[label])
    
    tokenized['labels'] = examples['class_name']
    tokenized['file_path'] = examples['file_path']
    return tokenized

def extract_text_from_word_coords(word_coordinates: list):
    all_text = ' '.join([item['word'] for item in word_coordinates])
    return all_text

def get_dataset():
    # Prepare data
    logger.info("Starting data preparation")

    # Get class labels from folder names
    labels = sorted(list(os.listdir(DATASET_PATH)))  # Sort for consistency
    idx2label = dict(enumerate(labels))
    label2idx = {k: v for v, k in enumerate(labels)}

    # Save label mapping for reference
    with open('label.txt', 'w') as label_file:
        label_file.write(json.dumps(label2idx))

    # Collect data
    file_paths = []
    contexts = []
    class_names = []

    for label in tqdm(os.listdir(DATASET_PATH), desc="Processing classes"):
        class_path = os.path.join(DATASET_PATH, label)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_name.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            wc = f.read()
                        f.close()
                        wc = ast.literal_eval(wc)
                        context = extract_text_from_word_coords(wc['word_coordinates'])
                        file_paths.append(file_path)
                        contexts.append(context)
                        class_names.append(label)
                    except Exception as e:
                        logger.warning(f"Failed to process file {file_path}: {e}")
                        continue

    # Create DataFrame
    data = pd.DataFrame({
        'file_path': file_paths,
        'context': contexts,
        'class_name': class_names
    })

    # Save for debugging
    data.to_csv("context_data_and_labels.csv", index=False)
    logger.info(f"DataFrame columns: {data.columns.tolist()}")

    # Log unique class names for debugging
    unique_classes = sorted(data['class_name'].unique())
    logger.info(f"Unique class names in dataset: {unique_classes}")
    logger.info(f"Expected labels from label2idx: {sorted(label2idx.keys())}")

    # Define dataset features with ClassLabel for class_name
    features = Features({
        'file_path': Value('string'),
        'context': Value('string'),
        'class_name': ClassLabel(names=labels)
    })

    # Convert to Dataset with specified features
    dataset = Dataset.from_pandas(data, features=features)

    # Tokenize data, passing label2idx
    dataset = dataset.map(lambda examples: tokenize_function(examples, label2idx), batched=False)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'file_path'])

    # Split dataset
    train_test = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='class_name')
    dataset = {
        'train': train_test['train'],
        'val': train_test['test']
    }

    # Save splits for debugging
    pd.DataFrame(dataset['train']).to_csv('training_set.csv', index=False)
    pd.DataFrame(dataset['val']).to_csv('testing_set.csv', index=False)

    logger.info(f"Dataset columns: {dataset['train'].column_names}")
    logger.info(f"Train size: {len(dataset['train'])}, Val size: {len(dataset['val'])}")
    return dataset


def get_costom_batch(split, dataset, batch_size):
    """
    Get a batch of data from the specified split.

    Args:
        split (str): Either 'train' or 'val'.
        dataset (dict): Dictionary containing 'train' and 'val' datasets (Hugging Face Dataset objects).
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (idx, targets) where idx is input_ids tensor [batch_size, 512] and targets is labels tensor [batch_size].
    """
    if split not in ['train', 'val']:
        raise ValueError("Split must be 'train' or 'val'")

    # Select the appropriate dataset split
    data = dataset[split]
    
    # Ensure batch_size is an integer and doesn't exceed dataset size
    batch_size = int(batch_size)
    data_size = len(data)
    if batch_size > data_size:
        batch_size = data_size
        print(f"Warning: batch_size ({batch_size}) exceeds {split} dataset size ({data_size}). Using full dataset.")

    # Generate random indices for batch
    ix = torch.randint(0, data_size, (batch_size,), dtype=torch.long)
    
    # Extract batch data
    idx = torch.stack([data[int(i)]['input_ids'].squeeze() for i in ix])
    targets = torch.tensor([data[int(i)]['labels'] for i in ix], dtype=torch.long)
    
    # Move to device
    if device_type == 'cuda':
        idx = idx.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        idx = idx.to(device)
        targets = targets.to(device)
    
    # Debugging shapes
    print(f"idx shape: {idx.shape}")
    print(f"targets shape: {targets.shape}")
    print('Loaded batch data')
    
    return idx, targets
