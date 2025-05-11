import torch
machine_device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_PATH = "/home/ntlpt19/personal_projects/data/test_data"
numclasses = 5
