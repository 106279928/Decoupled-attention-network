import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime

from utils import *
import cfgs_ic as cfgs
from main import *
from dataset_own import OwnDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ic_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train']) 
    return train_loader, train_loader

if __name__ == "__main__":
    model = load_network()
    display_cfgs(model)
    optimizers, optimizer_schedulers = generate_optimizer(model)
    criterion_CE = nn.CrossEntropyLoss().to(device)
    train_loader, test_loader = load_ic_dataset()

    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    loss_counter = Loss_counter(cfgs.global_cfgs['show_interval'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])

    test((test_loader), 
        model, 
        [encdec,
        flatten_label,
        test_acc_counter]
    )