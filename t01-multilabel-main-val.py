#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
from torch import sigmoid
from torch.nn import Linear
import torch.nn.init as nn_init
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib as matplt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
from os import path
from sklearn.metrics import roc_auc_score
import datetime
import time

# for importing external python files
import importlib

#delay_hrs = 3
#print('Delaying start for {0} hrs'.format(delay_hrs))
#time.sleep(delay_hrs*3600)
#print('Continue with the rest of the script ...')

path_join = path.join

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[12]:


# !!! REPLACE THIS VARIABLE IF NEEDS !!!
#cur_dir = '/Users/marcustan/Library/Mobile Documents/com~apple~CloudDocs/0_MCS/CS598 Deep Learning for Healthcare/CS598_Project/CS598LHO-Team1521-Project/'
#os.chdir(cur_dir)
ALL_IMAGE_DIR = '/Users/marcustan/Data/ChestXray14/images'; # directory containing all images
BASE_PATH_LABELS = './labels'; # training, validation and test lists

#print(os.getcwd())
#ALL_IMAGE_DIR = '/home/ubuntu/data/cs598-dlh/images'; # directory containing all images
#BASE_PATH_LABELS = '/home/ubuntu/data/cs598-dlh/labels'; # training, validation and test lists

TRAIN_LISTS = ['train_A_0.csv', 'train_A_1.csv', 'train_A_2.csv']
VAL_LISTS = ['val_A_0.csv', 'val_A_1.csv', 'val_A_2.csv']
TEST_LIST = ['test_A.csv']

MODEL_BASE_PATH = '../models'
MODEL_NAME = 'densenet121'
#MODEL_NAME = 'resnet50'
#MODEL_NAME = 'densenet121attA'
#MODEL_NAME = 'densenet121attB'

# Labels excluding the no-finding label
CLASSES = ['No Finding', 'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
          'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

# !!! HYPERPARAMETERS !!!
BATCH_SIZE = 8
#BATCH_SIZE = 16
#BATCH_SIZE = 32
TRAIN_EPOCH = 8
#IMAGE_RESIZE = 1024

LEARNING_RATE = 0.001
MOMENTUM = 0.9
ADAM_BETAS = (0.9, 0.999)
SEED = 0

REINITIALIZE_METHOD = None
#REINITIALIZE_METHOD = 'xavierNormal'

# <hr>
# 
# ## Preparation
# 
# ### Load and transform images

# ##  Create custom dataset and load training and validation data

# In[13]:

# Set seed
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSES = np.array(CLASSES)
# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, im_dir, im_names, im_labels, im_transforms=None):
        self.im_dir = im_dir
        self.im_labels = im_labels
        self.im_names = im_names
        if im_transforms:
            self.im_transforms = im_transforms
        else:
            self.im_transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        
        return len(self.im_labels)

    def __getitem__(self, idx):
        im_file = os.path.join(self.im_dir,
                               self.im_names[idx])
        #print(im_file)
        im = Image.open(im_file).convert('RGB')

        input_tensor = self.im_transforms(im)

        return input_tensor, self.im_labels[idx]

def load_data(all_image_dir, train_lists, base_path_labels, classes, batchsize, im_transforms):
    train_sets = []
    train_loaders = []
    for train_list in train_lists:
        full_path_list = path_join(base_path_labels, train_list)
        df = pd.read_csv(full_path_list)
        im_names = df['Image'].to_numpy()
        im_labels = torch.tensor(df[classes].to_numpy(), dtype=torch.float)
        #assert im_labels.shape[1] == n_classes, 'Number of classes from train list not consistent with provided N_CLASSES'
        train_sets.append(CustomDataset(all_image_dir, im_names, im_labels, im_transforms))
        train_loaders.append(DataLoader(train_sets[-1], batch_size=batchsize, shuffle=True))
        print('Size for {0} = {1}'.format(train_list, len(im_names)))
    
    return train_loaders
    
# Load data
im_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create training loaders
train_loaders = load_data(ALL_IMAGE_DIR, TRAIN_LISTS, BASE_PATH_LABELS, CLASSES, BATCH_SIZE, im_transforms)
val_loaders = load_data(ALL_IMAGE_DIR, VAL_LISTS, BASE_PATH_LABELS, CLASSES, BATCH_SIZE, im_transforms)


# In[14]:
# Visualize some of the images and show corresponding labels
# This part can be found in the ipynb version

## Models
def initialize_model(model_name, reinitialize_method):
    if 'densenet121att' in model_name:
        model_module = importlib.import_module(model_name)
        # print(model_module)
        model = getattr(model_module, model_name)(pretrained=True,
                                                  num_classes=len(CLASSES))
        # model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, len(CLASSES)),
        #                                 nn.Softmax(dim=1))
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, len(CLASSES)),
                                         nn.Sigmoid())
        # Initialize the weights of the classifier layer
        print('Initialize weights of classifier layer')
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn_init.xavier_normal_(m.weight, gain=1)

    elif 'densenet' in model_name:
        import densenet_models
        importlib.reload(densenet_models)
        # model = densenet121(pretrained=True)
        model = getattr(densenet_models, model_name)(pretrained=True)
        # model.classifier = nn.Sequential(nn.Linear(1024, len(CLASSES)), nn.Softmax(dim=1))
        model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, len(CLASSES)),
                                         nn.Sigmoid())
        # Initialize the weights of the classifier layer
        print('Initialize weights of classifier layer')
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn_init.xavier_normal_(m.weight, gain=1)

    elif 'resnet' in model_name:
        import resnet_models
        importlib.reload(resnet_models)
        # model = resnet50(pretrained=True)
        model = getattr(resnet_models, model_name)(pretrained=True)
        # model.fc = nn.Sequential(nn.Linear(model.fc.in_features, len(CLASSES)),
        #                         nn.Softmax(dim=1))
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, len(CLASSES)),
                                 nn.Sigmoid())
        # Initialize the weights of the classifier layer
        print('Initialize weights of classifier layer')
        model_fc = model.fc
        for m in model_fc.modules():
            if isinstance(m, nn.Linear):
                nn_init.xavier_normal_(m.weight, gain=1)
    else:
        raise NotImplementedError('Unknown model name')

    if reinitialize_method == None:
        pass
    elif reinitialize_method == 'xavierNormal':
        print('!!!!!! Reinitialize weights with xavier normal !!!!!!')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn_init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn_init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn_init.normal_(m.weight, 0, 0.01)
                nn_init.constant_(m.bias, val=0.)
            elif isinstance(m, nn.Linear):
                nn_init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn_init.constant_(m.bias, val=0.)
    else:
        raise NotImplementedError('Unknown initialization method')

    return model


model = initialize_model(MODEL_NAME, REINITIALIZE_METHOD)

loader_iter = iter(train_loaders[0])
images, labels = next(loader_iter)

print('{0} model output test'.format(MODEL_NAME))
# print(images[0, 0, 0, :5])
y_hat = model(images)
n_params = sum(p.numel() for p in model.parameters())
print('number of parameters = ', n_params)
print(y_hat)
print('y_hat.shape = ', y_hat.shape)
# print(model.classifier.in_features)
print(model)


## Set and train model

# In[31]:
def make_dir_if_not_exist(folder):
    try:
        os.makedirs(folder)
    except:
        pass


def train_model(model, train_loader, device, n_epoch, optimizer, criterion, model_base_path, split=0):
    start_time = time.time()

    model.train()  # prep model for training

    n_batches = len(train_loader)
    Ypred = []
    Ytruth = []
    stats = {'all_loss': [], 'epoch_end_ind': [], 'epoch_loss': [],
             'epoch_auroc_ave': [], 'epoch_auroc_classes': []}

    #max_count = 2; # for debugging

    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for count, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)
            # data = data.to(device)

            # Step 1. clear gradients
            optimizer.zero_grad()
            # Step 2. perform forward pass using `model`, save the output to y_hat;
            # Convert output to probability vector using softmax function
            ypred = model(data)
            # ypred = F.softmax(ypred, dim=1)
            Ypred.append(ypred.detach().cpu().numpy())
            Ytruth.append(target.detach().cpu().numpy())
            # Step 3. calculate the loss using `criterion`, save the output to loss.
            loss = criterion(ypred, target)
            # Step 4. backward pass
            loss.backward()
            # Step 5. optimization
            optimizer.step()
            # Step 6. record loss
            loss_scalar = float(loss.detach().cpu().numpy())
            curr_epoch_loss.append(loss_scalar)
            stats['all_loss'].append(loss_scalar)

            if count % 10 == 0:
                print('Split {0}, Epoch {1}, Batch {2}/{3}: elapsed {4:.2f}s,'
                      'batch loss {5:g}, curr mean epoch loss {6:g}'
                      ''.format(split, epoch, count + 1, n_batches, time.time() - start_time,
                                loss_scalar, np.mean(curr_epoch_loss)))

            # for debugging
            #if count >= max_count:
            #    break

        Ypred1 = np.concatenate(Ypred, axis=0)
        Ytruth1 = np.concatenate(Ytruth, axis=0)
        try:
            auroc_ave = roc_auc_score(Ytruth1, Ypred1, average='weighted')
            auroc_classes = roc_auc_score(Ytruth1, Ypred1, average=None)
        except ValueError:
            print('WARNING: AUC undefined as only one sample is available for 1 or more of the classes')
            auroc_ave = 0.0
            auroc_classes = np.zeros(target.shape[1])

        stats['epoch_auroc_ave'].append(auroc_ave)
        stats['epoch_auroc_classes'].append(auroc_classes)
        stats['epoch_end_ind'].append(len(stats['all_loss']) - 1)
        mean_epoch_loss = np.mean(curr_epoch_loss)
        stats['epoch_loss'].append(mean_epoch_loss)
        print('Epoch {0}: elapsed {1:.2f}s, curr mean epoch loss={2:g},'
              'curr average auroc={3:g}'.format(epoch, time.time() - start_time,
                                                mean_epoch_loss, auroc_ave))
        auroc_classes_dict = dict(zip(CLASSES, np.round(auroc_classes, 3)))
        print('Class auroc = ', auroc_classes_dict)
        print('Saving model ...')

        model_file = path_join(model_base_path, 'model_split' + str(split) + '_epoch' + str(epoch))
        torch.save(model.state_dict(), model_file)
        # for debugging
        # if count >= 3:
        #    break

    return model, stats


def eval_model(model, dataloader, device):
    model.eval()
    Ypred = []
    Ytruth = []
    n_batches = len(dataloader)
    # max_count = 3; # for debugging
    for count, (data, truth) in enumerate(dataloader):
        data = data.to(device)
        truth = truth.to(device)
        if count % 10 == 0:
            print('Batch {0}/{1}'.format(count + 1, n_batches))
        ypred = model(data)
        # ypred = ypred.softmax(dim=1)
        Ypred.append(ypred.detach().cpu().numpy())
        Ytruth.append(truth.detach().cpu().numpy())
        # if count >= max_count:
        #    break

    Ypred = np.concatenate(Ypred, axis=0)

    Ytruth = np.concatenate(Ytruth, axis=0)

    try:
        auroc_ave = roc_auc_score(Ytruth, Ypred, average='weighted')
        auroc_classes = roc_auc_score(Ytruth, Ypred, average=None)
    except ValueError:
        print('WARNING: AUC undefined as only one sample is available for 1 or more of the classes')
        auroc_ave = 0.0
        auroc_classes = np.zeros(truth.shape[1])

    return auroc_ave, auroc_classes


criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

n_epochs = TRAIN_EPOCH

datetime1 = datetime.datetime

np.random.seed(SEED)
torch.manual_seed(SEED)

# Directory for saving the models and output stats
time_stamp = datetime1.now().strftime("%Y-%m-%d-%H-%M-%S")
model_dir = MODEL_BASE_PATH + '-' + MODEL_NAME + '-' + time_stamp
make_dir_if_not_exist(model_dir)

# Copy this Python script to the output directory
this_file = path.basename(__file__)
from shutil import copyfile
copyfile(__file__, path_join(model_dir, this_file))

val_stats = pd.DataFrame(columns=['Split', 'Ave AUROC', *CLASSES])
train_stats = pd.DataFrame(columns=['Split', 'Epoch', 'Epoch_loss', 'Epoch_AUROC', *CLASSES])
train_loss = pd.DataFrame(columns=['Split', 'Batch_loss'])

fig1, ax1 = plt.subplots(figsize=(15, 7))
matplt.rcParams.update({'font.size': 16})
lnstyles = ['-o', '-^', '-v', '-<', '->', '-h',
            '--o', '--^', '--v', '--<', '-->', '--h',
            '-.o', '-.^', '-.v', '-.<', '-.>', '-.h']

for split, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
    print('\n********Split = ', split, ', training model ...')
    # Train model
    start_time_tot = time.time()

    # Re-initialize model for every split
    model = initialize_model(MODEL_NAME, REINITIALIZE_METHOD)

    # Must reset optimizer whenever model is re-initialized
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()

    model, stats = train_model(model, train_loader, DEVICE, n_epochs,
                               optimizer, criterion, model_dir, split)
    print('Total training time {0:.2f}s'.format(time.time() - start_time_tot))

    # Save training stats in data frame
    row_start = split * n_epochs
    row_end = (split + 1) * n_epochs
    for count, row in enumerate(range(row_start, row_end)):
        train_stats.loc[row] = [split, count, stats['epoch_loss'][count], stats['epoch_auroc_ave'][count],
                                *list(stats['epoch_auroc_classes'][count])]
    out_file = path_join(model_dir, 'train_stats.csv')

    train_stats.to_csv(out_file, index=False)
    print(train_stats)
    train_loss = train_loss.append(pd.DataFrame({'Split': split, 'Batch_loss': stats['all_loss']}))
    out_file = path_join(model_dir, 'train_loss.csv')
    train_loss.to_csv(out_file)

    # Plot training statistics
    str_prefix = 'Split_' + str(split)
    iterx = np.arange(len(stats['all_loss']))
    epoch_end_ind = np.array(stats['epoch_end_ind'])
    ax1.plot(iterx, stats['all_loss'], '--', label=str_prefix)
    ax1.plot(iterx[epoch_end_ind], stats['epoch_loss'], 'o-', label=str_prefix)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend(prop={'size': 16})
    out_file = path_join(model_dir, 'train_loss.png')
    fig1.savefig(out_file)

    fig2, ax2 = plt.subplots(figsize=(15, 7))
    epochs = np.arange(n_epochs)
    ax2.plot(epochs, stats['epoch_auroc_ave'], 's-', label='Ave')
    for jj, name in enumerate(CLASSES):
        ax2.plot(epochs, train_stats.loc[train_stats['Split'] == split, name],
                 lnstyles[jj], label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_xlim(-0.1, n_epochs)
    ax2.set_ylabel('AUROC')
    ax2.set_title(str_prefix)
    ax2.legend(prop={'size': 16})
    out_file = path_join(model_dir, 'split' + str(split) + '_train_auroc.png')
    fig2.savefig(out_file)
    # Validate model
    start_time_tot = time.time()
    print('\n********Validating model ...')
    auroc_ave, auroc_classes = eval_model(model, val_loader, DEVICE)
    print('Total validation time {0:.2f}s'.format(time.time() - start_time_tot))
    # print('Val average auroc={0:g}'.format(auroc_ave))
    # auroc_classes_dict = dict(zip(CLASSES, np.round(auroc_classes,3)))
    # print('Val class auroc=', auroc_classes_dict)

    # Save validation stats in a data frame
    val_stats.loc[split] = [split, auroc_ave, *auroc_classes]
    out_file = path_join(model_dir, 'val_stats.csv')
    # Stack the class AUROC
    val_stats_stacked = pd.melt(val_stats, id_vars=['Split'], var_name='Class', value_name='AUROC')
    val_stats_stacked.to_csv(out_file, index=False)
    # print(val_stats_stacked.loc[val_stats_stacked['Split']==split])
    print(val_stats_stacked)

# In[30]:

## Test model
#model_filename = path_join(model_dir, 'model_split0_epoch1')
#model.load_state_dict(torch.load(model_filename, map_location='cpu'))
#model.to(DEVICE)
test_loader = load_data(ALL_IMAGE_DIR, TEST_LIST, BASE_PATH_LABELS, CLASSES, BATCH_SIZE, im_transforms)[0]
start_time_tot = time.time()
auroc_ave, auroc_classes = eval_model(model, test_loader, DEVICE)
print('Total test time {0:.2f}s'.format(time.time() - start_time_tot))
print('Test average auroc={0:g}'.format(auroc_ave))
auroc_classes_dict = dict(zip(CLASSES, np.round(auroc_classes,3)))
auroc_classes_dict['Ave'] = auroc_ave
print('Test class auroc=', auroc_classes_dict)
out_file = path_join(model_dir, 'test_stats.csv')
with open(out_file, 'w') as file:
    for key, value in auroc_classes_dict.items():
        file.write('%s, %g\n'%(key, value))



