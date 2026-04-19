import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import time
import os
import copy
import argparse
import multiprocessing
from torchsummary import summary
from matplotlib import pyplot as plt
from torchsampler import ImbalancedDatasetSampler
# import albumentations as A
from tqdm import tqdm

from PIL import Image, ImageOps
import random
import cv2


from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from grain_im_utils import sharpen_img

# from img_to_sketch import img_to_sketch


from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
import napari
from lion_pytorch import Lion

from image_augmentation import UnNormalize, CannyDetection, \
HistogramEqualize, ImageToSketch, ImageToSketch1, ImgCustomRotate, \
ImgInvert, ImgSharpen, ImgShift,SharpRegionDetector, SobelDetection, \
Tenengrad_filter, UnNormalize 
    

# ****************************************************************************************

def test_on_field_data(model, field_dataset, field_data_loader, writer, epoch):
    model.eval()   # Set model to evaluate mode

    # Number of classes and dataset-size
    # num_classes=len(field_dataset.classes)
    dsize = len(field_dataset)

    # Initialize the prediction and label lists
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    pred_conf = torch.zeros(0, dtype=torch.long, device='cpu')

    # Evaluate the model accuracy on the dataset
    correct = 0
    total = 0
    with torch.no_grad():
        start = time.time()
        for images, labels in tqdm(field_data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            conf, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predlist = torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
            pred_conf = torch.cat([pred_conf, conf.view(-1).cpu()])

        stop = time.time() - start
        print('Average inference time: ', stop / dsize)

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())

    # Per-class accuracy
    class_accuracy = 100*conf_mat.diagonal()/conf_mat.sum(1)
    print('Per class accuracy')
    print('-'*18)
    acc_dict = {}

    field_data_class_name = ['Alnus', 'Betula',
                             'Carpinus', 'Corylus', 'Dactylis']
    for label, accuracy in zip(field_dataset.classes, class_accuracy):
        class_name = label

        if class_name in field_data_class_name:
            print('Accuracy of class %8s : %0.2f %%' % (class_name, accuracy))
            acc_dict[class_name] = accuracy

    writer.add_scalars('FieldTestAcc', acc_dict, epoch)
    writer.flush()

    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
                                                                            overall_accuracy))


# ****************************************************************************************
def test_on_field_data_no_writer(model, field_dataset, field_data_loader):
    model.eval()   # Set model to evaluate mode

    # Number of classes and dataset-size
    # num_classes=len(field_dataset.classes)
    dsize = len(field_dataset)

    # Initialize the prediction and label lists
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    pred_conf = torch.zeros(0, dtype=torch.long, device='cpu')

    # Evaluate the model accuracy on the dataset
    correct = 0
    total = 0
    with torch.no_grad():
        start = time.time()
        for images, labels in tqdm(field_data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            conf, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predlist = torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
            pred_conf = torch.cat([pred_conf, conf.view(-1).cpu()])

        stop = time.time() - start
        print('Average inference time: ', stop / dsize)

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())

    # Per-class accuracy
    class_accuracy = 100*conf_mat.diagonal()/conf_mat.sum(1)
    print('Per class accuracy')
    print('-'*18)
    acc_dict = {}

    field_data_class_name = ['Alnus', 'Betula',
                             'Carpinus', 'Corylus', 'Dactylis']
    for label, accuracy in zip(field_dataset.classes, class_accuracy):
        class_name = label
        if class_name in field_data_class_name:

            print('Accuracy of class %8s : %0.2f %%' % (class_name, accuracy))
            acc_dict[class_name] = accuracy

    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
                                                                            overall_accuracy))


unorm = UnNormalize(mean=(0.5758556,  0.5859324,  0.47398305),
                    std=(0.14691205, 0.17191792, 0.1579929))

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=False, default='finetune',
                help="Training mode: finetue/transfer/scratch")
args = vars(ap.parse_args())

# Set training mode
train_mode = args["mode"]

# img_dir = 'img_pollen'
img_dir = 'img_pollen'
# Set the train and validation directory paths
# dataset_folder = 'images_16_types'
# dataset_folder = 'images_16_types_included_dry'

# dataset_folder =  'images_16_types_plus_unknown_and_dust'
# dataset_folder =  'images_16_types_plus_unknown_plus_field_data'
# dataset_folder = 'images_16_types_plus_unknown_plus_field_data_2021'
# dataset_folder = 'images_16_types_included_dry_evaluate_hydrated'
# dataset_folder = 'images_16_types_evaluate_hydrated'

# 'images_5_types_multi_layers_9010',
data_folder_list = ['images_5_types_multi_layers_9010',
                    'images_5_types_multi_layers_8020',
                    'images_5_types_multi_layers_7030',
                    'images_5_types_multi_layers_6040',
                    'images_5_types_multi_layers_5050',
                    'images_5_types_multi_layers_4060',
                    'images_5_types_multi_layers_3070',
                    'images_5_types_multi_layers_2080',
                    'multi_datasets_5_types',
                    'images_7_types_imscramble',
                    'images_7_types_sketch',
                    'images_7_types_online_library',
                    'pollen_dust',
                    'images_7_types',
                    'images_5_types_multi_layers',
                    'images_multi_library',
                    'images_16_types',
                    'images_7_types_sketch',
                    'polleninfo_org',
                    'pollen_7_types_masked_monodepth',
                    'images_7_types_masked',
                    'images_2_types',
                    'images_16_types_remake_MIXED',
                    'images_16_types_plus_unknown_plus_field_data',
                    'images_16_types_plus_unknown_plus_field_data_2021',
                    'corrupted_30_lables_images_16_types',  
                    'corrupted_20_lables_images_16_types',
                    'corrupted_10_lables_images_16_types',
                    'images_16_types_plus_unknown_plus_field_data_2021',
                    'images_16_types_plus_unknown_plus_field_data_2022',
                    'pollen_non_pollen'
                    ]

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    foo = image_transforms['name']

    # Tensorboard summary
    
    writer = SummaryWriter(
        comment=f'{MODEL_TYPES[MODEL_ID]}_BS_{bs}_{data_folder_list[dataset_id]}_{img_size}_{foo}')

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch: ', epoch)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # for item in inputs:
                #     # img_to_display = unorm(item)
                #     transform_ = transforms.ToPILImage()
                #     img__ = transform_(item)
                #     numpydata = np.asarray(item.cpu())
                #     print(numpydata)
                    
                #     plt.imshow(img__)
                #     plt.show()

                # batch_mean = np.array([np.mean(np.array(layer)) for layer in inputs.cpu()]).mean(axis=0)
                # batch_std = np.array([np.std(np.array(layer)) for layer in inputs.cpu()]).mean(axis=0)     
                # 
                # print(batch_mean, batch_std)
                
                # print('*****************************************************')
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # loss_T = criterion_T(outputs, labels)
                    
                    # print(loss, loss_T)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                # writer.add_scalars('Train/LossAndAcc', {'Loss':epoch_loss, 'Acc':epoch_acc}, epoch)
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                # writer.add_scalars('Valid/LossAndAcc', {'Loss':epoch_loss, 'Acc':epoch_acc}, epoch)
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


            foo = image_transforms['name']
            if epoch % 10 == 0:
                torch.save(
                    model, f'models/{folder_name}_{foo}_{model_type}_ep{epoch}.pth')
                print('Save checkpoint ',
                      f'models/{folder_name}_{foo}_{model_type}_ep{epoch}.pth')
                # test_on_field_data(model, dataset['test'], dataloaders['test'], writer, epoch)
                test_on_field_data(
                    model, dataset['field_data'], dataloaders['field_data'], writer, epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ****************************************************************************
class CrossEntropyWithTemperature(nn.Module):
    def __init__(self, T):
        super(CrossEntropyWithTemperature, self).__init__()
        self.T = T

    def softmax_with_temperature(self, logits, temperature=1.0):
        """
            Applies softmax with temperature to the given logits.
            :param logits: input logits
            :param temperature: temperature parameter (default: 1.0)
            :return: softmax output
        """
        
        exp_logits = np.exp(logits.cpu().detach().numpy() / temperature)
        exp_logits_sum = np.sum(exp_logits)
        softmax_output = exp_logits / exp_logits_sum

        return torch.from_numpy(softmax_output).to(0)

    def forward(self, inputs, targets):
        outputs = self.softmax_with_temperature(inputs, self.T)
        
        # loss = 
        
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(outputs, targets)
        
        loss = nn.functional.nll_loss(outputs, targets)
        
        return loss
    
# ****************************************************************************
class CustomTransform(object):
    def __init__(self, prob=0.5):
                
        self.transform_seq = transforms.Compose([
            transforms.Resize(size = img_size + 4),
            transforms.CenterCrop(size = img_size),
            
            ImgCustomRotate(),
            SharpRegionDetector(p = 0.5),

            
            # ImageToSketch(p = 0.5 , dim = (img_size, img_size)),
            # transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),  # , scale = (0.25, 1.0)

            ImgShift(p=0.7, translate=(0.05, 0.05)),

            transforms.ColorJitter(
                brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
            
            transforms.ToTensor(),
            
            transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                  [0.1434643,  0.16687445, 0.15344492]),
            
            ])
        

        self.transform_seq_1 = transforms.Compose([
            transforms.Resize(size=img_size + 4),
            transforms.CenterCrop(size=img_size),
            
            ImgCustomRotate(),

            Tenengrad_filter(p=0.4, ksize = 3),
            
            # ImageToSketch(p = 0.5 , dim = (img_size, img_size)),
            # transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),  # , scale = (0.25, 1.0)
            # transforms.RandomResizedCrop(size=img_size, scale = (0.85, 1.2)), #, scale = (0.25, 1.0)

            ImgShift(p=0.7, translate=(0.05, 0.05)),

            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(size=img_size),

            transforms.ColorJitter(
                brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
            
            transforms.ToTensor(),
            
            transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                  [0.1434643,  0.16687445, 0.15344492]),
            
            ])
        
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            x = self.transform_seq(x)
        else:
            x = self.transform_seq_1(x)
            
       
        return x

# ****************************************************************************
            
for dataset_id in range(8, 9):
    
    dataset_folder = data_folder_list[dataset_id]
    dataset_root_folder = f'{img_dir}/{dataset_folder}'
    
    print(dataset_root_folder)
    
    # Set the train and validation directory path
    train_directory = f'{dataset_root_folder}_train'
    valid_directory = f'{dataset_root_folder}_val'
    test_directory = f'{dataset_root_folder}_test'
    
    # For corrupt test
    # valid_directory = f'{img_dir}/images_16_types_val'
    # test_directory = f'{img_dir}/images_16_types_val'
    
    # field_data_directory = f'{img_dir}/manual_labeled_field_data'
    field_data_directory = 'img_pollen/manual_labeled_field_data_v2'
    
    # train_directory = 'img_pollen/images_16_types_plus_unknown_plus_field_data_2021_train'
    # valid_directory = 'img_pollen/images_16_types_plus_unknown_plus_field_data_2021_val'
    # Set the model save path
    # MODEL_ID = 4
    
    # ****************************************************************************************
    
    MODEL_TYPES = ['vgg16',
                   'inceptionv3',
                   'mobile_netv2',
                   'resnet18',
                   'resnet50',
                   'eficientnet_b0',
                   'eficientnet_b4',
                   'densenet121',
                   ]
    
    device_id = 0
    
        
    for MODEL_ID in range(3, 4):
        model_type = MODEL_TYPES[MODEL_ID]
        print('Model type: ', model_type)
        # PATH="models/pollen_model_inceptionv3.pth"
        # PATH="models/pollen_model_mobile_netv2.pth"
        # PATH="models/pollen_model_resnet18.pth"
    
        folder_name = train_directory.split('/')[-1][:-6]
    
        PATH = f'models/{folder_name}_{model_type}.pth'
        print('MODEL PATH: ', PATH)
    
        # Batch size
        bs = 24
        # Number of epochs
        num_epochs = 51
        # Number of classes
        num_classes = 5
        # Number of workers
        num_cpu = multiprocessing.cpu_count()
    
        
        if MODEL_ID == 1:
            img_size = 224
        else:
            img_size = 224
            
                
        # Applying transforms to the data
        image_transforms_normal = {
            'name': 'image_transforms_normal',
            
            'train': CustomTransform(),
            
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
    
        # ****************************************************************************************
        
        # Applying transforms to the data
        image_transforms_normal_1 = {
            'name': 'image_transforms_normal_1',

            'train': transforms.Compose([
                ImgCustomRotate(),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                
                Tenengrad_filter(p=0.4, ksize = 3),
                
                # ImageToSketch(p = 0.5 , dim = (img_size, img_size)),
                # transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),  # , scale = (0.25, 1.0)
                # transforms.RandomResizedCrop(size=img_size, scale = (0.85, 1.2)), #, scale = (0.25, 1.0)
    
                ImgShift(p=0.7, translate=(0.05, 0.05)),
    
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(size=img_size),
    
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                      [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                Tenengrad_filter(p=1.0, ksize = 3),
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
    
        # ****************************************************************************************
        
        # Applying transforms to the data
        image_transforms_normal_2 = {
            'name': 'image_transforms_normal_2',
            
            'train': transforms.Compose([
                ImgCustomRotate(),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                
                ImageToSketch(p = 0.4 , dim = (img_size, img_size)),
                # transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),  # , scale = (0.25, 1.0)
                # transforms.RandomResizedCrop(size=img_size, scale = (0.85, 1.2)), #, scale = (0.25, 1.0)
    
                ImgShift(p=0.7, translate=(0.05, 0.05)),
    
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(size=img_size),
    
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                      [0.1434643,  0.16687445, 0.15344492]),
            ]),
            
    
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
    
        # ****************************************************************************************
        
        # Applying transforms to the data
        image_transforms_normal_3 = {
            'name': 'image_transforms_normal_3',
            
            'train': transforms.Compose([
                ImgCustomRotate(),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                
                Tenengrad_filter(p=0.9, ksize = 3),
                
                # ImageToSketch(p = 0.4 , dim = (img_size, img_size)),
   
                ImgShift(p=0.7, translate=(0.05, 0.05)),
    
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(size=img_size),
    
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                      [0.1434643,  0.16687445, 0.15344492]),
            ]),
            
    
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                Tenengrad_filter(p=1.0, ksize = 3),
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
    
        # ****************************************************************************************
        
        # Applying transforms to the data
        image_transforms_normal_4 = {
            'name': 'image_transforms_normal_4',
            
            'train': transforms.Compose([
                ImgCustomRotate(),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                
                Tenengrad_filter(p = 1.0, ksize = 3),
                
                # ImageToSketch(p = 0.4 , dim = (img_size, img_size)),
   
                ImgShift(p=0.7, translate=(0.05, 0.05)),
    
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(size=img_size),
    
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                      [0.1434643,  0.16687445, 0.15344492]),
            ]),
            
    
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                Tenengrad_filter(p=1.0, ksize = 3),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
        
        # ****************************************************************************************
        
        # Applying transforms to the data
        image_transforms_normal_5 = {
            'name': 'image_transforms_normal_5',
            
            'train': transforms.Compose([
                ImgCustomRotate(),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                
                ImageToSketch(p = 1.0 , dim = (img_size, img_size)),
                # transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),  # , scale = (0.25, 1.0)
                # transforms.RandomResizedCrop(size=img_size, scale = (0.85, 1.2)), #, scale = (0.25, 1.0)
    
                ImgShift(p=0.7, translate=(0.05, 0.05)),
    
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(size=img_size),
    
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                      [0.1434643,  0.16687445, 0.15344492]),
            ]),
            
    
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
        
        
        # ****************************************************************************************
    
        image_transforms_imscramble = {
            'name': 'image_transforms_imscramble',
            
            'train': transforms.Compose([
    
                ImgCustomRotate(),
    
                # ImgShift(p=0.7, translate = (0.05, 0.15)),
                # transforms.Resize(size=img_size),
                # transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=img_size),
    
                # transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'valid': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ]),
    
            'field': transforms.Compose([
                # ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                transforms.Resize(size=img_size),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5732364, 0.5861441, 0.4746769],
                                     [0.1434643,  0.16687445, 0.15344492]),
            ])
        }
    
        # ****************************************************************************************
        image_transforms_sketch = {
            'name': 'image_transforms_sketch',
            
            'train': transforms.Compose([
    
                ImgCustomRotate(),
                # ImageToSketch1(p=1.0, dim=(img_size, img_size)),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                
                ImageToSketch(p = 1.0, dim = (img_size, img_size)),
                # transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)), #, scale = (0.25, 1.0)
                # transforms.RandomResizedCrop(size=img_size, scale = (0.85, 1.2)), #, scale = (0.25, 1.0)
    
                ImgShift(p=0.1, translate=(0.05, 0.05)),
    
                transforms.RandomHorizontalFlip(),
    
                # transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                # transforms.Normalize( [0.56662977, 0.56662977, 0.56662977],
                #                       [0.26335555, 0.26335555, 0.26335555]),
                
                transforms.Normalize([0.21624915, 0.21624915, 0.21624915],
                                      [0.32675993, 0.32675993, 0.32675993]),
                
            ]),
    
            'valid': transforms.Compose([
    
                ImageToSketch(p=1.0, dim=(img_size, img_size)),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
    
                transforms.Normalize([0.21624915, 0.21624915, 0.21624915],
                                      [0.32675993, 0.32675993, 0.32675993]),
                
                # transforms.Normalize( [0.56662977, 0.56662977, 0.56662977],
                #                       [0.26335555, 0.26335555, 0.26335555]),
                
                transforms.Normalize([0.21624915, 0.21624915, 0.21624915],
                                      [0.32675993, 0.32675993, 0.32675993]),
            ]),
    
            'field': transforms.Compose([
                ImageToSketch(p=1.0, dim=(img_size, img_size)),
                transforms.Resize(size=img_size + 4),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
    
                # transforms.Normalize( [0.56662977, 0.56662977, 0.56662977],
                #                       [0.26335555, 0.26335555, 0.26335555]),
                
                transforms.Normalize([0.21624915, 0.21624915, 0.21624915],
                                      [0.32675993, 0.32675993, 0.32675993]),
            ])
        }
    
    
    ###############################################################################
        # image_transforms = image_transforms_normal
        # image_transforms = image_transforms_normal_1
        # image_transforms = image_transforms_normal_2
        # image_transforms = image_transforms_normal_3
        image_transforms = image_transforms_normal_4
        # image_transforms = image_transforms_normal_5
        # image_transforms = image_transforms_imscramble
    
        # image_transforms = image_transforms_sketch
    
        # Load data from folders
        dataset = {
            'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
            'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
            # 'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['valid']),
            'field_data': datasets.ImageFolder(root=field_data_directory, transform=image_transforms['field']),
        }
    
        # Size of train and validation data
        dataset_sizes = {
            'train': len(dataset['train']),
            'valid': len(dataset['valid'])
        }
    
        # Create iterators for data loading
        dataloaders = {
            'train': data.DataLoader(dataset['train'],
                                     sampler=ImbalancedDatasetSampler(
                                         dataset['train']),
                                     batch_size=bs),
    
            'valid': data.DataLoader(dataset['valid'], batch_size=8, shuffle=False,
                                     # sampler = ImbalancedDatasetSampler(dataset['train']),
                                     num_workers=num_cpu, pin_memory=True, drop_last=False),
    
            # 'test':data.DataLoader(dataset['test'], batch_size=bs, shuffle=True,
            #                         # sampler = ImbalancedDatasetSampler(dataset['train']),
            #                         num_workers=num_cpu, pin_memory=True, drop_last=True),
    
            'field_data': data.DataLoader(dataset['field_data'], batch_size=bs, shuffle=True,
                                          # sampler = ImbalancedDatasetSampler(dataset['train']),
                                          num_workers=num_cpu, pin_memory=True, drop_last=True)
        }
    
        # Class names or target labels
        class_names = dataset['train'].classes
        print("Classes:", class_names)
    
        # Save classe names to file
        class_file = open(f"{dataset_folder}_class_names.txt", "w")
    
        print('Save to class file', f"{dataset_folder}_class_names.txt")
    
        for i in range(len(class_names) - 1):
            class_file.write(class_names[i] + '\t')
    
        class_file.write(class_names[len(class_names) - 1])
        class_file.close()
    
        # Print the train and validation data sizes
        print("Training-set size:", dataset_sizes['train'],
              "\nValidation-set size:", dataset_sizes['valid'])
    
        # Set default device as gpu, if available
        device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(device)
    
        if train_mode == 'finetune':
            # ***********************************************************************
            if MODEL_ID == 0:
                # VGG16
                print("\nLoading VGG16 model ...\n")
                model_ft = models.vgg16(pretrained=True)
                # model_ft.aux_logits=False
                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
                # num_ftrs = model_ft.classifier[6].out_features
                # model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
            # ***********************************************************************
            if MODEL_ID == 1:
                # Inception
                print("\nLoading inception v3 model ...\n")
                model_ft = models.inception_v3(pretrained=True)
                model_ft.aux_logits = False
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
            # ************************************************************************
            if MODEL_ID == 2:
                # Mobilenet
                print("\nLoading mobilenetv2 ...\n")
                model_ft = models.mobilenet_v2(pretrained=True)
    
                # Freeze all the required layers (i.e except last conv block and fc layers)
                # for params in list(model_ft.parameters())[0:-5]:
                #     params.requires_grad = False
    
                # Modify fc layers to match num_classes
                num_ftrs = model_ft.classifier[-1].in_features
                model_ft.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=False),
                    nn.Linear(in_features=num_ftrs,
                              out_features=num_classes, bias=True)
                )
    
            # ************************************************************************
            if MODEL_ID == 3:
                # Load a pretrained model - Resnet18
                # Resnet 18
                print("\nLoading resnet18 model ...\n")
                model_ft = models.resnet18(pretrained=True)
    
                # Modify fc layers to match num_classes
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
                # model_ft.fc = nn.Sequential(
                #             nn.Linear(num_ftrs, 128),
                #             nn.ReLU(inplace=True),
                #             nn.Linear(128, num_classes))
    
    
            # ***********************************************************************
            if MODEL_ID == 4:
                # Resnet 50
                print("\nLoading resnet50 model ...\n")
                model_ft = models.resnet50(pretrained=True)
    
                # if resume == True:
                #     EVAL_MODEL='models/images_16_types_stylized_resnet50_ep20.pth'
                #     # EVAL_MODEL='models/images_16_types_resnet50_ep10.pth'
                #     print('Load resume model')
                #     gpuDevice = 0
                #     model_ft = torch.load(EVAL_MODEL, map_location=f'cuda:{gpuDevice}')
    
                # Modify fc layers to match num_classes
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
                # model_ft.fc = nn.Sequential(
                #             nn.Linear(num_ftrs, 128),
                #             nn.ReLU(inplace=True),
                #             nn.Linear(128, num_classes))
    
            # ************************************************************************
            if MODEL_ID == 5:
                # efficient net b0
                print("\nLoading efficient net b0 model ...\n")
                # model_ft = models.resnet18(pretrained=True)
                model_ft = torch.hub.load(
                    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    
                # Modify fc layers to match num_classes
                num_ftrs = model_ft.classifier.fc.in_features
                model_ft.classifier.fc = nn.Linear(num_ftrs, num_classes)
    
            # ************************************************************************
            if MODEL_ID == 6:
                # efficient net b4
                print("\nLoading efficient net b4 model ...\n")
                # model_ft = models.resnet18(pretrained=True)
                model_ft = torch.hub.load(
                    'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
    
                # Modify fc layers to match num_classes
                num_ftrs = model_ft.classifier.fc.in_features
                model_ft.classifier.fc = nn.Linear(num_ftrs, num_classes)
    
            # ************************************************************************
            if MODEL_ID == 7:
                # densenet 121
                print("\nLoading model.densenet 121 model ...\n")
                model_ft = models.densenet121(pretrained=True)
                # model_ft.aux_logits=False
                # num_ftrs = model_ft.fc.in_features
                # num_ftrs = model_ft.classifier.in_features
                model_ft.classifier = nn.Linear(model_ft.classifier.in_features, num_classes)
                
               
        
            # ***********************************************************************
            # models.densenet161
            # print("\nLoading models.densenet161 model...\n")
            # model_ft = models.densenet161(pretrained=True)
            # # model_ft.aux_logits=False
            # # num_ftrs = model_ft.fc.in_features
            # num_ftrs = model_ft.classifier.in_features
    
            # # model_ft.fc = nn.Linear(num_ftrs, num_classes)
            # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    
        # elif train_mode=='scratch':
        #     # Load a custom model - VGG11
        #     print("\nLoading VGG11 for training from scratch ...\n")
        #     model_ft = MyVGG11(in_ch=3,num_classes=16)
    
        #     # Set number of epochs to a higher value
        #     num_epochs=30
    
        # elif train_mode=='transfer':
        #     # Load a pretrained model - MobilenetV2
        #     print("\nLoading mobilenetv2 as feature extractor ...\n")
        #     model_ft = models.mobilenet_v2(pretrained=True)
    
        #     # Freeze all the required layers (i.e except last conv block and fc layers)
        #     for params in list(model_ft.parameters())[0:-5]:
        #         params.requires_grad = False
    
        #     # Modify fc layers to match num_classes
        #     num_ftrs=model_ft.classifier[-1].in_features
        #     model_ft.classifier=nn.Sequential(
        #         nn.Dropout(p=0.2, inplace=False),
        #         nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        #         )
    
        # summary(model_ft, input_size=(3, img_size, img_size))
    
        # Transfer the model to GPU
        model_ft = model_ft.to(device)
    
    
        # Print model summary
        print('Model Summary:-\n')
        for num, (name, param) in enumerate(model_ft.named_parameters()):
            print(num, name, param.requires_grad)
        # summary(model_ft.cuda(), inputiuytr_size=(3, img_size, img_size))
        # print(model_ft)
    
        # Loss function
        criterion = nn.CrossEntropyLoss()
        # T = 1.0
        # criterion = CrossEntropyWithTemperature(T)
        
        
        # Optimizer
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = torch.optim.Adam(model_ft.parameters())
        # optimizer_ft = Lion(model_ft.parameters(), lr = 1e-3)
               
        
        # Learning rate decay
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001)
        # exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=4)
    
        # Model training routine
        print("\nTraining:-\n")
    
        # Train the model
    
        # test_on_field_data_no_writer(model_ft, dataset['field_data'], dataloaders['field_data'])
    
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=num_epochs)
        # Save the entire model
        print("\nSaving the model...", PATH)
        torch.save(model_ft, PATH)
    
        '''
        Sample run: python train.py --mode=finetue
        '''
