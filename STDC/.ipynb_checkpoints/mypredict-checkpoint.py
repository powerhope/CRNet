import matplotlib
matplotlib.use('Agg')
import scipy
from scipy import ndimage
import torch, cv2
import numpy as np
import numpy.ma as ma
import sys
import pdb
import torch

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
# from dataset import get_segmentation_dataset
# from network import get_segmentation_model
# from config import Parameters
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage

import matplotlib.pyplot as plt
import torch.nn as nn
from models.model_stages import BiSeNet
from cityscapes import CityScapes
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)       # 0: 'road' 
    palette[3:6] = (244, 35,232)        # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)         # 2''building'
    palette[9:12] = (102,102,156)       # 3 wall
    palette[12:15] =  (190,153,153)     # 4 fence
    palette[15:18] = (153,153,153)      # 5 pole
    palette[18:21] = (250,170, 30)      # 6 'traffic light'
    palette[21:24] = (220,220, 0)       # 7 'traffic sign'
    palette[24:27] = (107,142, 35)      # 8 'vegetation'
    palette[27:30] = (152,251,152)      # 9 'terrain'
    palette[30:33] = ( 70,130,180)      # 10 sky
    palette[33:36] = (220, 20, 60)      # 11 person
    palette[36:39] = (255, 0, 0)        # 12 rider
    palette[39:42] = (0, 0, 142)        # 13 car
    palette[42:45] = (0, 0, 70)         # 14 truck
    palette[45:48] = (0, 60,100)        # 15 bus
    palette[48:51] = (0, 80,100)        # 16 train
    palette[51:54] = (0, 0,230)         # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)      # 18 'bicycle'
    palette[57:60] = (105, 105, 105)
    return palette

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img
def predict_whole_img_w_label(net, image, classes, method, scale, label):
    """
         Predict the whole image w/o using multiple crops.
         The scale specify whether rescale the input image before predicting the results.
    """
#     N_, C_, H_, W_ = image.shape
#     if torch_ver == '0.4':
#         interp = nn.Upsample(size=(H_, W_), mode='bilinear', align_corners=True)
#     else:
#     interp = nn.Upsample(size=(H_, W_), mode='bilinear')

#     bug
#     if scale > 1:
#         scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
#     else:
#         scaled_img = image

#     scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    
    full_prediction_= net(Variable(torch.from_numpy(image), volatile=True).cuda())
#     full_prediction_ = net('STDCNet1446',19)
    if 'dsn' in method or 'center' in method or 'fuse' in method:
        full_prediction = full_prediction_[-1]
    else:
        full_prediction = full_prediction_[0]

    full_prediction = F.upsample(input=full_prediction, size=(1024, 2048), mode='bilinear', align_corners=True)
    result = full_prediction.cpu().data.numpy().transpose(0,2,3,1)
    return result
def get_confusion_matrix(gt_label, pred_label, class_num):
        """
        Calcute the confusion matrix by given label and pred
        :param gt_label: the ground truth label
        :param pred_label: the pred label
        :param class_num: the nunber of class
        :return: the confusion matrix
        """
        index = (gt_label * class_num + pred_label).astype('int32')
        label_count = np.bincount(index)
        confusion_matrix = np.zeros((class_num, class_num))

        for i_label in range(class_num):
            for i_pred_label in range(class_num):
                cur_index = i_label * class_num + i_pred_label
                if cur_index < len(label_count):
                    confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

        return confusion_matrix
def main():
    """Create the model and start the evaluation process."""
#     args = Parameters().parse()

    # file_log = open(args.log_file, "w")
    # sys.stdout = sys.stderr = file_log

#     print("Input arguments:")
#     sys.stdout.flush()
#     for key, val in vars(args).items():
#         print("{:16} {}".format(key, val))

    h, w = map(int, [768,1536])
    input_size = (h, w)

    output_path = './my_predict_222'
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

    CRnet = BiSeNet('STDCNet1446', 19,use_boundary_2=False, use_boundary_4=False, 
     use_boundary_8=True, use_boundary_16=False, 
     use_conv_last=False)

    ignore_label = 255
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
          3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
          7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
          14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
          18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
          28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    methodName = 'train_STDC2-Seg_depthwise57/pths'
    print('loading parameters...')
    respth = '/xiaoou/STDC-Seg-master/STDC-Seg-master/checkpoint/{}/'.format(methodName)
    save_pth = os.path.join(respth, 'model_maxmIOU{}.pth'.format(75))
    store_output = 'True'
    
    saved_state_dict = torch.load(save_pth)
    CRnet.load_state_dict(saved_state_dict,False)

    model = nn.DataParallel(CRnet)
    model.eval()
    model.cuda()
    dspth = '/xiaoou/my_last_project/dataset/cityscapes'
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
    dsval = CityScapes(dspth,cropsize=[1024,2048],mode='val',randomscale=randomscale)
    testloader = data.DataLoader(dsval,
                    batch_size = 2,
                    shuffle = False,
                    num_workers = 2,
                    drop_last = False)

    data_list = []
    confusion_matrix = np.zeros((19,19))

    palette = get_palette(20)

    image_id = 0
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
            sys.stdout.flush()
        image, label,name = batch
        size = image.shape
#         size = size[0]
        with torch.no_grad():
            if 'val' in 'val':
                output= predict_whole_img_w_label(model, image.numpy(), 19, 
                            'val', scale=float(1), label=Variable(label.long().cuda()))
                
            else:
                output = predict_whole_img(model, image.numpy(),19, 
                    'val', scale=float(1))
#         print(output.shape)
        seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        m_seg_pred = ma.masked_array(seg_pred, mask=torch.eq(label, 255))
        ma.set_fill_value(m_seg_pred, 20)
        seg_pred = m_seg_pred
#         seg_pred = np.expand_dims(seg_pred, 1)
#         print(seg_pred.shape)

        for i in range(image.size(0)): 
            image_id += 1
            print('%d th segmentation map generated ...'%(image_id))
            sys.stdout.flush()
            if store_output == 'True':
                output_im = PILImage.fromarray(seg_pred[i])
                output_im.putpalette(palette)
                output_im.save(output_path+'/'+name[i]+'_gtFine_labelIds.png')
#         print(label.shape)
        seg_pred = np.expand_dims(seg_pred, 1)
        seg_gt = np.asarray(label.numpy())#[:,:size[0],:size[1]], dtype=np.int)
#         print(seg_gt.shape)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, 19)
            
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()
    
    print({'meanIU':mean_IU, 'IU_array':IU_array})

    print("confusion matrix\n")
#     print(confusion_matrix)
    sys.stdout.flush()

main()