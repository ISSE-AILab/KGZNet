import os
import numpy as np
import utils.array_tool as at
from utils.Config import opt
from imageio import imwrite
#
from PIL import Image




#

def compute_iou(pred_masks, gt_masks):
    pred_masks, gt_masks = np.squeeze(at.tonumpy(pred_masks)), np.squeeze(at.tonumpy(gt_masks))
    ious = []
    for i in range(len(pred_masks)):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i]

        union = np.sum(np.logical_or(pred_mask, gt_mask))
        intersection = np.sum(np.logical_and(pred_mask, gt_mask))
        iou = intersection/union
        ious.append(iou)
    batch_iou = np.sum(np.array(ious))

    return batch_iou

import cv2
def save_pred_result(img_ids, images, pred_masks):
    for i in range(len(img_ids)):
        if not os.path.exists(os.path.join(opt.result_root, img_ids[i])):
            os.mkdir(os.path.join(opt.result_root, img_ids[i]))

        image = np.squeeze(images[i]).astype(np.uint8)
        imwrite(os.path.join(opt.result_root, img_ids[i], 'image.png'), image)

        pred_mask = np.squeeze(pred_masks[i])
        imwrite(os.path.join(opt.result_root, img_ids[i], 'pred_mask.png'), pred_mask)

        ind = np.argwhere(pred_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        # image = images[i].reshape(512,512,1)

        # print(minh)
        # print(minw)
        # print(maxh)
        # print(maxw)
        blue = (255, 0, 0)

        # image = image[int(512*0.334):int(512*0.667),int(512*0.334):int(512*0.667),:]

        # image = cv2.resize(image, (512,512))
        cv2.rectangle(image, (minw, minh), (maxw, maxh), blue, 4)
        cv2.imwrite(os.path.join(opt.result_root, img_ids[i], 'rectangle.png'), image)
        img_crop=image[minh:maxh,minw:maxw]
        img_crop=cv2.resize(img_crop, (512, 512))
        cv2.imwrite(os.path.join(opt.result_root, img_ids[i], 'combined.png'), img_crop)





        combined = np.multiply(image, pred_mask)
        imwrite(os.path.join(opt.result_root, img_ids[i], 'combined1.png'), combined)
