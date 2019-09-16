import keras
import tensorflow as tf
import keras.backend as K

def dice(y_true, y_pred, classes=None, smooth=1.):
    loss = 0.
    classes =  classes if classes else K.int_shape(y_pred)[-1]
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    
    if classes > 1:
        for num_label in range(classes):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return loss / classes
    else:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return loss

def dice_target(y_true, y_pred):
    return dice(y_true[...,1], y_pred[...,1], classes=1)

def dice_cancer(y_true, y_pred):
    return dice(y_true[...,2], y_pred[...,2], classes=1)
    
def iou(y_true, y_pred, classes=None):
    loss = 0.
    classes =  classes if classes else K.int_shape(y_pred)[-1]
    y_pred = K.cast(y_pred > 0.5, dtype=y_pred.dtype)
    
    if classes > 1:
        print('multi class')
        for num_label in range(1, classes):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            notTrue = 1 - y_true_f
            union = K.sum(y_true_f + (notTrue * y_pred_f))
            loss += intersection / (union + K.epsilon())
        return loss / (classes-1)
    else:
        print('single class')
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        notTrue = 1 - y_true_f
        union = K.sum(y_true_f + (notTrue * y_pred_f))
        loss += intersection / (union + K.epsilon())
        return loss

def iou_vessel(y_true, y_pred):
    return iou(y_true[...,1], y_pred[...,1], classes=1)

