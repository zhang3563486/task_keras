import keras
import keras.backend as K

def dice_loss(y_true, y_pred, smooth=1.):
    loss = 0.
    classes = K.int_shape(y_pred)[-1]
    y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
    if classes > 1:
        # multi classes
        for num_label in range(classes):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - loss/classes
    else:
        # single class
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - loss

def dice_loss_wo(y_true, y_pred, smooth=1.):
    loss = 0.
    classes = K.int_shape(y_pred)[-1]
    y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
    if classes > 1:
        for num_label in range(1,classes):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1-loss / (classes-1)
    else:
        raise ValueError('Model must have multi classes output.')

def crossentropy(y_true, y_pred):
    classes = K.int_shape(y_pred)[-1]
    if classes > 1:
        # multi classes
        loss = K.categorical_crossentropy(y_true, y_pred)
    else:
        # single classes
        loss = K.binary_crossentropy(y_true, y_pred)
    return loss

def focal(alpha=0.25, gamma=2.0):
    def _loss(y_true, y_pred):
        loss = 0.
        classes = K.int_shape(y_pred)[-1]
        y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
        if classes > 1:
            # multi classes
            for num_label in range(classes):
                y_true_f = K.flatten(y_true[...,num_label])
                y_pred_f = K.flatten(y_pred[...,num_label])
                pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
                pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))
                loss += -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            return loss / classes
        else:
            # single classes
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
            pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))
            loss += -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
            return loss
    return _loss

def ce_dice_loss(y_true, y_pred):
    return crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def focal_dice_loss(y_true, y_pred):
    return focal()(y_true, y_pred) + dice_loss(y_true, y_pred)

def ce_logdice_loss(y_true, y_pred):
    return crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def focal_logdice_loss(y_true, y_pred):
    return focal()(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))