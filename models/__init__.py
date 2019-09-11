from __future__ import print_function

class Backbone(object):
    def __init__(self, backbone):
        self.custom_objects = {
            'dice'          : losses.dice_loss(),
            'dicewo'        : losses.dice_loss_wo(),
            'crossentropy'  : losses.crossentropy,
            'focal'         : losses.focal()
        }

        self.backbone = backbone

    def unet(self, *args, **kwargs):
        raise NotImplementedError('U-Net method is not implemented.')