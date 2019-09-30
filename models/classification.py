

def set_model(main_args, sub_args, base_filter=32):
    if 'vgg' in sub_args['mode']['model']:
        from .vgg3d import VGG3D
        model = VGG3D(main_args, sub_args, base_filter=base_filter)
    
    elif 'res' in sub_args['mode']['model']:
        from .resnet3d import ResNet3D
        model = ResNet3D(main_args, sub_args, base_filter=base_filter)

    return model