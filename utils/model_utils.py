import torch
from models.alexNet import AlexNet
from models.resNet50 import ResNet, BasicBlock
from models.inceptionV1 import InceptionV1
from models.DenseNet import DenseNet

def load_model(filename):
    state = torch.load(filename)
    config = state['config']
    
    # Initialize the model based on the saved configuration
    if config['model_name'] == 'AlexNet':
        model = AlexNet(num_classes=36)
    elif config['model_name'] == 'ResNet':
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=36)
    elif config['model_name'] == 'InceptionV1':
        model = InceptionV1(num_classes=36)
    elif config['model_name'] == 'DenseNet':
        model = DenseNet(num_classes=36)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    model.load_state_dict(state['model_state_dict'])
    
    return model, config
