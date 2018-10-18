"""Residual Networks"""

"""Learning from PyTorch tutorials available at https://pytorch.org/"""

# import the required libraries

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


"""Define the building blocks of the Residual Network architecture"""

# create a generic 3 by 3 convolutional block

def conv3x3(in_planes, out_planes, stride = 1):
    
    """3x3 convolution with padding; it'll usually return the 'same' convolutions"""
    
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)



# define the BasicBlock class

class BasicBlock(nn.Module):
    
    expansion = 1
    
    # define the __init__ constructor for the class
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)        # defines a 3 by 3 convolution block
        
        self.bn1 = nn.BatchNorm2d(planes)                     # batch normalization with learnable parameters
        
        self.relu = nn.ReLU(inplace = True)                   # perform the ReLU activation in-place
        
        self.conv2 = conv3x3(planes, planes)                  # defines a 3 by 3 convolution block
        
        self.bn2 = nn.BatchNorm2d(planes)                     # batch normalization with learnable parameters
        
        self.downsample = downsample
        
        self.stride = stride
        
    # define the residual block structure
    
    def forward(self, x):
        
        residual = x                                          # start with the input, initialize the residual 
        
        out = self.conv1(x)                                   # first convolutional layer
        
        out = self.bn1(out)                                   # batch normalize the layer
        
        out = self.relu(out)                                  # pass through the ReLU activation function
        
        out = self.conv2(out)                                 # perform the convolution again while keeping the same #channels
        
        out = self.bn2(out)                                   # batch normalize the layer
        
        if self.downsample is not None:
            
            residual = self.downsample(x)                     # downsample the input residual
            
        out += residual                                       # add the residual to the output of the two convolutions
        
        out = self.relu(out)                                  # pass through the ReLU activation function
        
        return out                                            # return the output value
    
    
    
# define the Bottleneck class

class Bottleneck(nn.Module):
    
    expansion = 4
    
    # define the __init__ constructor for the class
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        
        super(Bottleneck, self).__init__()
        
        # defines 1 by 1 convolution to reduce the #channels in the input
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        
        self.bn1 = nn.BatchNorm2d(planes)                    # batch normalization with learnable parameters
        
        # defines 3 by 3 convolution block
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        
        self.bn2 = nn.BatchNorm2d(planes)                    # batch normalization with learnable parameters
        
        # defines a 1 by 1 filter to expand the #channels in the input
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1, bias = False)
        
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)   # batch normalization with learnable parameters
        
        self.relu = nn.ReLU(inplace = True)                  # perform the ReLU activation in place
        
        self.downsample = downsample
        
        self.stride = stride
        
    # define the residual block structure
    
    def forward(self, x):
        
        residual = x                                         # start with the input, initialize the residual
        
        out = self.conv1(x)                                  # first convolutional layer, 1 by 1 (mostly compress)
        
        out = self.bn1(out)                                  # batch normalize the layer
        
        out = self.relu(out)                                 # pass through the ReLU activation function
        
        out = self.conv2(out)                                # perform the convolution again while keeping the same #channels
        
        out = self.bn2(out)                                  # batch normalize the layer
        
        out = self.relu(out)                                 # pass through the ReLU activation function
        
        out = self.conv3(out)                                # third convolution layer, 1 by 1 (expand by a factor of 4)
        
        out = self.bn3(out)                                  # batch normalize the layer
        
        if self.downsample is not None:
            
            residual = self.downsample(x)                    # downsample the input residual
            
        out += residual                                      # add the residual to the output of the three convolutions
        
        out = self.relu(out)                                 # pass through the ReLU activation function
        
        return out                                           # return the output value


"""Define the Residual Network architecture"""

# define the residual network class

class ResNet(nn.Module):
    
    # define the __init__ constructor for the class
    
    def __init__(self, block, layers, num_classes = 1000):
        
        self.inplanes = 64
        
        super(ResNet, self).__init__()
        
        # define the first convolution
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
         
        self.bn1 = nn.BatchNorm2d(64)                        # batch normalize the layer
        
        self.relu = nn.ReLU(inplace = True)                  # pass through ReLU activation function
        
        # perform maximum pooling to produce the reduced output dimensions
        
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # definition of the layers of sets of residual blocks
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        
        # perform average pooling to reduce the dimensions for the fully connected layers
        
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        
        # define the fully connected layer to produce the output for the multiclass classification problem
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # initialize the parameters using He initialization for the convolutional, 
        # and (0, 1) for the batch normalization layers
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant_(m.weight, 1)
                
                nn.init.constant_(m.bias, 0)
                
    # define the _make_layer function to produce the layer that'll constitute the set of residual blocks 
                
    def _make_layer(self, block, planes, blocks, stride = 1):
        
        downsample = None
        
        # it's purpose is to match the size of the residual and the filters after two convolutions before adding them together
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, 
                                                 stride = stride, bias = False), 
                                       nn.BatchNorm2d(planes * block.expansion),)
            
        layers = []
        
        # first layer reduces the size of the filters according to the stride
            
        layers.append(block(self.inplanes, planes, stride, downsample))
            
        self.inplanes = planes * block.expansion
        
        # adds the later residual blocks where the size of the filters remain the same
            
        for i in range(1, blocks):
                
            layers.append(block(self.inplanes, planes))
                
        return nn.Sequential(*layers)                       # returns the final output of the layer definition
    
    # define the forward function to produce the overall architecture
        
    def forward(self, x):
        
        # let's say that the images are of the size 224 by 224
        
        # define the first convolutions with filter/kernel size at 7, stride at 2, and padding at 3
        # this results in the output of 112 by 112 (64)
        
        x = self.conv1(x)                                     
        
        x = self.bn1(x)                                     # perform batch normalization
        
        x = self.relu(x)                                    # pass through the ReLU activation function
        
        # perform maximum pooling with filter/kernel size at 3, stride at 2, and padding at 1
        # this results in the output of 56 by 56 (64)
        
        x = self.maxpool(x)
        
        # this executes the first set of residual blocks where the dimensions (height and width) remain the 
        # same with each layer; there are two options of BasicBlock and Bottleneck that could be executed
        # for instance, for ResNet 18, it adds 2 BasicBlocks where it stays at 56 by 56 (64)
        
        x = self.layer1(x)
        
        # this executes the second set of residual blocks where the dimensions (height and width) remain the 
        # same with each layer; there are two options of BasicBlock and Bottleneck that could be executed
        # for instance, for ResNet 50, it adds 3 Bottlenecks where it stays at 28 by 28 (128)
        
        x = self.layer2(x)
        
        # this executes the thrid set of residual blocks where the dimensions (height and width) remain the 
        # same with each layer; there are two options of BasicBlock and Bottleneck that could be executed
        # it stays at 14 by 14 (256)
        
        x = self.layer3(x)
        
        # this executes the thrid set of residual blocks where the dimensions (height and width) remain the 
        # same with each layer; there are two options of BasicBlock and Bottleneck that could be executed
        # it stays at 7 by 7 (512)
        
        x = self.layer4(x)
        
        # perform average pooling to reduce the dimension to 1 by 1 from 7 by 7
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)                           # create the input for the fully connected layer
        
        # fully connected layer to give the final output, as the number of classes
        
        x = self.fc(x)                                      
        
        return x


"""Define the different ResNet models based on the number of layers (BasicBlock or Bottleneck)"""

# define the ResNet 18 model

def resnet18(pretrained = False, **kwargs):
    
    """Construct a ResNet-18 model as defined by the number of layers"""
    
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    if pretrained:
        
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
    return model

# define the ResNet 34 model

def resnet34(pretrained = False, **kwargs):
    
    """Construct a ResNet-34 model as defined by the number of layers"""
    
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        
    return model

# define the ResNet 50 model

def resnet50(pretrained = False, **kwargs):
    
    """Construct a ResNet-50 model as defined by the number of layers"""
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        
    return model

# define the ResNet 101 model

def resnet101(pretrained = False, **kwargs):
    
    """Construct a ResNet-101 model as defined by the number of layers"""
    
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    
    if pretrained:
        
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        
    return model

# define the ResNet 152 model

def resnet152(pretrained = False, **kwargs):
    
    """Construct a ResNet-152 model as defined by the number of layers"""
    
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    
    if pretrained:
        
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        
    return model