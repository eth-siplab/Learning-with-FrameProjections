import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=2, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(InceptionBlock, self).__init__()

        self.conv1 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv2 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.conv3 = conbr_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Adjust the convolutional layer when input and output channels differ
        if in_channels != out_channels:
            self.conv_res = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        # Apply the residual connection if the input and output channels differ
        if residual.shape[1] != out.shape[1]:
            residual = self.conv_res(residual)
        
        out += residual
        out = self.relu(out)
        
        return out
    
class UNET_1D_simp(nn.Module):
    def __init__(self, input_dim, output_dim, layer_n, kernel_size, depth, args):
        super(UNET_1D_simp, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        self.output_dim = output_dim
        self.args = args

        self.AvgPool1D1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.AvgPool1D2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=2)
        self.AvgPoolOut = nn.AvgPool1d(kernel_size=6, stride=2, padding=2)

        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size, 1, 1)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n * 2), self.kernel_size, 2, 2)
        self.layer3 = self.down_layer(int(self.layer_n * 2) + int(self.input_dim), int(self.layer_n * 3),
                                      self.kernel_size, 2, 2)
        self.layer4 = self.down_layer(int(self.layer_n * 3) + int(self.input_dim), int(self.layer_n * 4),
                                      self.kernel_size, 2, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n * 7), int(self.layer_n * 3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n * 5), int(self.layer_n * 2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n * 3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.outcov = nn.Conv1d(self.layer_n, 1, kernel_size=self.kernel_size, stride=1, padding=2)
        self.outcov2 = nn.Conv1d(in_channels=128, out_channels=181, kernel_size=1)
        self.fc = nn.Linear(output_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, 181)
        self.out_act = nn.ReLU()

    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        return nn.Sequential(*block)
    
    def forward(self, x): # x -> (batch_size, channels, time steps)
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        #############Encoder#####################

        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)

        x1 = torch.cat([out_1, pool_x1], 1)
        out_2 = self.layer3(x1)

        x2 = torch.cat([out_2, pool_x2], 1)
        x3 = self.layer4(x2)

        #############Decoder####################
        up = self.upsample(x3)
        up = torch.cat([up, out_2], 1)
        up = self.cbr_up1(up)

        up = self.upsample(up)
        up = torch.cat([up, out_1], 1)
        up = self.cbr_up2(up)

        up = self.upsample(up)
        up = torch.cat([up, out_0], 1)
        up = self.cbr_up3(up)

        out = self.outcov(up)
        #out1 = self.fc(torch.flatten(out,start_dim=1))
        out1 = torch.tanh(out.squeeze())
        #filtered_signal = torch.mul(x,out1.unsqueeze(1)).squeeze()
        #out2 = torch.nn.Softmax(dim=1)(self.fc2(torch.flatten(out,start_dim=1)))
        #mapped_freq = map_freq(out2, self.args.cuda)
        return out1, None
    
################## convnet ###################
class convnet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, linear_unit, args):
        super(convnet, self).__init__()

        self.lin_unit = linear_unit
        self.args = args

        self.AvgPool1D0 = nn.AvgPool1d(kernel_size=int(self.args.fs/4), stride=None) if not args.data_type == 'ppg' else nn.AvgPool1d(kernel_size=int(args.fs/5), stride=1)

        self.conv1 = conbr_block(in_channels, out_channels, kernel_size, stride, dilation=1)
        self.incept1 = InceptionBlock(out_channels, 12, kernel_size, stride, dilation=1)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.incept2 = InceptionBlock(36, 36, kernel_size, stride, dilation=1)
        self.pool2 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.incept3 = InceptionBlock(108, 108, kernel_size, stride, dilation=1)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=3)        
        self.conv_out = conbr_block(324, 324, 1, stride=1, dilation=1)
        self.fc1 = nn.Linear(3564, linear_unit)

    def forward(self, x):
        x = self.AvgPool1D0(x)
        x = self.conv1(x)
        x = self.incept1(x)
        x = self.pool1(x)
        x = self.incept2(x)
        x = self.pool2(x)
        x = self.incept3(x)
        x = self.pool3(x)      
        x = self.conv_out(x) 
        x = self.fc1(torch.flatten(x,start_dim=1))
        out = torch.tanh(x)
        return out.squeeze(), None

############################################### Analyze Layer ############################
class analyze_layer(nn.Module):
    def __init__(self, input_layer, out_layer, kernel_size, stride, linear_unit):
        super(analyze_layer, self).__init__()

        self.first_conv = conbr_block(1, out_layer, kernel_size, stride, dilation=1)
        self.second_conv = conbr_block(out_layer, out_layer*2, kernel_size, stride, dilation=1)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(linear_unit, input_layer)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.pool1(x)
        x = self.fc1(torch.flatten(x,start_dim=1))
        out = 0.01+F.sigmoid(x)
        return out.squeeze()
    
############################################### DCL Arch ############################

class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, data_type='ppg', conv_kernels=64, kernel_size=5, LSTM_units=128):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units
        
        if data_type == 'ppg':
            self.fc1 = nn.Linear(128, 200)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = torch.tanh(x.squeeze())
        return x, None
    
############## Setup models #################

def setup_model(args, DEVICE):
    if args.model == 'unet':
        return UNET_1D_simp(input_dim=1, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args).cuda(DEVICE)
    elif args.model == 'resunet':
        return resunet(args=args).cuda(DEVICE)
    elif args.model == 'convnet':
        return convnet(in_channels=1, out_channels=8, kernel_size=5, stride=1, linear_unit=args.out_dim, args=args).cuda(DEVICE)
    elif args.model == 'dcl':
        return DeepConvLSTM(n_channels=1, data_type=args.data_type, conv_kernels=64, kernel_size=5, LSTM_units=128).cuda(DEVICE)
    elif args.model == 'resnet1d':
        args.model == ResNet1D(in_channels=1, base_filters=32, kernel_size=5, stride=1, groups=1, n_block=3, n_classes=args.out_dim, downsample_gap=2, increasefilter_gap=4, use_do=True).cuda(DEVICE)
    else:
        NotImplementedError

############## Setup the other model ################
class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x
      
class res_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        #conv layer
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        #Shortcut Connection (Identity Mapping)
        self.s = nn.Conv1d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip
      
class decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.r = res_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x
      
class resunet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        """ Encoder 1 """
        self.c11 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv1d(1, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = res_block(64, 128, stride=2)
        self.r3 = res_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = res_block(256, 512, stride=2)

        """ Decoder """
        self.d1 = decoder(512, 256)
        self.d2 = decoder(256, 128)
        self.d3 = decoder(128, 64)

        """ Output """
        self.output = nn.Conv1d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(200, 181)
    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ output """
        output = self.output(d3)
        output1 = self.sigmoid(output)
        out2 = torch.nn.Softmax(dim=1)(self.fc2(torch.flatten(output,start_dim=1)))
        mapped_freq = map_freq(out2, self.args.cuda)
        return output1.squeeze(), mapped_freq

############ parameter count ###########
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

############## RES-NET1D ################
"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
    """
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False, backbone=False, output_dim=200):
        super(ResNet1D, self).__init__()
        
        self.out_dim = output_dim
        self.backbone = backbone
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        self.dense2 = nn.Linear(out_channels, self.out_dim)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.transpose(-1,-2) # RESNET 1D takes channels first
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.backbone:
            out = self.dense2(out)
            return None, out
        # out = self.do(out)
        out_class = self.dense(out)
        # out = self.softmax(out)
        
        return out_class, out    
    
############## Attention ################
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
########################### Spect ############################

class ResidualBlock_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), downsample=None, ch_expansion=False):
        super(ResidualBlock_2d, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2) 
        
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample  # If needed to match dimensions
        self.downsample_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=stride)  
        self.ch_expansion = ch_expansion  # If needed to expand channels
        self.expansion = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        identity = x  # Save input for residual connection
        if self.downsample is not None:
            identity = self.downsample_pool(x)
        if self.ch_expansion:
            identity = self.expansion(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Add residual connection
        out = F.relu(out)  # Activation after summing
        
        return out

class FourierEncoder(nn.Module):
    def __init__(self, in_channels, in_length, out_channels, kernel_size=3, initial_stride=1):
        super(FourierEncoder, self).__init__()
        
        # --- Amplitude branch ---
        # Process absolute values (amplitude)
        self.abs_conv = nn.Conv1d(in_channels, 16, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)
        self.abs_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.abs_block1 = BasicBlock(
            in_channels=16, out_channels=32, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=True
        )
        self.abs_block2 = BasicBlock(
            in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=False
        )

        # --- Phase branch ---
        # Process phase values (angle)
        self.ang_conv = nn.Conv1d(in_channels, 16, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)
        # self.ang_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.ang_conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)
        # self.ang_conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=initial_stride, padding=kernel_size//2)

        self.ang_block1 = BasicBlock(
            in_channels=16, out_channels=32, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=True
        )
        self.ang_block2 = BasicBlock(
            in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, groups=1, 
            downsample=True, use_bn=False, use_do=False, is_first_block=False
        )

        if in_length == 200:
            self.fc_abs = nn.Linear(26, 1)
            self.fc_angle = nn.Linear(26, 1)
        elif in_length == 128:
            self.fc_abs = nn.Linear(17, 1)
            self.fc_angle = nn.Linear(17, 1)
        elif in_length == 100:
            self.fc_abs = nn.Linear(13, 1)
            self.fc_angle = nn.Linear(13, 1)
        elif in_length == 480:
            self.fc_abs = nn.Linear(61, 1)
            self.fc_angle = nn.Linear(61, 1)
        elif in_length == 1000:
            self.fc_abs = nn.Linear(126, 1)
            self.fc_angle = nn.Linear(126, 1)
        
        # Fully connected layer combining both branches.
        # Each branch outputs 64 channels after global pooling.
        self.fc = nn.Linear(64, 128) # Isoalign
        # self.fc = nn.Linear(64, 192) # CLIP
        
    def forward(self, x):
        # Compute amplitude and phase from complex input.
        x_abs = torch.abs(x).float()    # shape: (B, C, T)
        x_ang = torch.angle(x).float()  # shape: (B, C, T)
        # Process amplitude branch.
        a = self.abs_conv(x_abs)
        a = F.relu(a)
        a = self.abs_block1(a)
        a = self.abs_block2(a)
        # Global average pooling over time dimension.
        a = self.fc_abs(a).squeeze(-1) # shape: (B, 64)
        
        # # Process phase branch.
        p = self.ang_conv(x_ang)
        p = F.relu(p)
        p = self.ang_block1(p)
        p = self.ang_block2(p)
        p = self.fc_angle(p).squeeze(-1) # shape: (B, 64)
        
        # Concatenate features from both branches.
        out = torch.cat([a, p], dim=1)  # shape: (B, 128)
        # out = self.fc(a)
        return out

class FourierAutoencoder(nn.Module):
    def __init__(self, encoder, input_channels, in_length, latent_dim_each=64):
        super(FourierAutoencoder, self).__init__()

        self.encoder = encoder
        
        # Define decoders to reconstruct the original amplitude and phase.
        # Here, we assume the original reconstruction target has 101 dimensions.
        self.decoder_abs = nn.Sequential(
            nn.Linear(latent_dim_each, 64),
            nn.ReLU(),
            nn.Linear(64, 101)
        )
        self.decoder_ang = nn.Sequential(
            nn.Linear(latent_dim_each, 64),
            nn.ReLU(),
            nn.Linear(64, 101)
        )
        
    def forward(self, x):
        # Compute amplitude and phase from the complex input.
        # --- Amplitude branch ---
        latent = self.encoder(x)      # e.g. (B, 16, T)
        
        # Now decode each branch separately.
        recon_abs = self.decoder_abs(latent[:, :64])  # shape: (B, 101)
        recon_ang = self.decoder_ang(latent[:, 64:])
        return (recon_abs, recon_ang), latent

def conbr_block_2d(in_channels, out_channels, kernel_size, stride, padding):
    """A 2D convolutional block: Conv2d -> BatchNorm2d -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ConvMapping(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, kernel_size=3):
        """
        A convolutional mapping that transforms a vector of dimension in_channels to hidden_channels and then back to in_channels.
        Kernel size is chosen to preserve the length dimension.
        """
        super(ConvMapping, self).__init__()
        # We treat the input as a 1D signal with one channel and length=in_length.
        # Conv1d: 1 -> hidden_dim with stride 2 will approximately reduce the length by half.
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=2, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        # ConvTranspose1d: hidden_dim -> 1 to upsample back to in_length.
        # We use output_padding=1 to exactly recover the original length (if needed).
        self.deconv = nn.ConvTranspose1d(in_channels=hidden_channels, out_channels=1,
                                         kernel_size=kernel_size, stride=2,
                                         padding=padding, output_padding=1)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, in_length) or (B, 1, in_length).
        Returns:
            Output tensor of shape (B, in_length).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # shape: (B, 1, in_length)
        out = self.conv1(x)      # shape: (B, hidden_dim, ~in_length/2)
        out = self.relu(out)
        out = self.deconv(out)   # shape: (B, 1, in_length)
        return out.squeeze(1)    # shape: (B, in_length)

class UNET_2D_simp(nn.Module):
    def __init__(self, input_channels, output_channels, layer_n, spect_freq, spect_time, kernel_size):
        """
        A simplified U-Net for 2D data (e.g. wavelet spectrograms).

        Parameters:
        - input_channels: Number of input channels (e.g., 1 for a single spectrogram channel)
        - output_channels: Number of output channels (e.g., number of classes or reconstruction channels)
        - layer_n: Base number of filters
        - kernel_size: Size of the convolution kernel (assumed square)
        - depth: (Not used in this simplified version, but could control number of layers)
        - args: Additional arguments (if needed)
        """
        super(UNET_2D_simp, self).__init__()
        self.input_channels = input_channels
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.spec_freq = spect_freq
        self.spec_time = spect_time

        # Pooling layers (2D average pooling)
        self.AvgPool2D1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.AvgPool2D2 = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))
        
        # Encoder
        self.layer1 = self.down_layer_2d(self.input_channels, self.layer_n, self.kernel_size, stride=1, padding=self.kernel_size//2)
        self.layer2 = self.down_layer_2d(self.layer_n, self.layer_n * 2, self.kernel_size, stride=2, padding=self.kernel_size//2)
        # Concatenate with pooled input (similar to adding residual skip from input)
        self.layer3 = self.down_layer_2d(self.layer_n * 2 + self.input_channels, self.layer_n * 3, self.kernel_size, stride=2, padding=self.kernel_size//2)
        self.layer4 = self.down_layer_2d(self.layer_n * 3 + self.input_channels, self.layer_n * 4, self.kernel_size, stride=2, padding=self.kernel_size//2)
        
        # Decoder
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.cbr_up1 = conbr_block_2d(self.layer_n * 4 + self.layer_n * 3, self.layer_n * 3, self.kernel_size, stride=1, padding=self.kernel_size//2)
        # self.cbr_up2 = conbr_block_2d(self.layer_n * 3 + self.layer_n * 2, self.layer_n * 2, self.kernel_size, stride=1, padding=self.kernel_size//2)
        # self.cbr_up3 = conbr_block_2d(self.layer_n * 2 + self.layer_n, self.layer_n, self.kernel_size, stride=1, padding=self.kernel_size//2)
        
        # Final output convolution layer to map to desired output channels
        # self.outcov = nn.Conv2d(self.layer_n, output_channels, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

        if self.spec_freq == 48 and self.spec_time == 200:
            self.fc = nn.Linear(25, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 128:
            self.fc = nn.Linear(16, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 100:
            self.fc = nn.Linear(13, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 480:
            self.fc = nn.Linear(60, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 1000:
            self.fc = nn.Linear(125, 1)
            self.fc2 = nn.Linear(6, 1)

        if self.spec_freq == 48 and self.spec_time == 200:
            self.fc = nn.Linear(25, 1)
            self.fc2 = nn.Linear(6, 1)

    def down_layer_2d(self, in_channels, out_channels, kernel_size, stride, padding):
        """Creates a downsampling layer using a conv block."""
        return nn.Sequential(
            conbr_block_2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        # x is expected to have shape: (batch, channels, freq, time)
        # Create multi-scale pooled inputs from the original signal for skip connections.
        pool_x1 = self.AvgPool2D1(x)  # Downsample by factor of 2
        pool_x2 = self.AvgPool2D2(x)  # Downsample by factor of 4
        # Encoder path
        out_0 = self.layer1(x)          # (B, layer_n, F, T)
        out_1 = self.layer2(out_0)        # (B, layer_n*2, F/2, T/2)
        
        # Concatenate skip connection from original input (pooled) with out_1
        x1 = torch.cat([out_1, pool_x1], dim=1)  # (B, layer_n*2 + input_channels, F/2, T/2)
        out_2 = self.layer3(x1)          # (B, layer_n*3, F/4, T/4)
        
        x2 = torch.cat([out_2, pool_x2], dim=1)  # (B, layer_n*3 + input_channels, F/4, T/4)
        x3 = self.layer4(x2)             # (B, layer_n*4, F/8, T/8)

        # Decoder path
        # up = self.upsample(x3)           # Upsample to (B, layer_n*4, F/4, T/4)
        # up = torch.cat([up, out_2], dim=1)  # (B, layer_n*4 + layer_n*3, F/4, T/4)
        # up = self.cbr_up1(up)            # (B, layer_n*3, F/4, T/4)
        
        # up = self.upsample(up)           # Upsample to (B, layer_n*3, F/2, T/2)
        # up = torch.cat([up, out_1], dim=1)   # (B, layer_n*3 + layer_n*2, F/2, T/2)
        # up = self.cbr_up2(up)            # (B, layer_n*2, F/2, T/2)
        
        # up = self.upsample(up)           # Upsample to (B, layer_n*2, F, T)
        # up = torch.cat([up, out_0], dim=1)   # (B, layer_n*2 + layer_n, F, T)
        # up = self.cbr_up3(up)            # (B, layer_n, F, T)
        
        # out = self.outcov(up)            # (B, output_channels, F, T)

        return None, self.fc2(self.fc(x3).squeeze()).squeeze()