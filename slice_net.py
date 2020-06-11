import torch
import torch.nn as nn

class SE_module(nn.Module):
    def __init__(self, channel, r):
        super(SE_module, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.__avg_pool(x)
        y = self.__fc(y)
        return x * y

class Channel_Attention(nn.Module):
    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return x * y

class Spartial_Attention(nn.Module):
    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask

class ClassHead(nn.Module):
    def __init__(self, input_channel, num_classes=2, attention=None):
        super(ClassHead, self).__init__()
        
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.attention = attention
        
        self.classifier_conv = nn.Conv2d(
            input_channel,
            num_classes,
            1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True
        )
        
    def forward(self, x):
        if self.attention is not None:
            x = self.attention(x)
            
        x = self.classifier_conv(x)
        
        return x

class SliceNet(nn.Module):
    def __init__(self, resnet, num_classes=2, attention=None):
        super(SliceNet, self).__init__()
        
        self.num_classes = num_classes
        self.resnet = resnet
        
        self.head = ClassHead(resnet.feature_size, num_classes, attention)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.head(x)
        
        return x
    
class BitBalanceHardMiningLoss(nn.Module):
    def __init__(self):
        super(BitBalanceHardMiningLoss, self).__init__()
        
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        n, _, _, _ = logits.shape
        
        logits = logits.reshape(n, 2, -1)
        targets = targets.reshape(n, -1).to(dtype=int)
        
        ce_loss = self.ce(logits, targets)
        
        # create grad mask
        with torch.no_grad():
            grad_masks = []
            
            pos_loss = torch.zeros_like(ce_loss).copy_(ce_loss)
            pos_loss[targets<1] = 0.
            
            neg_loss = torch.zeros_like(ce_loss).copy_(ce_loss)
            neg_loss[targets>1] = 0.
            
            neg_masks = 1 - targets

            for i in range(n):
                pos_count = torch.nonzero(targets[i, :]).shape[0]
                neg_count = torch.nonzero(neg_masks[i, :]).shape[0]
                min_count = pos_count if pos_count < neg_count else neg_count
                
                _, pos_indices = torch.topk(pos_loss[i, :], min_count)
                _, neg_indices = torch.topk(neg_loss[i, :], min_count)
                
                grad_mask = torch.zeros_like(ce_loss[i, :], dtype=int)
                
                for pos_index in pos_indices:
                    grad_mask[pos_index] = 1
                
                for neg_index in neg_indices:
                    grad_mask[neg_index] = 1
                    
                grad_masks.append(grad_mask)
                
            grad_masks = torch.stack(grad_masks)
        
        ce_loss = ce_loss[grad_masks]
        
        return ce_loss.mean()
        