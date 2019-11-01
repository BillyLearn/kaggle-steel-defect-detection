from efficientnet  import *
from include import *


class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x

class Net(nn.Module):
    def load_pretrain(self, skip=['logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)



    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = EfficientNetB5(drop_connect_rate)
        self.stem   = e.stem
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        self.block5 = e.block5
        self.block6 = e.block6
        self.block7 = e.block7
        self.last   = e.last
        e = None  #dropped

        #---
        self.lateral0 = nn.Conv2d(2048, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d( 176, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(  64, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(  40, 64,  kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d( 64, 64),
        )
        self.top4 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit_mask = nn.Conv2d(64,num_class+1,kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        x = self.stem(x)            #; print('stem  ',x.shape)
        x = self.block1(x)    ;x0=x #; print('block1',x.shape)
        x = self.block2(x)    ;x1=x #; print('block2',x.shape)
        x = self.block3(x)    ;x2=x #; print('block3',x.shape)
        x = self.block4(x)          #; print('block4',x.shape)
        x = self.block5(x)    ;x3=x #; print('block5',x.shape)
        x = self.block6(x)          #; print('block6',x.shape)
        x = self.block7(x)          #; print('block7',x.shape)
        x = self.last(x)      ;x4=x #; print('last  ',x.shape)

        # segment
        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #128x128
        t2 = self.top2(t2) #128x128
        t3 = self.top3(t3) #128x128

        t = torch.cat([t1,t2,t3],1)
        t = self.top4(t)
        logit_mask = self.logit_mask(t)
        logit_mask = F.interpolate(logit_mask, scale_factor=2.0, mode='bilinear', align_corners=False)

        return logit_mask


#### loss
#########################################################################
def criterion_mask(logit, truth, weight=None):
    if weight is None: weight=[1,1,1,1]
    weight = torch.FloatTensor([1]+weight).to(truth.device).view(1,-1 )

    batch_size,num_class,H,W = logit.shape

    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)

    log_probability = -F.log_softmax(logit,-1)
    probability = F.softmax(logit,-1)

    onehot = torch.zeros(batch_size*H*W,num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1,1),value=1) #F.one_hot(truth,5).float()

    loss = log_probability*onehot

    probability = probability.view(batch_size,H*W,5)
    truth  = truth.view(batch_size,H*W,1)
    weight = weight.view(1,1,5)

    alpha  = 2
    focal  = torch.gather(probability, dim=-1, index=truth.view(batch_size,H*W,1))
    focal  = (1-focal)**alpha
    focal_sum = focal.sum(dim=[1,2],keepdim=True)
    weight = weight*focal/focal_sum.detach() *H*W
    weight = weight.view(-1,5)

    loss = loss*weight
    loss = loss.mean()
    return loss


def metric_label(probability, truth, threshold=0.5):
    batch_size=len(truth)

    with torch.no_grad():
        probability = probability.view(batch_size,4)
        truth = truth.view(batch_size,4)

        #----
        neg_index = (truth==0).float()
        pos_index = 1-neg_index
        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)

        #----
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)

        #----
        tn = tn.data.cpu().numpy()
        tp = tp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return tn,tp, num_neg,num_pos

def truth_to_onehot(truth, num_class=4):
    onehot = truth.repeat(1,num_class,1,1)
    arange = torch.arange(1,num_class+1).view(1,num_class,1,1).to(truth.device)
    onehot = (onehot == arange).float()
    return onehot

def predict_to_onehot(predict, num_class=4):
    value, index = torch.max(predict, 1, keepdim=True)
    value  = value.repeat(1,num_class,1,1)
    index  = index.repeat(1,num_class,1,1)
    arange = torch.arange(1,num_class+1).view(1,num_class,1,1).to(predict.device)
    onehot = (index == arange).float()
    value  = value*onehot
    return value

def metric_mask(logit, truth, threshold=0.5, sum_threshold=100):

    with torch.no_grad():
        probability = torch.softmax(logit,1)
        truth = truth_to_onehot(truth)
        probability = predict_to_onehot(probability)

        batch_size,num_class,H,W = truth.shape
        probability = probability.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2*(p*t).sum(-1)/((p+t).sum(-1)+1e-12)

        neg_index = (t_sum==0).float()
        pos_index = 1-neg_index

        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)
        dn = (neg_index*d_neg).sum(0)
        dp = (pos_index*d_pos).sum(0)

        #----
        dn = dn.data.cpu().numpy()
        dp = dp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return dn,dp, num_neg,num_pos

def probability_mask_to_probability_label(probability):
    batch_size,num_class,H,W = probability.shape
    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 5)
    value, index = probability.max(1)
    probability = value[:,1:]
    return probability
