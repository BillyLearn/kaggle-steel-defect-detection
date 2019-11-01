from resnet  import *
from include import *

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]


####################################################################################################
class Net(nn.Module):
    def load_pretrain(self, skip):
        conversion=copy.copy(CONVERSION)
        for i in range(0,len(conversion)-8,4):
            conversion[i] = 'block.' + conversion[i][5:]
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=conversion)

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None  #dropped
        self.feature = nn.Conv2d(512,32, kernel_size=1) #dummy conv for dim reduction
        self.logit = nn.Conv2d(32,num_class, kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

        x = F.dropout(x,0.5,training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit

### loss ###################################################################
# metric
def metric_hit(logit, truth, threshold=0.5):
    batch_size, num_class, H, W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, num_class, -1)

        probability = torch.sigmoid(logit)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        tp = ((p + t) == 2).float()
        tn = ((p + t) == 0).float()

        tp = tp.sum(dim=[0, 2])
        tn = tn.sum(dim=[0, 2])

        num_pos = t.sum(dim=[0,2])
        num_neg = batch_size*H*W - num_pos

        tp = tp.data.cpu().numpy()
        tn = tn.data.cpu().numpy().sum()
        num_pos = num_pos.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().sum()

        tp = np.nan_to_num(tp/(num_pos+1e-12),0)
        tn = np.nan_to_num(tn/(num_neg+1e-12),0)

        tp = list(tp)
        num_pos = list(num_pos)

    return tn,tp, num_neg,num_pos

def criterion(logit, truth, weight=None):
    batch_size,num_class, H,W = logit.shape
    logit = logit.view(batch_size,num_class)
    truth = truth.view(batch_size,num_class)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if weight is None:
        loss = loss.mean()

    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).sum()

    return loss