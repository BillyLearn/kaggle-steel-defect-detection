from utils import *
from model import *
from resnet import *
from include import *

def get_train_valid_loader(batch_size):
    train_dataset = SteelDataSet(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_11568.npy',],
        augment = train_augment,
    )

    train_loader  = DataLoader(
        train_dataset,
        sampler    = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 2,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = SteelDataSet(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['valid_1000.npy',],
        augment = None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler    = SequentialSampler(valid_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    assert(len(train_dataset)>=batch_size)

    return train_dataset, train_loader, valid_loader



## batch整理
def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_mask  = []
    truth_label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_mask.append(batch[b][1])
        infor.append(batch[b][2])

        label = (batch[b][1].reshape(4,-1).sum(1)>8).astype(np.int32)
        truth_label.append(label)

    input = np.stack(input)
    input = image_to_input(input, IMAGE_RGB_MEAN,IMAGE_RGB_STD)
    input = torch.from_numpy(input).float()

    truth_mask = np.stack(truth_mask)
    truth_mask = (truth_mask>0.5).astype(np.float32)
    truth_mask = torch.from_numpy(truth_mask).float()

    truth_label = np.array(truth_label)
    truth_label = torch.from_numpy(truth_label).float()

    return input, truth_mask, truth_label, infor

def do_valid(net, valid_loader):
    valid_num  = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)

    for t, (input, truth_mask, truth_label, infor) in enumerate(valid_loader):
        net.eval()
        input = input.cuda()
        truth_mask  = truth_mask.cuda()
        truth_label = truth_label.cuda()

        with torch.no_grad():
            logit = net(input) #data_parallel(net, input)
            loss  = criterion(logit, truth_label)
            tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)

        batch_size = len(infor)
        l = np.array([ loss.item(), tn,*tp])
        n = np.array([ batch_size, num_neg,*num_pos])
        valid_loss += l*n
        valid_num  += n
        print('\r %8d /%8d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/valid_num

    return valid_loss

# Class which is used by the infor object in __get_item__
class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                except Exception:
                    setattr(self, key, value)

    def __str__(self):
        text =''
        for k,v in self.__dict__.items():
            text += '\t%s : %s\n'%(k, str(v))
        return text

# 创建mask
def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask

class SteelDataSet(Dataset):
    def __init__(self, split, csv, mode, augment=None):
        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.uid = list(np.concatenate([np.load(TRAIN_NPY + '/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s'%f) for f in csv])
        df.fillna('', inplace=True)
        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in self.uid for c in [1,2,3,4] ])
        self.df = df

    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()

        length = len(self)
        num = len(self)*4
        pos = (self.df['Label']==1).sum()
        neg = num-pos

        #---

        string  = ''
        string += '\tmode    = %s\n'%self.mode
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\t\tlen   = %5d\n'%len(self)
        if self.mode == 'train':
            string += '\t\tnum   = %5d\n'%num
            string += '\t\tneg   = %5d  %0.3f\n'%(neg,neg/num)
            string += '\t\tpos   = %5d  %0.3f\n'%(pos,pos/num)
            string += '\t\tpos1  = %5d  %0.3f  %0.3f\n'%(pos1,pos1/length,pos1/pos)
            string += '\t\tpos2  = %5d  %0.3f  %0.3f\n'%(pos2,pos2/length,pos2/pos)
            string += '\t\tpos3  = %5d  %0.3f  %0.3f\n'%(pos3,pos3/length,pos3/pos)
            string += '\t\tpos4  = %5d  %0.3f  %0.3f\n'%(pos4,pos4/length,pos4/pos)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])

        infor = Struct(
            index    = index,
            folder   = folder,
            image_id = image_id,
        )

        if self.augment is None:
            return image, mask, infor
        else:
            return self.augment(image, mask, infor)