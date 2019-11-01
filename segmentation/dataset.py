from include import *
from model import *
from utils import *

def get_train_valid_loader(batch_size):
    train_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_11568.npy',],
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        sampler     = FiveBalanceClassSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    valid_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['valid_1000.npy',],
        augment = None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    assert(len(train_dataset)>=batch_size)

    return train_dataset, train_loader, valid_loader

class FiveBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #5x
        self.num_image = len(self.dataset.df)//4
        self.length = self.num_image*5


    def __iter__(self):
        neg  = np.random.choice(self.neg_index,  self.num_image, replace=True)
        pos1 = np.random.choice(self.pos1_index, self.num_image, replace=True)
        pos2 = np.random.choice(self.pos2_index, self.num_image, replace=True)
        pos3 = np.random.choice(self.pos3_index, self.num_image, replace=True)
        pos4 = np.random.choice(self.pos4_index, self.num_image, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_label = []
    truth_mask  = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_label.append(batch[b][1])
        truth_mask.append(batch[b][2])
        infor.append(batch[b][3])

    input = np.stack(input).astype(np.float32)/255
    input = input.transpose(0,3,1,2)
    truth_label = np.stack(truth_label)
    truth_mask  = np.stack(truth_mask)

    input = torch.from_numpy(input).float()
    truth_label = torch.from_numpy(truth_label).float()
    truth_mask = torch.from_numpy(truth_mask).long().unsqueeze(1)

    with torch.no_grad():
        arange = torch.FloatTensor([1,2,3,4]).to(truth_mask.device).view(1,4,1,1).long()
        m = truth_mask.repeat(1,4,1,1)
        m = (m==arange).float()

        #relabel for augmentation cropping, etc
        truth_label = m.sum(dim=[2,3])
        truth_label = (truth_label > 1).float()

    return input, truth_label, truth_mask, infor

def do_valid(net, valid_loader):

    valid_loss = np.zeros(17, np.float32)
    valid_num  = np.zeros_like(valid_loss)

    for t, (input, truth_label, truth_mask, infor) in enumerate(valid_loader):

        batch_size = len(infor)

        net.eval()
        input = input.cuda()
        truth_label = truth_label.cuda()
        truth_mask  = truth_mask.cuda()

        with torch.no_grad():
            logit_mask = data_parallel(net, input)
            loss = criterion_mask(logit_mask, truth_mask)

            probability_mask  = F.softmax(logit_mask,1)
            probability_label = probability_mask_to_probability_label(probability_mask)
            tn,tp, num_neg,num_pos = metric_label(probability_label, truth_label)
            dn,dp, num_neg,num_pos = metric_mask(logit_mask, truth_mask)

        #---
        l = np.array([ loss.item()*batch_size, *tn, *tp, *dn, *dp])
        n = np.array([ batch_size, *num_neg, *num_pos, *num_neg, *num_pos])
        valid_loss += l
        valid_num  += n

        print('\r %4d/%4d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

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

class SteelDataset(Dataset):
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
        self.num_image = len(df)//4


    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()
        neg1 = num1-pos1
        neg2 = num2-pos2
        neg3 = num3-pos3
        neg4 = num4-pos4

        length = len(self)
        num = len(self)
        pos = (self.df['Label']==1).sum()
        neg = num-pos

        string  = ''
        string += '\tmode    = %s\n'%self.mode
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\tnum_image = %8d\n'%self.num_image
        string += '\tlen       = %8d\n'%len(self)
        if self.mode == 'train':
            string += '\t\tpos1, neg1 = %5d  %0.3f,  %5d  %0.3f\n'%(pos1,pos1/num,neg1,neg1/num)
            string += '\t\tpos2, neg2 = %5d  %0.3f,  %5d  %0.3f\n'%(pos2,pos2/num,neg2,neg2/num)
            string += '\t\tpos3, neg3 = %5d  %0.3f,  %5d  %0.3f\n'%(pos3,pos3/num,neg3,neg3/num)
            string += '\t\tpos4, neg4 = %5d  %0.3f,  %5d  %0.3f\n'%(pos4,pos4/num,neg4,neg4/num)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        label = [ 0 if r=='' else 1 for r in rle]
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=c) for c,r in zip([1,2,3,4],rle)])
        mask  = mask.max(0, keepdims=0)

        infor = Struct(
            index    = index,
            folder   = folder,
            image_id = image_id,
        )

        if self.augment is None:
            return image, label, mask, infor
        else:
            return self.augment(image, label, mask, infor)