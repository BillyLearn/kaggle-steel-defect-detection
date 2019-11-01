from include import *

def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort = pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    df = df.drop('sort', axis=1)
    return df

def train_augment(image, mask, infor):

    u=np.random.choice(3)
    if u==0:
        pass
    elif u==1:
        image, mask = do_random_crop_rescale(image, mask, 1600-(256-224), 224)
    elif u==2:
        image, mask = do_random_crop_rotate_rescale(image, mask, 1600-(256-224), 224)

    if np.random.rand()>0.5:
        image = do_random_log_contast(image)

    if np.random.rand()>0.5:
        image, mask = do_flip_lr(image, mask)

    if np.random.rand()>0.5:
        image, mask = do_flip_ud(image, mask)

    if np.random.rand()>0.5:
        image, mask = do_noise(image, mask)

    return image, mask, infor

def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [:,y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)

        mask = mask.transpose(1,2,0)
        mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2,0,1)

    return image, mask

def do_random_crop_rotate_rescale(image, mask, w, h):
    H,W = image.shape[:2]

    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1,0.1,2)

    dscale_x = np.random.uniform(-0.00075,0.00075)
    dscale_y = np.random.uniform(-0.25,0.25)

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift*min(H,W)

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+W/2
    y = (src*[sin, cos]).sum(1)+H/2

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = mask.transpose(1,2,0)
    mask = cv2.warpPerspective( mask, transform, (W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    mask = mask.transpose(2,0,1)


    return image, mask

def do_random_log_contast(image):
    gain = np.random.uniform(0.70,1.30,1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255,0,255).astype(np.uint8)
    return image

def do_noise(image, mask, noise=8):
    H,W = image.shape[:2]
    image = image + np.random.uniform(-1,1,(H,W,1))*noise
    image = np.clip(image,0,255).astype(np.uint8)
    return image, mask


def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = mask[:,:,::-1]
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = mask[:,::-1,:]
    return image, mask


def image_to_input(image,rbg_mean,rbg_std):#, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-rbg_mean[0])/rbg_std[0]
    input[:,1] = (input[:,1]-rbg_mean[1])/rbg_std[1]
    input[:,2] = (input[:,2]-rbg_mean[2])/rbg_std[2]
    return input


def input_to_image(input,rbg_mean,rbg_std):#, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = input.data.cpu().numpy()
    input[:,0] = (input[:,0]*rbg_std[0]+rbg_mean[0])
    input[:,1] = (input[:,1]*rbg_std[1]+rbg_mean[1])
    input[:,2] = (input[:,2]*rbg_std[2]+rbg_mean[2])
    input = input.transpose(0,2,3,1)
    input = input[...,::-1]
    image = (input*255).astype(np.uint8)
    return image

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string