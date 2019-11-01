from dataset import *
from torch.optim.optimizer import Optimizer
import math

# Learning Rate Adjustments
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]
    return lr

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])  # noqa
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def run_train():
    ## hyperparam and initialize  --------------------------------------
    schduler = NullScheduler(lr=0.001)
    iter_accum = 1
    batch_size = 5 #8

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 200
    iter_valid  = 200
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 1000))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0

    iter = 0
    i    = 0

    valid_loss = np.zeros(17,np.float32)
    train_loss = np.zeros( 6,np.float32)
    batch_loss = np.zeros_like(valid_loss)

    ## dataset ----------------------------------------
    train_dataset, train_loader, valid_loader = get_train_valid_loader(batch_size)

    ## net ----------------------------------------
    net = Net().cuda()
    if INITIAL_CHECKPOINT is not None:
        state_dict = torch.load(INITIAL_CHECKPOINT, map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict,strict=False)  #True
    else:
        net.load_pretrain(is_print=False)

    ## optimiser ----------------------------------
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    # optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    # optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)


    ## load model ------------------------------------------------------------------------
    if INITIAL_CHECKPOINT is not None:
        initial_optimizer = INITIAL_CHECKPOINT.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
        pass

    ## run train ------------------------------------------------------------------------
    start_timer = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (input, truth_label, truth_mask, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


            if (iter % iter_valid==0):
                valid_loss = do_valid(net, valid_loader)

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                asterisk = '*' if iter in iter_save else ' '
                print('%0.5f %5.1f%s %5.1f | %5.3f  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f]  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f] | %5.3f  [%0.2f : %0.2f %0.2f %0.2f %0.2f] | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss,
                         *train_loss,
                         time_to_str((timer() - start_timer),'min'))
                )
                print('\n')

            # save model --------------------------------------------------------------------
            if iter in iter_save:
                torch.save({
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, EFFICIENTNET_B5_MODEL_PATH  +'/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),  EFFICIENTNET_B5_MODEL_PATH + '/%08d_model.pth'%(iter))
                    pass


            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            net.train()
            input = input.cuda()
            truth_label = truth_label.cuda()
            truth_mask  = truth_mask.cuda()

            logit_mask = data_parallel(net, input)
            loss = criterion_mask(logit_mask, truth_mask)

            probability_mask  = F.softmax(logit_mask,1)
            probability_label = probability_mask_to_probability_label(probability_mask)
            tn,tp, num_neg,num_pos = metric_label(probability_label, truth_label)

            ((loss)/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  --------
            l = np.array([ loss.item()*batch_size,tn.sum(),*tp ])
            n = np.array([ batch_size, num_neg.sum(),*num_pos ])
            batch_loss      = l/(n+1e-8)
            sum_train_loss += l
            sum_train      += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0


            print('\r',end='',flush=True)
            asterisk = ' '
            print('%0.5f %5.1f%s %5.1f | %5.3f  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f]  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f] | %5.3f  [%0.2f : %0.2f %0.2f %0.2f %0.2f] | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss,
                         *batch_loss,
                         time_to_str((timer() - start_timer),'min'))
            , end='',flush=True)
            i=i+1

if __name__ == '__main__':
    print('                     |------------------------------------------- VALID------------------------------------------------|---------------------- TRAIN/BATCH ---------------------\n')
    print('rate     iter  epoch |  loss           [tn1,2,3,4  :  tp1,2,3,4]                    [dn1,2,3,4  :  dp1,2,3,4]          |  loss    [tn :  tp1,2,3,4]          | time             \n')
    print('\n')
    run_train()
    print('\nsucess!')