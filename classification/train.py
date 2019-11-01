from datasets import *

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

def run_train():
    ## hyperparam and initialize  --------------------------------------
    batch_size = 4
    iter_accum = 8

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log = 500
    iter_valid  = 1500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 1500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0

    train_loss = np.zeros(20,np.float32)
    valid_loss = np.zeros(20,np.float32)
    batch_loss = np.zeros(20,np.float32)
    iter = 0
    i    = 0

    schduler = NullScheduler(lr=0.001)

    ## dataset -------------------------------------------------------------
    train_dataset, train_loader, valid_loader = get_train_valid_loader(batch_size)

    ## net ----------------------------------------
    net = Net().cuda()
    if INITIAL_CHECKPOINT is not None:
        state_dict = torch.load(INITIAL_CHECKPOINT, map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict,strict=False)
    else:
        net.load_pretrain(skip=['logit'])

    ## optimiser ----------------------------------
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    ## load model --------------------------------
    if INITIAL_CHECKPOINT is not None:
        initial_optimizer = INITIAL_CHECKPOINT.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
        pass

    ## run train ---------------------------------------------
    start = timer()
    while  iter<=num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = np.zeros(20,np.float32)

        optimizer.zero_grad()
        for t, (input, truth_mask, truth_label, infor) in enumerate(train_loader):
            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch

            if (iter % iter_valid==0):
                valid_loss = do_valid(net, valid_loader)

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                asterisk = '*' if iter in iter_save else ' '
                print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (\
                            rate, iter/1000, asterisk, epoch,
                            *valid_loss[:6],
                            *train_loss[:6],
                            time_to_str((timer() - start),'min'))
                )
                print('\n')

            # save model --------------------------------------------------------------------
            if iter in iter_save:
                torch.save(net.state_dict(), RESNET34_MODEL_PATH + '/%08d_model.pth'%(iter))
                torch.save({
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, RESNET34_MODEL_PATH + '/%08d_optimizer.pth'%(iter))
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

            logit =  net(input) #data_parallel(net,input)
            loss = criterion(logit, truth_label)

            tn,tp, num_neg,num_pos = metric_hit(logit, truth_label)

            (loss/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            l = np.array([ loss.item(), tn,*tp ])
            n = np.array([ batch_size, num_neg,*num_pos ])

            batch_loss[:6] = l
            sum_train_loss[:6] += l*n
            sum[:6] += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum+1e-12)
                sum_train_loss[...] = 0
                sum[...]            = 0


            print('\r',end='',flush=True)
            asterisk = ' '
            print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  |  %5.3f   %4.2f [%4.2f,%4.2f,%4.2f,%4.2f]  | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:6],
                         *batch_loss[:6],
                         time_to_str((timer() - start),'min'))
            , end='',flush=True)
            i=i+1

if __name__ == '__main__':
    print('rate     iter   epoch |  loss    tn, [tp1,tp2,tp3,tp4]       |  loss    tn, [tp1,tp2,tp3,tp4]       | time           ')
    print('--------------------------------------------------------------------------------------------------------------------\n')
    run_train()
    print('\nsucess!')
