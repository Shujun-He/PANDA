import numpy as np


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class lr_AIAYN():
    '''
    Learning rate scheduler from the paper:
    Attention is All You Need
    '''
    def __init__(self,optimizer,d_model,warmup_steps=4000,factor=1):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0
        self.factor=factor

    def step(self):
        self.step_num+=1
        lr=self.d_model**-0.5*np.min([self.step_num**-0.5,
                                      self.step_num*self.warmup_steps**-1.5])*self.factor
        update_lr(self.optimizer,lr)
