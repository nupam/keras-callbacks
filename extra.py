import keras
import numpy as np
from matplotlib import pyplot as plt
import os
import keras.backend as K
import math

class RecorderCallback(keras.callbacks.Callback):
    
    """
    parameters:
        alpha: float, smoothness factor, weight for exponetially weighted average to smooth loss
    	alpha must be bwtween [0,1)
    	for no smoothing us alpha=0

    keras callback to store training losses, learning rate and momentum (if applicable) during training of any model
    """
    
    def __init__(self, alpha=0.9):
        super(RecorderCallback, self).__init__()
        self.lr_list, self.loss_list, self.mom_list = [], [], []
        self.alpha = alpha
        
    def __clip__(self, x, low=0.025, high=0.975):
        x = np.clip(x, np.quantile(x, low), np.quantile(x, high))
        return x
    
    def on_train_begin(self, logs=None):
        if self.model.optimizer.__class__.__name__ in ('Adam', 'Nadam', 'Adamax'):
            self.store_mom = True
    
    def on_batch_end(self, batch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        loss = logs.get('loss')
        self.loss_list.append(loss)
        self.lr_list.append(lr)
        if self.store_mom:
            self.mom_list.append(K.get_value(self.model.optimizer.beta_1))
        
    def plot_losses(self, log=False, clip_losses=False):
        
        """
    	plots losses
    	parameters:
			log:  boolean, logscale for loss, pass parameter log=True
			clip_losses: boolean, to clip losses between 2.5 and 97.5 percentile, pass parameter clip_losses=True
    	"""
        
        y = self.loss_list
        tmp_y = [y[0]]
        for i in range(1, len(y)):
            tmp_y.append(self.alpha*tmp_y[i-1] + (1-self.alpha)*y[i])
        
        if clip_losses:
            tmp_y = self.__clip__(tmp_y)
            
        if log:
            plt.yscale('log')
            
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.plot(tmp_y)
        plt.show()
        
    def plot_lr(self):
        """
        plots learning rate w.r.t steps(batches)
        """
        plt.xlabel('steps')
        plt.ylabel('lr')
        x = self.lr_list
        plt.plot(x)
        plt.show()
        
    def plot_mom(self):
        """
        plots momentum (if applicable) w.r.t steps(batches)
        """
        if not self.store_mom:
            raise NotImplementedError('only implemented for Adam optimizer and derivatives\nunable to use with ', self.model.optimizer.__class__.__name__)
        plt.xlabel('steps')
        plt.ylabel('momentum')
        x = self.mom_list
        plt.plot(x)
        plt.show()
    
    def reset(self):
        self.lr_list, self.loss_list = [], []
        
class CyclicLRCallback(keras.callbacks.Callback):
    
    """
	Warning: This callback is implemented for Adam family of optimizers, i.e, Adam, Adamax, Nadam, with parameter beta_1 as momentum
	
	parameters:
		max_lr: maximum value of learning rate, if not provided fetched from optimizer
		min_lr: minimum value of learning rate
		cycles: number of cycles to repeat of CLR
		pct_start: ratio of cycle to increase the learning rate from min_lr to max_lr, remaining to decrease
		moms: momentum range to be used
		decay: decay value of max_lr after each cycle, max_lr after each cycle becomes max_lr*decay
			it is possible to use decay > 1, no warning will be issued

	keras callback for cyclic learning rate, for more details on working see original paper: https://arxiv.org/abs/1506.01186
	Learning rate is linearly increased to max_lr from zero in pct_start (start percentage-[0,1]) part of cycle then decreases to zero as cosine dunction in (1-pct_start) part of cycle.
	This is repeated as number of cycles.
	In similar manner momentum is decresed from moms[0] to moms[1]

	"""
    
    def __init__(self, max_lr=None, min_lr=K.epsilon(), cycles=1, pct_start = 0.3, moms=(0.95, 0.85), decay=1.0, verbose=None):
        super(CyclicLRCallback, self).__init__()
        assert cycles > 0
        assert max_lr is None or max_lr >= 0
        assert min_lr >= 0
        assert pct_start >= 0 and pct_start <= 1.0
        assert len(moms)==2 and moms[0] >= 0.0 and moms[0] <= 1.0 and moms[1] >= 0.0 and moms[1] <= 1.0
        assert decay > 0 ## it is possible to use decay > 1, no warning will be issued
        
        self.cycles = cycles
        self.min_lr = min_lr+K.epsilon()
        self.max_lr = max_lr
        self.moms = moms
        self.decay = decay
        self.pct_start = pct_start+K.epsilon()
        self.current_batch = 0
        self.verbose = verbose
        
        
    def on_train_begin(self, logs=None):
        if self.model.optimizer.__class__.__name__ not in ('Adam', 'Nadam', 'Adamax'):
            raise NotImplementedError('only implemented for Adam optimizer and derivatives\nunable to use with ', self.model.optimizer.__class__.__name__)
        self.verbose = self.verbose if self.verbose is not None else self.params['verbose']
        self.epochs = self.params['epochs']
        self.steps = self.params['steps'] if self.params['steps'] is not None else math.ceil(self.params['samples'])//self.params['batch_size']
        self.lr_original = float(K.get_value(self.model.optimizer.lr))
        self.max_lr = self.max_lr+K.epsilon() if self.max_lr is not None else float(K.get_value(self.model.optimizer.lr))
        self.steps_per_cycle = (self.steps*self.epochs)//self.cycles
        self.current_batch = 0
        self.beta_1_original =  float(K.get_value(self.model.optimizer.beta_1))
        
        if self.verbose: 
            print('epochs:', self.epochs, ', steps per cycle:', self.steps_per_cycle, ', total steps:',
              self.epochs*self.steps, ', cycles:', self.cycles, ', max_lr:', self.max_lr)
        
    def on_batch_begin(self, batch, logs=None):
        pct_start = self.pct_start
        step = self.current_batch%self.steps_per_cycle
        
        ##decay max_lr
        if step == 0:
            if self.current_batch > 0:
                self.max_lr *= self.decay
            if self.verbose and abs(self.decay-1.0) > K.epsilon() and self.current_batch//self.steps_per_cycle < self.cycles:
                print('\ncycle:', self.current_batch//self.steps_per_cycle, ', setting lr to:', self.max_lr)
            
            
        pct_now = step/self.steps_per_cycle
        increasing_lr =  pct_now <= pct_start
        
        up = pct_now/pct_start
        down = np.cos((pct_now-pct_start)/(1.-pct_start+K.epsilon()) *np.pi/2.)
        moms_diff = self.moms[0] - self.moms[1]
        
        lr_now = (increasing_lr*up + (not increasing_lr)*down)*self.max_lr
        curr_mom = self.moms[0] - increasing_lr*up*moms_diff - (not increasing_lr)*down*moms_diff
        
        
        ## update parameters, i.e, learning rate and momentum of optimizer
        K.set_value(self.model.optimizer.lr, lr_now)
        K.set_value(self.model.optimizer.beta_1, curr_mom)

        self.current_batch += 1
        
    def on_train_end(self, logs=None):
        """
        reset optimizer learning rate and momentum
        """
        K.set_value(self.model.optimizer.beta_1, self.beta_1_original)
        K.set_value(self.model.optimizer.lr, self.lr_original)
    
    
class LRFindCallback(keras.callbacks.Callback):
    
    """
	parameters:
		max_lr: float, maximum value of learning rate to test on
		min_lr: float, manimum value of learning rate to test on
		max_epochs: integer, maximum number of epochs to run test upto
		multiplier: float, ratio to increase learning rate from previous step
		max_loss: float, maximum loss of model till which learning could be increased

	keras callback which gradually increases leraning rate from min_lr to max_lr and records loss at each step
	Stores learning rate nad respective loss in python lists lr_list, loss_list respectively
	Uses disk to write temporary model weights, read write permission and enough disk space is required
	"""
    
    def __init__(self, max_lr=1., min_lr=8e-5, max_epochs=10, multiplier=1.015, max_loss=None):
        super(LRFindCallback, self).__init__()
        assert max_epochs >= 1
        assert max_lr >= 0
        assert min_lr >= 0
        assert multiplier > 1
        
        self.max_epochs = max_epochs
        self.min_lr, self.max_lr = min_lr + K.epsilon(), max_lr + K.epsilon()
        self.multiplier = multiplier + K.epsilon()
        self.lr_list, self.loss_list = [], []
        self.tmp_filename = 'temporary_model_weights_do_not_delete_lrfindcallback.h5'
        self.max_loss = max_loss
#         print('multiplier ',self.multiplier)
#         print('initial lr ', self.min_lr)
        
    def on_train_begin(self, batch, logs=None):
        ##save state
        try:
            self.model.save_weights(self.tmp_filename)
        except:
            raise OSError('unable to save weights as temporary file on disk')
        self.original_lr = float(K.get_value(self.model.optimizer.lr))
        #print('original_lr ', self.original_lr)
        
        K.set_value(self.model.optimizer.lr, self.min_lr)
    
    def on_batch_end(self, batch, logs=None):
        lr_now = float(K.get_value(self.model.optimizer.lr))
        #print(lr_now)
        loss = logs.get('loss')
        if self.max_loss is None:
            self.max_loss = 10*loss
         
        ##save loss and lr
        self.loss_list.append(loss)
        self.lr_list.append(lr_now)
        
        ##stopping condition
        if np.isnan(loss) or np.isinf(loss) or loss > self.max_loss or lr_now > self.max_lr:
            print('stopping')
            #print('status ',np.isnan(loss), np.isinf(loss), loss > self.max_loss, lr_now)
            self.model.stop_training = True
            
        lr_now *= self.multiplier
        
        K.set_value(self.model.optimizer.lr, lr_now)
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.max_epochs:
            self.model.load_weights(self.tmp_filename)
            K.set_value(self.model.optimizer.lr, self.original_lr)
            self.model.stop_training = True
            os.remove(self.tmp_filename)
            
    def on_train_end(self, logs=None):
        self.model.load_weights(self.tmp_filename)
        #print('original_lr ', self.original_lr)
        K.set_value(self.model.optimizer.lr, self.original_lr)
        #print('lr after: ', K.get_value(self.model.optimizer.lr))
        self.model.stop_training = True
        os.remove(self.tmp_filename)
        
    def reset(self):
        self.lr_list, self.loss_list = [], []
        
        
        
        
def lr_find(model, data, generator=False, max_epochs = 10, steps_per_epoch=None, alpha=0.9, logloss=True, clip_loss=False, **kwargs):
    
    
    """
		uses LRFindCallback defined above to plot lr vs loss graph
		parameters:
			model: keras model object to test on
			data: numpy arrays (x, y) or data_generator yeilding mini-batches as such
			max_epochs: maximum number of epochs run test to
			steps_per_epoch: number of steps to take per epoch, only uses when generator=True is provided
			alpha: shooting factor(parameter for smoothing loss, use 0 for no smoothing)
			logloss: plots loss in logarithmic scale
			clip_loss: clips loss between 2.5 and 97.5 percentile
			max_lr: maximum value of learning rate to test on
			min_lr: manimum value of learning rate to test on
			multiplier: ratio to increase learning rate from previous step
			max_loss: maximum loss of model till which learning could be increased

	"""
    
    lr_cb = LRFindCallback(max_epochs=max_epochs, **kwargs)
    if generator:
        model.fit_generator(data, steps_per_epoch=steps_per_epoch, epochs=max_epochs, callbacks=[lr_cb])
    else:
        model.fit(data[0], data[1], epochs=max_epochs, callbacks=[lr_cb])
    
    lr, loss = lr_cb.lr_list, lr_cb.loss_list
    for i in range(1, len(lr)):
        loss[i] = alpha*loss[i-1] + (1-alpha)*loss[i]
        
    if clip_loss:
         loss = np.clip(loss, np.quantile(loss, 0.5), np.quantile(loss, 0.95))
    
    plt.xlabel('lr')
    plt.ylabel('loss')
    if logloss:
        plt.yscale('log')
    plt.xscale('log')
    plt.plot(lr, loss)
