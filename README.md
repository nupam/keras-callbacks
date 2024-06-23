# keras-callbacks
## Contains callbacks for cyclic learning rate, recording loss/lr/momentum, finding lr
### For example usage see example_usage.ipynb(crazy high lr is to exaggerate decay)
### This module is inspired from fastaiv1 learner and cyclic learning rate scheduler

This module is tested on keras 2.2.4 with tensorflow backend<br><br>
Callbacks and functions defined:

1. RecorderCallback:<br>
keras callback to store training losses, learning rate and momentum (if applicable) during training of any model<br>
    parameters:<br>
     > alpha: float, smoothness factor, weight for exponetially weighted average to smooth loss<br>
    	  alpha must be between \[0,1)<br>
    	  for no smoothing us alpha=0<br>

 2. CyclicLRCallback:<br>
	  Warning: This callback is only implemented for Adam family of optimizers, i.e, Adam, Adamax, Nadam, with parameter beta_1 as momentum<br>
    keras callback for cyclic learning rate.<br>For more details on working see original paper: https://arxiv.org/abs/1506.01186<br>
    This callback also features auto decay option that will decay learning rate after patience cycles if no improvement in monitored metric/loss is observed, In such case epochs must be multiple of cycles.<br>
	<br>Learning rate is linearly increased to max_lr from zero in pct_start (start percentage-[0,1]) part of cycle then decreases to zero as cosine function in (1-pct_start) part of cycle.
	<br>This is repeated as number of cycles.
	<br>In similar manner momentum is decresed from moms[0] to moms[1]<br>
	parameters:<br>
    > max_lr: maximum value of learning rate, if not provided fetched from optimizer<br>
		min_lr: minimum value of learning rate<br>
		cycles: number of cycles to repeat of CLR<br>
		auto_decay: set True to decay lr automatically at end of cycle<br>
		patience: number of cycles to wait before decaying lr<br>
		monitor: monitered metric/loss for auto decay<br>
		pct_start: ratio of cycle to increase the learning rate from min_lr to max_lr, remaining to decrease<br>
		moms: momentum range to be used<br>
		decay: decay value of max_lr after each cycle, max_lr after each cycle becomes max_lr\*decay<br> it is possible to use decay > 1, no warning will be issued<br>
    
  3. LRFindCallback:<br>
  keras callback which gradually increases leraning rate from min_lr to max_lr and records loss at each step<br>
	Stores learning rate nad respective loss in python lists lr_list, loss_list respectively<br
	Uses disk to write temporary model weights, read write permission and enough disk space is required<br>
	parameters:<br>
		> max_lr: float, maximum value of learning rate to test on<br>
		min_lr: float, manimum value of learning rate to test on<br>
		max_epochs: integer, maximum number of epochs to run test upto<br>
		multiplier: float, ratio to increase learning rate from previous step<br>
		max_loss: float, maximum loss of model till which learning could be increased<br>

	
	
  4. lr_find:<br>
  Function<br>
   uses LRFindCallback defined above to plot lr vs loss graph<br>
   parameters:<br>
		> model: keras model object to test on<br>
			data: numpy arrays (x, y) or data_generator yeilding mini-batches as such<br>
			max_epochs: maximum number of epochs run test to<br>
			steps_per_epoch: number of steps to take per epoch, only uses when generator=True is provided<br>
			batch_size: batchsize to use in model.fit, not applicable if generator is used<br>
			alpha: shooting factor(parameter for smoothing loss, use 0 for no smoothing)<br>
			logloss: plots loss in logarithmic scale<br>
			clip_loss: clips loss between 2.5 and 97.5 percentile<br>
			max_lr: maximum value of learning rate to test on<br>
			min_lr: manimum value of learning rate to test on<br>
			multiplier: ratio to increase learning rate from previous step<br>
			max_loss: maximum loss of model till which learning could be increased<br>

It was a great learning experiecne <3
