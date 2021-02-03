#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Nov 2019 with Keras 2.2.4 and Tensorflow 1.13

AdamToSGD

Modified version of Keras' EarlyStopping callback that switches to the SGD
optimizer from Adam following arXiv 1712.07628:

    Keskar, N. S., & Socher, R. (2017).
    Improving generalization performance by switching from adam to sgd.
    arXiv preprint arXiv:1712.07628.
       
- The callback monitors learning rate according to (4) from arXiv 1712.07628
- If (4) from that paper is satisfied, the callback stops training early and
starts training using separate SWATS function (Switching from Adam To SGD) with
SGD optimizer that uses the learning rate that satisfied (4).
       
Usage:

    model = Sequential()
    ...
    model.compile(...)    
    
    AdamToSGD_ = [AdamToSGD(on_train_end=SWATS(x=train_x, y=train_y, ...))]
    
    def SWATS(x=train_x,
              y=train_y,
              ...,
              *args,
              **kwargs)
    
    result = model.fit(train_x,
                       train_y,
                       batch_size=16,
                       epochs=4,
                       verbose=1,
                       callbacks=[AdamToSGD_],
                       validation_split=0.05)

MIT License
"""



from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import keras.backend as K


class AdamToSGD(EarlyStopping):
    """Modified version of EarlyStopping callback that switches from SGD to Adam.
   
    Args:
        SWATS (fcn): Function that re-compiles model with SGD optimizer
        and restarts training (see documentation for example)
        
    Returns:
        Restarted training using SGD optimizer once conditions are met
    """

    def __init__(self, on_train_end, **kwargs):
        self.on_train_end = SWATS
        super(AdamToSGD, self).__init__(**kwargs)
        
    def on_epoch_end(self, epoch, logs={}):
        lr = self.model.optimizer.lr
        bias_corrected_exponential_average = lr / (1. - self.model.optimizer.beta_2)
        if (K.abs(bias_corrected_exponential_average - lr) < 1e-9) is not None:
            if self.restore_best_weights:
                 self.best_weights = self.model.get_weights()
        else:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of the best epoch')
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        super(AdamToSGD, self).on_train_end()
        self.on_train_end()
