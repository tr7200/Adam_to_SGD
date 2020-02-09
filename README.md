# Adam2SGD

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
    
    
    AdamToSGD_ = [AdamToSGD(after_training_with_Adam=SWATS(x=train_x, 
                                                           y=train_y,
                                                           ...))]
                                                           
                                                           
    def SWATS(momentum=0.0,    # SGD optimizer arguments
              nesterov=False
              ...,
              loss='mse',      # compile arguments
              ...,
              x=None,          # model.fit statements
              y=None,
              ...):
        """
        This functions restarts training if condition 4 from 
        1712.07628 is satisfied in the callback
        """
        lr = float(K.get_value(model.optimizer.lr))
        bias_corrected_exponential_avg = lr / (1. - K.get_value(model.optimizer.beta_2))
     
        if (K.abs(bias_corrected_exponential_avg - lr) < 1e-9) is not None:
            return
        else:
            SGD_optimizer = SGD(lr=bia_corrected_exponential_avg,
                                ...)
         
            model.compile(optimizer=SGD_optimizer,
                          ...)
                       
            print('\nNow switching to SGD...\n')
         
            model.fit(x=x,
                      y=y,
                      ...)
      
     
    result = model.fit(train_x,
                       train_y,
                       callbacks=[AdamToSGD_, ...],
                       ...)


If condition (4) from arXiv 1712.07628 is satisfied, training will end early and
restart with the SGD optimizer using the last learning rate value from Adam before 
that condition.

This callback is more suitable for training with image or text data for hundreds of 
epochs.

`python setup.py install` to install.

MIT License
