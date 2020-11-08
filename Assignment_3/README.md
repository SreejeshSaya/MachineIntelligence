## Designing Artificial Neural Networks for classification of LBW Cases from Scratch


### What's left!?
- [ ] Number of layers, number of neurons in each layer, number of iterations, learning rate
- [ ] ~~Creating a layer class - (looks like there is no need)~~
- [ ] Choice of error and activation functions - later ???
- [x] regularization -- (L2 reg used)
- [x] initialisation of weights -- (He & Xavier init used)
- [ ] ~~Batch vs Stochastic gradient descent vs MiniBatch Gradient descent~~
- [ ] RMS Prop/Adam optimization algo on mini batches - later ????
- [x] Normalization ()
- [x] Control verbose & actvn func
- [x] Moved testing to main.py
- [ ] Clean/Put comments
- [ ] *SettingWithCopyWarning, as of for now used ignore in Neural_Net*
- [ ] DivBy0 might arise in confusion matrix computation, but DivBy0 in error computation is taken care of (error initialised to 0 or l2,to avoid DivBy0,try-except)
