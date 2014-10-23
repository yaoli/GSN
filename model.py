import numpy, os, sys, cPickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time
import argparse

cast32      = lambda x : numpy.cast['float32'](x)
trunc       = lambda x : str(x)[:8]
logit       = lambda p : numpy.log(p / (1 - p) )
binarize    = lambda x : cast32(x >= 0.5)
sigmoid     = lambda x : cast32(1. / (1 + numpy.exp(-x)))

def SaltAndPepper(X, rate=0.3):
    # Salt and pepper noise
    
    drop = numpy.arange(X.shape[1])
    numpy.random.shuffle(drop)
    sep = int(len(drop)*rate)
    drop = drop[:sep]
    X[:, drop[:sep/2]]=0
    X[:, drop[sep/2:]]=1
    return X

def get_shared_weights(n_in, n_out, interval, name):
    #val = numpy.random.normal(0, sigma_sqr, size=(n_in, n_out))
    val = numpy.random.uniform(-interval, interval, size=(n_in, n_out))
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def get_shared_bias(n, name, offset = 0):
    val = numpy.zeros(n) - offset
    val = cast32(val)
    val = theano.shared(value = val, name = name)
    return val

def load_mnist(path):
    data = cPickle.load(open(os.path.join(path,'mnist.pkl'), 'r'))
    return data

def load_mnist_binary(path):
    data = cPickle.load(open(os.path.join(path,'mnist.pkl'), 'r'))
    data = [list(d) for d in data] 
    data[0][0] = (data[0][0] > 0.5).astype('float32')
    data[1][0] = (data[1][0] > 0.5).astype('float32')
    data[2][0] = (data[2][0] > 0.5).astype('float32')
    data = tuple([tuple(d) for d in data])
    return data
    
def load_tfd(path):
    import scipy.io as io
    data = io.loadmat(os.path.join(path, 'TFD_48x48.mat'))
    X = cast32(data['images'])/cast32(255)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    labels  = data['labs_ex'].flatten()
    labeled = labels != -1
    unlabeled   =   labels == -1  
    train_X =   X[unlabeled]
    valid_X =   X[unlabeled][:100] # Stuf
    test_X  =   X[labeled]

    del data

    return (train_X, labels[unlabeled]), (valid_X, labels[unlabeled][:100]), (test_X, labels[labeled])

def experiment(state, channel):
    if state.test_model and 'config' in os.listdir('.'):
        print 'Loading local config file'
        config_file =   open('config', 'r')
        config      =   config_file.readlines()
        try:
            config_vals =   config[0].split('(')[1:][0].split(')')[:-1][0].split(', ')
        except:
            config_vals =   config[0][3:-1].replace(': ','=').replace("'","").split(', ')
            config_vals =   filter(lambda x:not 'jobman' in x and not '/' in x and not ':' in x and not 'experiment' in x, config_vals)
        
        for CV in config_vals:
            print CV
            if CV.startswith('test'):
                print 'Do not override testing switch'
                continue        
            try:
                exec('state.'+CV) in globals(), locals()
            except:
                exec('state.'+CV.split('=')[0]+"='"+CV.split('=')[1]+"'") in globals(), locals()
     


    else:
        # Save the current configuration
        # Useful for logs/experiments
        print 'Saving config'
        f = open('config', 'w')
        f.write(str(state))
        f.close()


    print state
    # Load the data, train = train+valid, and shuffle train
    # Targets are not used (will be misaligned after shuffling train
    if state.dataset == 'MNIST':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist(state.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
        
    elif state.dataset == 'MNIST_binary':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist_binary(state.data_path)
        train_X = numpy.concatenate((train_X, valid_X))
        
    elif state.dataset == 'TFD':
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_tfd(state.data_path)
    
    N_input =   train_X.shape[1]
    root_N_input = numpy.sqrt(N_input)
    numpy.random.seed(1)
    numpy.random.shuffle(train_X)
    train_X = theano.shared(train_X)
    valid_X = theano.shared(valid_X)
    test_X  = theano.shared(test_X)

    # Theano variables and RNG
    X       = T.fmatrix()
    index   = T.lscalar()
    MRG = RNG_MRG.MRG_RandomStreams(1)
    
    # Network and training specifications
    K               =   state.K # N hidden layers
    N               =   state.N # number of walkbacks 
    layer_sizes     =   [N_input] + [state.hidden_size] * K # layer sizes, from h0 to hK (h0 is the visible layer)
    learning_rate   =   theano.shared(cast32(state.learning_rate))  # learning rate
    annealing       =   cast32(state.annealing) # exponential annealing coefficient
    momentum        =   theano.shared(cast32(state.momentum)) # momentum term

    # THEANO VARIABLES
    X       = T.fmatrix()   # Input of the graph
    index   = T.lscalar()   # index to minibatch
    MRG = RNG_MRG.MRG_RandomStreams(1)
 

    # PARAMETERS : weights list and bias list.
    # initialize a list of weights and biases based on layer_sizes
    weights_list    =   [get_shared_weights(layer_sizes[i], layer_sizes[i+1], numpy.sqrt(6. / (layer_sizes[i] + layer_sizes[i+1] )), 'W') for i in range(K)]
    bias_list       =   [get_shared_bias(layer_sizes[i], 'b') for i in range(K + 1)]

    if state.test_model:
        # Load the parameters of the last epoch
        # maybe if the path is given, load these specific attributes 
        param_files     =   filter(lambda x:'params' in x, os.listdir('.'))
        max_epoch_idx   =   numpy.argmax([int(x.split('_')[-1].split('.')[0]) for x in param_files])
        params_to_load  =   param_files[max_epoch_idx]
        PARAMS = cPickle.load(open(params_to_load,'r'))
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[:len(weights_list)], weights_list)]
        [p.set_value(lp.get_value(borrow=False)) for lp, p in zip(PARAMS[len(weights_list):], bias_list)]

    # Util functions
    def dropout(IN, p = 0.5):
        noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
        OUT     =   (IN * noise) / cast32(p)
        return OUT

    def add_gaussian_noise(IN, std = 1):
        print 'GAUSSIAN NOISE : ', std
        noise   =   MRG.normal(avg  = 0, std  = std, size = IN.shape, dtype='float32')
        OUT     =   IN + noise
        return OUT

    def corrupt_input(IN, p = 0.5):
        # salt and pepper? masking?
        noise   =   MRG.binomial(p = p, n = 1, size = IN.shape, dtype='float32')
        IN      =   IN * noise
        return IN

    def salt_and_pepper(IN, p = 0.2):
        # salt and pepper noise
        print 'DAE uses salt and pepper noise'
        a = MRG.binomial(size=IN.shape, n=1,
                              p = 1 - p,
                              dtype='float32')
        b = MRG.binomial(size=IN.shape, n=1,
                              p = 0.5,
                              dtype='float32')
        c = T.eq(a,0) * b
        return IN * a + c

    # Odd layer update function
    # just a loop over the odd layers
    def update_odd_layers(hiddens, noisy):
        for i in range(1, K+1, 2):
            print i
            if noisy:
                simple_update_layer(hiddens, None, i)
            else:
                simple_update_layer(hiddens, None, i, add_noise = False)

    # Even layer update
    # p_X_chain is given to append the p(X|...) at each update (one update = odd update + even update)
    def update_even_layers(hiddens, p_X_chain, noisy):
        for i in range(0, K+1, 2):
            print i
            if noisy:
                simple_update_layer(hiddens, p_X_chain, i)
            else:
                simple_update_layer(hiddens, p_X_chain, i, add_noise = False)

    # The layer update function
    # hiddens   :   list containing the symbolic theano variables [visible, hidden1, hidden2, ...]
    #               layer_update will modify this list inplace
    # p_X_chain :   list containing the successive p(X|...) at each update
    #               update_layer will append to this list
    # add_noise     : pre and post activation gaussian noise

    def simple_update_layer(hiddens, p_X_chain, i, add_noise=True):
        # Compute the dot product, whatever layer
        post_act_noise  =   0

        if i == 0:
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + bias_list[i]           

        elif i == K:
            hiddens[i]  =   T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]
            
        else:
            # next layer        :   layers[i+1], assigned weights : W_i
            # previous layer    :   layers[i-1], assigned weights : W_(i-1)
            hiddens[i]  =   T.dot(hiddens[i+1], weights_list[i].T) + T.dot(hiddens[i-1], weights_list[i-1]) + bias_list[i]

        # Add pre-activation noise if NOT input layer
        if i==1 and state.noiseless_h1:
            print '>>NO noise in first layer'
            add_noise   =   False

        # pre activation noise            
        if i != 0 and add_noise:
            print 'Adding pre-activation gaussian noise'
            hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)
       
        # ACTIVATION!
        if i == 0:
            print 'Sigmoid units'
            hiddens[i]  =   T.nnet.sigmoid(hiddens[i])
        else:
            print 'Hidden units'
            hiddens[i]  =   hidden_activation(hiddens[i])

        # post activation noise            
        if i != 0 and add_noise:
            print 'Adding post-activation gaussian noise'
            hiddens[i]  =   add_gaussian_noise(hiddens[i], state.hidden_add_noise_sigma)

        # build the reconstruction chain
        if i == 0:
            # if input layer -> append p(X|...)
            p_X_chain.append(hiddens[i])
            
            # sample from p(X|...)
            if state.input_sampling:
                print 'Sampling from input'
                sampled     =   MRG.binomial(p = hiddens[i], size=hiddens[i].shape, dtype='float32')
            else:
                print '>>NO input sampling'
                sampled     =   hiddens[i]
            # add noise
            sampled     =   salt_and_pepper(sampled, state.input_salt_and_pepper)
            
            # set input layer
            hiddens[i]  =   sampled

    def update_layers(hiddens, p_X_chain, noisy = True):
        print 'odd layer update'
        update_odd_layers(hiddens, noisy)
        print
        print 'even layer update'
        update_even_layers(hiddens, p_X_chain, noisy)

 
    ''' F PROP '''
    #X = T.fmatrix()
    if state.act == 'sigmoid':
        print 'Using sigmoid activation'
        hidden_activation = T.nnet.sigmoid
    elif state.act == 'rectifier':
        print 'Using rectifier activation'
        hidden_activation = lambda x : T.maximum(cast32(0), x)
    elif state.act == 'tanh':
        hidden_activation = lambda x : T.tanh(x)    
   
    
    ''' Corrupt X '''
    X_corrupt   = salt_and_pepper(X, state.input_salt_and_pepper)

    ''' hidden layer init '''

    hiddens     = [X_corrupt]
    p_X_chain   = [] 
    print "Hidden units initialization"
    for w,b in zip(weights_list, bias_list[1:]):
        # init with zeros
        print "Init hidden units at zero before creating the graph"
        hiddens.append(T.zeros_like(T.dot(hiddens[-1], w)))

    # The layer update scheme
    print "Building the graph :", N,"updates"
    for i in range(N):
        update_layers(hiddens, p_X_chain)
    

    # COST AND GRADIENTS    

    print 'Cost w.r.t p(X|...) at every step in the graph'
    #COST        =   T.mean(T.nnet.binary_crossentropy(reconstruction, X))
    #COST        =   [T.mean(T.nnet.binary_crossentropy(rX, X)) for rX in p_X_chain]
    COST = [T.mean(T.sqr(rX-X)) for rX in p_X_chain]
    show_COST   =   COST[-1] 
    COST        =   numpy.sum(COST)
    #COST = T.mean(COST)
    
    params          =   weights_list + bias_list
    
    gradient        =   T.grad(COST, params)
                
    gradient_buffer =   [theano.shared(numpy.zeros(x.get_value().shape, dtype='float32')) for x in params]
    
    m_gradient      =   [momentum * gb + (cast32(1) - momentum) * g for (gb, g) in zip(gradient_buffer, gradient)]
    g_updates       =   [(p, p - learning_rate * mg) for (p, mg) in zip(params, m_gradient)]
    b_updates       =   zip(gradient_buffer, m_gradient)
        
    updates         =   OrderedDict(g_updates + b_updates)
    
    f_cost      =   theano.function(inputs = [X], outputs = show_COST)
    
    indexed_batch   = train_X[index * state.batch_size : (index+1) * state.batch_size]
    sampled_batch   = MRG.binomial(p = indexed_batch, size = indexed_batch.shape, dtype='float32')
    
    f_learn     =   theano.function(inputs  = [index], 
                                    updates = updates, 
                                    givens  = {X : indexed_batch},
                                    outputs = show_COST)
    
    f_test      =   theano.function(inputs  =   [X],
                                    outputs =   [X_corrupt] + hiddens[0] + p_X_chain,
                                    on_unused_input = 'warn')


    #############
    # Denoise some numbers  :   show number, noisy number, reconstructed number
    #############
    import random as R
    R.seed(1)
    random_idx      =   numpy.array(R.sample(range(len(test_X.get_value())), 100))
    numbers         =   test_X.get_value()[random_idx]
    
    f_noise = theano.function(inputs = [X], outputs = salt_and_pepper(X, state.input_salt_and_pepper))
    noisy_numbers   =   f_noise(test_X.get_value()[random_idx])

    # Recompile the graph without noise for reconstruction function
    hiddens_R     = [X]
    p_X_chain_R   = []

    for w,b in zip(weights_list, bias_list[1:]):
        # init with zeros
        hiddens_R.append(T.zeros_like(T.dot(hiddens_R[-1], w)))

    # The layer update scheme
    for i in range(N):
        update_layers(hiddens_R, p_X_chain_R, noisy=False)

    f_recon = theano.function(inputs = [X], outputs = p_X_chain_R[-1]) 


    ############
    # Sampling #
    ############
    
    # the input to the sampling function
    network_state_input     =   [X] + [T.fmatrix() for i in range(K)]
   
    # "Output" state of the network (noisy)
    # initialized with input, then we apply updates
    #network_state_output    =   network_state_input
    
    network_state_output    =   [X] + network_state_input[1:]

    visible_pX_chain        =   []

    # ONE update
    update_layers(network_state_output, visible_pX_chain, noisy=True)

    if K == 1: 
        f_sample_simple = theano.function(inputs = [X], outputs = visible_pX_chain[-1])
    
    
    # WHY IS THERE A WARNING????
    # because the first odd layers are not used -> directly computed FROM THE EVEN layers
    # unused input = warn
    f_sample2   =   theano.function(inputs = network_state_input, outputs = network_state_output + visible_pX_chain, on_unused_input='warn')

    def sample_some_numbers_single_layer():
        x0    =   test_X.get_value()[:1]
        samples = [x0]
        x  =   f_noise(x0)
        for i in range(399):
            x = f_sample_simple(x)
            samples.append(x)
            x = numpy.random.binomial(n=1, p=x, size=x.shape).astype('float32')
            x = f_noise(x)
        return numpy.vstack(samples)
            
    def sampling_wrapper(NSI):
        out             =   f_sample2(*NSI)
        NSO             =   out[:len(network_state_output)]
        vis_pX_chain    =   out[len(network_state_output):]
        return NSO, vis_pX_chain

    def sample_some_numbers(N=400):
        # The network's initial state
        init_vis    =   test_X.get_value()[:1]

        noisy_init_vis  =   f_noise(init_vis)

        network_state   =   [[noisy_init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [noisy_init_vis]

        for i in range(N-1):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)
    
    def plot_samples(epoch_number):
        to_sample = time.time()
        if K == 1:
            # one layer model
            V = sample_some_numbers_single_layer()
        else:
            V, H0 = sample_some_numbers()
        img_samples =   PIL.Image.fromarray(tile_raster_images(V, (root_N_input,root_N_input), (20,20)))
        
        fname       =   'samples_epoch_'+str(epoch_number)+'.png'
        img_samples.save(fname) 
        print 'Took ' + str(time.time() - to_sample) + ' to sample 400 numbers'
   
    ##############
    # Inpainting #
    ##############
    def inpainting(digit):
        # The network's initial state

        # NOISE INIT
        init_vis    =   cast32(numpy.random.uniform(size=digit.shape))

        #noisy_init_vis  =   f_noise(init_vis)
        #noisy_init_vis  =   cast32(numpy.random.uniform(size=init_vis.shape))

        # INDEXES FOR VISIBLE AND NOISY PART
        noise_idx = (numpy.arange(N_input) % root_N_input < (root_N_input/2))
        fixed_idx = (numpy.arange(N_input) % root_N_input > (root_N_input/2))
        # function to re-init the visible to the same noise

        # FUNCTION TO RESET HALF VISIBLE TO DIGIT
        def reset_vis(V):
            V[0][fixed_idx] =   digit[0][fixed_idx]
            return V
        
        # INIT DIGIT : NOISE and RESET HALF TO DIGIT
        init_vis = reset_vis(init_vis)

        network_state   =   [[init_vis] + [numpy.zeros((1,len(b.get_value())), dtype='float32') for b in bias_list[1:]]]

        visible_chain   =   [init_vis]

        noisy_h0_chain  =   [init_vis]

        for i in range(49):
           
            # feed the last state into the network, compute new state, and obtain visible units expectation chain 
            net_state_out, vis_pX_chain =   sampling_wrapper(network_state[-1])


            # reset half the digit
            net_state_out[0] = reset_vis(net_state_out[0])
            vis_pX_chain[0]  = reset_vis(vis_pX_chain[0])

            # append to the visible chain
            visible_chain   +=  vis_pX_chain

            # append state output to the network state chain
            network_state.append(net_state_out)
            
            noisy_h0_chain.append(net_state_out[0])

        return numpy.vstack(visible_chain), numpy.vstack(noisy_h0_chain)
    
  
  
   
    
    def save_params(n, params):
        print 'saving parameters...'
        save_path = 'params_epoch_'+str(n)+'.pkl'
        f = open(save_path, 'wb')
        try:
            cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        finally:
            f.close() 

    # TRAINING
    n_epoch     =   state.n_epoch
    batch_size  =   state.batch_size
    STOP        =   False
    counter     =   0

    train_costs =   []
    valid_costs =   []
    test_costs  =   []
    
    if state.vis_init:
        bias_list[0].set_value(logit(numpy.clip(0.9,0.001,train_X.get_value().mean(axis=0))))

    if state.test_model:
        # If testing, do not train and go directly to generating samples, parzen window estimation, and inpainting
        print 'Testing : skip training'
        STOP    =   True


    while not STOP:
        counter     +=  1
        t = time.time()
        print counter,'\t',

        #train
        train_cost  =   []
        for i in range(len(train_X.get_value(borrow=True)) / batch_size):
            #train_cost.append(f_learn(train_X[i * batch_size : (i+1) * batch_size]))
            #training_idx = numpy.array(range(i*batch_size, (i+1)*batch_size), dtype='int32')
            train_cost.append(f_learn(i))
        train_cost = numpy.mean(train_cost) 
        train_costs.append(train_cost)
        print 'Train : ',trunc(train_cost), '\t',


        #valid
        valid_cost  =   []
        for i in range(len(valid_X.get_value(borrow=True)) / 100):
            valid_cost.append(f_cost(valid_X.get_value()[i * 100 : (i+1) * batch_size]))
        valid_cost = numpy.mean(valid_cost)
        #valid_cost  =   123
        valid_costs.append(valid_cost)
        print 'Valid : ', trunc(valid_cost), '\t',

        #test
        test_cost  =   []
        for i in range(len(test_X.get_value(borrow=True)) / 100):
            test_cost.append(f_cost(test_X.get_value()[i * 100 : (i+1) * batch_size]))
        test_cost = numpy.mean(test_cost)
        test_costs.append(test_cost)
        print 'Test  : ', trunc(test_cost), '\t',

        if counter >= n_epoch:
            STOP = True

        print 'time : ', trunc(time.time() - t),

        print 'MeanVisB : ', trunc(bias_list[0].get_value().mean()),
        
        print 'W : ', [trunc(abs(w.get_value(borrow=True)).mean()) for w in weights_list]

        if (counter % 5) == 0:
            # Checking reconstruction
            reconstructed   =   f_recon(noisy_numbers) 
            # Concatenate stuff
            stacked         =   numpy.vstack([numpy.vstack([numbers[i*10 : (i+1)*10], noisy_numbers[i*10 : (i+1)*10], reconstructed[i*10 : (i+1)*10]]) for i in range(10)])
        
            number_reconstruction   =   PIL.Image.fromarray(tile_raster_images(stacked, (root_N_input,root_N_input), (10,30)))
            #epoch_number    =   reduce(lambda x,y : x + y, ['_'] * (4-len(str(counter)))) + str(counter)
            number_reconstruction.save('number_reconstruction'+str(counter)+'.png')
    
            #sample_numbers(counter, 'seven')
            plot_samples(counter)
    
            #save params
            save_params(counter, params)
     
        # ANNEAL!
        new_lr = learning_rate.get_value() * annealing
        learning_rate.set_value(new_lr)

    # Save
    state.train_costs = train_costs
    state.valid_costs = valid_costs
    state.test_costs = test_costs

    # if test

    # 10k samples
    print 'Generating 10,000 samples'
    samples, _  =   sample_some_numbers(N=10000)
    f_samples   =   'samples.npy'
    numpy.save(f_samples, samples)
    print 'saved digits'


    # parzen
    print 'Evaluating parzen window'
    import likelihood_estimation_parzen
    likelihood_estimation_parzen.main(0.20,'mnist') 

    # Inpainting
    print 'Inpainting'
    test_X  =   test_X.get_value()

    numpy.random.seed(2)
    test_idx    =   numpy.arange(len(test_Y))

    for Iter in range(10):

        numpy.random.shuffle(test_idx)
        test_X = test_X[test_idx]
        test_Y = test_Y[test_idx]

        digit_idx = [(test_Y==i).argmax() for i in range(10)]
        inpaint_list = []

        for idx in digit_idx:
            DIGIT = test_X[idx:idx+1]
            V_inpaint, H_inpaint = inpainting(DIGIT)
            inpaint_list.append(V_inpaint)

        INPAINTING  =   numpy.vstack(inpaint_list)

        plot_inpainting =   PIL.Image.fromarray(tile_raster_images(INPAINTING, (root_N_input,root_N_input), (10,50)))

        fname   =   'inpainting_'+str(Iter)+'.png'
        #fname   =   os.path.join(state.model_path, fname)

        plot_inpainting.save(fname)

        if False and __name__ ==  "__main__":
            os.system('eog inpainting.png')
 



    if __name__ == '__main__':
        import ipdb; ipdb.set_trace()
    
    return 
