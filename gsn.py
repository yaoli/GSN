import numpy, os, sys, cPickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
from theano.printing import pprint, debugprint
from tools import RAB_tools
from tools.RAB_tools import apply_act, constantX
import PIL.Image
from collections import OrderedDict
from image_tiler import *
import time
import argparse
import utils
from collections import OrderedDict

floatX = 'float32'

class GSN(object):
    def __init__(self, state, channel):
        model_config = state.GSN
        self.n_in = model_config.n_in
        self.n_out = model_config.n_out
        self.n_hiddens = model_config.n_hiddens
        self.n_hidden_layers = len(self.n_hiddens)
        self.rng_numpy, self.rng_theano = RAB_tools.get_two_rngs()
        self.act = model_config.hidden_act
        self.init_weights = model_config.init_weights
        self.input_noise_level = model_config.input_noise_level
        self.n_hprop = model_config.n_hprop
        self.noiseless_h1 = model_config.noiseless_h1
        self.hidden_add_noise_sigma = model_config.hidden_add_noise_sigma
        self.add_noise_to_hiddens = model_config.add_noise_to_hiddens
        self.input_sampling = model_config.input_sampling
        
        train_config = state.GSN.train
        self.n_epochs = train_config.n_epochs
        self.sgd_type = train_config.sgd_type
        self.adadelta_epsilon = train_config.adadelta_epsilon
        self.lr = train_config.lr
        self.lr_ts = theano.shared(numpy.float32(self.lr), name='learning_rate')
        self.momentum = train_config.momentum
        self.lr_decrease = self.lr / self.n_epochs
        self.minibatch_size = train_config.minibatch_size
        self.valid_freq = train_config.valid_freq
        self.channel = channel
        self.verbose = train_config.verbose
        self.save_model_path = state.save_model_path
        RAB_tools.create_dir_if_not_exist(self.save_model_path)

        self.costs = []

    def build_theano_fn(self):
        self.x = T.fmatrix('x')
        self.x.tag.test_value = numpy.random.normal(
            size=(self.minibatch_size, self.n_in)).astype(floatX)
        layer_sizes = [self.n_in] + self.n_hiddens
        # init params
        self.Ws = []
        for i in range(self.n_hidden_layers):
            W = RAB_tools.build_weights(
                n_row=layer_sizes[i], n_col=layer_sizes[i+1], style=self.init_weights,
                name='W%d'%i, rng_numpy=self.rng_numpy)
            self.Ws.append(W)
        self.bs = []
        for i in range(len(layer_sizes)):
            b = RAB_tools.build_bias(size=layer_sizes[i], name='b%d'%i)
            self.bs.append(b)
        self.params = self.Ws + self.bs
        # now build the computational graph
        x_corrupt = utils.salt_and_pepper(self.x, self.rng_theano, self.input_noise_level)
        # states include [x_corrupt, h1, h2, h3, ...]
        states_all = [x_corrupt]
        p_x_chain   = []
        # init h0 with 0
        for w,b,i,n in zip(self.Ws, self.bs[1:], range(len(self.n_hiddens)),self.n_hiddens):
            # init with zeros
            print "Init hidden units at zero before creating the graph"
            state_h = theano.shared(numpy.zeros((self.minibatch_size,n),
                                                dtype=floatX),'S%d'%(i+1))
            states_all.append(state_h)
        
        # The layer update scheme
        print "Building the graph :", self.n_hprop,"updates"
        for i in range(self.n_hprop):
            if i == 0:
                states = states_all
            states, p_x = self.update_layers(
                states, add_noise=self.add_noise_to_hiddens)
            p_x_chain.append(p_x)
        # now have new hiddens and p_X_chain
        # build the cost
        cost_steps = T.stacklists(
            [T.mean(T.nnet.binary_crossentropy(rx, self.x)) for rx in p_x_chain])
        cost = T.mean(cost_steps)
        updates = OrderedDict()
        consider_constant = None
        updates = RAB_tools.build_updates_with_rules(
            cost, self.params,
            consider_constant, updates,
            self.lr_ts, self.adadelta_epsilon,
            self.lr_decrease, constantX(self.momentum),
            floatX, self.sgd_type
        )
        # now compiling
        self.train_fn = theano.function(
            inputs=[self.x],
            outputs=cost,
            updates=updates,
            name='train_fn'
        )
        # for sampling
        new_states, p_x = self.update_layers(states_all, self.add_noise_to_hiddens)
        self.sampling_one_step_fn = theano.function(inputs=[self.x],
                                           outputs=[p_x]+new_states,
                                           name='sampling_one_step_fn'
                                           )
        
    def update_layers(self, current_states, add_noise):
        def simple_update_layer(states, i, add_noise):
            # Compute the dot product, whatever layer
            post_act_noise  =   0
            # store the updated state
            s = None
            if i == 0:
                # this is update to the inputs
                s = T.dot(states[i+1], self.Ws[i].T) + self.bs[i]
            elif i == self.n_hidden_layers:
                s = T.dot(states[i-1], self.Ws[i-1]) + self.bs[i]
            else:
                # next layer        :   layers[i+1], assigned weights : W_i
                # previous layer    :   layers[i-1], assigned weights : W_(i-1)
                s = T.dot(states[i+1], self.Ws[i].T) + \
                  T.dot(states[i-1], self.Ws[i-1]) + self.bs[i]
                    
            # Add pre-activation noise if NOT input layer
            if i==1 and self.noiseless_h1:
                print '>>NO noise in first layer'
                add_noise = False

            # pre activation noise            
            if i != 0 and add_noise:
                print 'Adding pre-activation gaussian noise'
                s = utils.add_gaussian_noise(
                    s, self.rng_theano, self.hidden_add_noise_sigma)
            # ACTIVATION!
            if i == 0:
                print 'Sigmoid units'
                s = T.nnet.sigmoid(s)
            else:
                print 'Hidden units'
                s = apply_act(s, self.act)

            # post activation noise            
            if i != 0 and add_noise:
                print 'Adding post-activation gaussian noise'
                s = utils.add_gaussian_noise(
                    s, self.rng_theano, self.hidden_add_noise_sigma)
            return s
                
        def update_odd_layers(states, add_noise):
            print 'odd layer update'
            new_states = [None]*len(states)
            # first copy the states for even layers
            for i in range(0, self.n_hidden_layers+1, 2):
                new_states[i] = states[i]
            for i in range(1, self.n_hidden_layers+1, 2):
                print i
                new_states[i] = simple_update_layer(new_states, i, add_noise)
            return new_states
        
        def update_even_layers(states, add_noise):
            print 'even layer update'
            new_states = [None]*len(states)
            p_x = None
            # first copy the states for odd layers
            for i in range(1, self.n_hidden_layers+1, 2):
                new_states[i] = states[i]
            for i in range(0, self.n_hidden_layers+1, 2):
                print i
                s = simple_update_layer(new_states, i, add_noise)
                if i == 0:
                    p_x = s
                    if self.input_sampling:
                        print 'Sampling from input'
                        s = self.rng_theano.binomial(
                                p=s, size=s.shape, dtype='float32')
                    
                # add noise
                new_states[i] = utils.salt_and_pepper(s, self.rng_theano,
                                                      self.input_noise_level)
            return new_states, p_x
        
        # hiddens and p_X_chain are modified inplace!!!
        # x is layer 0, and so on
        new_states = update_odd_layers(current_states, add_noise)
        new_states, p_x = update_even_layers(new_states, add_noise)
        return new_states, p_x
    
    def prepare_dataset(self):
        print 'prepare the dataset'
        (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = utils.load_mnist_binary()
        self.train_x = train_X
        self.valid_x = valid_X
        self.test_x = test_X
        self.marginal = numpy.mean(self.train_x,axis=0)
        # This is for the reason that we need to decide the dimension of all
        # H states beforehand, making it a theano shared variable.
        assert self.train_x.shape[0] % self.minibatch_size == 0
        
    def simple_sgd(self):
        #set up marginal
        self.bs[0].set_value(self.marginal)
        idx = range(self.train_x.shape[0])
        epoch = 0
        epoch_end = self.n_epochs
        minibatch_idx_overall = RAB_tools.generate_minibatch_idx(
            self.train_x.shape[0], self.minibatch_size)
        while (epoch < epoch_end):
            costs_epoch = []
            for k, use_idx in enumerate(minibatch_idx_overall):
                if self.verbose:
                    sys.stdout.write('\rTraining minibatches %d/%d'%(
                             k, len(minibatch_idx_overall)))
                    sys.stdout.flush()
                minibatch_data = self.train_x[use_idx,:]

                cost = self.train_fn(minibatch_data)
                
                if numpy.isnan(cost):
                    print 'cost is NaN'
                    import ipdb; ipdb.set_trace()
                costs_epoch.append(cost)
            # now linearly decrease the learning rate
            current_lr = self.lr_ts.get_value()
            new_lr = current_lr - numpy.float32(self.lr_decrease)
            self.lr_ts.set_value(new_lr)
            cost_epoch_avg = numpy.mean(costs_epoch)
            self.costs.append([epoch, cost_epoch_avg])
            print '\rTraining %d/%d epochs, cost %.5f, lr %.5f'%(
            epoch, epoch_end, cost_epoch_avg, current_lr)
            if epoch != 0 and (epoch+1) % self.valid_freq == 0:
                numpy.savetxt(self.save_model_path+'epoch_costs.txt', self.costs)
                if self.channel:
                    self.channel.save()
                self.make_plots(self.costs)
                #self.visualize_filters(epoch)
                #self.compute_LL(epoch, save_nothing=False)
                #self.inpainting(epoch, self.k)
                self.generate_samples(epoch)
                self.save_model(epoch)
            epoch += 1
        # end of training
        print

    def generate_samples(self, epoch):
        print 'generate samples'
        N = 100
        # init state
        h_states = [numpy.zeros((1,n),dtype=floatX) for n in self.n_hiddens]
        x = self.train_x[0][numpy.newaxis, :]
        states = [x] + h_states
        for i in range(N):
            self.sampling_one_step_fn()
    def make_plots(self, costs):
        costs = numpy.asarray(costs)
        plt.plot(costs[:,0],costs[:,1])
        plt.savefig(self.save_model_path+'costs.png')
        
    def save_model(self, epoch):
        print 'saving model params'
        params = [param.get_value() for param in self.params]
        RAB_tools.dump_pkl(params, self.save_model_path + 'model_params_e%d.pkl'%epoch)

    def generate_samples(self):
        pass

    def load_model_params(self, params_path):
        print '======================================'
        print 'loading learned parameters from %s'%params_path
        params = RAB_tools.load_pkl(params_path)
        assert len(self.params) == len(params)
        for param_new, param_old in zip(params, self.params):
            assert param_new.shape == param_old.get_value().shape
            param_old.set_value(param_new)
        print 'trained model loaded success!'

    def train_valid_test(self):
        self.build_theano_fn()
        self.prepare_dataset()
        self.simple_sgd()

    def load_and_evaluate(self, params_path):
        print 'load params and evaluate the trained model...'
        self.build_theano_fn()
        self.prepare_dataset()
        self.load_model_params(params_path)
        self.generate_samples(how_many=1000,
                              save_path=os.path.dirname(params_path)+'/samples.png')
        
def train_from_scratch(state, data_engine, channel):
    model = GSN(state, channel)
    model.train_valid_test()
    
def evaluate_trained(state, data_engine, params_path, channel):
    model = GSN(state, channel)
    model.load_and_evaluate(params_path)
    
def continue_train(state, data_engine, params_path, channel):
    raise NotImplementedError()
    pass
