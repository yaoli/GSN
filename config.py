# -*- coding: utf-8 -*-
from jobman import DD
from tools import RAB_tools

exp_path = RAB_tools.get_rab_exp_path()
load_path_0 = exp_path + '/gsn_jmlr/test/'
load_path_1 = exp_path + '//'
load_path_2 = exp_path + '//'

config = DD({
    'model': 'GSN',
    'load_trained': DD({
        # action: 0 standard train, 1 load trained model and evaluate, 2 continue training
        'action': 0,
        'from_path': load_path_0,
        'epoch': 2, 
        }),
    'random_seed': 1234,
    'save_model_path': exp_path + \
                      'gsn_jmlr/h3000_h3000_h3000_ni04_nh2_w6_old_ada_reproduce_0/',
    'dataset': DD({
        # MNIST_binary, MNIST_real
        'signature': 'MNIST_binary',
        }),
    'GSN': DD({
        'n_in': None,
        'n_out': None,
        'n_hiddens': [4000,1000,1000],
        'hidden_act': 'tanh',
        # 1:0.01 gau\ssian,2: formula
        'init_weights': 2,
        'input_noise_level': 0.4,
        'add_noise_to_hiddens': True,
        'noiseless_h1': True,
        'hidden_add_noise_sigma': 2,
        'n_hprop': 6,
        # sampling v in the computational graph
        'input_sampling': True,
        'force_h_states': False,
        'train': DD({
            # valid once every 'valid_freq' epochs
            'valid_freq': 10,
            # compute valid and test LL over this many of orderings
            'n_epochs': 1000,
            'minibatch_size': 100,
            # 0 for momentum, 1 for adadelta, 3 for rmsprop
            'sgd_type': 1,
            'adadelta_epsilon': 1e-8,
            'momentum': 0.5,
            'lr': 0.1,
            'l2':0.,
            'use_dropout': False,
            'verbose': True,
            'fine_tune': DD({
                'activate': False,
                'n_epochs': 4000,
                })
            })
        })
    })
