'''
This scripts produces the model trained on MNIST discussed in the paper:

Deep Generative Stochastic Networks Trainable by Backprop
http://arxiv.org/abs/1306.1091
'''
import argparse
import model

def main():
    parser = argparse.ArgumentParser()
    # Add options here
    args = parser.parse_args()
    
    args.K          =   2
    args.N          =   1
    args.n_epoch    =   1000
    args.batch_size =   100

    #args.hidden_add_noise_sigma =   1e-10
    
    args.hidden_add_noise_sigma =   2
    args.input_salt_and_pepper  =   0.4

    args.learning_rate  =   0.25
    args.momentum       =   0.5
    args.annealing      =   0.995

    args.hidden_size    =   1500

    args.input_sampling =   True
    args.noiseless_h1   =   True

    args.vis_init       =   False

    #args.act            =   'rectifier'
    args.act            =   'tanh'

    args.dataset        = 'MNIST'
    args.data_path      =   '.'

    model.experiment(args, None)
    
if __name__ == '__main__':
    main()