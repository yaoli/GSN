"""
This script will train a single layer model known as Generalized Denoising Auto-Encoder
WITH 5 steps of 'walkback' training.

Reference paper: 
'Generalized Denoising Auto-Encoders as Generative Models'
Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent
http://arxiv.org/abs/1305.6663
  
"""
import argparse
import model

def main():
    parser = argparse.ArgumentParser()
    # Add options here
    parser.add_argument('--K', type=int, default=1) # nubmer of hidden layers
    parser.add_argument('--N', type=int, default=5) # number of walkbacks
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_add_noise_sigma', type=float, default=0)
    parser.add_argument('--input_salt_and_pepper', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=10)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--annealing', type=float, default=1.)
    parser.add_argument('--hidden_size', type=float, default=2000)
    parser.add_argument('--act', type=str, default='sigmoid')
    parser.add_argument('--dataset', type=str, default='MNIST_binary')
    parser.add_argument('--data_path', type=str, default='.')
   
    # argparse does not deal with bool 
    parser.add_argument('--vis_init', type=int, default=0)
    parser.add_argument('--noiseless_h1', type=int, default=1)
    parser.add_argument('--input_sampling', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=0)
  
    args = parser.parse_args()
    
    model.experiment(args, None)
    
if __name__ == '__main__':
    main()