This package contains the accompanying code for the following two papers:

* \[1\] Yoshua Bengio, Éric Thibodeau-Laufer, Jason
  Yosinski. [Deep Generative Stochastic Networks Trainable by Backprop](http://arxiv.org/abs/1306.1091). _arXiv
  preprint arXiv:1306.1091._ ([PDF](http://arxiv.org/pdf/1306.1091v3),
  [BibTeX](https://raw.github.com/yaoli/GSN/master/doc/gsn.bib))

* \[2\] Yoshua Bengio, Li Yao, Guillaume Alain, Pascal
  Vincent. [Generalized Denoising Auto-Encoders as Generative Models](http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models). _NIPS,
  2013._ ([PDF](http://media.nips.cc/nipsbooks/nipspapers/paper_files/nips26/491.pdf),
  [BibTeX](https://raw.github.com/yaoli/GSN/master/doc/dae.bib))



Setup
---------------------

#### Install Theano

Download Theano and make sure it's working properly.  All the
information you need can be found by following this link:
http://deeplearning.net/software/theano/

#### Prepare the MNIST dataset

1. Download the MNIST dataset from http://deeplearning.net/data/mnist/mnist.pkl.gz

2. Unzip the file to generate mnist.pkl using `gunzip mnist.pkl.gz`

3. (Optional) To visualize MNIST, run `python image_tiler.py`



Reproducing the Results
---------------------

The below commands are given in two formats: the first will run on the
GPU and the second on the CPU. Choose whichever is most appropriate
for your setup.  Of course, the GPU versions will only work if Theano
is being used on a machine with a compatible GPU (more about
[using the GPU in Theano](http://deeplearning.net/software/theano/tutorial/using_gpu.html)).

1. To run a two layer Generative Stochastic Network (paper \[1\])

        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_gsn.py
        THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_gsn.py

2. To run a one layer Generalized Denoising Autoencoder with a walkback procedure (paper \[2\])

        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dae_walkback.py
        THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dae_walkback.py

3. To run a one layer Generalized Denoising Autoencoder without a walkback procedure (paper \[2\])

        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dae_no_walkback.py
        THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dae_no_walkback.py

4. Getting the log-likelihood estimation and inpainting (as described in paper \[1\])

    To test a model, by generating inpainting pictures, and 10000
    samples used by the parzen density estimator, run this command in
    this directory, with the `params_epoch_X.pkl` and `config` file
    that was generated when training the model. If multiple
    `params_epoch_X.pkl` files are present, the one with the largest
    epoch number is used.

        THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_gsn.py --test_model 1



#### Important notes on running the code

* (1), (2) and (3) will generate images for both the denoising and
  pseudo-Gibbs sampling, and save parameters every 5 epochs. We have
  provided some examples of the reconstruction and generated samples
  (consecutive Gibbs samples) under the directory 'images/' for 3
  types of models. By just looking at the pictures there, it is clear
  that 2-layer model beats the 1-layer model with walkback training,
  which then beats the 1-layer model without walkback training.

* The code is written such that it produces better results on the
  estimated log-likelihood by Parzen density estimator than in our
  paper \[2\]. For example, (2) produces a log-likelihood of around
  150 and (3) produces 50. Both number could be higher if the model is
  trained longer. Trust this number with precaution, as the estimation
  from the Parzen density estimator is biased and tends to prefer
  rigid samples. You will notice this number is high even when the
  generated images do not look good. Trust the visualizations more.

* The codes outputs a lot of information on the screen. This is meant
  to show the progression. Also you can safely ignore the warning
  message from Theano. The training starts when the following is
  printed out:

        1    Train :  0.607192    Valid :  0.367054    Test  :  0.364999    time :  20.40169 MeanVisB :  -0.22522 W :  ['0.024063', '0.022423']
        2    Train :  0.302400    Valid :  0.277827    Test  :  0.277751    time :  20.33490 MeanVisB :  -0.32510 W :  ['0.023877', '0.022512']
        3    Train :  0.292427    Valid :  0.267693    Test  :  0.268585    time :  20.45896 MeanVisB :  -0.38779 W :  ['0.023882', '0.022544']
        4    Train :  0.268086    Valid :  0.267201    Test  :  0.268247    time :  20.37603 MeanVisB :  -0.43271 W :  ['0.023856', '0.022535']
        5    Train :  0.266533    Valid :  0.264087    Test  :  0.265572    time :  20.26944 MeanVisB :  -0.47086 W :  ['0.023840', '0.022517']

  For each training epoch, the first 3 numbers are the cost on the
  training, validation, and test sets, followed by the training time
  (in seconds, on GPU Nvidia GeForce GTX580, 300 seconds in Intel(R)
  Core(TM) i7-2600K CPU @ 3.40GHz), then the mean of the visible bias,
  and mean of the magnitude of weights.


#### Contact

Questions? Contact us: li.yao@umontreal.ca
