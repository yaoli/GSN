This package contains the accompanied codes for the following two papers:

A. Deep Generative Stochastic Networks Trainable by Backprop

Yoshua Bengio, Éric Thibodeau-Laufer

B. Generalized Denoising Auto-Encoders as Generative Models

Yoshua Bengio, Li Yao, Guillaume Alain, Pascal Vincent

SETUPS:

(1) Download Theano and make sure it's working properly. 
All the information you need can be found by following this link.
http://deeplearning.net/software/theano/

(2) Prepare MNIST dataset
download MNIST datasets from http://deeplearning.net/data/mnist/mnist.pkl.gz

unzip the file to generate mnist.pkl using 'gunzip mnist.pkl.gz'

to visualize MNIST: 'python image_tiler.py'

(3) to run a two layer Deep Stochastic Network in the paper A

run on gpu: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dsn.py

run on cpu: THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dsn.py

(4) to run a one layer Generalized Denoising Autoencoder with a walkback procedure in the paper B 

run on gpu: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dae_walkback.py

run on cpu: THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dae_walkback.py

(5) to run a one layer Generalized Denoising Autoencoder without a walkback procedure in the paper B 

run on gpu: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dae_no_walkback.py

run on cpu: THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dae_no_walkback.py 

(6) Getting the log-likelihood estimation and inpainting as described in the paper A. 

To test a model, by generating inpainting pictures, and 10000 samples used by the parzen density estimator, run this command in this directory, with the params_epoch_X.pkl and config file that was generated when training the model. If multiple params_epoch_X.pkl files are present, the one with the largest epoch number is used.

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dsn.py --test_model 1

---------------------------
Important tips for running the codes:

* (4) and (5) will generate images for both the denoising and pseudo-Gibbs sampling, and save parameters every 5 epochs. We have provided some examples of the reconstruction and generated samples(consecutive Gibbs samples) under the directory 'images/' for 3 types of models.  


* The code is written such that it produces better results on the estimated log-likelihood by Parzen density estimator than in our paper B. For example, (4) produces a log-likelihood of around 150 and (5) produces 50. Both number could be higher if the model is trained longer. Trust this number with precaution. As the estimation from the Parzen density estimator is biased and tends to prefer rigid samples. You will notice this number is high even when the generated images do not look good. Trust the visulizations more. 

* The codes outputs a lot of information on the screen. This is meant to show the progression. Also you can safely ignore the warning message from Theano. The training starts when the following is printed out:

1       Train :  0.607192       Valid :  0.367054       Test  :  0.364999       time :  20.40169 MeanVisB :  -0.22522 W :  ['0.024063', '0.022423']

2       Train :  0.302400       Valid :  0.277827       Test  :  0.277751       time :  20.33490 MeanVisB :  -0.32510 W :  ['0.023877', '0.022512']

3       Train :  0.292427       Valid :  0.267693       Test  :  0.268585       time :  20.45896 MeanVisB :  -0.38779 W :  ['0.023882', '0.022544']

4       Train :  0.268086       Valid :  0.267201       Test  :  0.268247       time :  20.37603 MeanVisB :  -0.43271 W :  ['0.023856', '0.022535']

5       Train :  0.266533       Valid :  0.264087       Test  :  0.265572       time :  20.26944 MeanVisB :  -0.47086 W :  ['0.023840', '0.022517']

For each training epoch, the first 3 numbers are the training cost, followed by the training time (in seconds), then the mean of the visable bias, and mean of the magnitude of weights. 
