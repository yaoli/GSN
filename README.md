(1) Download Theano and make sure it's working properly. 
All the information you need can be found in this link.
http://deeplearning.net/software/theano/

(2) Prepare MNIST dataset
download MNIST datasets from http://deeplearning.net/data/mnist/mnist.pkl.gz

unzip the file to generate mnist.pkl using 'gunzip mnist.pkl.gz'

to visualize MNIST: 'python image_tiler.py'

(3) Run DeepStochasticNetwork in the paper.

run on gpu: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dsn.py

run on cpu: THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dsn.py

---------------------------
The training will generate both the reconstruction images and samples, and save parameters every 5 epochs.



To test a model, by generating inpainting pictures, and 10000 samples used by the parzen density estimator,
run this command in this directory, with the params_epoch_X.pkl and config file that was generated when training
the model. If multiple params_epoch_X.pkl files are present, the one with the largest epoch number is used.

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dsn.py --test_model 1
