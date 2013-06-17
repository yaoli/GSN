(1) Download Theano and make sure it's working properly. 
All the information you need can be found in this link.
http://deeplearning.net/software/theano/

(2) Prepare MNIST dataset
download MNIST datasets from http://deeplearning.net/data/mnist/mnist.pkl.gz

unzip the file to generate mnist.pkl using 'gunzip mnist.pkl.gz'

(3) Run DeepStochasticNetwork in the paper.

run on gpu: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run_dsn.py

run on cpu: THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_dsn.py

---------------------------
The training will generate both the reconstruction images and samples.  

