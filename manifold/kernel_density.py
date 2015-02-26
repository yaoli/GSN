"""
conditional distribution p(x|z) as parsen window denstity astimator. 
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
import numpy
import data_tools.data_provider as data_provider
import cPickle
rng_numpy = numpy.random.RandomState(1234)

def load_pkl(path):
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def plot_manifold_samples(sample, data, save_path):
    assert sample.shape[1] == data.shape[1]
    print 'using first %d samples to generate the plot'%data.shape[0]
    n_dim = data.shape[1]
    sample = sample[:data.shape[0]]
    fig = plt.figure()
    for i in range(n_dim-1):
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(2,9,i+1)
        ax.plot(sample[:,i], sample[:, i+1], '*r')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('samples')
        ax.set_axis_off()

        ax = fig.add_subplot(2,9,9+i+1)
        ax.plot(data[:,i], data[:, i+1], '*b')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)
    
def get_toy_manifold_dataset(n_dim=10):
    print 'loading manifold dataset'
    # 5K + 5K + 10K
    path = ''
    train_x = load_pkl(path+'train_samples.pkl').astype('float32')
    train_y = numpy.zeros((train_x.shape[0],)).astype('int32')
    valid_x = load_pkl(path+'valid_samples.pkl').astype('float32')
    valid_y = numpy.zeros((valid_x.shape[0],)).astype('int32')
    test_x = load_pkl(path+'test_samples.pkl').astype('float32')
    test_y = numpy.zeros((test_x.shape[0],)).astype('int32')
    return train_x, train_y, valid_x, valid_y, test_x, test_y

class DenoisingParzenWindow(object):
    def __init__(self):
        self.load_dataset()
        self.n_dim = self.train_x.shape[1]

        # the window width
        self.sigma = 0.01
        # std of adding noise, train_x std = 0.64, mean=0.30
        self.gamma = 0.1

        self.train_x_noisy = self.sample_z_given_x(self.train_x)
        self.valid_x_noisy = self.sample_z_given_x(self.valid_x)
        self.test_x_noisy = self.sample_z_given_x(self.test_x)
        
    def load_dataset(self):
        (self.train_x,_,
         self.valid_x,_, self.test_x,_) = get_toy_manifold_dataset(n_dim=10)
    
    def sample_z_given_x(self, x):
        mu = numpy.zeros((self.n_dim,))
        cov = numpy.identity(self.n_dim) * (self.gamma**2)
        if x.ndim == 1:
            size = 1
            
        else:
            size = x.shape[0]
            
        noise = rng_numpy.multivariate_normal(mean=mu, cov=cov, size=size)
        if size == 1:
            noise = noise.flatten()
        return x+noise

    def predict(self, D):
        D_noisy = self.sample_z_given_x(D)
        nll = []
        for d, d_noisy in zip(D, D_noisy):
            p_d_given_z = self.p_x_given_z(d, d_noisy)
            nll.append(-numpy.log(p_d_given_z))

        nll = nll.sum()
        return nll
        
    def sample_x_given_z(self, z):
        p_component = self.get_p_component(z)
        which_component = numpy.argmax(rng_numpy.multinomial(1, p_component, size=1))
        mu = self.train_x[which_component]
        cov = numpy.identity(self.n_dim) * self.sigma**2
        x = rng_numpy.multivariate_normal(mean=mu, cov=cov, size=1)
        return x.flatten()
        
    def get_p_component(self, x_noisy):
        w_up = numpy.exp(-((x_noisy - self.train_x_noisy)**2).sum(axis=1) / (2 * self.sigma**2))
        w_down = numpy.exp(-((x_noisy - self.train_x_noisy)**2).sum(axis=1) / (2 * self.sigma**2)).sum()
        p_component = w_up / w_down

        return p_component
        
    def p_x_given_z(self, x, x_noisy):
        
        p_component = self.get_p_component(x_noisy)
        p_gaussian_term = numpy.exp(
            -((x - self.train_x)**2).sum(axis=1) / (2 * self.sigma**2)) \
            / (numpy.sqrt(2*numpy.pi) * self.sigma)

        p_all = (p_component * p_gaussian_term).sum()
        
        return p_component, p_gaussian_term, p_all
        
    
def gibbs(steps, model):
    # initial x
    x = model.train_x[0]

    # sampling
    collection = []
    for i in range(steps):
        z = model.sample_z_given_x(x)
        x = model.sample_x_given_z(z)
        collection.append(x)

    return numpy.asarray(collection[1000:])

def plot(sample, data, gamma=None):
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 15}

    matplotlib.rc('font', **font)
    
    n_dim = X.shape[1]
    
    for i in range(n_dim-1):
        if i != 8:
            continue
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.5)
        ax = fig.add_subplot(1,2,1)
        ax.plot(sample[:,i], sample[:, i+1], '*')
        #ax.set_xlabel(r'$x_{%d}$'%i)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 3))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1))
        ax.set_title('samples, noise=%.2f'%gamma)
        
        ax = fig.add_subplot(1,2,2)
        ax.plot(data[:,i], data[:, i+1], '*')
        #ax.set_xlabel(r'$x_{%d}$'%i)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 3))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1))
        ax.set_title('data')
        
        plt.show()
        
if __name__ == '__main__':
    i = DenoisingParzenWindow()
    i.load_dataset()
    n_gibbs_steps = i.train_x.shape[0]
    X = gibbs(n_gibbs_steps, i)
    #plot(X, i.train_x, i.gamma)
    plot_manifold_samples(X, i.train_x, 'test_samples.png')

    
