from config import config
from jobman import DD, expand
from tools import RAB_tools
import gsn
import os, sys, socket
import os.path

class Logger(object):
    def __init__(self, stdout_file):
        self.terminal = sys.stdout
        self.log = stdout_file
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

class Unbuffered(object):
    def __init__(self, stream, stdout_file):
        self.stream = stream
        self.log_file = open(stdout_file, "w")
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.log_file.write(data)    # Write the data of stdout here to a text file as well
            
def set_config(conf, args, add_new_key=False):
    # add_new_key: if conf does not contain the key, creates it
    for key in args:
        if key != 'jobman':
            v = args[key]
            if isinstance(v, DD):
                set_config(conf[key], v)
            else:
                if conf.has_key(key):
                    conf[key] = convert_from_string(v)
                elif add_new_key:
                    # create a new key in conf
                    conf[key] = convert_from_string(v)
                else:
                    raise KeyError(key)

def convert_from_string(x):
        """
        Convert a string that may represent a Python item to its proper data type.
        It consists in running `eval` on x, and if an error occurs, returning the
        string itself.
        """
        try:
            return eval(x, {}, {})
        except Exception:
            return x

def evaluate_trained(config, state, channel):
    config_path = config.load_trained.from_path + 'model_config.pkl'
    epoch = config.load_trained.epoch
    params_path = config.load_trained.from_path + 'model_params_e%d.pkl'%(epoch) 
    assert config_path is not None
    assert params_path is not None
    assert os.path.isfile(params_path)
    assert os.path.isfile(config_path)
    print 'load the config options from the best trained model'
    used_config = RAB_tools.load_pkl(config_path)
    action = config.load_trained.action
    assert action == 1
    from_path = config.load_trained.from_path
    epoch = config.load_trained.epoch
    save_model_path = config.load_trained.from_path
    set_config(config, used_config)
    config.load_trained.action = action
    config.load_trained.from_path = from_path
    config.load_trained.epoch = epoch
    config.save_model_path = save_model_path
    
    model_type = config.model
    # set up automatically some fields in config
    if 'MNIST' in config.dataset.signature:
        config[model_type].n_in = 784
        config[model_type].n_out = 784

    # Also copy back from config into state.
    for key in config:
        setattr(state, key, config[key])

    print 'Model Type: %s'%model_type
    print 'Host:    %s' % socket.gethostname()
    print 'Command: %s' % ' '.join(sys.argv)
    
    print 'initializing data engine'
    input_dtype = 'float32'
    target_dtype = 'int32'
    data_engine = None
    gsn.evaluate_trained(state, data_engine, params_path, channel)
    
def continue_train(config, state, channel):
    config_path = config.load_trained.from_path + 'model_config.pkl'
    epoch = config.load_trained.epoch
    params_path = config.load_trained.from_path + 'model_params_e%d.pkl'%(epoch) 
    assert config_path is not None
    assert params_path is not None
    assert os.path.isfile(params_path)
    assert os.path.isfile(config_path)
    print 'load the config options from the best trained model %s'%config_path
    used_config = RAB_tools.load_pkl(config_path)
    action = config.load_trained.action
    assert action == 2
    from_path = config.load_trained.from_path
    epoch = config.load_trained.epoch
    save_model_path = config.save_model_path
    set_config(config, used_config)
    config.load_trained.action = 0
    config.load_trained.from_path = from_path
    config.load_trained.epoch = epoch
    config.save_model_path = save_model_path
    model_type = config.model
    # set up automatically some fields in config
    if "MNIST" in config.dataset.signature:
        config[model_type].n_in = 784
        config[model_type].n_out = 784
    # Also copy back from config into state.
    for key in config:
        setattr(state, key, config[key])
    print 'Model Type: %s'%model_type
    print 'Host:    %s' % socket.gethostname()
    print 'Command: %s' % ' '.join(sys.argv)
    
    print 'initializing data engine'
    input_dtype = 'float32'
    target_dtype = 'int32'
    data_engine = None
    gsn.continue_train(state, data_engine, params_path, channel)
    
def train_from_scratch(config, state, channel):
    model_type = config.model
    # set up automatically some fields in config
    if "MNIST" in config.dataset.signature:
        config[model_type].n_in = 784
        config[model_type].n_out = 784

    # manipulate the 'state
    # save the config file
    save_model_path = config.save_model_path

    if save_model_path == 'current':
        config.save_model_path = './'
        # to facilitate the use of cluster for multiple jobs
        save_path = './model_config.pkl'
    else:
        # run locally, save locally
        save_path = save_model_path + 'model_config.pkl'

    RAB_tools.create_dir_if_not_exist(config.save_model_path)
    # for stdout file logging
    #sys.stdout = Unbuffered(sys.stdout, state.save_model_path + 'stdout.log')
    print 'saving model config into %s'%save_path
    RAB_tools.dump_pkl(config, save_path)
    # Also copy back from config into state.
    for key in config:
        setattr(state, key, config[key])
    
    print 'Model Type: %s'%model_type
    print 'Host:    %s' % socket.gethostname()
    print 'Command: %s' % ' '.join(sys.argv)
    
    print 'initializing data engine'
    input_dtype = 'float32'
    target_dtype = 'int32'
    data_engine = None    
    gsn.train_from_scratch(state, data_engine, channel)
    
def main(state, channel=None):
    
    # copy state to config
    set_config(config, state)
    
    action = config.load_trained.action
    if action == 0:
        # normal training
        train_from_scratch(config, state, channel)
        return 1
    elif action == 1:
        # load trained model and evaluate
        evaluate_trained(config, state, channel)
        return 1
    elif action == 2:
        # load trained model, continue training
        continue_train(config, state, channel)
        return 1
    else:
        raise NotImplementedError()    

def experiment(state, channel):
    # called by jobman
    main(state, channel)
    return channel.COMPLETE
    
if __name__ == '__main__':
    args = {}
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            args[k] = v
    except:
        print 'args must be like a=X b.c=X'
        exit(1)
    
    state = expand(args)
    
    sys.exit(main(state))
