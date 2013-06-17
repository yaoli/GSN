""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.

"""


import numpy, os
#import data_provider
from PIL import Image

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array

def visualize_mnist():
    train, _,_,_,_,_ = data_provider.load_mnist()
    design_matrix = train
    images = design_matrix[0:2500, :]
    channel_length = 28 * 28
    to_visualize = images
                    
    image_data = tile_raster_images(to_visualize,
                                    img_shape=[28,28],
                                    tile_shape=[50,50],
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(numpy.uint8(image_data))
    im_new.save('samples_mnist.png')
    os.system('eog samples_mnist.png')

def visualize_cifar10():
    import cifar10_wrapper
    train, test = cifar10_wrapper.get_cifar10_raw()
    

    images = train.X[0:2500, :]
    channel_length = 32 * 32
    to_visualize = (images[:, 0:channel_length],
                    images[:, channel_length:channel_length * 2],
                    images[:,channel_length*2:channel_length * 3],
                    None)
                    
    image_data = tile_raster_images(to_visualize,
                                    img_shape=[32,32],
                                    tile_shape=[50,50],
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(numpy.uint8(image_data))
    im_new.save('samples_cifar10.png')
    os.system('eog samples_cifar10.png')

def visualize_weight_matrix_single_channel(img_shape, tile_shape, to_visualize):
    # the weights learned from the black-white image, e.g., MNIST
    # W: inputs by hidden
    # img_shape = [28,28]
    # tile_shape = [10,10]
    
    image_data = tile_raster_images(to_visualize,
                                    img_shape=img_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=(2,2))
    
    im_new = Image.fromarray(numpy.uint8(image_data))
    im_new.save('l0_weights.png')
    #im_new.save('ica_weights.png')
    #os.system('eog ica_weights.png')

def visualize_convNet_weights(params):
    W = params[-4]
    a,b,c,d = W.shape
    to_visualize = W.reshape((a*b, c*d))
    image_data = tile_raster_images(to_visualize,
                                    img_shape=[c,d],
                                    tile_shape=[10,5],
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(numpy.uint8(image_data))
    im_new.save('convNet_weights.png')
    
def visualize_first_layer_weights(W, dataset_name=None):
    imgs = W.T
    if dataset_name == 'MNIST':
        img_shape = [28,28]
        to_visualize = imgs
    elif dataset_name == 'TFD_unsupervised':
        img_shape = [48,48]
        to_visualize = imgs

    elif dataset_name == 'CIFAR10':
        img_shape = [32,32]
        channel_length = 32 * 32
        to_visualize = (imgs[:, 0:channel_length],
            imgs[:, channel_length:channel_length * 2],
            imgs[:,channel_length*2:channel_length * 3],
            None)
    else:
        raise NotImplementedError('%s does not support visulization of W'%self.dataset_name)

    t = int(numpy.ceil(numpy.sqrt(W.shape[1])))

    tile_shape = [t,t]

    visualize_weight_matrix_single_channel(img_shape,
                                                   tile_shape, to_visualize)
    
def visualize_reconstruction_quality_ae(x, x_tilde, x_reconstructed, image_shape):    
    # to visualize the reconstruction quality of MNIST on DAE
    assert x.shape == x_tilde.shape
    assert x_tilde.shape == x_reconstructed.shape
    
    n_show = 400
    tile_shape = [20, 20]
    tile_spacing = (2,2)
    image_shape = image_shape
    
    idx = range(x.shape[0])
    numpy.random.shuffle(idx)
    
    use = idx[:n_show]
    channel_length = image_shape
    to_visualize = x[use]
    image_data = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)

    to_visualize = x_tilde[use]
    image_corrupted = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)

    to_visualize = x_reconstructed[use]
    image_reconstructed = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)
    
    vertical_bar = numpy.zeros((image_data.shape[0], 5))
    vertical_bar[:,2] += 255

    image = numpy.concatenate((image_data, vertical_bar, image_corrupted,
                               vertical_bar, image_reconstructed), axis=1)
    
    im_new = Image.fromarray(numpy.uint8(image))
    #im_new.save('reconstruction_mnist.png')
    #os.system('eog reconstruction_mnist.png')
    return im_new
    
def visualize_gibbs_chain(data, samples, x_noisy, x_reconstruct, jumps, image_shape):
    # jumps is a binary matrix
    # randomly pick to visualize
    #assert data.shape == samples.shape
    
    n_show = 400
    tile_shape = [20, 20]
    tile_spacing = (2,2)
    image_shape = image_shape
    
    idx = range(data.shape[0])
    numpy.random.shuffle(idx)
    
    use = idx[:n_show]
    
    to_visualize = data[use]
    image_data = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)
    
    use = range(n_show)
    
    to_visualize = samples[use]
    image_1 = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)
    to_visualize = x_noisy[use]
    image_2 = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)
    to_visualize = x_reconstruct[use]
    image_3 = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)
    # now masking those intermediate steps in the chain
    jumps = jumps.flatten()
    jumps[0] = 0
    mask = numpy.zeros((n_show)) != 0
    for idx, jump in enumerate(jumps):
        if jumps[idx] ==1 and jumps[idx-1]==0:
            mask[idx-1] = True

    to_visualize = numpy.zeros(x_reconstruct.shape)
    for i, m in enumerate(mask):
        if m:
           to_visualize[i] = x_reconstruct[i]
    image_4 = tile_raster_images(to_visualize,
                                    img_shape=image_shape,
                                    tile_shape=tile_shape,
                                    tile_spacing=tile_spacing)
    
    vertical_bar = numpy.zeros((image_data.shape[0], 5))
    vertical_bar[:,2] += 255
    
    image = numpy.concatenate((image_data, vertical_bar, image_1,
                               vertical_bar, image_2, vertical_bar,
                               image_3, vertical_bar, image_4), axis=1)
    im_new = Image.fromarray(numpy.uint8(image))
    #im_new.save('samples_mnist%.png')
    #os.system('eog samples_mnist.png')
    return im_new
    
if __name__ == '__main__':
    #visualize_mnist()
    #visualize_cifar10()
    #W = RAB_tools.load_pkl('convNet_saved_params.pkl')
    #visualize_convNet_weights(W)
    visualize_weight_matrix_single_channel(W)

    
