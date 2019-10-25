'''
miscellaneous functions: learning
'''
import os
import datetime

import numpy as np

import torch
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


''' for monitoring lr '''
def get_lrs(optimizer):
    lrs = [float(param_group['lr']) for param_group in optimizer.param_groups]
    lr_max = max(lrs)
    lr_min = min(lrs)
    return lr_min, lr_max


''' save and load '''
def save_checkpoint(state, opt, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(opt.path, filename)
    print("=> save checkpoint '{}'".format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

#def load_checkpoint(model, optimizer, opt, filename='checkpoint.pth.tar', verbose=True):
#    filename = os.path.join(opt.path, filename)
#    if os.path.isfile(filename):
#        if verbose:
#            print("=> loading checkpoint '{}'".format(filename))
#        checkpoint = torch.load(filename)
#        opt.start_epoch = checkpoint['epoch']
#        opt.start_batch_idx = checkpoint['batch_idx']
#        opt.best_val_loss = checkpoint['best_val_loss']
#        model.load_state_dict(checkpoint['state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer'])
#        if verbose:
#            print("=> loaded checkpoint '{}'".format(filename))
#    else:
#        print("=> no checkpoint found at '{}'".format(filename))

def load_checkpoint(model, optimizer, opt, filename='checkpoint.pth.tar', verbose=True):
    filename = os.path.join(opt.path, filename)
    if os.path.isfile(filename):
        if verbose:
            print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        opt.start_epoch = checkpoint['epoch']
        opt.start_batch_idx = checkpoint['batch_idx']
        opt.best_val1_loss = checkpoint['best_val1_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if verbose:
            print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))



''' log '''
def logging(s, path=False, filename='log.txt', print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        assert path, 'path is not define. path: {}'.format(path)
        with open(os.path.join(path, filename), 'a+') as f_log:
            f_log.write(s + '\n')

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')


''' visualization '''
def get_image_from_values(input, batch_size, num_channels, num_height):
    '''
    input : b x c x h x w (where h = w = 1)
    '''
    assert num_height == 1
    input = input.detach()
    output = input.view(1, 1, batch_size, num_channels*num_height*num_height).clone().cpu()
    output = vutils.make_grid(output, normalize=True, scale_each=True)
    return output

def get_grid_image(input, batch_size, num_channels, num_height, nrow=8, pad_value=0):
    '''
    input : b x c x h x w (where h = w)
    '''
    input = input.detach()
    output = input.view(batch_size, num_channels, num_height, num_height).clone().cpu()
    output = vutils.make_grid(output, nrow=nrow, normalize=True, scale_each=True, pad_value=pad_value)
    #output = vutils.make_grid(output, normalize=False, scale_each=False)
    return output

def get_plot(preds, seq_length, batch_size, num_channels, num_height, cond_length=None):
    '''
    input : s x b x c x h x w (where h = w = 1)
    '''
    assert num_height == 1
    preds = preds.detach()
    preds = preds.view(seq_length, batch_size*num_channels*num_height*num_height).clone().cpu()  # temporary

    # init
    if cond_length is not None:
        cond_length = min(cond_length, seq_length)
    else:
        cond_length = seq_length

    # convert to numpy
    preds = preds.numpy()
    #data = data.numpy()

    # plot
    scale = 0.5
    fig = plt.figure(figsize=(scale*30,scale*10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=scale*30)
    plt.xlabel('x', fontsize=scale*20)
    plt.ylabel('y', fontsize=scale*20)
    plt.xticks(fontsize=scale*20)
    plt.yticks(fontsize=scale*20)
    def draw(yi, color):
        plt.plot(np.arange(cond_length), yi[:cond_length], color, linewidth = scale*2.0)
        plt.plot(np.arange(cond_length, len(yi)), yi[cond_length:], color + ':', linewidth = scale*2.0)
    #draw(data[:, 0], 'k')
    draw(preds[:, 0], 'r')
    draw(preds[:, 1], 'g')
    draw(preds[:, 2], 'b')
    #plt.savefig('predict%d.pdf'%batch_idx)
    #plt.savefig('predict.pdf')
    #plt.close()

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_numpy_plot(data, title=None, xlabel='x', ylabel='y'):
    '''
    input : batch_size x seq_length
    '''
    batch_size, seq_length = data.shape

    # plot
    scale = 0.5
    fig = plt.figure(figsize=(scale*30,scale*10))
    if title is not None:
        plt.title(title, fontsize=scale*30)
    plt.xlabel(xlabel, fontsize=scale*20)
    plt.ylabel(ylabel, fontsize=scale*20)
    plt.xticks(fontsize=scale*20)
    plt.yticks(fontsize=scale*20)
    def draw(yi):
        plt.plot(np.arange(seq_length), yi, linewidth = scale*2.0)
    for i in range(batch_size):
        draw(data[i])

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image


''' latent plots '''
def get_unit_normal_samples(num_samples=5000):
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    x, y = np.random.multivariate_normal(mean, cov, num_samples).T
    return x, y

def get_latent_kde_plot(latents, labels, priors=None, random_select_axis=False, num_samples=1000):
    '''
    latents : inferred latents, batch_size x latent_dim (numpy array)
    labels : labels, batch_size (numpy array)
    priors : prior samples of latents, num_samples x latent_dim
    '''
    batch_size, latent_dim = latents.shape

    # init figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set(xlim=(-4, 4), ylim=(-4, 4))

    # select plot indices
    if random_select_axis:
        indices = np.random.permutation(latent_dim)
        ix = indices[0]
        iy = indices[1]
    else:
        ix = 0
        iy = 1

    # inferred
    dx, dy = latents[:, ix], latents[:, iy]
    ax = sns.kdeplot(dx, dy, cmap="Reds", shade=True, shade_lowest=False, ax=axes[0])

    # prior
    if priors is None:
        px, py = get_unit_normal_samples(num_samples=num_samples)
    else:
        px, py = priors[:, ix], priors[:, iy]
    ax = sns.kdeplot(px, py, cmap="Blues", shade=True, shade_lowest=False, ax=axes[1])

    ## add labels to the plot
    #red = sns.color_palette("Reds")[-2]
    #blue = sns.color_palette("Blues")[-2]
    #ax.text(2.5, 8.2, "virginica", size=16, color=blue)
    #ax.text(3.8, 4.5, "setosa", size=16, color=red)

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_latent_tsne_plot(latents, labels, n_classes=10, priors=None, num_samples=1000):
    '''
    latents : inferred latents, batch_size x latent_dim (numpy array)
    labels : labels, batch_size (numpy array)
    priors : prior samples of latents, num_samples x latent_dim
    '''
    batch_size, latent_dim = latents.shape
    num_samples = min(num_samples, batch_size)

    # subsamples
    indices = np.random.permutation(batch_size)
    latents = latents[indices[:num_samples]]
    labels  = labels[indices[:num_samples]]

    # init figure
    fig, ax = plt.subplots(figsize=(5, 5))
    palette = sns.color_palette(n_colors=n_classes)
    palette = [palette[i] for i in np.unique(labels)]

    # t-SNE
    latents_embedded = TSNE(n_components=2, verbose=True).fit_transform(latents)

    # plot
    if labels is not None:
        data = {'x': latents_embedded[:, 0],
                'y': latents_embedded[:, 1],
                'class': labels}
        sns.scatterplot(x='x', y='y', hue='class', data=data, palette=palette)
    else:
        data = {'x': latents_embedded[:, 0],
                'y': latents_embedded[:, 1]}
        sns.scatterplot(x='x', y='y', data=data, palette=palette)

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_latent_2d_plot(latents, labels, n_classes=10, priors=None, num_samples=1000):
    '''
    latents : inferred latents, batch_size x latent_dim (numpy array)
    labels : labels, batch_size (numpy array)
    priors : prior samples of latents, num_samples x latent_dim
    '''
    batch_size, latent_dim = latents.shape
    num_samples = min(num_samples, batch_size)

    # subsamples
    indices = np.random.permutation(batch_size)
    latents = latents[indices[:num_samples]]
    labels  = labels[indices[:num_samples]]

    # init figure
    fig, ax = plt.subplots(figsize=(5, 5))
    palette = sns.color_palette(n_colors=n_classes)
    palette = [palette[i] for i in np.unique(labels)]

    # plot
    if labels is not None:
        data = {'x': latents[:, 0],
                'y': latents[:, 1],
                'class': labels}
        sns.scatterplot(x='x', y='y', hue='class', data=data, palette=palette)
    else:
        data = {'x': latents[:, 0],
                'y': latents[:, 1]}
        sns.scatterplot(x='x', y='y', data=data, palette=palette)

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image
