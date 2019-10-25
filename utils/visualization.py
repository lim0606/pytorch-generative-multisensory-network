import numpy as np

import torch
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
sns.set_palette('colorblind')

from sklearn.manifold import TSNE

#from utils import get_grid_image, get_image_from_values
from utils import get_image_from_values

linestyles = ['-', ':', '-:']
color_palette = sns.color_palette()[0:1]+sns.color_palette()[3:4]+sns.color_palette()[2:3]+sns.color_palette()[1:2]+sns.color_palette()[4:]


def get_grid_image(input, batch_size, num_channels, num_height, num_width, nrow=8, pad_value=0):
    '''
    input : b x c x h x w (where h = w)
    '''
    input = input.detach()
    output = input.view(batch_size, num_channels, num_height, num_width).clone().cpu()
    output = vutils.make_grid(output, nrow=nrow, normalize=True, scale_each=True, pad_value=pad_value)
    #output = vutils.make_grid(output, normalize=False, scale_each=False)
    return output

# visualize predictions (image)
# data split order (rgb -> lr -> ul), and thus combine order (ul -> lr -> rgb)
def get_combined_visualization_image_data(dataset, dims, img_gens, num_img_queries, num_plots=4, nrow=8, pad_value=1):
    # init
    dataset = dataset.split('haptix-shepard_metzler_5_parts-multi')[-1]
    num_episodes = len(img_gens[0]) // num_img_queries
    new_img_gens = []

    # repacking generated images
    for idx in range(len(img_gens)):
        # get idx
        img_gen = img_gens[idx]

        # repacking generated images
        img_gen = [img_gen[i*num_img_queries:(i+1)*num_img_queries] for i in range(num_episodes)]

        # append to new_img_gens
        new_img_gens += [img_gen]

    xs = []
    for i in range(num_plots):
        if 'ul' in dataset:
            cmb_ul_img = []
            cmb_ul_dim = []
            for idx in range(0, len(img_gens), 2):
                cmb_ul_img += [torch.cat([
                    new_img_gens[idx][i],
                    new_img_gens[idx+1][i]
                    ], dim=2)]
                cmb_ul_dim += [(
                    dims[idx][0],
                    dims[idx][1] + dims[idx+1][1],
                    dims[idx][2],
                    )]
                assert dims[idx][0] == dims[idx+1][0]
                assert dims[idx][2] == dims[idx+1][2]
        else:
            cmb_ul_img = []
            cmb_ul_dim = []
            for idx in range(len(img_gens)):
                cmb_ul_img += [new_img_gens[idx][i]]
                cmb_ul_dim += [(dims[idx][0], dims[idx][1], dims[idx][2])]

        if 'lr' in dataset:
            cmb_lr_img = []
            cmb_lr_dim = []
            for idx in range(0, len(cmb_ul_img), 2):
                cmb_lr_img += [torch.cat([
                    cmb_ul_img[idx],
                    cmb_ul_img[idx+1],
                    ], dim=3)]
                cmb_lr_dim += [(
                    cmb_ul_dim[idx][0],
                    cmb_ul_dim[idx][1],
                    cmb_ul_dim[idx][2] + cmb_ul_dim[idx+1][2],
                    )]
                assert cmb_ul_dim[idx][0] == cmb_ul_dim[idx+1][0]
                assert cmb_ul_dim[idx][1] == cmb_ul_dim[idx+1][1]
        else:
            cmb_lr_img = cmb_ul_img
            cmb_lr_dim = cmb_ul_dim

        if 'rgb' in dataset:
            cmb_rgb_img = []
            cmb_rgb_dim = []
            for idx in range(0, len(cmb_lr_img), 3):
                cmb_rgb_img += [torch.cat([
                    cmb_lr_img[idx],
                    cmb_lr_img[idx+1],
                    cmb_lr_img[idx+2]], dim=1)]
                cmb_rgb_dim += [(
                    cmb_lr_dim[idx][0] + cmb_lr_dim[idx+1][0] + cmb_lr_dim[idx+2][0],
                    cmb_lr_dim[idx][1],
                    cmb_lr_dim[idx][2],
                    )]
                assert cmb_lr_dim[idx][1] == cmb_lr_dim[idx+1][1] == cmb_lr_dim[idx+2][1]
                assert cmb_lr_dim[idx][2] == cmb_lr_dim[idx+1][2] == cmb_lr_dim[idx+2][2]
        else:
            cmb_rgb_img = cmb_lr_img
            cmb_rgb_dim = cmb_lr_dim

        # check condition
        assert len(cmb_rgb_img) == 1
        assert len(cmb_rgb_dim) == 1

        # get the resulting image
        cmb_img = cmb_rgb_img[0]
        nchannels, nheight, nwidth = cmb_rgb_dim[0]
        x = get_grid_image(
                cmb_img,
                cmb_img.size(0),
                nchannels, nheight, nwidth, nrow=nrow, pad_value=pad_value)
        xs += [x]

    return xs

## visualize predictions (image) (old)
#def get_combined_visualization_image_data(dataset, dims, img_gens, num_img_queries, num_plots=4):
#    # init
#    dataset = dataset.split('haptix-shepard_metzler_5_parts-multi')[-1]
#    num_episodes = len(img_gens[0]) // num_img_queries
#    new_img_gens = []
#
#    # repacking generated images
#    for idx in range(len(img_gens)):
#        # get idx
#        img_gen = img_gens[idx]
#
#        # repacking generated images
#        img_gen = [img_gen[i*num_img_queries:(i+1)*num_img_queries] for i in range(num_episodes)]
#
#        # append to new_img_gens
#        new_img_gens += [img_gen]
#
#    xs = []
#    for i in range(num_plots):
#        if 'rgb' in dataset:
#            cmb_rgb_img = []
#            cmb_rgb_dim = []
#            for idx in range(0, len(img_gens), 3):
#                cmb_rgb_img += [torch.cat([
#                    new_img_gens[idx][i],
#                    new_img_gens[idx+1][i],
#                    new_img_gens[idx+2][i]], dim=1)]
#                cmb_rgb_dim += [(
#                    dims[idx][0] + dims[idx+1][0] + dims[idx+2][0],
#                    dims[idx][1],# + dims[idx+1][1] + dims[idx+2][1],
#                    dims[idx][2],# + dims[idx+1][2] + dims[idx+2][2],
#                    #dims[idx][3] + dims[idx+1][3] + dims[idx+2][3],
#                    #'image'
#                    )]
#                assert dims[idx][1] == dims[idx+1][1] == dims[idx+2][1]
#                assert dims[idx][2] == dims[idx+1][2] == dims[idx+2][2]
#        else:
#            cmb_rgb_img = []
#            cmb_rgb_dim = []
#            for idx in range(len(img_gens)):
#                cmb_rgb_img += [new_img_gens[idx][i]]
#                cmb_rgb_dim += [(dims[idx][0], dims[idx][1], dims[idx][2])]
#
#        if 'ul' in dataset:
#            cmb_ul_img = []
#            cmb_ul_dim = []
#            for idx in range(0, len(cmb_rgb_img), 2):
#                cmb_ul_img += [torch.cat([
#                    cmb_rgb_img[idx],
#                    cmb_rgb_img[idx+1]
#                    ], dim=2)]
#                cmb_ul_dim += [(
#                    cmb_rgb_dim[idx][0],
#                    cmb_rgb_dim[idx][1] + cmb_rgb_dim[idx+1][1],
#                    cmb_rgb_dim[idx][2],
#                    )]
#                assert cmb_rgb_dim[idx][0] == cmb_rgb_dim[idx+1][0]
#                assert cmb_rgb_dim[idx][2] == cmb_rgb_dim[idx+1][2]
#        else:
#            cmb_ul_img = cmb_rgb_img
#            cmb_ul_dim = cmb_rgb_dim
#
#        if 'lr' in dataset:
#            cmb_lr_img = []
#            cmb_lr_dim = []
#            for idx in range(0, len(cmb_ul_img), 2):
#                cmb_lr_img += [torch.cat([
#                    cmb_ul_img[idx],
#                    cmb_ul_img[idx+1],
#                    ], dim=3)]
#                cmb_lr_dim += [(
#                    cmb_ul_dim[idx][0],
#                    cmb_ul_dim[idx][1],
#                    cmb_ul_dim[idx][2] + cmb_ul_dim[idx+1][2],
#                    )]
#                assert cmb_rgb_dim[idx][0] == cmb_rgb_dim[idx+1][0]
#                assert cmb_rgb_dim[idx][2] == cmb_rgb_dim[idx+1][2]
#        else:
#            cmb_lr_img = cmb_ul_img
#            cmb_lr_dim = cmb_ul_dim
#
#        # check condition
#        assert len(cmb_lr_img) == 1
#        assert len(cmb_lr_dim) == 1
#
#        # get the resulting image
#        cmb_img = cmb_lr_img[0]
#        nchannels, nheight, nwidth = cmb_lr_dim[0]
#        x = get_grid_image(
#                cmb_img,
#                cmb_img.size(0),
#                nchannels, nheight, nrow=8, pad_value=1)
#        xs += [x]
#
#    return xs

# visualize predictions (image)
#def get_visualization_image_data(idx, nchannels, nheight, nwidth, device, train_context, train_target, img_output, img_gen, num_img_queries, nviews, num_plots=4):
def get_visualization_image_data(idx, nchannels, nheight, nwidth, device, train_context, train_target, img_output, img_gen, num_img_queries, nviews, num_plots=4, nrow=8):
    # init
    num_episodes = len(train_context)
    blank_image = torch.zeros(1, nchannels, nheight, nwidth).to(device)

    # repacking output images
    batch_sizes = [target[idx*2].size(0) for target in train_target if target[idx*2] is not None]
    img_output = img_output.detach()
    img_output = [img_output[sum(batch_sizes[:i]):sum(batch_sizes[:i])+batch_sizes[i]] for i in range(len(batch_sizes))]

    # repacking generated images
    img_gen = [img_gen[i*num_img_queries:(i+1)*num_img_queries] for i in range(num_episodes)]

    def pad_blank(images):
        if images is None:
            new_images = blank_image.expand(nviews, nchannels, nheight, nwidth)
        else:
            new_images = torch.cat([
                images,
                blank_image.expand(nviews - images.size(0), nchannels, nheight, nwidth)
                ],
                dim=0)
        return new_images

    xs = []
    for i in range(min(num_plots, len(batch_sizes))):
        _context = get_grid_image(
                pad_blank(train_context[i][idx*2]),
                nviews, nchannels, nheight, nwidth, nrow=4, pad_value=1)
        _target = get_grid_image(
                pad_blank(train_target[i][idx*2]),
                nviews, nchannels, nheight, nwidth, nrow=4, pad_value=1)
        _output = get_grid_image(
                pad_blank(img_output[i]),
                nviews, nchannels, nheight, nwidth, nrow=4, pad_value=1)
        _gen    = get_grid_image(
                img_gen[i],
                img_gen[i].size(0),
                nchannels, nheight, nwidth, nrow=nrow, pad_value=1)
        x = torch.cat([_context, _target, _output, _gen], dim=2)
        xs += [x]

    return xs
    #    writer.add_image(
    #            'train/cond-target-recon-gensh-genrd-i{}/img'.format(i),
    #            x, (epoch-1)*len(train_loader) + batch_idx)

# visualize predictions (haptic)
def get_visualization_haptic_data(idx, nchannels, nheight, device, train_context, train_target, hpt_output, hpt_gen, num_hpt_queries, nviews=32, num_plots=4):
    # init
    num_episodes = len(train_context)
    blank_image = hpt_output.new_zeros(1, nchannels*nheight*nheight)

    # repacking output images
    batch_sizes = [target[idx*2].size(0) for target in train_target if target[idx*2] is not None]
    hpt_output = hpt_output.detach()
    #hpt_output = hpt_output * ppc_std + ppc_mean  # rescale
    hpt_output = [hpt_output[sum(batch_sizes[:i]):sum(batch_sizes[:i])+batch_sizes[i]] for i in range(len(batch_sizes))]

    # repacking generated images
    hpt_gen = [hpt_gen[i*num_hpt_queries:(i+1)*num_hpt_queries] for i in range(num_episodes)]

    def pad_blank(images):
        if images is None:
            new_images = blank_image.expand(nviews, nchannels*nheight*nheight)
        else:
            new_images = torch.cat([
                images,
                blank_image.expand(nviews - images.size(0), nchannels*nheight*nheight)
                ],
                dim=0)
        return new_images

    xs = []
    for i in range(min(4, len(batch_sizes))):
        _output = get_image_from_values(
                pad_blank(hpt_output[i]),
                nviews, nchannels, nheight)
        _context = get_image_from_values(
                pad_blank(train_context[i][idx*2]),
                nviews, nchannels, nheight)
        _target = get_image_from_values(
                pad_blank(train_target[i][idx*2]),
                nviews, nchannels, nheight)
        _gen    = get_image_from_values(
                hpt_gen[i],
                hpt_gen[i].size(0),
                nchannels, nheight)
        x = torch.cat([_context, _target, _output, _gen], dim=2)
        xs += [x]

    return xs
    #    writer.add_image(
    #            'train/cond-target-recon-gensh-genrd-i{}/hpt'.format(i),
    #            x, (epoch-1)*len(train_loader) + batch_idx)

def convert_npimage_torchimage(image):
    return torch.transpose(torch.transpose(torch.from_numpy(image), 0, 2), 1, 2)

def get_latent_tsne_plot(latents, labels, mlabels=None, n_classes=10, priors=None, num_samples=1000):
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
    if mlabels is not None:
        mlabels = mlabels[indices[:num_samples]]

    # init palette
    palette = sns.color_palette(n_colors=n_classes)
    palette = [palette[i] for i in np.unique(labels)]
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'] 
    if mlabels is None:
        mpalette = {0: markers[0]}
        mlabels = np.zeros(labels.shape)
    else:
        mpalette = {}
        num_markers = 0
        for mlabel in mlabels:
            if not mlabel in mpalette:
                mpalette[mlabel] = markers[num_markers % len(markers)]
                num_markers += 1

    # init figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # t-SNE
    latents_embedded = TSNE(n_components=2, verbose=True).fit_transform(latents)

    # plot
    if labels is not None:
        data = {'x': latents_embedded[:, 0],
                'y': latents_embedded[:, 1],
                'class': labels,
                'marker': mlabels,
                }
        sns.scatterplot(x='x', y='y', hue='class', style='marker', markers=mpalette, data=data, palette=palette)
    else:
        data = {'x': latents_embedded[:, 0],
                'y': latents_embedded[:, 1],
                'marker': mlabels,
                }
        sns.scatterplot(x='x', y='y', style='marker', markers=mpalette, data=data, palette=palette)

    # draw to canvas
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # close figure
    plt.close()
    return image

def get_colors(keys):
    colors = {}
    num_colors = 0
    for key in keys:
        colors[key] = color_palette[num_colors]
        num_colors += 1
    return colors

def get_combined_visualization_haptic_data(
        hpt_tgt_gen_mask,
        title=None,
        fontsize=10,
        label_pad=1,
        tick_pad=-4,
        dpi=600,
        ):
    keys = list(hpt_tgt_gen_mask.keys())
    batch_size = hpt_tgt_gen_mask[keys[0]][0].size(0)
    colors = get_colors(['target'] + keys)

    images = []
    vals = []
    for i in range(batch_size):
        # plot
        fig = plt.figure(figsize=(3, 2), dpi=dpi)

        # get target
        hpt_tgts = hpt_tgt_gen_mask[keys[0]][0]
        hpt_tgt = hpt_tgts[i].cpu().numpy()
        x = torch.Tensor([dim for dim in range(len(hpt_tgt))]).long().numpy()

        # plot target
        label ='target'
        color = colors['target']
        linestyle = linestyles[0]
        plt.plot(x, hpt_tgt, linestyle=linestyle, color=color, label=label)

        # get and plot gen per num_context
        val = []
        for num_context in keys:
            # get data
            hpt_tgts, hpt_gens = hpt_tgt_gen_mask[num_context]
            hpt_gen = hpt_gens[i].cpu().numpy()

            # check
            assert np.allclose(hpt_tgts[i].cpu().numpy(), hpt_tgt)
            #try:
            #    assert np.allclose(hpt_tgts[i].cpu().numpy(), hpt_tgt)
            #except:
            #    ipdb.set_trace()

            # value
            diff = np.sum((hpt_tgt - hpt_gen)**2)
            val += [diff]

            # plot 
            label ='nc: {}'.format(num_context)
            color = colors[num_context]
            linestyle = linestyles[0]
            plt.plot(x, hpt_gen, linestyle=linestyle, color=color, label=label)

        # append to vals
        vals += [val]

        # set config
        plt.xlabel('index', fontsize=fontsize, labelpad=label_pad)
        plt.ylabel('val',   fontsize=fontsize, labelpad=label_pad)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize, pad=tick_pad)
        plt.tick_params(axis='both', which='minor', labelsize=fontsize-1, pad=tick_pad)

        # set legend
        plt.legend(loc='lower right', frameon=True, fontsize='x-small')

        # set xlim
        left, right = plt.xlim()
        plt.xlim((-1, int(right)+1))
        plt.xticks(np.array([int(key) for key in x if key % 20 == 0]))

        # draw to canvas
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images += [image]

        # close figure
        plt.close()

    return images, vals
