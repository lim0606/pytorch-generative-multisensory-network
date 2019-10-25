'''
miscellaneous functions: colors
'''
import torch

def rgb2gray(img):
    #R = img[:, 0:1, :, :]
    #G = img[:, 1:2, :, :]
    #B = img[:, 2:3, :, :]

    ##gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    #gray = 0.2125 * R + 0.7154 * G + 0.0721 * B
    #return gray

    h, s, v = rgb_to_hsv(img)
    return v

'''
copied and modified from https://github.com/enthought/Python-2.7.3/blob/master/Lib/colorsys.py
'''
#def rgb_to_hsv(r, g, b):
def rgb_to_hsv(img):
    batch_size, nchannels, nheight, nwidth = img.size()
    r = 255*img[:, 0:1, :, :].contiguous().view(-1)
    g = 255*img[:, 1:2, :, :].contiguous().view(-1)
    b = 255*img[:, 2:3, :, :].contiguous().view(-1)

    # get brightness
    maxc = torch.max(r, torch.max(g, b))
    minc = torch.min(r, torch.min(g, b))
    v = maxc

    # get mask
    #if minc == maxc:
    #    return 0.0, 0.0, v
    mask = minc == maxc

    # get saturation
    s = (maxc-minc) / maxc
    s.masked_fill_(mask, 0.)

    # get hue
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    h = v.new_zeros(v.size())
    #if r == maxc:
    #    h = bc-gc
    rmask = r == maxc
    h[rmask] = (bc-gc)[rmask]
    #elif g == maxc:
    #    h = 2.0+rc-bc
    gmask = g == maxc
    h[gmask] = (2.0+rc-bc)[gmask]
    #else:
    #    h = 4.0+gc-rc
    bmask = b == maxc
    h[bmask] = (4.0+gc-rc)[bmask]
    h = (h/6.0) % 1.0
    h.masked_fill_(mask, 0.)

    # reshape
    h = h.view(batch_size, 1, nheight, nwidth)
    s = s.view(batch_size, 1, nheight, nwidth)
    v = v.view(batch_size, 1, nheight, nwidth)
    return h, s, v
