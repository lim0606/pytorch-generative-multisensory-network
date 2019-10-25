'''
miscellaneous functions: prob
'''
import os
import errno
import datetime
import math
import gzip
import numpy as np
import pickle as pkl

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical, Normal

from PIL import Image

from utils.learning import logging
from utils.transformers import quaternion_from_matrix, rotation_matrix


''' for rl general '''
def get_random_variable(probs, dist='categorical'):
    if dist == 'categorical':
        rv = Categorical(probs)
    else:
        rv = Normal(probs[0], probs[1])

    return rv

def get_log_prob(rv, action):
    log_prob = rv.log_prob(action)
    return log_prob

def sample(probs, dist='categorical'):
    # get random variable
    rv = get_random_variable(probs, dist=dist)

    # sample
    action = rv.sample()

    ## measure log prob
    #log_prob = get_log_prob(rv, action)

    return action#, log_prob


''' for random action generation '''
def gen_ornstein_uhlenbeck_process(x, mu=1.0, sig=0.2, dt=1e-2):
    '''
    https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    https://gist.github.com/StuartGordonReid/961cd2b227d023aa51af
    '''
    th = 1
    mu = mu  # 1.2
    sig = sig  # 0.3
    dt = dt  # 1e-2
    dx = th * (mu-x) * dt + sig * math.sqrt(dt) * np.random.normal(size=x.size)
    x_tp1 = x + dx
    return x_tp1

'''
convert angle and axis to rotation matrix⋅
https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
'''
def r_from_angle_axis(angle, axis):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis[0], axis[1], axis[2]

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix
    matrix = np.zeros((3,3))
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca

    return matrix

'''
convert pitch, yaw to R
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def r_from_pitch_yaw(pitch, yaw):
    # apply global yaw -> R_z
    R_z = rotation_matrix(yaw, np.array([0., 0., 1.]))[:3, :3]

    # infer new rotation axis given R_z
    new_y_axis = R_z.dot(np.array([0., 1., 0.]))
    R_local_y = rotation_matrix(pitch, new_y_axis)[:3, :3]

    # combine rotations
    R = R_local_y.dot(R_z)

    return R

'''
convert pitch, yaw to quat
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def quat_from_pitch_yaw(pitch, yaw):
    # convert pitch, yaw to R
    R = r_from_pitch_yaw(pitch, yaw)

    # infer quaternion from rotation matrix
    quat = quaternion_from_matrix(R)

    return quat

'''
convert roll, yaw to R
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def r_from_roll_yaw(roll, yaw):
    # apply global yaw -> R_z
    R_z = rotation_matrix(yaw, np.array([0., 0., 1.]))[:3, :3]

    # infer new rotation axis given R_z
    new_x_axis = R_z.dot(np.array([1., 0., 0.]))
    R_local_x = rotation_matrix(roll, new_x_axis)[:3, :3]

    # combine rotations
    R = R_local_x.dot(R_z)

    return R

'''
convert roll, yaw to quat
https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
'''
def quat_from_roll_yaw(roll, yaw):
    # convert roll, yaw to R
    R = r_from_roll_yaw(roll, yaw)

    # infer quaternion from rotation matrix
    quat = quaternion_from_matrix(R)

    return quat


''' for environment settings '''
def load_env_info(env_name, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # load info
    filename = os.path.join(root, '{}-info.pkl'.format(env_name))
    if os.path.exists(filename):
        info = pkl.load(open(filename, 'rb'))
        return info
    else:
        return False

def save_env_info(info, env_name, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # make directory
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # load info
    filename = os.path.join(root, '{}-info.pkl'.format(env_name))
    if os.path.exists(filename):
        return False
    else:
        pkl.dump(info, open(filename, 'wb'))
        return True


''' for generative models '''
def save_image(i_exp, i_episode, t, img, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.png'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.png'.format(t))

    # make directory
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # save
    #plt.imsave(filename, img)
    img = Image.fromarray(img, mode='RGB')
    img.save(filename, format='PNG')

    # save to file list
    logging('{}.png'.format(t), path, filename='filenames.list', print_=False, log_=True)

def load_image(i_exp, i_episode, t, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.png'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.png'.format(t))

    # load img
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            img = Image.open(f)
            return np.array(img.convert('RGB'))
    else:
        return False

def save_observation_action(i_exp, i_episode, t, observation, action, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(t))

    # make directory
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # save
    np.save(filename, {'observation': observation, 'action': action})

def load_observation_action(i_exp, i_episode, t, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    if t < 0:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(i_episode))
    else:
        # init path
        path = os.path.join(root, 'env{}'.format(i_exp), 'ep{}'.format(i_episode))

        # init filename
        filename = os.path.join(path, '{}.npy'.format(t))

    # load observation and action
    if os.path.exists(filename):
        obj = np.load(filename)
        observation = obj.item().get('observation').astype(np.float32)
        action = obj.item().get('action').astype(np.float32)
        return observation, action
    else:
        return False

def save_scene(i_exp, scene, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # init filename
    filename = os.path.join(root, 'scene{}.pkl.gz'.format(i_exp))

    # make directory
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # save
    with gzip.open(filename, 'wb') as f:
        #torch.save(scene, f)
        pkl.dump(scene, f)

def load_scene(i_exp, suffix='simple'):
    # init root path
    root = os.path.join('data', 'haptix', suffix)

    # init filename
    filename = os.path.join(root, 'scene{}.pkl.gz'.format(i_exp))

    # load observation and action
    if os.path.exists(filename):
        with gzip.open(filename, 'rb') as f:
            #scene = torch.load(f)
            scene = pkl.load(f)
        return scene 
    else:
        return False
