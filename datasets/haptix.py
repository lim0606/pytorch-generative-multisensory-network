import os
import sys
import gzip
import pickle
import itertools
from PIL import Image
import math
import pickle as pkl
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from utils import normalize
from utils import sample_queries
from utils import get_label
from utils import get_masks
#from utils import rgb2gray


SEMANTIC = [
    torch.LongTensor([1, 2, 3, 23, 24, 25, 45, 46, 47, 58, 59, 60, 71, 72, 73]), # wrist
    torch.LongTensor([4, 5, 6, 7, 26, 27, 28, 29, 48, 49, 50, 51, 61, 62, 63, 64, 74, 75, 76, 77, 84, 85, 86, 99, 100, 101]), # thumb
    torch.LongTensor([8, 9, 10, 11, 30, 31, 32, 33, 52, 53, 65, 66, 78, 79, 87, 88, 89, 102, 103, 104]), # index
    torch.LongTensor([12, 13, 14, 34, 35, 36, 54, 67, 80, 90, 91, 92, 105, 106, 107]), # middle
    torch.LongTensor([15, 16, 17, 18, 37, 38, 39, 40, 55, 68, 81, 93, 94, 95, 108, 109, 110]), # ring
    torch.LongTensor([19, 20, 21, 22, 41, 42, 43, 44, 56, 57, 69, 70, 82, 83, 96, 97, 98, 111, 112, 113]), # pinky
    torch.LongTensor([114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]), # touch
]
SEMANTIC = [sem-1 for sem in SEMANTIC] # 0-base index

HALF = [
    torch.cat([
        torch.LongTensor([4, 5, 6, 7, 26, 27, 28, 29, 48, 49, 50, 51, 61, 62, 63, 64, 74, 75, 76, 77, 84, 85, 86, 99, 100, 101]), # thumb
        torch.LongTensor([8, 9, 10, 11, 30, 31, 32, 33, 52, 53, 65, 66, 78, 79, 87, 88, 89, 102, 103, 104]), # index
        torch.LongTensor([12, 13, 14, 34, 35, 36, 54, 67, 80, 90, 91, 92, 105, 106, 107]), # middle
    ]),
    torch.cat([
        torch.LongTensor([15, 16, 17, 18, 37, 38, 39, 40, 55, 68, 81, 93, 94, 95, 108, 109, 110]), # ring
        torch.LongTensor([19, 20, 21, 22, 41, 42, 43, 44, 56, 57, 69, 70, 82, 83, 96, 97, 98, 111, 112, 113]), # pinky
        torch.LongTensor([114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]), # touch
        torch.LongTensor([1, 2, 3, 23, 24, 25, 45, 46, 47, 58, 59, 60, 71, 72, 73]), # wrist
    ]),
]
HALF = [sem-1 for sem in HALF] # 0-base index

CHANNEL_WISE = [
    (0,   22), # joint pos
    (22,  44), # joint vel
    (44,  57), # actu pos
    (57,  70), # actu vel
    (70,  83), # actu force
    (83,  98), # acc
    (98,  113), # gyro
    (113, 132), # torch
]

def get_mask_from_combinations(combinations):
    n_mods_all = {}
    m_counts = 0
    for i, mask in enumerate(combinations):
        n_mods = sum(mask)
        if n_mods in n_mods_all:
            n_mods_all[n_mods] += [i]
        else:
            n_mods_all[n_mods] = [i]
            m_counts += 1

    # get mask
    m_idx = np.random.randint(m_counts)
    n_mods = m_idx+1
    indices = n_mods_all[n_mods]
    mask = combinations[indices[np.random.choice(len(indices))]]

    return mask

'''
define Scene class
'''
class Scene(object):
    def __init__(self,
                 images,
                 cameras,
                 proprioceptives,
                 hands,
                 hand_images,
                 sampled_dir_cameras,
                 sampled_dir_hands,
                 env_info,
                 ):
        self.images =              images
        self.cameras =             cameras
        self.proprioceptives =     proprioceptives
        self.hands =               hands
        self.hand_images =         hand_images
        self.sampled_dir_cameras = sampled_dir_cameras
        self.sampled_dir_hands =   sampled_dir_hands
        self.env_info =            env_info


'''
define SceneDataset classes
'''
class SceneDataset(Dataset):
    """
    copied and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/dataset.py
    """
    def __init__(self,
                 root, names, n_parts, split='train', transform=None,
                 target_sample_method='remaining', max_cond_size=20, max_target_size=20,
                 normalize_method=None,
                 use_touch=True,
                 do_zeromean_unitnormal=False,
                 do_minmax_norm=False,
                 img_size=64,
                 ):
        super().__init__()
        self.root = root
        assert type(names) == type([])
        assert len(names) == len(n_parts)
        self.n_parts = n_parts
        self.names = names
        self.name = '_'.join(self.names)
        self.split = split
        assert self.split in ['train', 'val1', 'val2', 'test']
        paths = []
        for name in names:
            if self.split == 'train':
                path = os.path.join(self.root, name, 'train')
            elif self.split in ['val1', 'val2']:
                path = os.path.join(self.root, name, 'val')
            else:
                path = os.path.join(self.root, name, 'test')
            paths += [path]
        self.paths = paths
        self.count_num_scenes()
        self.environment = 'ShepardMetzlerHandSingleObj-v0'
        self.transform = transform
        self.num_views = len(self.load_scene(0).images)
        assert target_sample_method in ['remaining', 'random', 'full']
        self.target_sample_method = target_sample_method
        self.max_cond_size = min(self.num_views, max_cond_size)
        self.max_target_size = min(self.num_views, max_target_size)
        self.normalize_method = normalize_method
        self.use_touch = use_touch
        self.do_zeromean_unitnormal = do_zeromean_unitnormal
        self.do_minmax_norm = do_minmax_norm
        self.img_size=img_size

        # load stat
        with open('cache/shepard_metzler_5_parts_stat.pt', 'rb') as f:
            stat = torch.load(f)
            self.ppc_mean = stat['mean'].view(132)
            self.ppc_std = stat['std'].view(132)
            self.ppc_max = stat['max'].view(132)
            self.ppc_min = stat['min'].view(132)
            buff = self.ppc_max.new_zeros(self.ppc_max.size()).fill_(0.1)
            self.ppc_05_max_minus_min = 0.5 * torch.max(buff, self.ppc_max - self.ppc_min)

        # load shape
        self.class_offset = [0]
        shapes = []
        for name in self.names:
            with open('data/haptix/{}/shapes.pkl'.format(name), 'rb') as f:
                _shapes = pkl.load(f) #torch.load(f)
            self.class_offset += [len(_shapes)]
            shapes += _shapes
        self.shapes = shapes

    def count_num_scenes(self):
        # split train and val
        os.system('mkdir -p cache/haptix/{}'.format(self.name))

        # read samples
        if self.split == 'train':
            split = 'train'
        elif self.split in ['val1', 'val2']:
            split = 'val'
        else:
            split = 'test'
        samples_filename = os.path.join('cache/haptix/{}'.format(self.name), 'samples-{}.pt'.format(split))
        if os.path.exists(samples_filename):
            self.samples = torch.load(samples_filename)
        else:
            samples = []
            for path in self.paths:
                scene_list = [scene for scene in os.listdir(path) \
                                 if os.path.splitext(scene)[-1] == '.gz']
                _samples = [os.path.join(path, scene) for scene in scene_list]
                samples += _samples
            torch.save(samples, open(samples_filename, 'wb'))
            self.samples = samples

    def load_scene(self, idx):
        sys.modules['__main__'].Scene = Scene
        filename = self.samples[idx]
        with gzip.open(filename, 'rb') as f:
            scene = pickle.load(f)
        return scene

    def load_image_camera(self, scene, frame_idx):
        # load image
        image = scene.images[frame_idx]
        if self.transform is not None:
            image = Image.fromarray(image, mode='RGB')
            image = self.transform(image)

        # load camera
        camera = scene.cameras[frame_idx]
        camera = torch.from_numpy(camera).float()

        return image, camera

    def load_hand_image(self, scene, frame_idx):
        # load image
        image = scene.hand_images[frame_idx]
        if self.transform is not None:
            image = Image.fromarray(image, mode='RGB')
            image = self.transform(image)

        return image

    def load_proprioceptive_hand(self, scene, frame_idx):
        # load proprioceptive
        proprioceptive = scene.proprioceptives[frame_idx]
        proprioceptive = torch.from_numpy(proprioceptive).float()

        if self.do_zeromean_unitnormal:
            proprioceptive = (proprioceptive - self.ppc_mean) / (self.ppc_std + 1e-8)

        if self.do_minmax_norm:
            proprioceptive = (proprioceptive - self.ppc_min) / self.ppc_05_max_minus_min - 1.

        #proprioceptive = proprioceptive[0:22] # joint pos
        #proprioceptive = proprioceptive[22:44] # joint vel
        #proprioceptive = proprioceptive[44:57] # actu pos
        #proprioceptive = proprioceptive[57:70] # actu vel
        #proprioceptive = proprioceptive[70:83] # actu force
        #proprioceptive = proprioceptive[83:98] # acc
        #proprioceptive = proprioceptive[98:113] # gyro
        #proprioceptive = proprioceptive[113:132] # torch

        if not self.use_touch:
            proprioceptive = proprioceptive[0:113] # w/o torch

        # load hand
        hand = scene.hands[frame_idx]
        hand = torch.from_numpy(hand).float()

        ''' prune un-necessary dims '''
        # the first 13 dimensions has the same value for all data
        hand = hand[-5:]

        return proprioceptive, hand

    def __getitem__(self, scene_idx):
        raise NotImplementedError 

    def __len__(self):
        return len(self.samples)

class ImageHapticSceneDataset(SceneDataset):
    def __init__(self,
                 root, names, n_parts, split='train', transform=None,
                 target_sample_method='remaining', max_cond_size=20, max_target_size=20,
                 use_image =True,
                 use_haptic=True,
                 use_hand_image=False,
                 normalize_method=None,
                 use_touch=True,
                 do_zeromean_unitnormal=False,
                 do_minmax_norm=False,
                 img_size=64,
                 img_split_leftright=False,
                 img_split_upperlower=False,
                 img_split_rgb=False,
                 hpt_split_method=None,
                 do_modal_exclusive_observation=False,
                 expname='',
                 train_min_num_modes=1,
                 train_max_num_modes=-1,
                 shuffle_combination=True,
                 ):
        super().__init__(root=root,
                         names=names,
                         n_parts=n_parts,
                         split=split,
                         transform=transform,
                         target_sample_method=target_sample_method,
                         max_cond_size=max_cond_size,
                         max_target_size=max_target_size,
                         normalize_method=normalize_method,
                         use_touch=use_touch,
                         do_zeromean_unitnormal=do_zeromean_unitnormal,
                         do_minmax_norm=do_minmax_norm,
                         img_size=img_size,
                         )

        self.use_image = use_image
        self.use_haptic = use_haptic
        assert self.use_image or self.use_haptic
        self.use_hand_image = use_hand_image

        self.img_split_leftright = img_split_leftright
        self.img_split_upperlower = img_split_upperlower
        self.img_split_rgb = img_split_rgb
        assert self.img_split_leftright in [True, False]
        assert self.img_split_upperlower in [True, False]
        assert self.img_split_rgb in [True, False]

        self.hpt_split_method = hpt_split_method
        assert self.hpt_split_method in [None, 'semantic', 'half', 'channelwise']

        self.num_img_modalities = self.get_num_img_modalities()
        self.num_hpt_modalities = self.get_num_hpt_modalities()
        self.num_modalities = self.num_img_modalities + self.num_hpt_modalities

        self.data_dims = self.get_img_data_dims() + self.get_hpt_data_dims()

        self.do_modal_exclusive_observation = do_modal_exclusive_observation
        self.expname = expname
        self.train_min_num_modes = train_min_num_modes
        self.train_max_num_modes = train_max_num_modes if train_max_num_modes > 0 else self.num_modalities
        assert self.train_min_num_modes > 0 and self.train_max_num_modes > 0
        assert self.train_max_num_modes >= self.train_min_num_modes

        self.set_possible_joint_observations()

        if do_modal_exclusive_observation:
            self.set_valid_combinations()
            self.set_jointobs()
        else:
            self.valid_combinations = self.combinations
            self.jointobs = -np.ones(len(self.samples))

        self.shuffle_combination = shuffle_combination

    def set_jointobs(self):
        # split train and val
        os.system('mkdir -p cache/haptix/{}'.format(self.name))

        # get split text
        if self.split == 'train':
            split = 'train'
        elif self.split in ['val1', 'val2']:
            split = 'val'
        else:
            split = 'test'

        # read jointobs
        if self.split in ['train', 'val1']:
            jointobs_filename = os.path.join(
                    'cache/haptix/{}'.format(self.name),
                    'jointobs{}-m{}-{}-prob.pt'.format(
                        '-{}'.format(self.expname) if self.expname != '' else '',
                        self.num_modalities,
                        split,
                        ))
            if os.path.exists(jointobs_filename):
                self.jointobs = torch.load(jointobs_filename)
            else:
                # prob combination agnostic
                mprob = [1./i for i in range(1,self.num_modalities+1)]
                mprob = [mprob[i]/sum(mprob) for i in range(self.num_modalities)]

                # prob
                prob = [0]*len(self.valid_combinations)
                for i in range(len(prob)):
                    prob[i] = mprob[sum(self.valid_combinations[i])-1]

                # normalize
                denom = sum(prob)
                for i in range(len(prob)):
                    prob[i] /= denom

                jointobs = np.random.choice([i for i in range(len(prob))], size=len(self.samples), p=prob)
                torch.save(jointobs, open(jointobs_filename, 'wb'))
                self.jointobs = jointobs
        else:
            self.jointobs = -np.ones(len(self.samples))

    def set_valid_combinations(self):
        # split train and val
        os.system('mkdir -p cache/haptix/{}'.format(self.name))

        # get split text
        if self.split == 'train':
            split = 'train'
        elif self.split in ['val1', 'val2']:
            split = 'val'
        else:
            split = 'test'

        # read jointobs
        if self.split in ['train', 'val1']:
            jointobs_filename = os.path.join(
                    'cache/haptix/{}'.format(self.name),
                    'jointobs{}-m{}-{}.pt'.format(
                        '-{}'.format(self.expname) if self.expname != '' else '',
                        self.num_modalities,
                        split, #'train' if self.split in ['train', 'val1', 'val2'] else 'test',
                        ))
            if os.path.exists(jointobs_filename):
                self.valid_combinations = torch.load(jointobs_filename)
            else:
                valid_combinations = get_masks(self.num_modalities, min_modes=self.train_min_num_modes, max_modes=self.train_max_num_modes)

                # save
                torch.save(valid_combinations, open(jointobs_filename, 'wb'))
                self.valid_combinations = valid_combinations
        else:
            self.valid_combinations = self.combinations

    def set_possible_joint_observations(self):
        # init
        mod_indices = [idx for idx in range(self.num_modalities)]

        # split joint observations
        combinations = []
        for i in range(1, len(mod_indices)+1):
            combinations += [idx for idx in itertools.combinations(mod_indices, i)]

        # get binary masks
        binary_combinations = []
        for comb in combinations:
            binary_comb = [0]*self.num_modalities
            for i in comb:
                binary_comb[i] = 1
            binary_combinations += [tuple(binary_comb)]

        self.combinations = binary_combinations

    def get_img_data_dims(self):
        dims = [(3, self.img_size, self.img_size, 7, 'image')]

        if self.img_split_leftright:
            new_dims = []
            for dim in dims:
                new_dims += [(dim[0], dim[1], dim[2]//2,        7, 'image')]
                new_dims += [(dim[0], dim[1], dim[2]-dim[2]//2, 7, 'image')]
            dims = new_dims

        if self.img_split_upperlower:
            new_dims = []
            for dim in dims:
                new_dims += [(dim[0], dim[1]//2,        dim[2], 7, 'image')]
                new_dims += [(dim[0], dim[1]-dim[1]//2, dim[2], 7, 'image')]
            dims = new_dims

        if self.img_split_rgb:
            new_dims = []
            for dim in dims:
                new_dims += [(dim[0]//3, dim[1], dim[2], 7, 'image')]
                new_dims += [(dim[0]//3, dim[1], dim[2], 7, 'image')]
                new_dims += [(dim[0]//3, dim[1], dim[2], 7, 'image')]
            dims = new_dims

        return dims

    def get_hpt_data_dims(self):
        if self.hpt_split_method is None:
            dims = [(132, 1, 1, 7, 'haptic')]
        elif self.hpt_split_method == 'semantic':
            dims = [(15, 1, 1, 7, 'haptic'),
                    (26, 1, 1, 7, 'haptic'),
                    (20, 1, 1, 7, 'haptic'),
                    (15, 1, 1, 7, 'haptic'),
                    (17, 1, 1, 7, 'haptic'),
                    (20, 1, 1, 7, 'haptic'),
                    (19, 1, 1, 7, 'haptic')]
        elif self.hpt_split_method == 'half':
            dims = [(61, 1, 1, 7, 'haptic'),
                    (71, 1, 1, 7, 'haptic'),
                    ]
        elif self.hpt_split_method == 'channelwise':
            dims = [(22 -0,   1, 1, 7, 'haptic'),
                    (44 -22,  1, 1, 7, 'haptic'),
                    (57 -44,  1, 1, 7, 'haptic'),
                    (70 -57,  1, 1, 7, 'haptic'),
                    (83 -70,  1, 1, 7, 'haptic'),
                    (98 -83,  1, 1, 7, 'haptic'),
                    (113-98,  1, 1, 7, 'haptic'),
                    (132-113, 1, 1, 7, 'haptic')]
        else:
            raise NotImplementedError

        return dims

    def get_num_img_modalities(self):
        # num_img_modalities
        num_img_modalities = 1

        if self.img_split_leftright:
            num_img_modalities = num_img_modalities*2

        if self.img_split_upperlower:
            num_img_modalities = num_img_modalities*2

        if self.img_split_rgb:
            num_img_modalities = num_img_modalities*3

        return num_img_modalities

    def get_num_hpt_modalities(self):
        # num_hpt_modalities
        num_hpt_modalities = 1

        if self.hpt_split_method is None:
            pass
        elif self.hpt_split_method == 'semantic':
            num_hpt_modalities = 7
        elif self.hpt_split_method == 'half':
            num_hpt_modalities = 2
        elif self.hpt_split_method == 'channelwise':
            num_hpt_modalities = 8
        else:
            raise NotImplementedError

        return num_hpt_modalities 

    def check_img_option(self, rgb, leftright, upperlower):
        # rgb
        if self.img_split_rgb:
            assert rgb in ['r', 'g', 'b']
        else:
            assert rgb is None

        # leftright
        if self.img_split_leftright:
            assert leftright in ['left', 'right']
        else:
            assert leftright is None

        # upperlower
        if self.img_split_upperlower:
            assert upperlower in ['upper', 'lower']
        else:
            assert upperlower is None

    def _getimagecontext(self, scene, frame_idx, rgb=None, leftright=None, upperlower=None):
        # init
        self.check_img_option(rgb, leftright, upperlower)

        # load image, camera
        image, camera = self.load_image_camera(scene, frame_idx)

        # get size
        nchannels, height, width = image.size()

        # split leftright
        if self.img_split_leftright:
            image = image[:, :, :width//2] if leftright == 'left' else image[:, :, width//2:]

        # split upperlower
        if self.img_split_upperlower:
            image = image[:, :height//2, :] if upperlower == 'upper' else image[:, height//2:, :]

        # split rgb
        if self.img_split_rgb:
            if rgb == 'r':
                image = image[0:1, :, :]
            elif rgb == 'g':
                image = image[1:2, :, :]
            else:
                image = image[2:3, :, :]

        return image, camera

    def check_hpt_option(self, channel):
        if self.hpt_split_method is None:
            assert channel is None
        elif self.hpt_split_method == 'semantic':
            assert type(channel) == type(0) and channel in range(7)
        elif self.hpt_split_method == 'half':
            assert type(channel) == type(0) and channel in range(2)
        elif self.hpt_split_method == 'channelwise':
            assert type(channel) == type(0) and channel in range(8)
        else:
            raise NotImplementedError

    def _gethapticcontext(self, scene, frame_idx, channel=None):
        # init
        self.check_hpt_option(channel)

        # proprioceptive, hand
        proprioceptive, hand = self.load_proprioceptive_hand(scene, frame_idx)

        # split
        if self.hpt_split_method is None:
            pass
        elif self.hpt_split_method == 'semantic':
            proprioceptive = proprioceptive.index_select(0, SEMANTIC[channel])
        elif self.hpt_split_method == 'half':
            proprioceptive = proprioceptive.index_select(0, HALF[channel])
        elif self.hpt_split_method == 'channelwise':
            proprioceptive = proprioceptive[CHANNEL_WISE[channel][0]:CHANNEL_WISE[channel][1]]
        else:
            raise NotImplementedError

        # hand_image
        hand_image = self.load_hand_image(scene, frame_idx)

        return proprioceptive, hand, hand_image

    def getimagecontext(self, scene, indices, rgb=None, leftright=None, upperlower=None):
        # get images, cameras
        images, cameras = [], []
        for frame_idx in indices:
            image, camera = self._getimagecontext(scene, frame_idx, rgb, leftright, upperlower)
            images  += [image.unsqueeze(0)]
            cameras += [camera.unsqueeze(0)]
        images  = torch.cat(images, dim=0) if len(images) > 0 else None
        cameras = torch.cat(cameras, dim=0) if len(cameras) > 0 else None

        #return context
        return images, cameras

    def gethapticcontext(self, scene, frame_indices, channel=None):
        # get proprioceptives, hands
        proprioceptives, hands, hand_images = [], [], []
        for frame_idx in frame_indices:
            proprioceptive, hand, hand_image = self._gethapticcontext(scene, frame_idx, channel)
            proprioceptives += [proprioceptive.unsqueeze(0)]
            hands           += [hand.unsqueeze(0)]
            hand_images     += [hand_image.unsqueeze(0)]
        proprioceptives = torch.cat(proprioceptives, dim=0) if len(proprioceptives) > 0 else None
        hands           = torch.cat(hands, dim=0) if len(hands) > 0 else None
        hand_images     = torch.cat(hand_images, dim=0) if len(hand_images) > 0 else None

        # normalize
        proprioceptives = normalize(proprioceptives, self.normalize_method)

        #return context
        if self.use_hand_image:
            return proprioceptives, hands, hand_images
        else:
            return proprioceptives, hands

    def sample_cond_target_size(self):
        # sample cond size
        # cond_size = start from 0 if allow_empty_context. 1 otherwise
        #             end at self.max_cond_size self.allow_cond_target_overlap;
        #             on the other hand, in no-overlap case,⋅
        #             end at self.max_cond_size-1 since the size of context can be 0, but target shouldn't be.
        cond_size = np.random.randint(
                0,
                self.max_cond_size+1)

        # target indices
        if self.target_sample_method == 'remaining':
            target_size = self.max_target_size-cond_size
        elif self.target_sample_method == 'random':
            target_size = np.random.randint(0,
                    min(max(self.num_views-cond_size,
                            self.max_target_size-cond_size),
                        self.max_target_size)+1)
        elif self.target_sample_method == 'full':
            target_size = self.max_target_size
        else:
            raise NotImplementedError('unknown target_sample_method: {}'.format(self.target_sample_method))

        return cond_size, target_size

    def get_indices(self, cond_size, target_size):
        # rand perm indices
        indices = np.random.permutation(self.num_views) if self.split in ['train', 'val2'] else np.arange(self.num_views)

        '''  select context frames '''
        indices_context = [index for index in indices[:cond_size]]

        ''' select target frames '''
        indices_target = [index for index in indices[-target_size:]] if target_size > 0 else []

        return indices_context, indices_target

    def __getitem__(self, scene_idx):
        '''
        Note: the size of context can be 0 (if allow_empty_context is True), but not target
              as a result, context can be (None, None)
              however, target shouldn't
        '''

        ''' load scene '''
        # load scene
        scene = self.load_scene(scene_idx)

        # get env_info
        info = scene.env_info
        info['environment'] = self.environment
        _n_parts = len(scene.env_info['obj_infos'][0]['parts_rel'])
        offset_idx = self.n_parts.index(_n_parts)
        class_offset = self.class_offset[offset_idx]
        info['class'] = class_offset+scene.env_info['class']

        # init image input types
        rgb, upperlower, leftright = [], [], []
        if self.use_image:
            if self.img_split_rgb:
                rgb = ['r', 'g', 'b']
            else:
                rgb = [None]

            if self.img_split_upperlower:
                upperlower = ['upper', 'lower']
            else:
                upperlower = [None]

            if self.img_split_leftright:
                leftright = ['left', 'right']
            else:
                leftright = [None]

        # init haptic input types
        channels = []
        if self.use_haptic:
            if self.hpt_split_method is None:
                channels = [None]
            else:
                channels = range(self.num_hpt_modalities)

        # get mask
        if not self.do_modal_exclusive_observation:
            if self.shuffle_combination and self.split in ['train', 'val2']:
                #mask = self.combinations[np.random.choice(len(self.combinations))]
                mask = get_mask_from_combinations(self.combinations)
            else:
                mask = self.combinations[-1]
        else:
            if self.jointobs[scene_idx] > -1:
                mask = self.valid_combinations[self.jointobs[scene_idx]]
            elif self.jointobs[scene_idx] == -1 and self.shuffle_combination:
                #mask = self.valid_combinations[np.random.choice(len(self.valid_combinations))]
                mask = get_mask_from_combinations(self.valid_combinations)
            else:
                mask = self.combinations[-1]
        info['mask'] = mask

        # get cond_target_sizes
        total_target_size = 0
        while total_target_size == 0:
            img_cond_target_sizes = []
            for i, (color, lr, ul) in enumerate(itertools.product(rgb, leftright, upperlower)):
                if mask[i]:
                    cond_size, target_size = self.sample_cond_target_size()
                else:
                    cond_size, target_size = 0, self.num_views
                img_cond_target_sizes += [(cond_size, target_size)]
                total_target_size += target_size

            hpt_cond_target_sizes = []
            for ii, ch in enumerate(channels):
                i = self.num_img_modalities+ii
                if mask[i]:
                    cond_size, target_size = self.sample_cond_target_size()
                else:
                    cond_size, target_size = 0, self.num_views
                hpt_cond_target_sizes += [(cond_size, target_size)]
                total_target_size += target_size

        # init
        context, target = [], []

        # get context and target (image)
        for (cond_size, target_size), (color, lr, ul) in zip(img_cond_target_sizes, itertools.product(rgb, leftright, upperlower)):
            indices_context, indices_target = self.get_indices(cond_size, target_size)
            context += list(self.getimagecontext(scene, indices_context, color, lr, ul))
            target  += list(self.getimagecontext(scene, indices_target , color, lr, ul))

        # get context and target (haptic)
        for (cond_size, target_size), ch in zip(hpt_cond_target_sizes, channels):
            indices_context, indices_target = self.get_indices(cond_size, target_size)
            context += list(self.gethapticcontext(scene, indices_context, ch))
            target  += list(self.gethapticcontext(scene, indices_target , ch))

        # convert to tuple
        context = tuple(context)
        target  = tuple(target)

        return info, context, target


'''
collate_fn for SceneDatasets
'''
def collate_fn(batch):
    '''
    Input:
      batch: list of tuples, each of which is
             (context, target)
             where context = (images, cameras)
                   target  = (images, cameras)
    Output:
      contexts: a list, whose element is context
                where context = (images, cameras)
      targets:  a list, whose element is target
                where context = (images, cameras)
    '''
    # init
    infos = [] 
    contexts = []
    targets  = []

    # append
    for info, context, target in batch:
        infos += [info]
        contexts += [context]
        targets  += [target]

    return infos, contexts, targets


'''
get dataset loaders
'''
def get_transform(img_size=64, do_crop=False):
    trfms = []
    if do_crop:
        trfms += [transforms.CenterCrop(132)]
    trfms += [transforms.Resize(img_size), #, interpolation=Image.NEAREST),
              transforms.ToTensor()]
    return transforms.Compose(trfms)

def get_haptix_shepard_metzler_n_parts(train_batch_size,
                                       eval_batch_size,
                                       kwargs,
                                       n_parts=[5],
                                       target_sample_method='remaining',
                                       max_cond_size=20,
                                       max_target_size=20,
                                       use_image=True,
                                       use_haptic=True,
                                       normalize_method=None,
                                       use_touch=True,
                                       do_zeromean_unitnormal=False,
                                       do_minmax_norm=False,
                                       img_size=64,
                                       do_crop=False,
                                       img_split_leftright=False,
                                       img_split_upperlower=False,
                                       img_split_rgb=False,
                                       hpt_split_method=None,
                                       do_modal_exclusive_observation=False,
                                       expname='',
                                       train_min_num_modes=1,
                                       train_max_num_modes=-1,
                                       ):

    # init name
    names = []
    for _n_parts in n_parts:
        name = 'shepard_metzler_{}_parts'.format(_n_parts)
        names += [name]

    # init dataset (train / val)
    train_dataset = ImageHapticSceneDataset(
            root='data/haptix',
            names=names,
            n_parts=n_parts,
            split='train',
            transform=get_transform(img_size, do_crop),
            img_size=img_size,
            target_sample_method=target_sample_method,
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            use_image=use_image,
            use_haptic=use_haptic,
            normalize_method=normalize_method,
            use_touch=use_touch,
            do_zeromean_unitnormal=do_zeromean_unitnormal,
            do_minmax_norm=do_minmax_norm,
            img_split_leftright=img_split_leftright,
            img_split_upperlower=img_split_upperlower,
            img_split_rgb=img_split_rgb,
            hpt_split_method=hpt_split_method,
            do_modal_exclusive_observation=do_modal_exclusive_observation,
            expname=expname,
            train_min_num_modes=train_min_num_modes,
            train_max_num_modes=train_max_num_modes,
            )
    val1_dataset = ImageHapticSceneDataset(
            root='data/haptix',
            names=names,
            n_parts=n_parts,
            split='val1',
            transform=get_transform(img_size, do_crop),
            img_size=img_size,
            target_sample_method='remaining', #'full',
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            use_image=use_image,
            use_haptic=use_haptic,
            normalize_method=normalize_method,
            use_touch=use_touch,
            do_zeromean_unitnormal=do_zeromean_unitnormal,
            do_minmax_norm=do_minmax_norm,
            img_split_leftright=img_split_leftright,
            img_split_upperlower=img_split_upperlower,
            img_split_rgb=img_split_rgb,
            hpt_split_method=hpt_split_method,
            do_modal_exclusive_observation=do_modal_exclusive_observation,
            expname=expname,
            train_min_num_modes=train_min_num_modes,
            train_max_num_modes=train_max_num_modes,
            )
    val2_dataset = ImageHapticSceneDataset(
            root='data/haptix',
            names=names,
            n_parts=n_parts,
            split='val2',
            transform=get_transform(img_size, do_crop),
            img_size=img_size,
            target_sample_method='remaining', #'full',
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            use_image=use_image,
            use_haptic=use_haptic,
            normalize_method=normalize_method,
            use_touch=use_touch,
            do_zeromean_unitnormal=do_zeromean_unitnormal,
            do_minmax_norm=do_minmax_norm,
            img_split_leftright=img_split_leftright,
            img_split_upperlower=img_split_upperlower,
            img_split_rgb=img_split_rgb,
            hpt_split_method=hpt_split_method,
            do_modal_exclusive_observation=do_modal_exclusive_observation,
            expname=expname,
            train_min_num_modes=train_min_num_modes,
            train_max_num_modes=train_max_num_modes,
            )
    test_dataset = ImageHapticSceneDataset(
            root='data/haptix',
            names=names,
            n_parts=n_parts,
            split='test',
            transform=get_transform(img_size, do_crop),
            img_size=img_size,
            target_sample_method='remaining', #'full',
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            use_image=use_image,
            use_haptic=use_haptic,
            normalize_method=normalize_method,
            use_touch=use_touch,
            do_zeromean_unitnormal=do_zeromean_unitnormal,
            do_minmax_norm=do_minmax_norm,
            img_split_leftright=img_split_leftright,
            img_split_upperlower=img_split_upperlower,
            img_split_rgb=img_split_rgb,
            hpt_split_method=hpt_split_method,
            do_modal_exclusive_observation=do_modal_exclusive_observation,
            expname=expname,
            train_min_num_modes=train_min_num_modes,
            train_max_num_modes=train_max_num_modes,
            )

    # init dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    val1_loader = torch.utils.data.DataLoader(val1_dataset,
            batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)
    val2_loader = torch.utils.data.DataLoader(val2_dataset,
            batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)

    # init info
    info = {}
    info['nviews'] = train_dataset.num_views
    info['max_cond_size'] = train_dataset.max_cond_size
    info['max_target_size'] = train_dataset.max_target_size
    info['target_sample_method'] = train_dataset.target_sample_method
    info['num_img_modalities'] = train_dataset.num_img_modalities
    info['num_hpt_modalities'] = train_dataset.num_hpt_modalities
    info['num_modalities'] = train_dataset.num_modalities
    info['use_image'] = train_dataset.use_image
    info['use_haptic'] = train_dataset.use_haptic
    info['dims'] = train_dataset.data_dims

    return train_loader, val1_loader, val2_loader, test_loader, info


'''
get dataset
'''
def get_haptix_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False):
    # init arguments
    if eval_batch_size is None:
        eval_batch_size = train_batch_size
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    ################ M=2
    if dataset == 'haptix-shepard_metzler_5_parts':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-extrapol11':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol11',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=1,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-intrapol22':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol22',
                                                  train_min_num_modes=2,
                                                  train_max_num_modes=2,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-trm1':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='trm1',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=1,
                                                  )

    ################ M=2, 48
    if dataset == 'haptix-shepard_metzler_5_parts-48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  expname='crop48',
                                                  )

    ################ M=3, 48
    if dataset == 'haptix-shepard_metzler_5_parts-48-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol12',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=2,
                                                  )

    ################ M=5
    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol12',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=2,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-intrapol35':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol35',
                                                  train_min_num_modes=3,
                                                  train_max_num_modes=5,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-intrapol45':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol45',
                                                  train_min_num_modes=4,
                                                  train_max_num_modes=5,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-inextrapol23':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol23',
                                                  train_min_num_modes=2,
                                                  train_max_num_modes=3,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-inextrapol34':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='inextrapol34',
                                                  train_min_num_modes=3,
                                                  train_max_num_modes=4,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-inextrapol15':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='inextrapol15',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=5,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-trm1':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='trm1',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=1,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-ul-lr-trm3':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='trm3',
                                                  train_min_num_modes=3,
                                                  train_max_num_modes=3,
                                                  )

    ################ M=8, 48
    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol12',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=2,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-extrapol14':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol14',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=4,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol48',
                                                  train_min_num_modes=4,
                                                  train_max_num_modes=8,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol78':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol78',
                                                  train_min_num_modes=7,
                                                  train_max_num_modes=8,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-inextrapol18':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='inextrapol18',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=8,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-inextrapol36':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='inextrapol36',
                                                  train_min_num_modes=3,
                                                  train_max_num_modes=6,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-lr-rgb-half-inextrapol45':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='inextrapol45',
                                                  train_min_num_modes=4,
                                                  train_max_num_modes=5,
                                                  )

    ################ M=14, 48
    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol12',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=2,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol14':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol14',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=4,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol17':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol17',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=7,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-intrapol814':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol814',
                                                  train_min_num_modes=8,
                                                  train_max_num_modes=14,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-intrapol1114':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol1114',
                                                  train_min_num_modes=11,
                                                  train_max_num_modes=14,
                                                  )

    elif dataset == 'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-inextrapol114':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[5],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  img_split_rgb=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  hpt_split_method='half',
                                                  do_modal_exclusive_observation=True,
                                                  expname='inextrapol1114',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=14,
                                                  )
    else:
        raise NotImplementedError('dataset: {}'.format(dataset))
