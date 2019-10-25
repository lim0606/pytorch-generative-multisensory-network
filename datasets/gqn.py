'''
copied and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/dataset.py
'''
import os
import io
import sys
import random
import shutil
import collections
import gzip
from typing import NamedTuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


Scene = collections.namedtuple('Scene', ['frames', 'cameras']) ## only for human reading

class SceneDataset(Dataset):
    """
    copied and modified from https://github.com/l3robot/pytorch-ConvDraw/blob/master/src/dataset.py
    """
    def __init__(self, root, name, img_size, shm=False, train=True,
                 allow_empty_context=False, target_sample_method='remaining', max_cond_size=20, max_target_size=20):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.name = name  # os.path.split(root)[-1]
        self.train = train
        self.path = os.path.join(self.root, name+'-torch-compressed', 'train' if self.train else 'test')
        if shm:
            shm_root = f'/dev/shm/{self.name}'
            try:
                shutil.copytree(self.root, shm_root)
            except FileExistsError:
                print(' [+] data already in RAM')
            self.root = shm_root
        self.transform = None
        self.__load_images_path()
        self.num_views = len(self.__load_scene(0).frames)
        self.allow_empty_context = allow_empty_context
        assert target_sample_method in ['remaining', 'random', 'full']
        self.target_sample_method = target_sample_method
        self.max_cond_size = min(self.num_views, max_cond_size)
        self.max_target_size = min(self.num_views, max_target_size)

    def __jpeg2numpy(self, raw):
        arr = np.fromstring(raw, np.uint8)
        return cv2.resize(cv2.imdecode(arr, cv2.IMREAD_COLOR), (self.img_size, self.img_size))[:,:,[2,1,0]] ## to RGB
        # return (img - np.array([0.046732664, 0.012828712, 0.047347005])) / np.array([0.14458965, 0.06529137, 0.15832743])

    def __jpeg2tensor(self, raw):
        img = self.__jpeg2numpy(raw)
        img = np.rollaxis(img, 2, 0) / 255
        return torch.from_numpy(img).float()

    def __load_images_path(self):
        img_path_list = [scene for scene in os.listdir(self.path) \
                         if os.path.splitext(scene)[-1] == '.gz']
        self.samples = [os.path.join(self.path, scene) for scene in img_path_list]

    def __load_scene(self, idx):
        sys.modules['__main__'].Scene = Scene ## hack because of the pickle saving..
        with gzip.open(self.samples[idx], 'rb') as f:
            scene = torch.load(f)
        return scene

    def getitem(self, scene_idx, frame_idx):
        scene = self.__load_scene(scene_idx)
        return (self.__jpeg2tensor(scene.frames[frame_idx]),
                torch.from_numpy(scene.cameras[0][frame_idx]))

    def __getitem__(self, scene_idx):
        '''
        Note: the size of context can be 0 (if allow_empty_context is True), but not target
              as a result, context can be (None, None)
              however, target shouldn't
        '''
        # rand perm indices
        indices = np.random.permutation(self.num_views)

        # select context frames
        # cond_size = start from 0 if allow_empty_context. 1 otherwise
        #             end at self.max_cond_size self.allow_cond_target_overlap;
        #             on the other hand, in no-overlap case, 
        #             end at self.max_cond_size-1 since the size of context can be 0, but target shouldn't be.
        cond_size = np.random.randint(
                0 if self.allow_empty_context else 1,
                self.max_cond_size if self.target_sample_method == 'remaining' else self.max_cond_size+1)
                #self.max_cond_size+1 if self.allow_cond_target_overlap else self.max_cond_size)
        indices_context   = [index for index in indices[:cond_size]]

        # select target frames
        #target_size = np.random.randint(self.max_target_size-cond_size, self.max_target_size+1) if self.allow_cond_target_overlap else self.max_target_size-cond_size
        if self.target_sample_method == 'remaining':
            target_size = self.max_target_size-cond_size
        elif self.target_sample_method == 'random':
            target_size = np.random.randint(self.max_target_size-cond_size, self.max_target_size+1)
        elif self.target_sample_method == 'full':
            target_size = self.max_target_size
        else:
            raise NotImplementedError('unknown target_sample_method: {}'.format(self.target_sample_method))
        indices_target = [index for index in indices[-target_size:]]

        # build context
        images, cameras = [], []
        for frame_idx in indices_context:
            image, camera = self.getitem(scene_idx, frame_idx)
            images  += [image.unsqueeze(0)]
            cameras += [camera.unsqueeze(0)]
        images  = torch.cat(images, dim=0) if len(images) > 0 else None
        cameras = torch.cat(cameras, dim=0) if len(cameras) > 0 else None
        context = (images, cameras)

        # build target 
        images, cameras = [], []
        for frame_idx in indices_target:
            image, camera = self.getitem(scene_idx, frame_idx)
            images  += [image.unsqueeze(0)]
            cameras += [camera.unsqueeze(0)]
        assert len(images) > 0 and len(cameras) > 0
        images  = torch.cat(images, dim=0)
        cameras = torch.cat(cameras, dim=0)
        target  = (images, cameras)

        return None, context, target

    def __len__(self):
        return len(self.samples)

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
    contexts = []
    targets  = []

    # append
    for context, target in batch:
        contexts += [context]
        targets  += [target]

    return None, contexts, targets

def get_gqn_dataset_with_name(name,
                              train_batch_size,
                              eval_batch_size,
                              kwargs,
                              allow_empty_context=False,
                              target_sample_method='remaining',
                              max_cond_size=20,
                              max_target_size=20,
                              num_data=None
                              ):
    # init dataset (train / val)
    train_dataset = SceneDataset(
            root='data/gqn-datasets',
            name=name,
            train=True,
            img_size=64,
            allow_empty_context=allow_empty_context,
            target_sample_method=target_sample_method,
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            )
    val_dataset = SceneDataset(
            root='data/gqn-datasets',
            name=name,
            train=True,
            img_size=64,
            allow_empty_context=allow_empty_context,
            target_sample_method='remaining', #'full',
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            )
    test_dataset = SceneDataset(
            root='data/gqn-datasets',
            name=name,
            train=False,
            img_size=64,
            allow_empty_context=allow_empty_context,
            target_sample_method='remaining', #'full',
            max_cond_size=max_cond_size,
            max_target_size=max_target_size,
            )

    # set num data
    if num_data is not None:
        num_data = min(len(train_dataset.samples), num_data)
        suffix = num_data
        train_dataset.samples = train_dataset.samples[:num_data]
        val_dataset.samples   = val_dataset.samples[:num_data]
    else:
        num_data = len(train_dataset.samples)
        suffix = None

    # split train and val
    os.system('mkdir -p cache/gqn-datasets/{}'.format(name))
    split_filename = os.path.join(
            'cache/gqn-datasets/{}'.format(name),
            'split-{}.pt'.format(suffix) if suffix is not None else 'split.pt'
            )
    if os.path.exists(split_filename):
        indices = torch.load(split_filename)
    else:
        indices = torch.from_numpy(np.random.permutation(num_data))
        torch.save(indices, open(split_filename, 'wb'))
    train_dataset.samples = [train_dataset.samples[index] for index in indices[:num_data-20000]]
    val_dataset.samples   = [val_dataset.samples[index]   for index in indices[num_data-20000:]]

    # init dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)

    # init info
    info = {}
    info['nviews'] = train_dataset.num_views
    info['max_cond_size'] = train_dataset.max_cond_size
    info['max_target_size'] = train_dataset.max_target_size
    info['allow_empty_context'] = train_dataset.allow_empty_context
    info['target_sample_method'] = train_dataset.target_sample_method

    return train_loader, val_loader, test_loader, info

def get_gqn_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False):
    # init arguments
    if eval_batch_size is None:
        eval_batch_size = train_batch_size
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # get dataset
    if dataset in [
            #'jaco',
            'shepard_metzler_5_parts',
            'shepard_metzler_7_parts',
            'mazes',
            ]:
        return get_gqn_dataset_with_name(dataset, train_batch_size, eval_batch_size, kwargs,
                                         allow_empty_context=False,
                                         target_sample_method='remaining', #'full',
                                         max_cond_size=20,
                                         max_target_size=20,
                                         )
    elif dataset in [
            'rooms_ring_camera',
            'rooms_free_camera_no_object_rotations',
            'rooms_free_camera_with_object_rotations',
            ]:
        return get_gqn_dataset_with_name(dataset, train_batch_size, eval_batch_size, kwargs,
                                         allow_empty_context=False,
                                         target_sample_method='remaining', #'full',
                                         max_cond_size=5,
                                         max_target_size=20,
                                         )
    elif dataset == 'shepard_metzler_5_parts-10k':
        return get_gqn_dataset_with_name('shepard_metzler_5_parts',
                                         train_batch_size, eval_batch_size, kwargs,
                                         allow_empty_context=False,
                                         target_sample_method='remaining', #'full',
                                         max_cond_size=20,
                                         max_target_size=20,
                                         num_data=10000
                                         )
    elif dataset == 'shepard_metzler_5_parts-100k':
        return get_gqn_dataset_with_name('shepard_metzler_5_parts',
                                         train_batch_size, eval_batch_size, kwargs,
                                         allow_empty_context=False,
                                         target_sample_method='remaining', #'full',
                                         max_cond_size=20,
                                         max_target_size=20,
                                         num_data=100000
                                         )
    else:
        raise NotImplementedError('dataset: {}'.format(dataset))
