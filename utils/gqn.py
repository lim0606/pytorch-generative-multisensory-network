'''
miscellaneous functions: gqn 
'''
import itertools
import numpy as np
import torch

from utils.rl import r_from_pitch_yaw, r_from_roll_yaw
from utils.rl import gen_ornstein_uhlenbeck_process


''' for gqn '''
def batch_to_device(batch, device):
    for i in range(len(batch)):
        _batch = []
        for j in range(len(batch[i])):
            _batch += [batch[i][j].to(device) if batch[i][j] is not None else None]
        batch[i] = tuple(_batch)
    return batch

def merge_two_batch(context, target):
    assert len(context) == len(target)
    assert len(context[0]) == len(target[0])
    num_episodes = len(context)
    tuple_size = len(context[0])
    batch = [None]*num_episodes
    for i in range(num_episodes):
        _batch = []
        for j in range(tuple_size):
            if context[i][j] is None and target[i][j] is None:
                #raise ValueError('empty tuple')
                _batch += [None]
            elif context[i][j] is None:
                _batch += [target[i][j]]
            elif target[i][j] is None:
                _batch += [context[i][j]]
            else:
                _batch += [torch.cat([context[i][j], target[i][j]], dim=0)]
        batch[i] = tuple(_batch)
    return batch

def pack_sequence(sequences):
    lengths = [v.size(0) for v in sequences]
    return torch.cat(sequences, dim=0)

def unpack_sequence(packed_sequences, lengths):
    i = 0
    sequences = []
    for length in lengths:
        sequences += [packed_sequences[i:i+length]]
        i += length
    return sequences

def pad_sequence(sequences, lengths):
    # init
    sizes = [1] + list(sequences[0].size())
    sizes[1] = 1
    max_length = max(lengths)
    buff = sequences[0].new_zeros(*sizes)

    _sequences = []
    for i, sequence in enumerate(sequences):
        # get pad_size
        pad_size = max_length - lengths[i]
        if pad_size > 0:
            sizes[1] = pad_size
            _sequence = torch.cat([
                sequence.unsqueeze(0),
                buff.expand(*sizes),
                ], dim=1)
        else:
            _sequence = sequence.unsqueeze(0)

        # append to list
        _sequences += [_sequence]

    # concat
    padded_sequence = torch.cat(_sequences, dim=0)
    return padded_sequence

def get_reversed_tensor(tensor):
    # init
    num_episodes = tensor.size(0)

    # get rev idx
    rev_idx = torch.arange(num_episodes-1, -1, -1).long()
    if rev_idx.device != tensor.device:
        rev_idx = rev_idx.to(tensor.device)

    # reverse
    rev_tensor = tensor.index_select(0, rev_idx)

    return rev_tensor

def get_reversed_sequence(sequences):
    rev_sequence = [get_reversed_tensor(sequence) for sequence in sequences]
    return rev_sequence

def sort_padded_sequence(padded_sequence, lengths):
    # init
    device = padded_sequence.device

    # get idx and inv_idx
    idx = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    inv_idx = sorted(range(len(idx)), key=idx.__getitem__)

    # update lengths with idx
    sorted_lengths = [lengths[idx[i]] for i in range(len(idx))]

    # convert to torch tensor
    idx = torch.Tensor(idx).long()
    inv_idx = torch.Tensor(inv_idx).long()
    if idx.device != device:
        idx = idx.to(device)
        inv_idx = inv_idx.to(device)

    # get packed_reps_context
    sorted_padded_sequence = padded_sequence.index_select(0, idx)

    return idx, inv_idx, sorted_padded_sequence, sorted_lengths


''' for renderer '''
def broadcast_representation(reps, num_episodes, batch_sizes, indices=[]):
    # init size
    #_, rch, rnh, rnw = reps.size()
    sizes = list(reps.size())
    indices = range(num_episodes) if len(indices) == 0 else indices

    # post-processing representation
    reps = torch.chunk(reps, num_episodes, 0)
    new_reps = []
    #for i in range(num_episodes):
    for i, idx in enumerate(indices):
        batch_size = batch_sizes[i]
        sizes[0] = batch_size
        #new_reps += [reps[idx].expand(batch_size, rch, rnh, rnw)]
        new_reps += [reps[idx].expand(*sizes)]
    new_reps = torch.cat(new_reps, dim=0)

    return new_reps

def sum_tensor_per_episode(tensor, batch_sizes, target_indices=None, num_episodes=None):
    if target_indices is None:
        num_episodes = len(batch_sizes) 
        target_indices = [i for i in range(num_episodes)]
    #if tensor.dim() > 1:
    #    num_data = tensor.size(0)
    #    tensor = tensor.view(num_data, -1)
    #tensors = []
    start_index = 0
    b_idx = 0
    value = tensor.new_zeros(num_episodes)
    #for i, batch_size in enumerate(batch_sizes):
    for i, t_idx in enumerate(target_indices):
        batch_size = batch_sizes[i]
        val = torch.sum(tensor[start_index:start_index+batch_size], dim=0, keepdim=True)
        value[t_idx:t_idx+1] += val
        start_index += batch_size
    #return torch.cat(tensors, dim=0)
    return value


''' for visualization '''
def normalize_angle(angle):
    return (angle + 2*np.pi) % (2*np.pi)

def sample_camera(pitch, yaw, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False, flip_pitch=True):
    # init
    x0 = 0
    y0 = 0
    z0 = 0 
    #radius = 1.0

    # normalize_angle
    if use_normalize_angle:
        pitch = normalize_angle(pitch)

    # flip
    if flip_pitch:
        _pitch = -pitch
    else:
        _pitch = pitch

    if flip_yaw:
        _yaw = -yaw
    else:
        _yaw = yaw

    # rot_mat
    rot_mat = r_from_pitch_yaw(_pitch, _yaw)

    # get camera head and tail
    #head = np.array([x0, y0, z0]) - rot_mat.dot(np.array([radius-0.17, 0., 0.]))
    tail = np.array([x0, y0, z0]) - rot_mat.dot(np.array([radius, 0., 0.]))

    if swap_pitch_yaw:
        return np.array([tail[0], tail[1], tail[2], yaw, pitch])
    else:
        return np.array([tail[0], tail[1], tail[2], pitch, yaw])

def sample_queries(nrow=4, ncol=8, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False, flip_pitch=True):
    '''
    # for haptix
    queries = sample_queries(nrow=4, ncol=8, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False, flip_pitch=True)

    # for deepmind's
    queries = sample_queries(nrow=4, ncol=8, radius=3.5, swap_pitch_yaw=True, use_normalize_angle=False, flip_yaw=True, flip_pitch=True)
    '''

    #margin = 0.02
    #pitch = np.repeat(np.linspace(-np.pi/2.+margin, np.pi/2.-margin, num=nrow, endpoint=True), (ncol,))
    #yaw = np.tile(np.linspace(0., 2.*np.pi, num=ncol, endpoint=True), (nrow,))
    pitch = np.repeat(np.linspace(-np.pi/4., np.pi/4., num=nrow, endpoint=True), (ncol,))
    yaw = np.tile(np.linspace(0+np.pi/5., 2.*np.pi-np.pi/5., num=ncol, endpoint=True), (nrow,))

    queries = []
    for p, y in zip(pitch, yaw):
        queries += [sample_camera(p, y, radius, swap_pitch_yaw, use_normalize_angle, flip_yaw, flip_pitch)]

    # conver to torch tensor
    queries = torch.from_numpy(np.array(queries)).float()

    return queries

def sample_wrist(pitch, yaw, radius=0.17):
    # init
    x0 = 0
    y0 = 0
    z0 = 0
    #radius = 0.17  # 0.38 - 0.21

    # sample height
    height_low = 0.01 #0.11
    height_high = 0.04 #0.14
    height = 0.05 #0.04 #np.random.uniform(height_low, height_high)

    ## sample pitch yaw
    #pitch = normalize_ang(np.random.uniform(low=-np.pi/2., high=np.pi/2))
    #yaw   = np.random.uniform(low=0., high=2*np.pi)
    
    # normalize_angle
    pitch = normalize_angle(pitch)

    # rot_mat
    #rot_mat = r_from_pitch_yaw(pitch, yaw)
    rot_mat = r_from_roll_yaw(pitch, yaw)

    # get camera head and tail
    head = np.array([x0, y0, z0]) - rot_mat.dot(np.array([0., radius-0.1, -height]))
    tail = np.array([x0, y0, z0]) - rot_mat.dot(np.array([0., radius, -height]))

    ## sample horizon
    #horizon_low = 20
    #horizon_high = 30
    #t_horizon = np.random.randint(horizon_low, horizon_high+1)
    t_horizon = 25
    return head, tail, pitch, yaw, t_horizon

#def sample_hand_queries(
#        nrow=4,
#        ncol=8,
#        radius=0.17,
#        dt=0.04,
#        env_action_space_high=np.array(
#            [  1.57     ,   0.79     ,   1.       ,   2.1      ,   1.       ,
#               1.       ,   1.3      ,   0.34     ,   1.6      ,   1.6      ,
#               1.6      ,   0.34     ,   1.6      , 100.       , 100.       ,
#               100.     ,   6.2831855,   6.2831855], dtype=np.float32)):
#    '''
#    # for haptix
#    queries = sample_hand_queries(nrow=4, ncol=8, radius=0.17)
#    '''
#    #margin = 0.02
#    #pitch = np.repeat(np.linspace(-np.pi/2.+margin, np.pi/2.-margin, num=nrow, endpoint=True), (ncol,))
#    #yaw = np.tile(np.linspace(0., 2.*np.pi, num=ncol, endpoint=True), (nrow,))
#    pitch = np.repeat(np.linspace(-np.pi/4., np.pi/4., num=nrow, endpoint=True), (ncol,))
#    yaw = np.tile(np.linspace(0+np.pi/5., 2.*np.pi-np.pi/5., num=ncol, endpoint=True), (nrow,))
#
#    queries = []
#    for p, y in zip(pitch, yaw):
#        # select action
#        hand_head, hand_tail, hand_pitch, hand_yaw, actuator_convergence_frames = sample_wrist(p, y, radius)
#
#        # run multiple step for actuators to converge
#        action = np.zeros(18)
#        for acf in range(actuator_convergence_frames):
#            # apply heuristic policy
#            action = gen_ornstein_uhlenbeck_process(action, dt=dt, sig=0., mu=env_action_space_high)
#
#            # apply wrist pose and direction (pitch, yaw)
#            action[13] = hand_tail[0] # hand pos
#            action[14] = hand_tail[1] # hand pos
#            action[15] = hand_tail[2] # hand pos  0.11 / 0.12 / 0.13 / 0.14 ellipsoid
#            action[16] = hand_pitch
#            action[17] = hand_yaw # hand pos  0.11 / 0.12 / 0.13 / 0.14 ellipsoid
#
#        # add to queries
#        queries += [action]
#
#    # conver to torch tensor
#    queries = torch.from_numpy(np.array(queries)).float()
#
#    return queries

def sample_hand_queries(nrow=4, ncol=8, radius=0.17):
    return sample_queries(nrow, ncol, radius)

def sample_random_queries(nqueries=5, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False, flip_pitch=True):
    '''
    # for haptix
    queries = sample_random_queries(nrow=4, ncol=8, radius=1.0, swap_pitch_yaw=False, use_normalize_angle=True, flip_yaw=False, flip_pitch=True)

    # for deepmind's
    queries = sample_random_queries(nrow=4, ncol=8, radius=3.5, swap_pitch_yaw=True, use_normalize_angle=False, flip_yaw=True, flip_pitch=True)
    '''

    #margin = 0.02
    #pitch = np.repeat(np.linspace(-np.pi/2.+margin, np.pi/2.-margin, num=nrow, endpoint=True), (ncol,))
    #yaw = np.tile(np.linspace(0., 2.*np.pi, num=ncol, endpoint=True), (nrow,))
    pitch = np.random.uniform(-np.pi/4., np.pi/4., size=(nqueries,))
    yaw = np.random.uniform(0+np.pi/5., 2.*np.pi-np.pi/5., size=(nqueries,))

    queries = []
    for p, y in zip(pitch, yaw):
        queries += [sample_camera(p, y, radius, swap_pitch_yaw, use_normalize_angle, flip_yaw, flip_pitch)]

    # conver to torch tensor
    queries = torch.from_numpy(np.array(queries)).float()

    return queries

def sample_random_hand_queries(nqueries=5, radius=0.17):
    return sample_random_queries(nqueries, radius)

# define msc functions
def compare_part(part1, part2):
    is_same = 0
    for i in range(3):
        is_same += int(abs(part1[i] - part2[i]) < 1e-3)
    return is_same == 3

def compare_shape(shape1, shape2):
    is_same = 0
    for i in range(len(shape1)):
        is_same += 1 if compare_part(shape1[i], shape2[i]) else 0
    return is_same == len(shape1)

def compare_shape_with_shapelist(shape, shapelist):
    is_same = False
    for j in range(len(shapelist)):
        is_same = compare_shape(shape, shapelist[j])
        if is_same:
            break
    return is_same

def get_label(shape, shapelist):
    is_same = False
    for j in range(len(shapelist)):
        is_same = compare_shape(shape, shapelist[j])
        if is_same:
            break
    return j

''' for evaluation '''
#def trim_context(context, num_context, use_img=True, use_hpt=True):
#    num_episodes = len(context)
#    if num_context > 0:
#        new_context = [(
#            context[i][0][:num_context] if use_img else None,
#            context[i][1][:num_context] if use_img else None,
#            context[i][2][:num_context] if use_hpt else None,
#            context[i][3][:num_context] if use_hpt else None,
#            ) for i in range(num_episodes)]
#    else:
#        new_context = [(None, None, None, None) for i in range(num_episodes)]
#    return new_context

def trim_context_target(batch, num_context, use_img=True, use_hpt=True, nviews=15):
    num_episodes = len(batch)
    if num_context == 0:
        new_context = [(None, None, None, None) for i in range(num_episodes)]
        new_target = [(batch[i][0], batch[i][1], batch[i][2], batch[i][3]) for i in range(num_episodes)]
    elif num_context > 0 and num_context < nviews:
        new_context = [(
            batch[i][0][:num_context] if use_img and batch[i][0] is not None else None,
            batch[i][1][:num_context] if use_img and batch[i][1] is not None else None,
            batch[i][2][:num_context] if use_hpt and batch[i][2] is not None else None,
            batch[i][3][:num_context] if use_hpt and batch[i][3] is not None else None,
            ) for i in range(num_episodes)]
        new_target = [(
            batch[i][0][num_context:] if use_img and batch[i][0] is not None else batch[i][0],
            batch[i][1][num_context:] if use_img and batch[i][1] is not None else batch[i][1],
            batch[i][2][num_context:] if use_hpt and batch[i][2] is not None else batch[i][2],
            batch[i][3][num_context:] if use_hpt and batch[i][3] is not None else batch[i][3],
            ) for i in range(num_episodes)]
    elif num_context == nviews:
        new_context = [(
            batch[i][0][:num_context] if use_img and batch[i][0] is not None else None,
            batch[i][1][:num_context] if use_img and batch[i][1] is not None else None,
            batch[i][2][:num_context] if use_hpt and batch[i][2] is not None else None,
            batch[i][3][:num_context] if use_hpt and batch[i][3] is not None else None,
            ) for i in range(num_episodes)]
        new_target = [(
            None if use_img and batch[i][0] is not None else batch[i][0],
            None if use_img and batch[i][1] is not None else batch[i][1],
            None if use_hpt and batch[i][2] is not None else batch[i][2],
            None if use_hpt and batch[i][3] is not None else batch[i][3],
            ) for i in range(num_episodes)]
        #new_context = [(
        #    batch[i][0][:num_context] if use_img and batch[i][0] is not None else None,
        #    batch[i][1][:num_context] if use_img and batch[i][1] is not None else None,
        #    batch[i][2][:num_context] if use_hpt and batch[i][2] is not None else None,
        #    batch[i][3][:num_context] if use_hpt and batch[i][3] is not None else None,
        #    ) for i in range(num_episodes)]
        #new_target = [(None, None, None, None) for i in range(num_episodes)]
    return new_context, new_target

def new_trim_context_target(batch, num_context, mask, num_modalities, nviews=15):
    assert len(mask) == num_modalities
    num_episodes = len(batch)
    if num_context == 0:
        new_context = [tuple([None]*num_modalities*2) for i in range(num_episodes)]
        new_target = [tuple([batch[i][j] for j in range(num_modalities*2)]) for i in range(num_episodes)]
    elif num_context > 0 and num_context < nviews:
        new_context = [tuple([
            batch[i][j][:num_context] if mask[j//2] and batch[i][j] is not None else None for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
        new_target = [tuple([
            batch[i][j][num_context:] if mask[j//2] and batch[i][j] is not None else batch[i][j] for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
    elif num_context == nviews:
        new_context = [tuple([
            batch[i][j][:num_context] if mask[j//2] and batch[i][j] is not None else None for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
        new_target = [tuple([
            None if mask[j//2] and batch[i][j] is not None else batch[i][j] for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
    return new_context, new_target

def binary_trim_context_target(batch, img_num_context, hpt_num_context, num_modalities, nviews=15):
    assert num_modalities == 2
    num_episodes = len(batch)
    #if num_context == 0:
    cxt_episodes = []
    tgt_episodes = []
    for i in range(num_episodes):
        #cxt_episode = []
        #tgt_episode = []

        # image
        if img_num_context == 0:
            img_cxt = None
            cam_cxt = None
            img_tgt = batch[i][0]
            cam_tgt = batch[i][1]
        elif img_num_context > 0 and img_num_context < nviews:
            img_cxt = batch[i][0][:img_num_context]# if mask[0] and batch[i][0] is not None else None
            cam_cxt = batch[i][1][:img_num_context]# if mask[0] and batch[i][1] is not None else None
            img_tgt = batch[i][0][img_num_context:]# if mask[0] and batch[i][0] is not None else batch[i][0]
            cam_tgt = batch[i][1][img_num_context:]# if mask[0] and batch[i][1] is not None else batch[i][1]
        elif img_num_context == nviews:
            img_cxt = batch[i][0][:img_num_context]# if mask[0] and batch[i][0] is not None else None
            cam_cxt = batch[i][1][:img_num_context]# if mask[0] and batch[i][1] is not None else None
            img_tgt = None# if mask[0] and batch[i][0] is not None else batch[i][0]
            cam_tgt = None# if mask[0] and batch[i][1] is not None else batch[i][1]

        # haptic
        if hpt_num_context == 0:
            hpt_cxt = None
            hnd_cxt = None
            hpt_tgt = batch[i][2]
            hnd_tgt = batch[i][3]
        elif hpt_num_context > 0 and hpt_num_context < nviews:
            hpt_cxt = batch[i][2][:hpt_num_context]# if mask[1] and batch[i][2] is not None else None
            hnd_cxt = batch[i][3][:hpt_num_context]# if mask[1] and batch[i][3] is not None else None
            hpt_tgt = batch[i][2][hpt_num_context:]# if mask[1] and batch[i][2] is not None else batch[i][0]
            hnd_tgt = batch[i][3][hpt_num_context:]# if mask[1] and batch[i][3] is not None else batch[i][1]
        elif hpt_num_context == nviews:
            hpt_cxt = batch[i][2][:hpt_num_context]# if mask[1] and batch[i][2] is not None else None
            hnd_cxt = batch[i][3][:hpt_num_context]# if mask[1] and batch[i][3] is not None else None
            hpt_tgt = None# if mask[1] and batch[i][2] is not None else batch[i][2]
            hnd_tgt = None# if mask[1] and batch[i][3] is not None else batch[i][3]

        # make a tuple
        #cxt_episode = tuple(cxt_episode)
        #tgt_episode = tuple(tgt_episode)
        cxt_episode = (img_cxt, cam_cxt, hpt_cxt, hnd_cxt)
        cxt_episodes += [cxt_episode]
        tgt_episode = (img_tgt, cam_tgt, hpt_tgt, hnd_tgt)
        tgt_episodes += [tgt_episode]
    #return new_context, new_target
    return cxt_episodes, tgt_episodes

def binary_trim_context_target_with_hand_image(batch, img_num_context, hpt_num_context, num_modalities, nviews=15):
    assert num_modalities == 2
    num_episodes = len(batch)
    #if num_context == 0:
    cxt_episodes = []
    tgt_episodes = []
    cxt_hand_images = []
    tgt_hand_images = []
    for i in range(num_episodes):
        #cxt_episode = []
        #tgt_episode = []

        # image
        if img_num_context == 0:
            img_cxt = None
            cam_cxt = None
            img_tgt = batch[i][0]
            cam_tgt = batch[i][1]
        elif img_num_context > 0 and img_num_context < nviews:
            img_cxt = batch[i][0][:img_num_context]# if mask[0] and batch[i][0] is not None else None
            cam_cxt = batch[i][1][:img_num_context]# if mask[0] and batch[i][1] is not None else None
            img_tgt = batch[i][0][img_num_context:]# if mask[0] and batch[i][0] is not None else batch[i][0]
            cam_tgt = batch[i][1][img_num_context:]# if mask[0] and batch[i][1] is not None else batch[i][1]
        elif img_num_context == nviews:
            img_cxt = batch[i][0][:img_num_context]# if mask[0] and batch[i][0] is not None else None
            cam_cxt = batch[i][1][:img_num_context]# if mask[0] and batch[i][1] is not None else None
            img_tgt = None# if mask[0] and batch[i][0] is not None else batch[i][0]
            cam_tgt = None# if mask[0] and batch[i][1] is not None else batch[i][1]

        # haptic
        if hpt_num_context == 0:
            hpt_cxt    = None
            hnd_cxt    = None
            hndimg_cxt = None
            hpt_tgt    = batch[i][2]
            hnd_tgt    = batch[i][3]
            hndimg_tgt = batch[i][4]
        elif hpt_num_context > 0 and hpt_num_context < nviews:
            hpt_cxt    = batch[i][2][:hpt_num_context]# if mask[1] and batch[i][2] is not None else None
            hnd_cxt    = batch[i][3][:hpt_num_context]# if mask[1] and batch[i][3] is not None else None
            hndimg_cxt = batch[i][4][:hpt_num_context]# if mask[1] and batch[i][3] is not None else None
            hpt_tgt    = batch[i][2][hpt_num_context:]# if mask[1] and batch[i][2] is not None else batch[i][0]
            hnd_tgt    = batch[i][3][hpt_num_context:]# if mask[1] and batch[i][3] is not None else batch[i][1]
            hndimg_tgt = batch[i][4][hpt_num_context:]# if mask[1] and batch[i][3] is not None else batch[i][1]
        elif hpt_num_context == nviews:
            hpt_cxt    = batch[i][2][:hpt_num_context]# if mask[1] and batch[i][2] is not None else None
            hnd_cxt    = batch[i][3][:hpt_num_context]# if mask[1] and batch[i][3] is not None else None
            hndimg_cxt = batch[i][4][:hpt_num_context]# if mask[1] and batch[i][3] is not None else None
            hpt_tgt    = None# if mask[1] and batch[i][2] is not None else batch[i][2]
            hnd_tgt    = None# if mask[1] and batch[i][3] is not None else batch[i][3]
            hndimg_tgt = None

        # make a tuple
        #cxt_episode = tuple(cxt_episode)
        #tgt_episode = tuple(tgt_episode)
        cxt_episode = (img_cxt, cam_cxt, hpt_cxt, hnd_cxt)
        cxt_episodes += [cxt_episode]
        tgt_episode = (img_tgt, cam_tgt, hpt_tgt, hnd_tgt)
        tgt_episodes += [tgt_episode]
        cxt_hand_images += [hndimg_cxt]
        tgt_hand_images += [hndimg_tgt]
    #return new_context, new_target
    return cxt_episodes, tgt_episodes, cxt_hand_images, tgt_hand_images

def get_masks(num_modalities, min_modes=1, max_modes=2):
    # init
    mod_indices = [idx for idx in range(num_modalities)]

    # split joint observations
    combinations = []
    for i in range(max(1, min_modes), min(max_modes, num_modalities)+1):
        combinations += [idx for idx in itertools.combinations(mod_indices, i)]

    # get binary masks
    binary_combinations = []
    for comb in combinations:
        binary_comb = [0]*num_modalities
        for i in comb:
            binary_comb[i] = 1
        binary_combinations += [tuple(binary_comb)]

    return binary_combinations
