'''
miscellaneous functions: gqn 
'''
import itertools
import numpy as np
import torch


def new_trim_context_target_mask_on_target(batch, num_context, mask, num_modalities, nviews=15):
    assert len(mask) == num_modalities
    num_episodes = len(batch)
    if num_context == 0:
        new_context = [tuple([None]*num_modalities*2) for i in range(num_episodes)]
        new_target = [tuple([batch[i][j] if mask[j//2] and batch[i][j] is not None else None for j in range(num_modalities*2)]) for i in range(num_episodes)]
    elif num_context > 0 and num_context < nviews:
        new_context = [tuple([
            batch[i][j][:num_context] if mask[j//2] and batch[i][j] is not None else None for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
        new_target = [tuple([
            batch[i][j][num_context:] if mask[j//2] and batch[i][j] is not None else None for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
    elif num_context == nviews:
        new_context = [tuple([
            batch[i][j][:num_context] if mask[j//2] and batch[i][j] is not None else None for j in range(num_modalities*2)
            ]) for i in range(num_episodes)]
        new_target = [tuple([None]*num_modalities*2) for i in range(num_episodes)]
    return new_context, new_target

def new_trim_context_target_nomask_on_target(batch, num_context, mask, num_modalities, nviews=15):
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

def new_trim_context_target(batch, num_context, mask, num_modalities, nviews=15, mask_on_target=False):
    if mask_on_target:
        return new_trim_context_target_mask_on_target(batch=batch, num_context=num_context, mask=mask, num_modalities=num_modalities, nviews=nviews)
    else:
        return new_trim_context_target_nomask_on_target(batch=batch, num_context=num_context, mask=mask, num_modalities=num_modalities, nviews=nviews)
