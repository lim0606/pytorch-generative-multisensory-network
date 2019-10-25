from datasets.haptix import get_haptix_shepard_metzler_n_parts as _get_haptix_shepard_metzler_n_parts

def get_haptix_shepard_metzler_n_parts(*argv, **kwargs):
    train_loader, val1_loader, _, test_loader, info = _get_haptix_shepard_metzler_n_parts(*argv, **kwargs)
    return train_loader, val1_loader, None, test_loader, info

'''
get dataset
'''
def get_haptix_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False):
    # init arguments
    if eval_batch_size is None:
        eval_batch_size = train_batch_size
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    ################ M=2
    if dataset == 'haptix-shepard_metzler_4_parts':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_4_parts-extrapol11':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol11',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=1,
                                                  )

    elif dataset == 'haptix-shepard_metzler_4_parts-intrapol22':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol22',
                                                  train_min_num_modes=2,
                                                  train_max_num_modes=2,
                                                  )

    elif dataset == 'haptix-shepard_metzler_4_parts-trm1':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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
    elif dataset == 'haptix-shepard_metzler_4_parts-48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  expname='crop48',
                                                  )

    ################ M=3, 48
    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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
    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-intrapol35':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-intrapol45':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-inextrapol23':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-inextrapol34':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-inextrapol15':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-trm1':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-ul-lr-trm3':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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
    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-extrapol14':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-intrapol48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-intrapol78':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-inextrapol18':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-inextrapol36':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-lr-rgb-half-inextrapol45':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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
    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half-extrapol14':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half-extrapol17':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half-intrapol814':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half-intrapol1114':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    elif dataset == 'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half-inextrapol114':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4],
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

    ################################################# 6-parts ############
    ################ M=2
    elif dataset == 'haptix-shepard_metzler_6_parts':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_6_parts-extrapol11':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='extrapol11',
                                                  train_min_num_modes=1,
                                                  train_max_num_modes=1,
                                                  )

    elif dataset == 'haptix-shepard_metzler_6_parts-intrapol22':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  do_modal_exclusive_observation=True,
                                                  expname='intrapol22',
                                                  train_min_num_modes=2,
                                                  train_max_num_modes=2,
                                                  )

    elif dataset == 'haptix-shepard_metzler_6_parts-trm1':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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
    elif dataset == 'haptix-shepard_metzler_6_parts-48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  expname='crop48',
                                                  )

    ################ M=3, 48
    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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
    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  )

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-intrapol35':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-intrapol45':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-inextrapol23':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-inextrapol34':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-inextrapol15':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-trm1':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-ul-lr-trm3':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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
    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-extrapol14':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-intrapol48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-intrapol78':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-inextrapol18':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-inextrapol36':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-lr-rgb-half-inextrapol45':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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
    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half-extrapol12':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half-extrapol14':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half-extrapol17':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half-intrapol814':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half-intrapol1114':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    elif dataset == 'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half-inextrapol114':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[6],
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

    ################ M=2
    if dataset == 'haptix-shepard_metzler_46_parts':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  )

    ################ M=2
    elif dataset == 'haptix-shepard_metzler_456_parts':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,5,6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  )

    ################ M=2, 48
    elif dataset == 'haptix-shepard_metzler_456_parts-48':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,5,6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  expname='crop48',
                                                  )

    ################ M=3, 48
    elif dataset == 'haptix-shepard_metzler_456_parts-48-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,5,6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  do_minmax_norm=True,
                                                  img_size=48,
                                                  do_crop=True,
                                                  )

    ################ M=5
    elif dataset == 'haptix-shepard_metzler_456_parts-ul-lr':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,5,6],
                                                  target_sample_method='remaining', #'full',
                                                  max_cond_size=15,
                                                  max_target_size=15,
                                                  img_split_leftright=True,
                                                  img_split_upperlower=True,
                                                  do_minmax_norm=True,
                                                  )

    ################ M=8, 48
    elif dataset == 'haptix-shepard_metzler_456_parts-48-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,5,6],
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

    ################ M=14, 48
    elif dataset == 'haptix-shepard_metzler_456_parts-48-ul-lr-rgb-half':
        return get_haptix_shepard_metzler_n_parts(train_batch_size, eval_batch_size, kwargs,
                                                  n_parts=[4,5,6],
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

    else:
        raise NotImplementedError('dataset: {}'.format(dataset))
