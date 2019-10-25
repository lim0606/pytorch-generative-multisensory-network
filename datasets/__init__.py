from datasets.gqn import get_gqn_dataset
from datasets.haptix import get_haptix_dataset as get_haptix_dataset
from datasets.haptix_for_cls import get_haptix_dataset as get_haptix_dataset_for_cls

def get_dataset(dataset, train_batch_size, eval_batch_size=None, cuda=False):
    '''
    get_<dataset>_dataset(...) returns (train_loader, val_loader, test_loader, info)
    each <split>_loader is iterator
    each item in the iterators is a (data, targets) tuple data: sequence length
    x batch size
    target: None for corpus dataset, class label for image dataset
    '''
    if dataset in [
            'rooms_ring_camera',
            'rooms_free_camera_no_object_rotations',
            'rooms_free_camera_with_object_rotations',
            #'jaco',
            'shepard_metzler_5_parts',
            'shepard_metzler_5_parts-10k',
            'shepard_metzler_5_parts-100k',
            'shepard_metzler_7_parts',
            'mazes',
            ]:
        return get_gqn_dataset(dataset, train_batch_size, eval_batch_size, cuda)
    elif dataset in [
            'haptix-shepard_metzler_46_parts', # M=2
            'haptix-shepard_metzler_456_parts', # M=2
            'haptix-shepard_metzler_456_parts-ul-lr', #M=5
            'haptix-shepard_metzler_456_parts-48-lr-rgb-half', #M=8
            'haptix-shepard_metzler_456_parts-48-ul-lr-rgb-half', #M=14
            'haptix-shepard_metzler_456_parts-48', # M=2, 48
            ]:
        return get_haptix_dataset_for_cls(dataset, train_batch_size, eval_batch_size, cuda)
    elif dataset in [
            'haptix-shepard_metzler_4_parts', # M=2
            'haptix-shepard_metzler_4_parts-ul-lr', #M=5
            'haptix-shepard_metzler_4_parts-48-lr-rgb-half', #M=8
            'haptix-shepard_metzler_4_parts-48-ul-lr-rgb-half', #M=14
            'haptix-shepard_metzler_4_parts-48', # M=2, 48

            'haptix-shepard_metzler_6_parts', # M=2
            'haptix-shepard_metzler_6_parts-ul-lr', #M=5
            'haptix-shepard_metzler_6_parts-48-lr-rgb-half', #M=8
            'haptix-shepard_metzler_6_parts-48-ul-lr-rgb-half', #M=14
            'haptix-shepard_metzler_6_parts-48', # M=2, 48
            ]:
        return get_haptix_dataset_for_cls(dataset, train_batch_size, eval_batch_size, cuda)
    elif dataset in [ # missing modality experiments
            'haptix-shepard_metzler_5_parts-extrapol11',
            'haptix-shepard_metzler_5_parts-intrapol22',
            'haptix-shepard_metzler_5_parts-trm1',

            'haptix-shepard_metzler_5_parts-ul-lr-trm1',
            'haptix-shepard_metzler_5_parts-ul-lr-trm3',
            'haptix-shepard_metzler_5_parts-ul-lr-intrapol35',
            'haptix-shepard_metzler_5_parts-ul-lr-intrapol45',
            'haptix-shepard_metzler_5_parts-ul-lr-extrapol12',
            'haptix-shepard_metzler_5_parts-ul-lr-inextrapol23',
            'haptix-shepard_metzler_5_parts-ul-lr-inextrapol34',
            'haptix-shepard_metzler_5_parts-ul-lr-inextrapol15',

            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-extrapol12',
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-extrapol14',
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol48',
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-intrapol78',
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-inextrapol18',
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-inextrapol36',
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half-inextrapol45',

            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol12',
            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol14',
            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-extrapol17',
            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-intrapol814',
            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-intrapol1114',
            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half-inextrapol114',
            ]:
        return get_haptix_dataset(dataset, train_batch_size, eval_batch_size, cuda)
    elif dataset in [
            'haptix-shepard_metzler_5_parts', # M=2
            'haptix-shepard_metzler_5_parts-ul-lr', #M=5
            'haptix-shepard_metzler_5_parts-48-lr-rgb-half', #M=8
            'haptix-shepard_metzler_5_parts-48-ul-lr-rgb-half', #M=14
            #'haptix-shepard_metzler_5_parts-48', # M=2, 48
            ]:
        return get_haptix_dataset(dataset, train_batch_size, eval_batch_size, cuda)
    else:
        raise NotImplementedError('dataset: {}'.format(dataset))
