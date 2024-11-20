import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result = dict()
    for key in dataset_items[0].keys():
        if 'path' in key:
            result[key] = list([item[key] for item in dataset_items])
        elif 'audio' in key:
            result[key] = pad_sequence([item[key].squeeze() for item in dataset_items], batch_first=True)
        else:
            result[key] = [item[key] for item in dataset_items]
    
    return result
