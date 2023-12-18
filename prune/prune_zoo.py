import torch
from torch_pruning.prune.strategy import round_pruning_amount


def group_l1prune(model, group, compress_rate, round_to=1):
    """
    The function of computing the saliency of a filter group, in which the channel number is
    constrained by all group members. The basic idea to evaluate the group saliency is by averaging
    the saliency score of each filter.
    :param model: the to-prune model
    :param group: the to-prune filter group
    :param compress_rate: the compression ratio
    :param round_to: the factor of the channel number for the filter to be pruned.
    :return: the index list of the to-prune channel in each group member.
    """
    filters = 0.
    for model_id in group:
        layers = eval(f'model.module.{model_id} if hasattr(model, "module") else model.{model_id}')
        filters += torch.mean(torch.abs(layers.weight), dim=(1, 2, 3))
    measures = [filters[k] for k in range(filters.shape[0])]
    measures = list(enumerate(measures))
    measures.sort(key=lambda item: item[1])
    num_to_prune = int(filters.shape[0] * compress_rate) if compress_rate < 1 else int(compress_rate)
    # num_to_prune = num_to_prune if num_to_prune else 8
    num_to_prune = min(filters.shape[0] - 1, num_to_prune)
    num_to_prune = round_pruning_amount(filters.shape[0], num_to_prune, round_to)
    return [measures[k][0] for k in range(num_to_prune)]
