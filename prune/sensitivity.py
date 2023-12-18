import numpy as np
from .prune_zoo import *
from utils.loss import ComputeLoss
from tqdm import tqdm
from copy import deepcopy
import gc
from .dependency import DependencyGraph
import torch_pruning as tp
from scipy.optimize import fsolve
from functools import reduce


def ratio_compute(initial_rate, layer_num, thres):
    def f(x, arg):
        it = [1 + arg[0] * pow(x, i) for i in range(arg[1])]
        return reduce(lambda x, y: x * y, it) - arg[2]
    return fsolve(f, 1, [initial_rate, layer_num, thres])[0]


class Sensitivity(object):
    """
    the core implementation of PAGCP.
    :param min_ratio: the initial masking ratio of each layer
    :param max_ratio: the maximal masking ratio of each layer
    :param num: the interval number between the initial and maximal masking ratio
    :param metric: filter saliency criterion
    :param round: pruning round
    :param exp: whether to scale the local performance drop of each layer
    :param topk: the filtering ratio
    :return: the pruned model
    """
    def __init__(self, min_ratio, max_ratio, num, metric, round, exp, topk, *args):
        self.args = args[0]
        self.ratio = np.linspace(min_ratio, max_ratio, num)
        self.metric = tp.strategy.L1Strategy() if metric.lower() == 'l1' else tp.strategy.L2Strategy()
        self.exp = exp
        self.topk = topk
        self.inputsize = args[0].imgsz
        self.logger = args[1]
        self.round = round
        self.func_rate = lambda x: args[0].initial_rate + args[0].rate_slope * x     # the computation of initial performance
        self.func_thres = lambda x: args[0].initial_thres + args[0].thres_slope * x

    # def set_group(self, model):
    #     bottleneck_index = [2, 4, 6]
    #     self.groups = [[f'model[{i}].m[{n}].cv2.conv' for n in
    #                     range(len(model.module[i].m if hasattr(model, 'module') else model[i].m))] + [
    #                        f'model[{i}].cv1.conv'] for i in bottleneck_index]

    def set_group(self, model):
        # bottleneck_index = [2, 4, 6]
        # self.groups = [[f'model[{i}].m[{n}].conv[0].cv2.conv' for n in
        #                 range(len(model.module[i].m if hasattr(model, 'module') else model[i].m))] + [f'model[{i}].m[{n}].conv[2].cv2.conv' for n in
        #                 range(len(model.module[i].m if hasattr(model, 'module') else model[i].m))] + [
        #                    f'model[{i}].cv1.conv'] for i in bottleneck_index]
        self.groups = []

    def __call__(self, model, dataloader, part, sensitivity=None):
        if hasattr(self.metric, 'dataloader'): self.metric.dataloader = dataloader
        self.model = model
        self.set_group(model.model)
        self.criterion = ComputeLoss(model)
        temp_m = model.cuda()
        temp_m.eval()
        example_inputs = torch.randn(1, 3, self.inputsize, self.inputsize)

        base_b_total, base_o_total, base_c_total = 0., 0., 0.
        with torch.no_grad():
            for imgs, targets, paths, _ in tqdm(dataloader):
                imgs = imgs.cuda().float() / 255.0
                _, pred = temp_m(imgs)
                _, base_losses = self.criterion(pred, targets.cuda())
                base_b, base_o, base_c = base_losses[0] * imgs.shape[0], base_losses[1] * imgs.shape[0], base_losses[2] * imgs.shape[0]

                base_b_total += base_b
                base_o_total += base_o
                base_c_total += base_c
        base_loss_total = base_b_total + base_o_total + base_c_total
        pruned_model = deepcopy(model).cuda()
        sensitivity = sensitivity if sensitivity is not None else {}
        del temp_m
        gc.collect()

        DG = DependencyGraph()
        thres = self.func_rate(self.round)

        FLOPs_sens = {}
        _, _, base_flops = self.model.cuda().info(False, self.inputsize)

        # ----------layer sorting based on FLOPs---------- #
        for id, g in enumerate(self.groups):
            k = g[0]
            DG.build_dependency(pruned_model, example_inputs=example_inputs)
            layers = eval(f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
            prune_list = self.metric(layers.weight, amount=0.3, round_to=1)
            if len(prune_list) >= layers.weight.shape[0]:
                prune_list = prune_list[:-1]
            pruning_plan = DG.get_pruning_plan(layers, tp.prune_conv, idxs=prune_list)
            pruning_plan.exec()
            self.logger.info(f'model_id: {g}')
            _, _, temp_flops = pruned_model.cuda().info(False, self.inputsize)
            pruned_model = deepcopy(model).cuda()
            contrib_m = base_flops - temp_flops
            FLOPs_sens[f'group{id + 1}'] = contrib_m
        _, _, base_flops = self.model.cuda().info(False, self.inputsize)

        for k, v in model.named_modules():
            if hasattr(v, 'weight') and not isinstance(v, torch.nn.BatchNorm2d) and any([k.startswith(p) for p in part]) and k not in sensitivity.keys():
                DG.build_dependency(pruned_model, example_inputs=example_inputs)
                have_layers = [i.isdigit() for i in k.split('.')]
                if any(have_layers):
                    model_id = []
                    for i, ele in enumerate(k.split('.')):
                        if have_layers[i]:
                            model_id[-1] = model_id[-1] + f'[{ele}]'
                        else:
                            model_id.append(ele)
                    model_id = '.'.join(model_id)
                else:
                    model_id = k
                if any([model_id in group for group in self.groups]):
                    continue
                else:
                    layers = eval(f'pruned_model.module.{model_id} if hasattr(pruned_model, "module") else pruned_model.{model_id}')
                    prune_list = self.metric(layers.weight, amount=0.3, round_to=1)
                    if len(prune_list) >= layers.weight.shape[0]:
                        prune_list = prune_list[:-1]
                    if isinstance(v, torch.nn.Conv2d):
                        pruning_plan = DG.get_pruning_plan(layers, tp.prune_conv, idxs=prune_list)
                    else: pruning_plan = None
                    pruning_plan.exec()
                self.logger.info(f'model_id: {model_id}')
                _, _, temp_flops = pruned_model.cuda().info(False, self.inputsize)
                pruned_model = deepcopy(model).cuda()
                contrib_m = base_flops - temp_flops
                FLOPs_sens[model_id] = contrib_m

        exp = ratio_compute(thres, len(FLOPs_sens), self.func_thres(self.round)) if self.exp else 1.0
        self.logger.info(f'lambda: {exp}')
        rank_modules = sorted(FLOPs_sens, key=lambda x: FLOPs_sens[x], reverse=exp <= 1)
        self.prune_sequence = rank_modules
        self.logger.info('prune_sequence:' + str(rank_modules))

        # ----------sequentially pruning each layer---------- #
        for num, k in enumerate(rank_modules):
            sensitivity[k] = {}
            sensitivity[k]['loss'] = []
            sensitivity[k]['base_loss'] = float(base_loss_total.data)
            for l, r in enumerate(self.ratio):
                self.logger.info(f'pruning {num}/{len(rank_modules)}: {k}, base_loss:{base_loss_total:4f}, base_b:{base_b_total:4f}, base_o:{base_o_total:4f}, base_c:{base_c_total:4f}, ratio:{r}, thres:{thres}')
                temp_model = deepcopy(pruned_model)
                DG.build_dependency(temp_model, example_inputs=example_inputs)

                # get pruning set of each layer
                if 'group' in k:
                    group_id = int(k[5:])
                    group = self.groups[group_id - 1]
                    prune_list = group_l1prune(temp_model, group, r, round_to=1)
                    layers = eval(
                        f'temp_model.module.{group[0]} if hasattr(temp_model, "module") else temp_model.{group[0]}')
                else:
                    layers = eval(f'temp_model.module.{k} if hasattr(temp_model, "module") else temp_model.{k}')
                    prune_list = self.metric(layers.weight, amount=r, round_to=1)

                # execute the pruning
                if len(prune_list):
                    if len(prune_list) >= layers.weight.shape[0]:
                        prune_list = prune_list[:-1]
                    if isinstance(layers, torch.nn.Conv2d):
                        prune_m = tp.prune_conv
                    else: prune_m = None
                    pruning_plan = DG.get_pruning_plan(layers, prune_m, idxs=prune_list)
                    pruning_plan.exec()

                    temp_model = temp_model.cuda()
                    temp_b_total, temp_o_total, temp_c_total = 0., 0., 0.
                    with torch.no_grad():
                        for imgs, targets, paths, _ in tqdm(dataloader):
                            imgs = imgs.cuda().float() / 255.0
                            _, pred = temp_model(imgs)
                            _, temp_losses = self.criterion(pred, targets.cuda())
                            temp_b, temp_o, temp_c = temp_losses[0] * imgs.shape[0], temp_losses[1] * imgs.shape[0], \
                                                     temp_losses[2] * imgs.shape[0]

                            temp_b_total += temp_b
                            temp_o_total += temp_o
                            temp_c_total += temp_c

                    temp_loss_total = temp_b_total + temp_o_total + temp_c_total
                    b_rel = temp_b_total / base_b_total
                    o_rel = temp_o_total / base_o_total
                    c_rel = temp_c_total / base_c_total
                    self.logger.info(f'temp_loss:{temp_loss_total:4f}, temp_b:{temp_b_total:4f}, temp_o:{temp_o_total:4f}, temp_c:{temp_c_total:4f}')

                    # ----------get the pruning ratio of each layer based on task choices---------- #
                    if max(b_rel, o_rel, c_rel) > (1 + thres):
                        idx = np.argmax([b_rel.cpu(), o_rel.cpu(), c_rel.cpu()])
                        if 'group' in k:
                            group_id = int(k[5:])
                            group = self.groups[group_id - 1]
                            prune_list = group_l1prune(pruned_model, group, self.ratio[l - 1], round_to=1) if l >= 1 else []
                            layers = eval(
                                f'pruned_model.module.{group[0]} if hasattr(pruned_model, "module") else pruned_model.{group[0]}')
                        else:
                            layers = eval(
                                f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
                            prune_list = self.metric(layers.weight, amount=self.ratio[l - 1], round_to=1) if l >= 1 else []

                        if len(prune_list):
                            DG.build_dependency(pruned_model,
                                                example_inputs=example_inputs)
                            pruning_plan = DG.get_pruning_plan(layers, prune_m, idxs=prune_list)
                            pruning_plan.exec()
                            thres *= exp
                            base_loss_total, base_b_total, base_o_total, base_c_total = max(base_loss_total, last_loss_total), max(base_b_total, last_b_total), max(base_o_total, last_o_total), max(base_c_total, last_c_total)

                        sensitivity[k]['loss'].append(['box', 'object', 'class'][idx])
                        pruned_model = pruned_model.cuda()

                        break

                    sensitivity[k]['loss'].append(float(temp_loss_total))
                    last_loss_total, last_b_total, last_o_total, last_c_total = temp_loss_total, temp_b_total, temp_o_total, temp_c_total
                    del temp_model
                else:
                    sensitivity[k]['loss'].append(float(base_b_total + base_o_total + base_c_total))
        del pruned_model
        gc.collect()
        return sensitivity

    def get_ratio(self, sensitivity):
        _, _, base_flops = self.model.cuda().info(False, self.inputsize)

        pruned_model = deepcopy(self.model).cuda()
        DG = DependencyGraph()

        sens_keys = sensitivity.keys()
        flops = {}

        for k in sens_keys:
            if len(sensitivity[k]['loss']) > 1:
                DG.build_dependency(pruned_model, example_inputs=torch.randn(1, 3, self.inputsize, self.inputsize))
                r = self.ratio[len(sensitivity[k]['loss']) - 2]
                if 'group' in k:
                    group_id = int(k[5:])
                    group = self.groups[group_id - 1]
                    prune_list = group_l1prune(pruned_model, group, r, round_to=1)
                    layers = eval(
                        f'pruned_model.module.{group[0]} if hasattr(pruned_model, "module") else pruned_model.{group[0]}')
                else:
                    layers = eval(f'pruned_model.module.{k} if hasattr(pruned_model, "module") else pruned_model.{k}')
                    prune_list = self.metric(layers.weight, amount=r, round_to=1)
                if len(prune_list) >= layers.weight.shape[0]:
                    prune_list = prune_list[:-1]

                pruning_plan = DG.get_pruning_plan(layers, tp.prune_conv, idxs=prune_list)
                pruning_plan.exec()
                _, _, fs = pruned_model.cuda().info(False, self.inputsize)
                flops[k] = (sensitivity[k]['loss'][-2] - sensitivity[k]['base_loss']) / (base_flops - fs + 1e-20)
                base_flops = fs

        rank_keys = sorted(flops, key=lambda x: flops[x])
        candidate_keys = sorted(rank_keys[:min(len(rank_keys), int(len(sens_keys) * self.topk))])
        sorted_index = sorted([list(sens_keys).index(k) for k in candidate_keys])
        ratio = {list(sens_keys)[i]: self.ratio[len(sensitivity[list(sens_keys)[i]]['loss']) - 2] for i in sorted_index} # if total, then - 1
        return ratio
