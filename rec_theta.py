"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision
import torch.nn as nn

import numpy as np

import inversefed
torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

from collections import defaultdict
import datetime
import time
import os
import json
import hashlib
import csv
import copy
import pickle

import inversefed.porting as porting

nclass_dict = {'I32': 1000, 'I64': 1000, 'I128': 1000, 
               'CIFAR10': 10, 'CIFAR100': 100, 'CA': 8, 'ImageNet':1000,
               'FFHQ': 10, 'FFHQ64': 10, 'FFHQ128': 10,}
# Parse input arguments
parser = inversefed.options()
parser.add_argument('--unsigned', action='store_true', help='Use signed gradient descent')
# parser.add_argument('--lr', default=None, type=float, help='Optionally overwrite default step sizes.')
parser.add_argument('--num_exp', default=10, type=int, help='Number of consecutive experiments')
parser.add_argument('--max_iterations', default=4800, type=int, help='Maximum number of iterations for reconstruction.')

parser.add_argument('--meta_lr', default=1e-2, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--setting', default='train', type=str, help='Local learning rate for federated averaging') # 'train', 'init', 'base'

args = parser.parse_args()
if args.target_id is None:
    args.target_id = 0
args.save_image = True
args.signed = not args.unsigned


# Parse training strategy
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs


def l2(est_gradient, target_gradient):
    grad_diff = 0
    for idx, (gx, gy) in enumerate(zip(est_gradient, target_gradient)): 
        if len(gx.shape) >= 1:
            layer_grad = ((gx - gy) ** 2).sum() # / math.sqrt(gx.shape[0])
        else:
            layer_grad = ((gx - gy) ** 2).sum() # / math.sqrt(gx.shape[0])

        grad_diff += layer_grad
    return grad_diff


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training

    # Get data:
    defs.augmentations=False
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)

    model, model_seed = inversefed.construct_model(args.model, num_classes=nclass_dict[args.dataset], num_channels=3)
    if args.dataset.startswith('FFHQ'):
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    else:
        dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]
    model.to(**setup)
    model.eval()


    if args.optim == 'ours':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      
                      generative_model=args.generative_model,
                      gen_dataset=args.gen_dataset,
                      giml=args.giml,
                      gias_lr=args.gias_lr)
    elif args.optim == 'yin':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      
                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias_lr=0.0)
    elif args.optim == 'geiping':
        config = dict(signed=args.signed,
                      cost_fn='sim',
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      
                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias_lr=0.0)      
    elif args.optim == 'zhu':
        config = dict(signed=False,
                      cost_fn='l2',
                      indices='def',
                      weights='equal',
                      lr=args.lr if args.lr is not None else 1.0,
                      optim='LBFGS',
                      restarts=args.restarts,
                      max_iterations=500,
                      total_variation=args.tv,
                      init=args.init,
                      lr_decay=False,
                      )
    # psnr list
    psnrs = []

    # hash configuration

    config_comp = config.copy()
    config_comp['optim'] = args.optim
    config_comp['dataset'] = args.dataset
    config_comp['model'] = args.model
    config_comp['trained'] = args.trained_model
    config_comp['num_exp'] = args.num_exp
    config_comp['num_images'] = args.num_images
    config_comp['accumulation'] = args.accumulation
    config_comp['bn_stat'] = args.bn_stat
    config_comp['image_norm'] = args.image_norm
    config_comp['group_lazy'] = args.group_lazy
    config_hash = hashlib.md5(json.dumps(config_comp, sort_keys=True).encode()).hexdigest()

    print(config_comp)

    os.makedirs(args.table_path, exist_ok=True)
    os.makedirs(os.path.join(args.table_path, f'{config_hash}'), exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(os.path.join(args.result_path, f'{config_hash}'), exist_ok=True)


    node_dataset_indexes = [8112, 4000, 4100, 4200, 4300, 4400, 4500, 4600] # I val
    # node_dataset_indexes = [0, 1, 9, 10, 11, 12]        # C10 val
    node_dataset = []

    
    dataset_images = {}
    prev_models = []
    prev_gradients = []
    prev_labels = []

    for i in node_dataset_indexes:
        img, label = validloader.dataset[i]
        node_dataset.append((img.to(**setup), torch.as_tensor((label,), device=setup['device'])))


    for i in range(args.num_exp):
        
        ground_truth, labels = [], []
        selected_idx = np.random.choice(len(node_dataset), args.num_images, replace=False)
        for idx in selected_idx:
            img, label = node_dataset[idx]
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))
            print(f"selected {node_dataset_indexes[idx]}")

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

        # Run reconstruction
        bn_layers = []
        if args.bn_stat > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_layers.append(inversefed.BNStatisticsHook(module))

        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]

        prev_models.append(copy.deepcopy(model))
        prev_gradients.append(input_gradient)
        prev_labels.append(labels)
        
        bn_prior = []
        if args.bn_stat > 0:
            for idx, mod in enumerate(bn_layers):
                mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
                bn_prior.append(mean_var)
        # with open(f'exp_{i}_bn_prior.pkl', 'wb') as f:
        #     pickle.dump(bn_prior, f)
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)

        if args.setting == 'base':
            output, stats = rec_machine.reconstruct_theta(prev_gradients[-1:], prev_labels[-1:], prev_models[-1:], dataset_images, img_shape=img_shape, dryrun=args.dryrun)
        else:
            output, stats = rec_machine.reconstruct_theta(prev_gradients, prev_labels, prev_models, dataset_images, img_shape=img_shape, dryrun=args.dryrun)

        print(len(dataset_images.keys()))


        # Compute stats and save to a table:
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        test_mse = (output_den - ground_truth_den).pow(2).mean().item()
        test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

        inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name=f'mul_exp_{args.name}', dryrun=args.dryrun,

                                       config_hash=config_hash,
                                       model=args.model,
                                       dataset=args.dataset,
                                       trained=args.trained_model,
                                       accumulation=args.accumulation,
                                       restarts=args.restarts,
                                       OPTIM=args.optim,
                                       cost_fn=args.cost_fn,
                                       indices=args.indices,
                                       weights=args.weights,
                                       init=args.init,
                                       tv=args.tv,

                                       rec_loss=stats["opt"],
                                       psnr=test_psnr,
                                       test_mse=test_mse,
                                       feat_mse=feat_mse,

                                       target_id=i,
                                       seed=model_seed,
                                       epochs=defs.epochs,
                                    #    val_acc=training_stats["valid_" + name][-1],
                                       )

        
        if args.setting == 'train':
            model.train()
            model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
            model_optimizer.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            target_loss.backward()
            model_optimizer.step()
            model.eval()
        if args.setting == 'init':
            model, model_seed = inversefed.construct_model(args.model, num_classes=nclass_dict[args.dataset], num_channels=3)
            model.to(**setup)
            model.eval()



        # Save the resulting image
        if args.save_image and not args.dryrun:
            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            for j in range(args.num_images):
                torchvision.utils.save_image(output_denormalized[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{node_dataset_indexes[selected_idx[j]]}_{i}.png'))
                torchvision.utils.save_image(ground_truth_den[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{node_dataset_indexes[selected_idx[j]]}_gt.png'))

        # Save psnr values
        psnrs.append(test_psnr)
        inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name='psnrs', dryrun=args.dryrun, target_id=i, psnr=test_psnr)


    # psnr statistics
    psnrs = np.nan_to_num(np.array(psnrs))
    psnr_mean = psnrs.mean()
    psnr_std = np.std(psnrs)
    psnr_max = psnrs.max()
    psnr_min = psnrs.min()
    psnr_median = np.median(psnrs)
    timing = datetime.timedelta(seconds=time.time() - start_time)
    inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name='psnr_stats', dryrun=args.dryrun,
                                   number_of_samples=len(psnrs),
                                   timing=str(timing),
                                   mean=psnr_mean,
                                   std=psnr_std,
                                   max=psnr_max,
                                   min=psnr_min,
                                   median=psnr_median)

    config_exists = False
    if os.path.isfile(os.path.join(args.table_path, 'table_configs.csv')):
        with open(os.path.join(args.table_path, 'table_configs.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[-1] == config_hash:
                    config_exists = True
                    break

    if not config_exists:
        inversefed.utils.save_to_table(args.table_path, name='configs', dryrun=args.dryrun,
                                       config_hash=config_hash,
                                       **config_comp,
                                       number_of_samples=len(psnrs),
                                       timing=str(timing),
                                       mean=psnr_mean,
                                       std=psnr_std,
                                       max=psnr_max,
                                       min=psnr_min,
                                       median=psnr_median)

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
