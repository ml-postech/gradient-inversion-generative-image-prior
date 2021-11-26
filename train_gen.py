"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision

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
               'FFHQ': 10, 'FFHQ64': 10, 'FFHQ128': 10,
               'PERM': 1000
               }
# Parse input arguments
parser = inversefed.options()
parser.add_argument('--unsigned', action='store_true', help='Use signed gradient descent')
# parser.add_argument('--lr', default=None, type=float, help='Optionally overwrite default step sizes.')
parser.add_argument('--num_exp', default=10, type=int, help='Number of consecutive experiments')
parser.add_argument('--max_iterations', default=5000, type=int, help='Maximum number of iterations for reconstruction.')
parser.add_argument('--gias_iterations', default=0, type=int, help='Maximum number of gias iterations for reconstruction.')

parser.add_argument('--meta_lr', default=1e-2, type=float, help='Learning rate for outer loop of meta learning')
parser.add_argument('--checkpoint_path', default='', type=str, help='Checkpoint path for G')

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
    for idx, (gx, gy) in enumerate(zip(est_gradient, target_gradient)): # TODO: fix the variables here
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
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)

    if args.dataset == 'PERM':
        model, model_seed = inversefed.construct_model(args.model, num_classes=1000, num_channels=3)
        dm = torch.as_tensor(getattr(inversefed.consts, f'i64_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'i64_std'), **setup)[:, None, None]
    else:
        model, model_seed = inversefed.construct_model(args.model, num_classes=10, num_channels=3)
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    model.to(**setup)
    model.eval()


    # Load a trained model?
    if args.trained_model:
        file = f'{args.model}_{args.epochs}.pth'
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, file), map_location=setup['device']))
            print(f'Model loaded from file {file}.')
        except FileNotFoundError:
            print('Training the model ...')
            print(repr(defs))
            inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup)
            torch.save(model.state_dict(), os.path.join(args.model_path, file))

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
                    z_norm=args.z_norm,
                    group_lazy=args.group_lazy,
                    init='randn',
                    lr_decay=True,
                    dataset=args.dataset,
                    
                    generative_model=args.generative_model,
                    gen_dataset=args.gen_dataset,
                    giml=args.giml,
                    gias=args.gias,
                    gias_lr=args.gias_lr,
                    gias_iterations=args.gias_iterations
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
    config_comp['bn_stat'] = args.bn_stat
    config_comp['image_norm'] = args.image_norm
    config_comp['z_norm'] = args.z_norm
    config_comp['group_lazy'] = args.group_lazy
    config_comp['meta_lr'] = args.meta_lr
    config_comp['checkpoint_path'] = args.checkpoint_path
    config_hash = hashlib.md5(json.dumps(config_comp, sort_keys=True).encode()).hexdigest()

    print(config_comp)

    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{config_hash}', exist_ok=True)


    if args.checkpoint_path:
        with open(args.checkpoint_path, 'rb') as f:
            G, latents = pickle.load(f)
            G = G.requires_grad_(True).to(setup['device'])
        G_optimizer = torch.optim.Adam(G.parameters(), lr=args.meta_lr)

    else:
        if args.generative_model == 'DCGAN':
            G = porting.load_decoder_dcgan(config, device=setup['device'], dataset='C10')
        elif args.generative_model == 'DCGAN-untrained':
            G = porting.load_decoder_dcgan_untrained(config, device=setup['device'], dataset=args.gen_dataset)
        elif args.generative_model == 'stylegan2-ada-untrained':
            G, G_mapping, G_synthesis = porting.load_decoder_stylegan2(config, device=setup['device'], dataset=args.gen_dataset, untrained=True, ada=True)
        
        if args.generative_model.startswith('DCGAN'):
            G_optimizer = torch.optim.Adam(G.parameters(), lr=args.meta_lr)
        elif args.generative_model.startswith('stylegan2-ada'):
            G_optimizer = torch.optim.Adam(G.synthesis.parameters(), lr=args.meta_lr)
        elif args.generative_model.startswith('stylegan2'):
            G_optimizer = torch.optim.Adam(G.G_synthesis.parameters(), lr=args.meta_lr)
        latents = []

    target_id = args.target_id
    for i in range(args.num_exp):
        target_id = args.target_id + i * args.num_images
        tid_list = []
        if args.num_images == 1:
            ground_truth, labels = validloader.dataset[target_id]
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
            target_id_ = target_id + 1
            print("loaded img %d" % (target_id_ - 1))
            tid_list.append(target_id_ - 1)
        else:
            ground_truth, labels = [], []
            target_id_ = target_id
            while len(labels) < args.num_images:
                if args.dataset == 'PERM':
                    target_id_ = target_id_ % len(validloader.dataset)
                img, label = validloader.dataset[target_id_]
                target_id_ += 1
                if (label not in labels):
                    print("loaded img %d" % (target_id_ - 1))
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                    ground_truth.append(img.to(**setup))
                    tid_list.append(target_id_ - 1)

            ground_truth = torch.stack(ground_truth)
            labels = torch.cat(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

        # Run reconstruction
        input_gradients = []
        for j in range(args.num_images):
            model.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth[j].unsqueeze(0)), labels[j].unsqueeze(0))
            input_gradient = torch.autograd.grad(target_loss, model.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]
            input_gradients.append(input_gradient)

        print("Creating Gradient Reconstructor")
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, G=G)

        
        print("Starting Reconstruction")
        output, stats = rec_machine.reconstruct(input_gradients, labels, img_shape=img_shape, dryrun=args.dryrun)
        
        # G = rec_machine.G
        
        G_optimizer.zero_grad()
        if args.generative_model.startswith('stylegan2-ada'):
            G_updated = rec_machine.G_synthesis
            G_synthesis.requires_grad_(True)
            diff = l2(list(G_updated.parameters()), list(G_synthesis.parameters()))
        else:
            G_updated = rec_machine.G
            diff = l2(list(G_updated.parameters()), list(G.parameters()))
        # diff = 0
        # for gg in G_updated:
        #     diff += l2(list(gg.parameters()), list(G.parameters()))
        diff.backward()
        G_optimizer.step()

        latents.append(rec_machine.dummy_z)

        if (i + 1) % 50 == 0:
            model.train()
            model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            model_optimizer.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            target_loss.backward()
            model_optimizer.step()
            model.eval()


        # Compute stats and save to a table:
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
        test_mse = (output_den - ground_truth_den).pow(2).mean().item()
        test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

        inversefed.utils.save_to_table(f'results/{config_hash}', name=f'mul_exp_{args.name}', dryrun=args.dryrun,

                                       config_hash=config_hash,
                                       model=args.model,
                                       dataset=args.dataset,
                                       trained=args.trained_model,
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

                                       target_id=target_id,
                                       seed=model_seed,
                                       epochs=defs.epochs,
                                       )


        # Save the resulting image
        if args.save_image and not args.dryrun:
            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            for j in range(args.num_images):
                torchvision.utils.save_image(output_denormalized[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}.png'))
                torchvision.utils.save_image(ground_truth_den[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}_gt.png'))

        # Save psnr values
        psnrs.append(test_psnr)
        inversefed.utils.save_to_table(f'results/{config_hash}', name='psnrs', dryrun=args.dryrun, target_id=target_id, psnr=test_psnr)

        # Update target id
        target_id = target_id_

        if i % 5 == 0:
            with open('results/G_{}_{}.pkl'.format(config_hash, i), 'wb') as f:
                pickle.dump((copy.deepcopy(G).eval().requires_grad_(False).cpu(), latents), f)



    # psnr statistics
    psnrs = np.nan_to_num(np.array(psnrs))
    psnr_mean = psnrs.mean()
    psnr_std = np.std(psnrs)
    psnr_max = psnrs.max()
    psnr_min = psnrs.min()
    psnr_median = np.median(psnrs)
    timing = datetime.timedelta(seconds=time.time() - start_time)
    inversefed.utils.save_to_table(f'results/{config_hash}', name='psnr_stats', dryrun=args.dryrun,
                                   number_of_samples=len(psnrs),
                                   timing=str(timing),
                                   mean=psnr_mean,
                                   std=psnr_std,
                                   max=psnr_max,
                                   min=psnr_min,
                                   median=psnr_median)

    with open('results/G_{}.pkl'.format(config_hash), 'wb') as f:
        pickle.dump((copy.deepcopy(G).eval().requires_grad_(False).cpu(), latents), f)


    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
