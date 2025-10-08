import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import numpy as np
#import wandb
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
# if TENSORBOARD_FOUND False still
# from tensorboardX import SummaryWriter

# utils
from utils.logger import Logger as Log
from utils.basic_utils import str2bool, save_args, int2bool
from utils.init import init


losses = ["l1_loss", "ssim_loss", "alpha_regul", "sh_sparsity_loss", "total_loss", "iter_time"]
dens_statistic_dict = {"n_points_cloned": 0, "n_points_split": 0, "n_points_mercied": 0, "n_points_pruned": 0, "redundancy_threshold": 0, "opacity_threshold": 0}

def training(model, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations ,checkpoint, debug_from, args_dict):
    first_iter = 0
    tb_writer = tb_init(args_dict)
    
    gaussians = GaussianModel(model.sh_degree)
    scene = Scene(model, gaussians, args_dict=args_dict)
    gaussians.training_setup(opt) 
    
    if args_dict["warmup_iter"] > 0:
        opt.densify_until_iter += args_dict["warmup_iter"]
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1  #
    fine_tune_start = opt.iterations  # reduced 3DGS
    if opt.mercy_points:
        fine_tune_start = opt.iterations - opt.reduce_period

    loss_aggregator = {loss : 0 for loss in losses}

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, model.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
                        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        report_dict = {}  # progress bar report
            
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, low_pass = 0.3, itr=iteration, args_dict=args_dict)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        if opt.lambda_alpha_regul == 0:
            Lalpha_regul = torch.tensor([0.], device=image.device)
        else:
            points_opacity = gaussians.get_opacity[visibility_filter]
            Lalpha_regul = points_opacity.abs().mean()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + Lalpha_regul * args.lambda_alpha_regul

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                report_dict['Loss'] = f"{ema_loss_for_log:.{7}f}"
                report_dict['num_gaussians'] = f"{gaussians.get_xyz.shape[0]}"
                progress_bar.set_postfix(report_dict)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            results_dict={"pixel_number": gaussians.num_primitives,
                          "l1_loss": Ll1,
                          "ssim_loss": Lssim,
                          "alpha_regul": Lalpha_regul,
                          "total_loss": loss,
                          "iter_time": iter_start.elapsed_time(iter_end)}

            training_report(tb_writer, iteration, l1_loss, testing_iterations, scene, render, (pipe, background), results_dict, args_dict, loss_aggregator, dens_statistic_dict)

            # Densification
            if iteration < opt.densify_until_iter:      
                # Keep track of max radii in image-space for pruning 
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels=pixels)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, N=2, state_dict=dens_statistic_dict, store_grads=opt.store_grads)         
                
                if iteration % opt.opacity_reset_interval == 0 or (model.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            elif args.prune_dead_points and iteration % opt.densification_interval == 0:
                gaussians.prune(1/255, scene.cameras_extent, None, dens_statistic_dict)

            if iteration > opt.densify_until_iter and args_dict.get("extra_densi_state", False):
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pixels=pixels)
                if iteration % opt.densification_interval == 0:
                    gaussians.gradient_info_reset()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                Log.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                point_ckpt_path = args_dict["path_dict"]["model_path"] + "/ckpt_point/chkpnt_" + str(iteration) + "_.pth"
                torch.save((gaussians.capture(), iteration), point_ckpt_path)

            if (iteration in saving_iterations):
                dead_num = str(int(gaussians.opacity_dead_points(1/255)))
                iter_str = str(iteration)
                Log.info("\n[ITER {}] Saving Gaussians, the dead num is {}".format(iteration, dead_num))
                with open(os.path.join(args_dict["path_dict"]["exp_path"], "iter_" + iter_str+ "_Deadnum:_"+ dead_num + ".txt"), 'w') as file:  
                    pass

                if args.prune_dead_points:
                    gaussians.prune(1/255, scene.cameras_extent, None, dens_statistic_dict)

                scene.save(iteration)

    pixel_num = str(int(gaussians.get_xyz.shape[0]))
    with open(os.path.join(args_dict["path_dict"]["exp_path"], 'GS_num:_'+pixel_num+"_.txt"), 'w') as file:
            Log.info("The total number of gaussians is {}".format(pixel_num))
                
    
def tb_init(args_dict):
    tb_writer = None   
    tb_path = args_dict["path_dict"]["summary_path"] # TODO
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(tb_path)  
        Log.info("Logging progress to Tensorboard at {}".format(tb_path))
    else:
        Log.info("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, results_dict, args_dict, loss_aggregator, density_statistics_dict):

    Ll1 = results_dict["l1_loss"]
    Lssim = results_dict["ssim_loss"]
    loss = results_dict["total_loss"]
    alpha_regul = results_dict["alpha_regul"]
    elapsed = results_dict["iter_time"]
    total_num = results_dict["pixel_number"]

    if iteration % args.densification_interval == 0:
        (average_l1_loss,
        average_ssim_loss,
        average_alpha_regul,
        average_total_loss,
        average_iter_time) = (loss_aggregator["l1_loss"]/args.densification_interval,
                              loss_aggregator["ssim_loss"]/args.densification_interval,
                              loss_aggregator["alpha_regul"]/args.densification_interval,
                              loss_aggregator["total_loss"]/args.densification_interval,
                              loss_aggregator["iter_time"]/args.densification_interval)
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/avg_l1_loss', average_l1_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/avg_ssim_loss', average_ssim_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/avg_alpha_regul', average_alpha_regul, iteration)
            tb_writer.add_scalar('train_loss_patches/avg_total_loss', average_total_loss, iteration)
            tb_writer.add_scalar('avg_iter_time', average_iter_time, iteration)
            tb_writer.add_scalar('total_points/points_cloned', density_statistics_dict['n_points_cloned'], iteration)
            tb_writer.add_scalar('total_points/points_split', density_statistics_dict['n_points_split'], iteration)
            tb_writer.add_scalar('total_points/points_mercied', density_statistics_dict['n_points_mercied'], iteration)
            tb_writer.add_scalar('total_points/points_mercied_%', density_statistics_dict['n_points_mercied'] / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/points_pruned', density_statistics_dict['n_points_pruned'], iteration)
            tb_writer.add_scalar('total_points/points_pruned_%', density_statistics_dict['n_points_pruned'] / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/redundancy_threshold', density_statistics_dict['redundancy_threshold'], iteration)
            tb_writer.add_scalar('total_points/opacity_threshold', density_statistics_dict['opacity_threshold'], iteration)

        density_statistics_dict['n_points_cloned'] = 0
        density_statistics_dict['n_points_split'] = 0
        density_statistics_dict['n_points_mercied'] = 0
        density_statistics_dict['n_points_pruned'] = 0
        loss_aggregator["l1_loss"] = 0
        loss_aggregator["ssim_loss"] = 0
        loss_aggregator["alpha_regul"] = 0
        loss_aggregator["sh_sparsity_loss"] = 0
        loss_aggregator["total_loss"] = 0
        loss_aggregator["iter_time"] = 0
    else:
        loss_aggregator["l1_loss"] += Ll1.detach().item()
        loss_aggregator["ssim_loss"] += Lssim.detach().item()
        loss_aggregator["alpha_regul"] += alpha_regul.detach().item()
        loss_aggregator["total_loss"] += loss.detach().item()
        loss_aggregator["iter_time"] += elapsed

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Lssim.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/alpha_regul', alpha_regul.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', total_num, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, len(scene.getTrainCameras()), 3)]})

        for config in validation_configs:

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, args_dict=args_dict, is_training=False)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                Log.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS(vgg) {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                with open(os.path.join(args.path_dict['exp_path'], 'log_file.txt'), 'a') as file:
                    file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS(vgg) {} SSIM {} Num {}\n".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test, total_num))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # provide extra add_argument in lp op pp
    parser.add_argument('--config', type=str, help="path of config file")
    parser.add_argument('--seed', type=int, metavar='S', help='random seed')
    parser.add_argument('--distributed', type=int2bool, default=False, help="disable distributed training")
    parser.add_argument('--deterministic', type=bool, default=True, help="")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6027)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # ***********  Params for logging. and screen  **********
    parser.add_argument('--logfile_level', default='info', type=str, help='To set the log level to files.')
    parser.add_argument('--stdout_level', default='info', type=str, help='To set the level to print to screen.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=False, help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        help='Whether to write logging into files.')
    parser.add_argument('--log_format', type=str, nargs='?', default="%(asctime)s %(levelname)-7s %(message)s"
                        , help='Whether to write logging into files.')
    
    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    init(args)
    lp.from_cfg(args)
    op.from_cfg(args)
    pp.from_cfg(args)
    args.save_iterations.append(args.iterations)
    args.white_background = args.white_bg
    safe_state(args.quiet)
  
    outdoor_scenes=['bicycle', 'flowers', 'garden', 'stump', 'treehill']
    indoor_scenes=['room', 'counter', 'kitchen', 'bonsai']
    for scene in outdoor_scenes:
        if scene in args.source_path:
            args.images = "images_4"
            Log.info("Using images_4 for outdoor scenes")
    for scene in indoor_scenes:
        if scene in args.source_path:
            args.images = "images_2"
            Log.info("Using images_2 for indoor scenes")
    if 'playroom' in args.source_path:
        args.images = "images"
        Log.info("reset to images")
    
    save_args(args, args.path_dict["exp_path"], lp, pp, op, filename='total_args.json')

    # Start GUI server, configure and run training
    while True :
        try:
            network_gui.init(args.ip, args.port)
            Log.info(f"GUI server started at {args.ip}:{args.port}")
            break
        except Exception as e:
            args.port = args.port + 1
            Log.info(f"Failed to start GUI server, retrying with port {args.port}...")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,  args.save_iterations, args.checkpoint_iterations ,args.start_checkpoint, args.debug_from, args.__dict__)
    
    Log.info("\nTraining complete.")