import torch
import math
import sys
sys.path.append('..')

import diff_gauss_dropout

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import torch.nn.functional as F
import numpy as np
# import torchvision.utils as vutils  

# utils
from utils.logger import Logger as Log
from utils.basic_utils import is_list_or_tuple, isNum

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, low_pass = 0.3, itr=-1, args_dict=None, is_training=True):   
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    drop_mask = None
    use_drop = False
    dense_constrained_kernel = 0  # play no rule when set to zero
    dense_constrained_thres = 0  
    use_depth_drop = False

    depth_drop_version = 0 
    if is_training and args_dict is not None and 'random_drop' in args_dict.keys() and args_dict['random_drop']:
        use_drop, adopt_list = adopt_drop(args_dict, itr)
        if use_drop:
            N = means3D.shape[0]
            drop_mask, dense_constrained_kernel, dense_constrained_thres, use_depth_drop, depth_drop_version = generate_random_drop_mask(N, args_dict, adopt_list, pc)

    depth_threshold = args_dict['depth_threshold']

    raster_settings = diff_gauss_dropout.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=depth_threshold,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        low_pass=low_pass,
        pixel_aware=args_dict['pixel_aware'],
        use_dropout=use_drop,
        seed= int(args_dict['seed']),
        depthdrop = use_depth_drop,
        depthdrop_version = depth_drop_version,
        densecount_kernel_size = dense_constrained_kernel,
        dense_thres = dense_constrained_thres
    )

    rasterizer = diff_gauss_dropout.GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, depth, check_dist = rasterizer(
                                                            means3D = means3D,
                                                            means2D = means2D,
                                                            opacities = opacity,
                                                            shs = shs,
                                                            colors_precomp = colors_precomp,
                                                            scales = scales,
                                                            rotations = rotations,
                                                            cov3D_precomp = cov3D_precomp,
                                                            drop_mask = drop_mask)
    pixels = check_dist.get('pixels', None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    visibility_filter = radii > 0

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : visibility_filter,
            "radii": radii,
            "depth": depth,
            "pixels": pixels,
            "more_info": None}

def generate_random_drop_mask(N, args_dict, adopt_list, gaussion_model=None):

    dropout_type = args_dict['random_threthod_mod']['type']
    dropout_prob = args_dict['random_threthod_mod']['prob']
    dropout_dense_kernel = args_dict['random_threthod_mod']['dksize']
    dropout_dense_thres = args_dict['random_threthod_mod']['dense_thres']
    drop_version_dict = args_dict['random_threthod_mod']['drop_version'] #['drop_version']

    if adopt_list is None:
        drop_mask, use_depth_drop = get_dropout_mask(dropout_type, dropout_prob, N, gaussion_model, args_dict)
        dense_constrained_kernel = dropout_dense_kernel # can only be odd and zero
        dense_constrained_thres = dropout_dense_thres
        drop_version = drop_version_dict

    elif adopt_list is not None and is_list_or_tuple(adopt_list):
        drop_mask = torch.ones(N, 1, device='cuda', dtype=torch.float)  # 1-> keep ; 0 --> drop

        dense_constrained_kernel = 0 # adopt maximize
        dense_constrained_thres = 0 # adopt maximize
        # check depth drop dict
        use_depth_drop = False
        drop_version = 0  # cant be None 
        for i in range(len(adopt_list)):
            if adopt_list[i]:
                drop_mask_i, use_depth_drop_i = get_dropout_mask(dropout_type[i], dropout_prob[i], N, gaussion_model, args_dict)
                if drop_mask_i is not None: 
                    drop_mask *= drop_mask_i
                assert not (use_depth_drop and use_depth_drop_i), "only one depth drop can adopted, use_depth_drop and use_depth_drop_i can not be True at the same time"
                use_depth_drop = use_depth_drop or use_depth_drop_i # Only one depth dropout strategy procedure is allowed

                if use_depth_drop :
                    drop_version = drop_version_dict[i]

                # When there are multiple values, take the maximum value, that is, use the largest dense constraint
                dense_constrained_kernel_i = dropout_dense_kernel[i] # can only be odd and zero
                dense_constrained_thres_i = dropout_dense_thres[i]
                dense_constrained_kernel = max(dense_constrained_kernel, dense_constrained_kernel_i)
                dense_constrained_thres = max(dense_constrained_thres, dense_constrained_thres_i)

    if drop_mask is not None:
        drop_mask = drop_mask.to(torch.float)

    return drop_mask, int(dense_constrained_kernel), float(dense_constrained_thres), use_depth_drop, drop_version


def get_dropout_mask(drop_type, drop_prob, N, gaussion_model=None, args_dict=None):
    assert drop_type in ['const', 'densifi_stage', 'depth_drop']

    if drop_type == 'const':
        const_thres = drop_prob
        random_tensor = torch.rand(N, 1, device='cuda')
        drop_mask = (random_tensor > const_thres).to(torch.float)  # when < thres, mask = 0 and drop
        use_depth_drop = False
    
    elif drop_type == 'densifi_stage':
        assert gaussion_model is not None, "gaussion_model is None"
        const_thres = drop_prob
        gradient_flag = gaussion_model.obtain_densifi_state
        densify_grad_threshold = args_dict['densify_grad_threshold']
        random_tensor = torch.rand(N, 1, device='cuda')
        gradient_flag = (gradient_flag > densify_grad_threshold).to(torch.float)  # if bigger than, adopt the ramdom drop
        drop_thres = gradient_flag * const_thres  # flag_verse
        drop_mask = (random_tensor > drop_thres).to(torch.float)
        use_depth_drop = False

    elif drop_type == 'depth_drop':
        use_depth_drop = True
        drop_mask = None

    if drop_mask is not None:
        drop_mask = drop_mask.detach()

    return drop_mask, use_depth_drop
    

def adopt_drop(args_dict, itr):
    ramdomdrop_end = args_dict['ramdomdrop_end']
    ramdomdrop_begin = args_dict['ramdomdrop_begin'] 
    ramdomdrop_mod = args_dict['random_threthod_mod']['type']
    if isinstance(ramdomdrop_end, list) and isinstance(ramdomdrop_begin, list):
        adopt_list = [False] * len(ramdomdrop_end)
        assert len(ramdomdrop_end) == len(ramdomdrop_begin), "len(ramdomdrop_end) != len(ramdomdrop_begin)"
        adopt_drop = False
        for i in range(len(ramdomdrop_end)):
            if itr < ramdomdrop_end[i] and itr > ramdomdrop_begin[i] - 1:
                if itr == ramdomdrop_begin[i]:
                    Log.info(f"Adopt {ramdomdrop_mod[i]} strategy for random drop from {ramdomdrop_begin[i]} to {ramdomdrop_end[i]}")
                adopt_drop = True
                adopt_list[i] = True
    elif isNum(ramdomdrop_end) and isNum(ramdomdrop_begin):
        if itr < ramdomdrop_end and itr > ramdomdrop_begin - 1:
            if itr == ramdomdrop_begin:
                Log.info(f"Adopt {ramdomdrop_mod} strategy for random drop from {ramdomdrop_begin} to {ramdomdrop_end}")
            adopt_drop = True
            adopt_list = None
    else:
        raise ValueError("ramdomdrop_end and ramdomdrop_begin should be list or int at the same time")
    
    return adopt_drop, adopt_list

