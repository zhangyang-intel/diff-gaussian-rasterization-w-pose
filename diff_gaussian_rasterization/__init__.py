#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,                    # 上下文对象，保存反向传播所需的参数
        means3D,                # 3D高斯位置均值
        means2D,                # 2D高斯位置均值
        sh,                     # sh特征
        colors_precomp,         # 3D高斯预计算的颜色
        opacities,              # 3D高斯的不透明度
        scales,                 # 3D高斯的缩放
        rotations,              # 3D高斯的旋转
        cov3Ds_precomp,         # 3D高斯预计算的协方差
        theta,                  # 相机旋转增量
        rho,                    # 相机平移增量
        raster_settings,        # 高斯光栅化设置
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,                 # 背景颜色
            means3D,                            # 3D高斯位置均值
            colors_precomp,                     # 3D高斯预计算的颜色
            opacities,                          # 3D高斯的不透明度
            scales,                             # 3D高斯的缩放
            rotations,                          # 3D高斯的旋转
            raster_settings.scale_modifier,     # 3D高斯的缩放修正
            cov3Ds_precomp,                     # 3D高斯预计算的协方差
            raster_settings.viewmatrix,         # 视图矩阵T_CW
            raster_settings.projmatrix,         # 经过视图矩阵和投影矩阵相乘后得到的最终投影变换矩阵T_IW
            raster_settings.projmatrix_raw,     # 原始投影矩阵T_IC
            raster_settings.tanfovx,            # 相机水平视场角
            raster_settings.tanfovy,            # 相机垂直视场角
            raster_settings.image_height,       # 图像高度
            raster_settings.image_width,        # 图像宽度
            sh,                                 # sh特征
            raster_settings.sh_degree,          # sh特征的阶数
            raster_settings.campos,             # 相机位置
            raster_settings.prefiltered,        # 是否预过滤
            raster_settings.debug,              # 是否调试
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                # 比原始3DGS代码多了depth, opacity, n_touched，分别为深度图，不透明度图（曝光），每个高斯触及到的像素数量
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 被渲染的高斯总数、输出的颜色图、2D高斯半径、高斯几何缓冲区、binning缓冲区、图像缓冲区、深度图、不透明度图、每个像素触及到的高斯的数量
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered         # 被渲染的高斯总数
        # 3D高斯预计算的颜色、3D高斯位置均值、3D高斯的缩放、3D高斯的旋转、3D高斯预计算的协方差、2D高斯的半径、球谐系数、高斯几何、光栅、图像缓冲区
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_opacity, grad_n_touched):

        # 上下文、图像颜色、2D高斯半径、图像深度、每个像素触及到的高斯的数量的梯度

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,             # 背景颜色
                means3D,                        # 3D高斯均值
                radii,                          # 2D高斯半径
                colors_precomp,                 # 预计算的3D高斯颜色
                scales,                         # 3D高斯的缩放
                rotations,                      # 3D高斯的旋转
                raster_settings.scale_modifier, # 缩放修正
                cov3Ds_precomp,                 # 预计算的3D高斯颜色
                raster_settings.viewmatrix,     # 视图矩阵T_CW
                raster_settings.projmatrix,     # 经过视图矩阵和投影矩阵相乘后得到的最终投影变换矩阵T_IW
                raster_settings.projmatrix_raw, # 原始投影矩阵T_IC
                raster_settings.tanfovx,        # 相机水平视场角
                raster_settings.tanfovy,        # 相机垂直视场角
                grad_out_color,                 # RGB图的梯度
                grad_out_depth,                 # 深度图的梯度
                sh,                             # sh特征
                raster_settings.sh_degree,      # sh特征的阶数
                raster_settings.campos,         # 相机位置
                geomBuffer,                     # 高斯几何的缓冲区
                num_rendered,                   # 被渲染的高斯数量
                binningBuffer,                  # 光栅化的缓冲区
                imgBuffer,                      # 图像信息的缓冲区
                raster_settings.debug)          # 是否调试

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                # 比原始3DGS代码多了grad_tau，为相机位姿的梯度
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             # 2D高斯位置均值、预计算3D高斯颜色、3D高斯不透明度、3D高斯位置均值、预计算的3D高斯协方差、sh特征、3D高斯缩放、3D高斯旋转、3D高斯预计算的协方差、相机位姿的梯度
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
        
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)


        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    projmatrix_raw : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, theta=None, rho=None):
        
        # 这里输入了相机的旋转增量theta和平移增量rho，在render的前向传播中没有用到，在反向传播函数中使用
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:               # 相机旋转增量初始化
            theta = torch.Tensor([])
        if rho is None:                 # 相机平移增量初始化
            rho = torch.Tensor([])
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings, 
        )

