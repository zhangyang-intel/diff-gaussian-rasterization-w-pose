/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,			// 背景
	const torch::Tensor& means3D,				// 3D高斯的位置均值
    const torch::Tensor& colors,				// 3D高斯预计算的颜色
    const torch::Tensor& opacity,				// 3D高斯的不透明度
	const torch::Tensor& scales,				// 3D高斯的缩放
	const torch::Tensor& rotations,				// 3D高斯的旋转
	const float scale_modifier,					// 3D高斯的缩放修正
	const torch::Tensor& cov3D_precomp,			// 3D高斯的协方差矩阵
	const torch::Tensor& viewmatrix,			// 视图矩阵T_CW
	const torch::Tensor& projmatrix,			// 经过视图矩阵和投影矩阵相乘后得到的最终投影变换矩阵T_IW
    const torch::Tensor& projmatrix_raw,		// 原始投影矩阵T_IC
    const float tan_fovx,						// 相机水平视场角
	const float tan_fovy,						// 相机垂直视场角
    const int image_height,						// 图像高度
    const int image_width,						// 图像宽度
	const torch::Tensor& sh,					// 球谐函数系数
	const int degree,							// 球谐函数的阶数
	const torch::Tensor& campos,				// 相机位置
	const bool prefiltered,						// 是否预过滤
	const bool debug)							// 是否调试
{、
  // 检查means3D的维度 (num_points, 3)
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);	// 3D高斯的数量
  const int H = image_height;
  const int W = image_width;

  // 张量的数据类型：int32和float32
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);			// 储存输出的图像颜色
  torch::Tensor radii = torch::full({P}, 0, int_opts);									// 储存2D高斯椭圆的半径
  torch::Tensor n_touched = torch::full({P}, 0, int_opts);								// 储存3D高斯触及到的像素数量
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);					// 储存输出的深度图
  torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);					// 储存输出的不透明度图

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  // 存储高斯几何信息、binning信息和图像信息的缓冲区函数
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }
	  /************************************render的前向传播*********************************************/
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_opaticy.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		n_touched.contiguous().data<int>(),
        debug);
  }
  // 返回被渲染的高斯总数、输出的颜色图、2D高斯半径、高斯几何缓冲区、binning缓冲区、图像缓冲区、深度图、不透明度图、每个像素触及到的高斯的数量
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth, out_opaticy, n_touched);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,			// 背景颜色
	const torch::Tensor& means3D,				// 3D高斯的位置均值
	const torch::Tensor& radii,					// 2D高斯的半径
    const torch::Tensor& colors,				// 预计算的3D高斯颜色
	const torch::Tensor& scales,				// 3D高斯的缩放
	const torch::Tensor& rotations,				// 3D高斯的旋转
	const float scale_modifier,					// 缩放修正
	const torch::Tensor& cov3D_precomp,			// 预计算3D高斯协方差
	const torch::Tensor& viewmatrix,			// 视图矩阵T_CW
    const torch::Tensor& projmatrix,			// 经过视图矩阵和投影矩阵相乘后得到的最终投影变换矩阵T_IW
    const torch::Tensor& projmatrix_raw,		// 原始投影矩阵T_IC
    const float tan_fovx,						// 相机水平视场角
	const float tan_fovy,						// 相机垂直视场角
    const torch::Tensor& dL_dout_color,			// RGB图像的梯度
	const torch::Tensor& dL_dout_depths,		// 深度图像的梯度
	const torch::Tensor& sh,					// sh特征
	const int degree,							// sh特征的阶数
	const torch::Tensor& campos,				// 相机位置
	const torch::Tensor& geomBuffer,			// 高斯几何的缓冲区
	const int R,								// 被渲染的高斯数量
	const torch::Tensor& binningBuffer,			// 光栅化的缓冲区
	const torch::Tensor& imageBuffer,			// 图像信息的缓冲区
	const bool debug) 							// 是否调试
{
  const int P = means3D.size(0);		// 高斯数量
  const int H = dL_dout_color.size(1);	// 图像大小
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);		// 球谐系数个数
  }
  // 初始化存放梯度的容器
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());				// 3D高斯位置均值梯度
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());				// 2D高斯位置均值梯度
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());	// 3D高斯RGB颜色梯度
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());				// 3D高斯在相机系下深度的梯度
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());				// 2D高斯协方差梯度
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());				// 3D高斯不透明度梯度
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());				// 3D高斯协方差梯度
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());				// sh特征梯度
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());				// 3D高斯缩放梯度
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());			// 3D高斯旋转梯度
  torch::Tensor dL_dtau = torch::zeros({P,6}, means3D.options());					// 相机位姿的梯度

  if(P != 0)
  {  
	/************************************光栅化的反向传播*********************************************/ 
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
      projmatrix_raw.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depths.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
      dL_dtau.contiguous().data<float>(),
	  debug);
  }
  // 返回2D高斯位置均值、3D高斯颜色、3D高斯不透明度、3D高斯位置均值、3D高斯协方差、sh特征、3D高斯缩放、3D高斯旋转、相机位姿的梯度
  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}