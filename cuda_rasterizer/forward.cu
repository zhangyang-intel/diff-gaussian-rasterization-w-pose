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

#include "forward.h"
#include "auxiliary.h"
#include "helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 当前3D高斯的位置均值在相机系下的坐标
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 限制3D高斯在相机视锥体范围内
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 当前2D高斯的2D协方差矩阵cov Σ′ = J^T * W^T * Σ * W * J
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,		// 3D高斯的数量，球谐系数的数量，球谐系数的最大度数
	const float* orig_points,							// 3D高斯的位置均值
	const glm::vec3* scales,							// 3D高斯的缩放
	const float scale_modifier,							// 3D高斯的缩放修正
	const glm::vec4* rotations,							// 3D高斯的旋转
	const float* opacities,								// 3D高斯的不透明度
	const float* shs,									// 球谐系数
	bool* clamped,										// 被剔除的3D高斯(RGB颜色为负值)
	const float* cov3D_precomp,							// 预计算的3D高斯协方差矩阵
	const float* colors_precomp,						// 预计算的3D高斯颜色
	const float* viewmatrix,							// 视图矩阵T_CW
	const float* projmatrix,							// 经过视图矩阵和投影矩阵相乘后得到的最终投影变换矩阵T_IW
	const glm::vec3* cam_pos,							// 相机位置
	const int W, int H,									// 图像宽度和高度
	const float tan_fovx, float tan_fovy,				// 相机水平和垂直视场角的正切值
	const float focal_x, float focal_y,					// 焦距
	int* radii,											// 2D高斯椭圆的半径
	float2* points_xy_image,							// 2D高斯的位置均值
	float* depths,										// 3D高斯的深度（3D高斯中心在相机系下z轴的坐标）
	float* cov3Ds,										// 3D高斯的协方差矩阵
	float* rgb,											// 3D高斯的颜色
	float4* conic_opacity,								// 2D高斯的协方差矩阵和不透明度
	const dim3 grid,									// 栅格大小（x和y方向上的栅格数）
	uint32_t* tiles_touched,							// 每个3D高斯触及到的栅格数量
	bool prefiltered)									// 是否预过滤
{
	// 获取当前线程的全局索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 初始化2D高斯椭圆的半径和当前高斯触及到的栅格数
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// 剔除距离相机太近的高斯，计算当前3D高斯在相机系下的坐标p_view
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	// 计算当前3D高斯位置均值在图像系下的坐标
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };  // (x,y,z) -> (x,y,z,w) -> (x/w,y/w,z/w)

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 计算或使用预计算当前的3D高斯的协方差矩阵
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;  // 对称矩阵，只存储上三角6个元素
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 计算当前2D高斯的协方差矩阵
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	// 计算当前2D高斯的协方差矩阵的逆矩阵conic Σ′^(-1)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 计算当前2D高斯协方差矩阵的特征值
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// 计算当前2D高斯椭圆(圆)的半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// 计算当前2D高斯在像素系下的像素坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	// 计算当前2D高斯外接矩形相交的栅格范围
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 使用球谐函数的系数计算当前3D高斯的RGB颜色存储到缓冲区
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// 存储当前高斯的几何信息到缓冲区
	depths[idx] = p_view.z;														// 当前3D高斯的深度
	radii[idx] = my_radius;														// 当前2D高斯的半径
	points_xy_image[idx] = point_image;											// 当前2D高斯的位置均值
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };			// 当前2D高斯的协方差矩阵和不透明度打包
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);	// 当前3D高斯触及到的栅格数量
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,						// 包含每个栅格排序的高斯的起始和结束位置索引的数组
	const uint32_t* __restrict__ point_list,				// 包含所有高斯的索引的数组
	int W, int H,											// 图像宽度和高度
	const float2* __restrict__ points_xy_image,				// 所有2D高斯的位置均值
	const float* __restrict__ features,						// 所有3D高斯的颜色
	const float4* __restrict__ conic_opacity,				// 所有2D高斯的协方差矩阵和不透明度
	float* __restrict__ final_T,							// 存储每个像素最终透光率的数组
	uint32_t* __restrict__ n_contrib,						// 存储对每个像素颜色有贡献的高斯计数
	const float* __restrict__ bg_color,						// 背景颜色
	float* __restrict__ out_color,							// 输出的图像颜色
	const float* __restrict__ depth,						// 所有3D高斯在相机系下的深度
	float* __restrict__ out_depth, 							// 输出的深度图	
	float* __restrict__ out_opacity,						// 输出的不透明度图（曝光）
	int * __restrict__ n_touched)							// 每个3D高斯触及到的像素数
{
	// Identify current tile and associated min/max pixel range.
	// 读取当前栅格block和相关的最小/最大像素范围pix_min～pix_max
	auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

	// 当前线程像素点的像素坐标pix和像素ID
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	// 判断当前像素点是否在图像范围内
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// 读取当前栅格中的高斯的起始和结束位置索引range，并计算需要处理的高斯数量toDo，对其进行分批次处理rounds
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// 为当前栅格中的每个线程分配共享存储空间
	__shared__ int collected_id[BLOCK_SIZE];				// 当前栅格中所有高斯的ID
	__shared__ float2 collected_xy[BLOCK_SIZE];				// 当前栅格中所有2D高斯的位置均值
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];	// 当前栅格中所有2D高斯的协方差矩阵和不透明度
	__shared__ float collected_depth[BLOCK_SIZE];			// 当前栅格中所有3D高斯的深度

	// Initialize helper variables
	// 初始化渲染相关变量
	float T = 1.0f;					// 当前高斯的透光率
	uint32_t contributor = 0;		// 对当前像素有贡献的高斯的ID
	uint32_t last_contributor = 0;	// 对当前像素有贡献的最后一个高斯的ID
	float C[CHANNELS] = { 0 };		// 当前像素的颜色
	float D = 0.0f;					// 当前像素的深度

	// Iterate over batches until all done or range is complete
	// 逐批次处理高斯，每批次处理256个高斯
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// 线程块内是否所有线程都已经完成渲染，退出循环
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 读取当前栅格中的高斯的数据到共享内存中
		int progress = i * BLOCK_SIZE + block.thread_rank();	// 当前线程处理的高斯索引
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;							// 当前高斯的ID
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];			// 当前2D高斯的位置均值
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];	// 当前2D高斯的协方差矩阵和不透明度
			collected_depth[block.thread_rank()] = depth[coll_id];					// 当前3D高斯在相机系下的深度
		}
		// 等待所有线程读取完毕再继续
		block.sync();

		// Iterate over current batch
		// 遍历当前批次的高斯
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			// 更新当前线程负责像素贡献高斯的ID
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// 计算当前2D高斯对于当前像素点的不透明度power：当前2D高斯的位置均值和当前像素位置之间的距离
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 计算当前高斯的不透明度alpha
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			// 计算当前高斯的透光率T
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			// 计算当前像素点的颜色C和深度D
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;	// C = C + ci * αi * T
			}
			D += collected_depth[j] * alpha * T;	// D = D + di * αi * T
			// Keep track of how many pixels touched this Gaussian.
			// 当前高斯的透光率T大于0.5时，当前3D高斯触及到的像素数+1
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			// 更新当前像素最终的贡献高斯的ID
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// 所有线程都处理完毕后，将最终渲染数据写入到输出缓冲区
	if (inside)
	{
		final_T[pix_id] = T;											// 当前像素的透光率
		n_contrib[pix_id] = last_contributor;							// 对当前像素有贡献的高斯数量
		for (int ch = 0; ch < CHANNELS; ch++) {							// 当前像素的颜色
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;											// 当前像素的深度
		out_opacity[pix_id] = 1 - T;									// 当前像素的不透明度
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth, 
	float* out_opacity,
	int* n_touched)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		out_opacity,
		n_touched);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}