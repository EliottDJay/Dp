#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


__global__ void zero_init(float* input, int H, int W)
{
	auto block = cg::this_thread_block();
	int x = block.group_index().x * BLOCK_X + block.thread_index().x;
	int y = block.group_index().y * BLOCK_Y + block.thread_index().y;
	int idx = y * W + x;

	if (idx >= W * H)
		return;

	input[idx] = 0.0f;
}

__device__ int clamp(int value, int min_val, int max_val) {  
	// return max(min_val, min(value, max_val - 1));  
    return (value < min_val) ? min_val : ((value >= max_val) ? max_val - 1 : value);  
} 

__global__ void gaussion_count(const float2* points_xy_image, int* output, int* radii, int P, int width, int height)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	if (radii[idx] > 0){
	float2 xy = points_xy_image[idx];
	int x = (int)xy.x;
	int y = (int)xy.y;
	x = clamp(x, 0, width);
	y = clamp(y, 0, height); 
	atomicAdd(&(output[y * width + x]), 1);
	}

}

__global__ void average_filter(int* input, float* output, int width, int height, int kernelSize)
{
	int halfKernel = kernelSize / 2; // odd
	auto block = cg::this_thread_block();
	int x = block.group_index().x * BLOCK_X + block.thread_index().x;
	int y = block.group_index().y * BLOCK_Y + block.thread_index().y;

	if (x < width && y < height)
	{
		float sum = 0.0f;
		for (int fy = -halfKernel; fy <= halfKernel; ++fy) 
		{ 
			for (int fx = -halfKernel; fx <= halfKernel; fx++)
			{
				int nx = x + fx; 
				int ny = y + fy;
				if (nx >= 0 && nx < width && ny >= 0 && ny < height) 
				{
					sum += static_cast<float>(input[ny * width + nx]);
				}
			}
		}
		output[y * width + x] = sum ; 
	}
}

__global__ void indentify(int* input, float* output, int width, int height)
{
	auto block = cg::this_thread_block();
	int x = block.group_index().x * BLOCK_X + block.thread_index().x;
	int y = block.group_index().y * BLOCK_Y + block.thread_index().y;

	if (x < width && y < height)
	{
		output[y * width + x] = static_cast<float>(input[y * width + x]);
	}
}


__global__ void redrop(
	int P, int width, int height, 
	uint32_t* tiles_touched, int* radii, 
	const float* densemap, const float2* points_xy_image, 
	float* drop_mask_vertifi, const float* drop_mask, 
	const float dense_drop_thres, const int kernelSize)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		
		if (drop_mask != nullptr)
		{
			if (drop_mask[idx] == 0.f){
				drop_mask_vertifi[idx] = 1.0f;
			}
		}

		if (drop_mask_vertifi[idx] == 1.0f)
		{
			float2 xy = points_xy_image[idx];
			int x = (int)xy.x;
			int y = (int)xy.y;
			x = clamp(x, 0, width);
			y = clamp(y, 0, height); 
			float dense = densemap[y * width + x];
			if (dense > dense_drop_thres)
			{
				radii[idx] = 0;
				tiles_touched[idx] = 0;
			}
			else
			{
				drop_mask_vertifi[idx] = 0.0f;
			}
		}
	}
}

__global__ void directlydrop(
	int P, uint32_t* tiles_touched, int* radii, 
	float* drop_mask_vertifi, const float* drop_mask)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		
		if (drop_mask != nullptr)
		{
			if (drop_mask[idx] == 0.f){
				drop_mask_vertifi[idx] = 1.0f;
			}
		}

		if (drop_mask_vertifi[idx] == 1.0f)
		{
			radii[idx] = 0;
			tiles_touched[idx] = 0;
		}
	}
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);  // 
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	const float low_pass,
	const bool use_dropout,
	const int seed,
	const bool depthdrop,
	const int depthdrop_verion,
	const int densecount_kernel_size,
	const float dense_thres,
	const float* drop_mask,
	float* pixels,
	int* radii,
	float* pcdepths,
	float* drop_mask_vertifi,
	int* point_num_check,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	if (drop_mask_vertifi == nullptr)
	{
		throw std::runtime_error("drop_mask_vertifi is nullptr");
	}

	if (pcdepths == nullptr)
	{
		throw std::runtime_error("pcdepths is nullptr");
	}

	if (point_num_check == nullptr)
	{
		throw std::runtime_error("point_num_check is nullptr");
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		pcdepths,
		drop_mask_vertifi,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		low_pass,
		seed,
		depthdrop,
		depthdrop_verion
	), debug)

	gaussion_count << <(P + 255) / 256, 256 >> > (
			geomState.means2D, point_num_check, radii, P, width, height);
	cudaDeviceSynchronize();
	CHECK_CUDA(, debug)

	if (use_dropout){

		if (densecount_kernel_size > 0 && dense_thres > 0.f)
		{
			float* point_dense_map;  
			cudaError_t err = cudaMalloc((void**)&point_dense_map, width * height * sizeof(float));
			if (err != cudaSuccess) 
			{  
        	std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;  
			throw std::runtime_error("CUDA malloc failed");
    		} 
			CHECK_CUDA(, debug)

			zero_init << < tile_grid, block >>> (point_dense_map, height, width);
			CHECK_CUDA(, debug)
			cudaDeviceSynchronize();
			// radii and tiles_touched set to 0 according to the drop mask
			if (densecount_kernel_size > 2)
			{
				average_filter << < tile_grid, block >> > (point_num_check, point_dense_map, width, height, densecount_kernel_size);
			}
			else if (densecount_kernel_size == 1)
			{
				indentify << < tile_grid, block >> > (point_num_check, point_dense_map, width, height);
			}
			CHECK_CUDA(, debug)
			cudaDeviceSynchronize();
			redrop << <(P + 255) / 256, 256 >> > (
				P, width, height, geomState.tiles_touched, 
				radii, point_dense_map, geomState.means2D, 
				drop_mask_vertifi, drop_mask, dense_thres, densecount_kernel_size);
			cudaDeviceSynchronize();
			CHECK_CUDA(, debug)
			cudaFree(point_dense_map);
		}
		else
		{
			directlydrop << <(P + 255) / 256, 256 >> > (P, geomState.tiles_touched, radii, drop_mask_vertifi, drop_mask);
			cudaDeviceSynchronize();
			CHECK_CUDA(, debug)
		}
	}

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered; 
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth,
		pixels), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* depth,
	bool debug,
	const float low_pass)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)
	
	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		depth,
		low_pass), debug)
}