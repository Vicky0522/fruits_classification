#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_interp_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void ROIInterpForward(const int nthreads, 
                                 const float* data, const float* rois_data, float* out,
                                 const int channels, const int data_height, const int data_width,
                                 const int interp_height, const int interp_width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ho, wo) is an element in the interp output
    int n = index;
    int wo = n % interp_width;
    n /= interp_width;
    int ho = n % interp_height;
    n /= interp_height;
    int c = n % channels;
    n /= channels;

    rois_data += n * 4;
    const int roi_width_start = rois_data[0];
    const int roi_height_start = rois_data[1];
    const int roi_width_end = rois_data[2];
    const int roi_height_end = rois_data[3];
    const int roi_height = roi_height_end - roi_height_start;
    const int roi_width = roi_width_end - roi_width_start;

    data += n * channels * data_height * data_width;
    int data_size = data_height * data_width;
    int hi, wi, data_index;

    //special case : just copy
    if (roi_height == interp_height && roi_width == interp_width) {
      hi = ho;
      wi = wo;
      data_index = c * data_size + (roi_height_start + hi) * data_width + (roi_width_start + wi);
      out[index] = data[data_index];
    }
    else {
      const float rheight = (interp_height > 1) ? static_cast<float>(roi_height - 1) / (interp_height - 1) : 0.f;
      const float rwidth = (interp_width > 1) ? static_cast<float>(roi_width - 1) / (interp_width - 1) : 0.f;
      const float h1r = rheight * ho;
      hi = h1r;
      const int h1p = (hi < roi_height - 1) ? 1 : 0;
      const float h1lambda = h1r - hi;
      const float h0lambda = 1.f - h1lambda;
      const float w1r = rwidth * wo;
      const int wi = w1r;
      const int w1p = (wi < roi_width - 1) ? 1 : 0;
      const float w1lambda = w1r - wi;
      const float w0lambda = 1.f - w1lambda;

      data_index = c * data_size + (roi_height_start + hi) * data_width + (roi_width_start + wi);
      out[index] = h0lambda * (w0lambda * data[data_index]            + w1lambda * data[data_index+w1p]) + 
                   h1lambda * (w0lambda * data[data_index + h1p * data_width] + 
                   w1lambda * data[data_index + h1p * data_width + w1p]);
    }
  }
}

int ROIInterpForwardLauncher(const float* input, const float* rois, float* output,
                            const int batch_size, const int channels, const int height, const int width,
                            const int interp_height, const int interp_width, 
                            cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * channels * interp_height * interp_width;
  cudaError_t err;

  ROIInterpForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, input, rois, output, channels, height, width, 
      interp_height, interp_width);

  err = cudaGetLastError();
  if(cudaSuccess != err) {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}

__global__ void ROIInterpBackward(const int nthreads, 
                                  const float* out_grad, const float* rois_data, float* data_grad,
                                  const int channels, const int data_height, const int data_width,
                                  const int interp_height, const int interp_width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ho, wo) is an element in the interp out_grad
    int n = index;
    int wo = n % interp_width;
    n /= interp_width;
    int ho = n % interp_height;
    n /= interp_height;
    int c = n % channels;
    n /= channels;

    rois_data += n * 4;
    const int roi_width_start = rois_data[0];
    const int roi_height_start = rois_data[1];
    const int roi_width_end = rois_data[2];
    const int roi_height_end = rois_data[3];
    const int roi_height = roi_height_end - roi_height_start;
    const int roi_width = roi_width_end - roi_width_start;

    int hi, wi, data_index;
    int data_size = data_height * data_width;

    data_grad += n * channels * data_size;
    // special case: just copy
    if (roi_height == interp_height && roi_width == interp_width) {
      hi = ho;
      wi = wo;
      data_index = c * data_size + (roi_height_start + hi) * data_width + (roi_width_start + wi);
      data_grad[data_index] += out_grad[index];
    }
    else {
      const float rheight = (interp_height > 1) ? static_cast<float>(roi_height - 1) / (interp_height - 1) : 0.f;
      const float rwidth = (interp_width > 1) ? static_cast<float>(roi_width - 1) / (interp_width - 1) : 0.f;
      const float h1r = rheight * ho;
      const int hi = h1r;
      const int h1p = (hi < roi_height - 1) ? 1 : 0;
      const float h1lambda = h1r - hi;
      const float h0lambda = 1.f - h1lambda;
      const float w1r = rwidth * wo;
      const int wi = w1r;
      const int w1p = (wi < roi_width - 1) ? 1 : 0;
      const float w1lambda = w1r - wi;
      const float w0lambda = 1.f - w1lambda;

      data_index = c * data_size + (roi_height_start + hi) * data_width + (roi_width_start + wi);
      atomicAdd(&data_grad[data_index], h0lambda * w0lambda * out_grad[index]);
      atomicAdd(&data_grad[data_index + w1p], h0lambda * w1lambda * out_grad[index]);
      atomicAdd(&data_grad[data_index + h1p * data_width], h1lambda * w0lambda * out_grad[index]);
      atomicAdd(&data_grad[data_index + h1p * data_width + w1p], h1lambda * w1lambda * out_grad[index]);
    }
  }
}

int ROIInterpBackwardLauncher(const float* out_grad, const float* rois, float* data_grad, 
                             const int batch_size, const int channels, const int data_height, const int data_width, 
                             const int interp_height, const int interp_width, 
                             cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * interp_height * interp_width * channels;
    cudaError_t err;

    ROIInterpBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, out_grad, rois, data_grad, channels, data_height, data_width, 
      interp_height, interp_width);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }
    return 1;
}


#ifdef __cplusplus
}
#endif



