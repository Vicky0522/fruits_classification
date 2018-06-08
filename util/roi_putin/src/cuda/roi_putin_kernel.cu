#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_putin_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void ROIPutinForward(const int nthreads, 
                                const float* data_front, const float* rois_data, float* out,
                                const int channels, const int front_height, const int front_width,
                                const int back_height, const int back_width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in the output
    int n = index;
    int w = n % back_width;
    n /= back_width;
    int h = n % back_height;
    n /= back_height;
    int c = n % channels;
    n /= channels;

    rois_data += n * 4;
    const int roi_width_start = rois_data[0];
    const int roi_height_start = rois_data[1];
    const int roi_width_end = rois_data[2];
    const int roi_height_end = rois_data[3];
    const int roi_height = roi_height_end - roi_height_start + 1;
    const int roi_width = roi_width_end - roi_width_start + 1;

    // judge if it's in roi
    bool in_roi = (h >= roi_height_start && h <= roi_height_end &&
                   w >= roi_width_start && w <= roi_width_end);
    if (!in_roi) continue;

    data_front += n * channels * front_height * front_width;
    int data_front_size = front_height * front_width;
    int hi, wi, ho, wo, front_index;
    ho = h - roi_height_start;
    wo = w - roi_width_start;

    //special case : just copy
    if (roi_height == front_height && roi_width == front_width) {
      hi = ho;
      wi = wo;
      front_index = c * data_front_size + hi * front_width + wi;
      out[index] = data_front[front_index];
    }
    else {
      const float rheight = (roi_height > 1) ? static_cast<float>(front_height - 1) / (roi_height - 1) : 0.f;
      const float rwidth = (roi_width > 1) ? static_cast<float>(front_width - 1) / (roi_width - 1) : 0.f;
      const float h1r = rheight * ho;
      hi = h1r;
      const int h1p = (hi < front_height - 1) ? 1 : 0;
      const float h1lambda = h1r - hi;
      const float h0lambda = 1.f - h1lambda;
      const float w1r = rwidth * wo;
      const int wi = w1r;
      const int w1p = (wi < front_width - 1) ? 1 : 0;
      const float w1lambda = w1r - wi;
      const float w0lambda = 1.f - w1lambda;

      front_index = c * data_front_size + hi * front_width + wi;
      out[index] = h0lambda * (w0lambda * data_front[front_index] + w1lambda * data_front[front_index+w1p]) + 
                   h1lambda * (w0lambda * data_front[front_index + h1p * front_width] + 
                   w1lambda * data_front[front_index + h1p * front_width + w1p]);
    }
  }
}

int ROIPutinForwardLauncher(const float* input_front, const float* rois, float* output,
                            const int batch_size, const int channels, const int height, const int width,
                            const int back_height, const int back_width, 
                            cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * channels * back_height * back_width;
  cudaError_t err;

  ROIPutinForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, input_front, rois, output, channels, height, width, 
      back_height, back_width);

  err = cudaGetLastError();
  if(cudaSuccess != err) {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}

__global__ void ROIPutinBackward_SetZero(const int nthreads, 
                                         float* out_grad, const float* rois_data, const int channels, 
                                         const int back_height, const int back_width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in the out_grad
    int n = index;
    int w = n % back_width;
    n /= back_width;
    int h = n % back_height;
    n /= back_height;
    int c = n % channels;
    n /= channels;

    rois_data += n * 4;
    const int roi_width_start = rois_data[0];
    const int roi_height_start = rois_data[1];
    const int roi_width_end = rois_data[2];
    const int roi_height_end = rois_data[3];

    // judge if it's in roi
    bool in_roi = (h >= roi_height_start && h <= roi_height_end &&
                   w >= roi_width_start && w <= roi_width_end);
    if (in_roi) 
      out_grad[index] = 0;
  }
}

__global__ void ROIPutinBackward(const int nthreads, 
                                 const float* out_grad, const float* rois_data, float* front_grad,
                                  const int channels, const int front_height, const int front_width,
                                  const int back_height, const int back_width) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in the out_grad
    int n = index;
    int w = n % back_width;
    n /= back_width;
    int h = n % back_height;
    n /= back_height;
    int c = n % channels;
    n /= channels;

    rois_data += n * 4;
    const int roi_width_start = rois_data[0];
    const int roi_height_start = rois_data[1];
    const int roi_width_end = rois_data[2];
    const int roi_height_end = rois_data[3];
    const int roi_height = roi_height_end - roi_height_start + 1;
    const int roi_width = roi_width_end - roi_width_start + 1;

    // judge if it's in roi
    bool in_roi = (h >= roi_height_start && h <= roi_height_end &&
                   w >= roi_width_start && w <= roi_width_end);
    if (!in_roi) continue;

    int hi, wi, ho, wo, front_index;
    int data_front_size = front_height * front_width;
    ho = h - roi_height_start;
    wo = w - roi_width_start;

    front_grad += n * channels * data_front_size;
    // special case: just copy
    if (roi_height == front_height && roi_width == front_width) {
      hi = ho;
      wi = wo;
      front_index = c * data_front_size + hi * front_width + wi;
      front_grad[front_index] += out_grad[index];
    }
    else {
      const float rheight = (roi_height > 1) ? static_cast<float>(front_height - 1) / (roi_height - 1) : 0.f;
      const float rwidth = (roi_width > 1) ? static_cast<float>(front_width - 1) / (roi_width - 1) : 0.f;
      const float h1r = rheight * ho;
      const int hi = h1r;
      const int h1p = (hi < front_height - 1) ? 1 : 0;
      const float h1lambda = h1r - hi;
      const float h0lambda = 1.f - h1lambda;
      const float w1r = rwidth * wo;
      const int wi = w1r;
      const int w1p = (wi < front_width - 1) ? 1 : 0;
      const float w1lambda = w1r - wi;
      const float w0lambda = 1.f - w1lambda;

      front_index = c * data_front_size + hi * front_width + wi;
      atomicAdd(&front_grad[front_index], h0lambda * w0lambda * out_grad[index]);
      atomicAdd(&front_grad[front_index + w1p], h0lambda * w1lambda * out_grad[index]);
      atomicAdd(&front_grad[front_index + h1p * front_width], h1lambda * w0lambda * out_grad[index]);
      atomicAdd(&front_grad[front_index + h1p * front_width + w1p], h1lambda * w1lambda * out_grad[index]);
    }
  }
}

int ROIPutinBackwardLauncher(float* out_grad, const float* rois, float* front_grad, 
                             const int batch_size, const int channels, const int height, const int width, 
                             const int back_height, const int back_width, 
                             cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * back_height * back_width * channels;
    cudaError_t err;

    ROIPutinBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, out_grad, rois, front_grad, channels, height, width, 
      back_height, back_width);

    ROIPutinBackward_SetZero<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, out_grad, rois, channels, back_height, back_width);

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



