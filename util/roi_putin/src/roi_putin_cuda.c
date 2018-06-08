/*************************************************************************
    > File Name: util/roi_interp/src/roi_interp_cuda.c
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 15时57分36秒
 ************************************************************************/
#include <THC/THC.h>
#include <math.h>
#include "cuda/roi_putin_kernel.h"

extern THCState *state;

int roi_putin_forward_cuda( int front_height, int front_width,
                            THCudaTensor * input_front, THCudaTensor * rois, 
                            THCudaTensor * output) {
  // Grab the input tensor
  float * data_front_flat = THCudaTensor_data(state, input_front);
  float * rois_flat = THCudaTensor_data(state, rois);
  float * out_flat = THCudaTensor_data(state, output);

  // Number of ROIs
  int batch_size = THCudaTensor_size(state, input_front, 0);
  int size_rois = THCudaTensor_size(state, rois, 1);
  if (size_rois != 4) {
        return 0;
  }
  int back_height = THCudaTensor_size(state, output, 2);
  int back_width = THCudaTensor_size(state, output, 3);
  int num_channels = THCudaTensor_size(state, input_front, 1);

  cudaStream_t stream = THCState_getCurrentStream(state);

  ROIPutinForwardLauncher(data_front_flat, rois_flat, out_flat, 
                          batch_size, num_channels, front_height, front_width, 
                          back_height, back_width, stream);

  return 1;
}

int roi_putin_backward_cuda( int front_height, int front_width,
                             THCudaTensor * out_grad, THCudaTensor * rois, 
                             THCudaTensor * front_grad) {
  // Grab the input tensor
  float * out_grad_flat = THCudaTensor_data(state, out_grad);
  float * rois_flat = THCudaTensor_data(state, rois);

  float * front_grad_flat = THCudaTensor_data(state, front_grad);

  // Number of ROIs
  int batch_size = THCudaTensor_size(state, rois, 0);
  int size_rois = THCudaTensor_size(state, rois, 1);
  if (size_rois != 4) {
      return 0;
  }

  int back_height = THCudaTensor_size(state, out_grad, 2);
  int back_width = THCudaTensor_size(state, out_grad, 3);
  int num_channels = THCudaTensor_size(state, front_grad, 1);

  cudaStream_t stream = THCState_getCurrentStream(state);
  ROIPutinBackwardLauncher(out_grad_flat, rois_flat, front_grad_flat, 
                           batch_size, num_channels, front_height, front_width,
                           back_height, back_width, stream);

  return 1;
}


