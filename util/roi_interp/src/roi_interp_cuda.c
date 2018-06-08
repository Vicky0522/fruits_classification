/*************************************************************************
    > File Name: util/roi_interp/src/roi_interp_cuda.c
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 15时57分36秒
 ************************************************************************/
#include <THC/THC.h>
#include <math.h>
#include "cuda/roi_interp_kernel.h"

extern THCState *state;

int roi_interp_forward_cuda(int interp_height, int interp_width,
                            THCudaTensor * input, THCudaTensor * rois, 
                            THCudaTensor * output) {
  // Grab the input tensor
  float * data_flat = THCudaTensor_data(state, input);
  float * rois_flat = THCudaTensor_data(state, rois);
  float * out_flat = THCudaTensor_data(state, output);

  // Number of ROIs
  int batch_size = THCudaTensor_size(state, input, 0);
  int size_rois = THCudaTensor_size(state, rois, 1);
  if (size_rois != 4) {
        return 0;
  }
  int data_height = THCudaTensor_size(state, input, 2);
  int data_width = THCudaTensor_size(state, input, 3);
  int num_channels = THCudaTensor_size(state, input, 1);

  cudaStream_t stream = THCState_getCurrentStream(state);

  ROIInterpForwardLauncher(data_flat, rois_flat, out_flat, 
                        batch_size, num_channels, data_height, data_width, 
                        interp_height, interp_width, stream);

  return 1;
}

int roi_interp_backward_cuda(int interp_height, int interp_width,
                              THCudaTensor * out_grad, THCudaTensor * rois, 
                              THCudaTensor * data_grad) {
  // Grab the input tensor
  float * out_grad_flat = THCudaTensor_data(state, out_grad);
  float * rois_flat = THCudaTensor_data(state, rois);

  float * data_grad_flat = THCudaTensor_data(state, data_grad);

  // Number of ROIs
  int batch_size = THCudaTensor_size(state, rois, 0);
  int size_rois = THCudaTensor_size(state, rois, 1);
  if (size_rois != 4) {
      return 0;
  }

  int data_height = THCudaTensor_size(state, data_grad, 2);
  int data_width = THCudaTensor_size(state, data_grad, 3);
  int num_channels = THCudaTensor_size(state, data_grad, 1);

  cudaStream_t stream = THCState_getCurrentStream(state);
  ROIInterpBackwardLauncher(out_grad_flat, rois_flat, data_grad_flat, 
                         batch_size, num_channels, data_height, data_width,
                         interp_height, interp_width, stream);

  return 1;
}


