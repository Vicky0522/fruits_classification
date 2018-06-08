/*************************************************************************
    > File Name: util/roi_interp/src/cuda/roi_interp_kernel.h
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 19时46分31秒
 ************************************************************************/
#ifndef _ROI_PUTIN_KERNEL
#define _ROI_PUTIN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int ROIPutinForwardLauncher(const float* input_front, const float* rois, float* output,
                            const int batch_size, const int channels, const int height, const int width,
                            const int back_height, const int back_width, 
                            cudaStream_t stream);

int ROIPutinBackwardLauncher(float* out_grad, const float* rois, float* front_grad, 
                             const int batch_size, const int channels, const int height, const int width, 
                             const int back_height, const int back_width, 
                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

