/*************************************************************************
    > File Name: util/roi_interp/src/cuda/roi_interp_kernel.h
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 19时46分31秒
 ************************************************************************/
#ifndef _ROI_INTERP_KERNEL
#define _ROI_INTERP_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int ROIInterpForwardLauncher(const float* input, const float* rois, float* output,
                            const int batch_size, const int channels, const int height, const int width,
                            const int interp_height, const int interp_width, 
                            cudaStream_t stream);

int ROIInterpBackwardLauncher(const float* out_grad, const float* rois, float* data_grad, 
                             const int batch_size, const int channels, const int data_height, const int data_width, 
                             const int interp_height, const int interp_width, 
                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

