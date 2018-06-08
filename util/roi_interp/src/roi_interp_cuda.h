/*************************************************************************
    > File Name: util/roi_interp/src/roi_interp_cuda.h
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 16时27分36秒
 ************************************************************************/
int roi_interp_forward_cuda(int interp_height, int interp_width,
                            THCudaTensor * input, THCudaTensor * rois, 
                            THCudaTensor * output);

int roi_interp_backward_cuda(int interp_height, int interp_width,
                              THCudaTensor * out_grad, THCudaTensor * rois, 
                              THCudaTensor * data_grad);


