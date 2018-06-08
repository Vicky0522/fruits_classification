/*************************************************************************
    > File Name: util/roi_interp/src/roi_interp_cuda.h
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 16时27分36秒
 ************************************************************************/
int roi_putin_forward_cuda( int front_height, int front_width,
                            THCudaTensor * input_front, THCudaTensor * rois, 
                            THCudaTensor * output);

int roi_putin_backward_cuda( int front_height, int front_width,
                             THCudaTensor * out_grad, THCudaTensor * rois, 
                             THCudaTensor * front_grad);



