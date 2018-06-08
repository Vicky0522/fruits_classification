/*************************************************************************
    > File Name: util/roi_interp/src/roi_interp.h
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 15时23分50秒
 ************************************************************************/
int roi_putin_forward(const int front_height, const int front_width,
                      THFloatTensor * input_front,
                      THFloatTensor * rois, THFloatTensor * output);

