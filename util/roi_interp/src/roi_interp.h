/*************************************************************************
    > File Name: util/roi_interp/src/roi_interp.h
    > Author: ouyangwenqi
    > Mail: vinkeyoy@gmail.com
    > Created Time: 2018年04月11日 星期三 15时23分50秒
 ************************************************************************/
int roi_interp_forward(int interp_height, int interp_width, 
                       THFloatTensor * input, THFloatTensor * rois, 
                       THFloatTensor * output);

