import torch
from torch.autograd import Function
from torch.nn import Module
from _ext import custom_lib

class InterpFunction(Function):
    def __init__(self, pad_beg_, pad_end_,
                 shrink_factor, zoom_factor,
                 height, width):
        assert pad_beg_<=0, "Only supports non-pos padding (cropping) for now"
        assert pad_end_<=0, "Only supports non-pos padding (cropping) for now"
        num_ = input.size(0)
        channels_ = input.size(1)
        height_in_ = input.size(2)
        width_in_ = input.size(3)
        height_in_eff_ = height_in_ + pad_beg_ + pad_end_
        width_in_eff_ = width_in_ + pad_beg_ + pad_end_

        if shrink_factor is not None and zoom_factor is None) :
            assert shrink_factor>=1, "Shrink factor must be positive"
            height_out_ = (height_in_eff_ - 1) / shrink_factor + 1
            width_out_ = (width_in_eff_ - 1) / shrink_factor + 1
        elif zoom_factor is not None and shrink_factor is None:
            assert zoom_factor>=1, "Zoom factor must be positive"
            height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1)
            width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1)
        elif height is not None and width is not None:
            height_out_  = height
            width_out_  = width
        elif shrink_factor is not None and zoom_factor is not None:
            assert shrink_factor>=1, "Shrink factor must be positive"
            assert zoom_factor>=1, "Zoom factor must be positive"
            height_out_ = (height_in_eff_ - 1) / shrink_factor + 1
            width_out_ = (width_in_eff_ - 1) / shrink_factor + 1
            height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
            width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
        else:
            assert True, "error while loading parameters"
  
        assert height_in_eff_>=0, "height should be positive"
        assert width_in_eff_>=0, "width should be positive"
        assert height_out_>=0, "height should be positive"
        assert width_out_>=0, "width should be positive"
        
        self.num_ = num_
        self.channels_ = channels_
        self.pad_beg_ = pad_beg_
        self.pad_end_ = pad_end_
        self.height_in_eff_ = height_in_eff_
        self.width_in_eff_ = width_in_eff_
        self.height_out_ = height_out_
        self.width_out_ = width_out_

    def forward(self, input):
        output = torch.zeros(self.num_, self.channels_, self.height_out_, self.width_out_)

        if not input.is_cuda:
            custom_lib.interp_forward(
                0,
                self.num_ * self.channels_, input, - self.pad_beg_, - self.pad_beg_, 
                self.height_in_eff_, self.width_in_eff_, 
                self.height_in_, self.width_in_,
                output, 0, 0, self.height_out_, self.width_out_, 
                self.height_out_, self.width_out_)
        else:
            output = output.cuda()
            custom_lib.interp_forward_cuda(
                input, output,
                self.num_, self.channels_, self.pad_beg_, self.pad_end_,
                self.height_in_eff_, self.width_in_eff,
                self.height_in_, self.width_in_, self.height_out_, self.width_out_)

        return output

    def backward(self, grad_output):
        grad_input = torch.zeros(self.num_, self.channels, self.height_in_, self.width_in_)

        if not grad_output.is_cuda:
            custom_lib.interp_backward(
                0,
                self.num_ * self.channels_, grad_input, - self.pad_beg_, - self.pad_beg_, 
                self.height_in_eff_, self.width_in_eff_, 
                self.height_in_, self.width_in_,
                grad_output, 0, 0, self.height_out_, self.width_out_, 
                self.height_out_, self.width_out_)
        else:
            grad_input = grad_input.cuda()
            custom_lib.interp_backward_cuda(
                grad_input, grad_output,
                self.num_, self.channels_, self.pad_beg_, self.pad_end_,
                self.height_in_eff_, self.width_in_eff,
                self.height_in_, self.width_in_, self.height_out_, self.width_out_)

        return grad_input, None

class Interp(Module):
    def __init__(self, pad_beg_, pad_end_,
                 shrink_factor = None, zoom_factor = None,
                 height = None, width = None):
        self.pad_beg_ = pad_beg_
        self.pad_end_ = pad_end_
        self.shrink_factor = shrink_factor
        self.zoom_factor = zoom_factor
        self.height = height
        self.width = width

    def forward(self, input):
        return InterpFunction(self.pad_beg_, self.pad_end_,
                              self.shrink_factor, self.zoom_factor,
                              self.height, self.width)(input)

    
