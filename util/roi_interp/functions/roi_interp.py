import torch
from torch.autograd import Function
from torch.autograd import Variable
from .. import roi_interp


class RoIInterpFunction(Function):
    def __init__(self, interp_height, interp_width):
        self.interp_width = int(interp_width)
        self.interp_height = int(interp_height)
        self.output = None
        self.rois = None
        self.input_size = None

    def forward(self, input, rois):
        batch_size, num_channels, data_height, data_width = input.size()
        output = torch.zeros(batch_size, num_channels, self.interp_height, self.interp_width)

        if not input.is_cuda:
            print(input)
            print(rois)
            roi_interp.roi_interp_forward(self.interp_height, self.interp_width,
                                          input, rois, output)
            # output = output.cuda()
        else:
            output = output.cuda()
            roi_interp.roi_interp_forward_cuda(self.interp_height, self.interp_width, 
                                               input, rois, output)
            self.output = output
            self.rois = rois
            self.input_size = input.size()

        return output

    def backward(self, grad_output):
        assert(self.input_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.input_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()
        roi_interp.roi_interp_backward_cuda(self.interp_height, self.interp_width,
                                            grad_output, self.rois, grad_input)

        # print grad_input

        return grad_input, None
