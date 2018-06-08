import torch
from torch.autograd import Function
from torch.autograd import Variable
from .. import roi_putin


class RoIPutinFunction(Function):
    def __init__(self, front_height, front_width):
        self.front_width = int(front_width)
        self.front_height = int(front_height)
        self.output = None
        self.rois = None
        self.input_size = None

    def forward(self, front, back, rois):
        if not front.is_cuda:
            roi_putin.roi_putin_forward(self.front_height, self.front_width,
                                        front, rois, back)
            # output = output.cuda()
        else:
            roi_putin.roi_putin_forward_cuda(self.front_height, self.front_width, 
                                             front, rois, back)
            self.rois = rois
            self.front_size = front.size()

        return back

    def backward(self, grad_output):
        assert(self.front_size is not None and grad_output.is_cuda)

        batch_size, num_channels, front_height, front_width = self.front_size

        grad_front = torch.zeros(batch_size, num_channels, front_height, front_width).cuda()
        roi_putin.roi_putin_backward_cuda(self.front_height, self.front_width,
                                          grad_output, self.rois, grad_front)

        # print grad_input

        return grad_front, grad_output, None
