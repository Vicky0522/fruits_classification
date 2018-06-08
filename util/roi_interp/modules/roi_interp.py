import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
from ..functions.roi_interp import RoIInterpFunction

class RoIInterp(Module):
    def __init__(self, interp_height, interp_width, pad_input=0):
        super(RoIInterp, self).__init__()

        self.interp_width = int(interp_width)
        self.interp_height = int(interp_height)
        self.pad_input = int(pad_input)

    def forward(self, input, rois):
        # filter out [0,0,0,0] in rois
        index = []
        for i in range(input.size()[0]):
            if rois.data[i,0]==0 and rois.data[i,2]==0:
                pass
            else:
                index.append(i)
        index = Variable(torch.LongTensor(index))
        index = index.cuda() if input.is_cuda else index
        input = input.index_select(0, index)
        rois = rois.index_select(0, index)

        # pad input
        input_cat = (input,)
        rois_cat = (rois,)
        for i in range(self.pad_input-1):
            input_cat += (input,)
            rois_cat += (rois,)
        input = torch.cat(input_cat, 0)
        rois = torch.cat(rois_cat, 0)

        return RoIInterpFunction(self.interp_height, self.interp_width)(input, rois)

