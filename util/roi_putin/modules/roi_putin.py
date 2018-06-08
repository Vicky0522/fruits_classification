import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
from ..functions.roi_putin import RoIPutinFunction

class RoIPutin(Module):
    def __init__(self, front_height, front_width):
        super(RoIPutin, self).__init__()

        self.front_width = int(front_width)
        self.front_height = int(front_height)

    def forward(self, front, back, rois):
        # filter out [0,0,0,0] in rois
        index_face = []
        index_noface = []
        for i in range(rois.size()[0]):
            if rois.data[i,0]==0 and rois.data[i,2]==0:
                index_noface.append(i)
            else:
                index_face.append(i)
        index_noface = Variable(torch.LongTensor(index_noface))
        index_face = Variable(torch.LongTensor(index_face))
        index_noface = index_noface.cuda() if front.is_cuda else index_noface
        index_face = index_face.cuda() if front.is_cuda else index_face

        back_noface = back.index_select(0, index_noface) if len(index_noface)>0 else None
        back_face = back.index_select(0, index_face)
        rois = rois.index_select(0, index_face)

        back_face = RoIPutinFunction(self.front_height, self.front_width)(front, back_face, rois)
        return torch.cat([back_face, back_noface], 0) if back_noface is not None else back_face

