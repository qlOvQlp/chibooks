import torch
from torch.nn.modules import loss
import torch.nn.functional as F


class Soft_CrossEntropy(loss._Loss):
    def forward(self, model_output, soft_output):
        size_average = True
        model_output_log_prob = F.log_softmax(model_output, dim=1)
        soft_output = soft_output.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(soft_output, model_output_log_prob)
        if size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        return cross_entropy_loss
    