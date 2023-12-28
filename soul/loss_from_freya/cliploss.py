import torch
from torch.nn.modules import loss
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def gather_label(labels,
                world_size=1,
                gather_with_grad=False,
                ):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_soft_labels = torch.cat(
            torch.distributed.nn.all_gather(labels), dim=0)
    else:
        gathered_labels = [
            torch.zeros_like(labels) for _ in range(world_size)
        ]

        dist.all_gather(gathered_labels, labels)
        all_labels = torch.cat(gathered_labels, dim=0)
        
        # # all_soft_labels = all_labels / torch.sum(all_labels,dim=-1)
        # all_soft_labels = F.softmax(all_labels,dim=-1)

    return all_labels


def gather_features(image_features,
                    text_features,
                    rank=0,
                    world_size=1,
                    ):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    
    # We gather tensors from all gpus
    gathered_image_features = [
        torch.zeros_like(image_features) for _ in range(world_size)
    ]
    gathered_text_features = [
        torch.zeros_like(text_features) for _ in range(world_size)
    ]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)

    # ensure grads for local rank when all_* features don't have a gradient
    gathered_image_features[rank] = image_features
    gathered_text_features[rank] = text_features

    all_image_features = torch.cat(gathered_image_features, dim=0)
    all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ClipLoss(torch.nn.Module):
    def __init__(
        self,
        rank=0,
        world_size=1,

    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}


    def forward(self,
                image_features,
                text_features,
                ground_labels=None,
                logit_scale=1.0):
        
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, self.rank, self.world_size,
            )
            all_image_features = torch.nn.functional.normalize(all_image_features,p=2,dim=-1)
            all_text_features = torch.nn.functional.normalize(all_text_features,p=2,dim=-1)

            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T

        else:
            image_features = torch.nn.functional.normalize(image_features,p=2,dim=-1)
            text_features = torch.nn.functional.normalize(text_features,p=2,dim=-1)

            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]

        if ground_labels is not None:
            all_soft_labels = gather_label(ground_labels,self.world_size)
            # equal_labels = torch.zeros((all_soft_labels.shape[0],all_soft_labels.shape[0]))
            # for i,ife in enumerate(all_soft_labels):
            #     for j,jfe in enumerate(all_soft_labels):
            #         if torch.equal(ife,jfe):
            #             equal_labels[i][j] = 1.0
            #         else:
            #             equal_labels[i][j] = 0

            ground_labels_repeated = all_soft_labels.view(1, -1).repeat(
                image_features.shape[0]*self.world_size, 1)
            equal_labels = (ground_labels_repeated == all_soft_labels.view(
                -1, 1)).type(torch.float)

        
            labels = equal_labels / torch.sum(equal_labels, dim=1).view(-1, 1)
            labels = labels.cuda()
            total_loss = (F.cross_entropy(logits_per_image, labels) +
                            F.cross_entropy(logits_per_text, labels)) / 2
            
        else:
            ## TODO
            pass
        return total_loss