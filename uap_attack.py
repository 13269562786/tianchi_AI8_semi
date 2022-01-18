import torch
import torch.nn as nn

from torchattacks.attack import Attack
from torch.nn.modules.loss import _WeightedLoss, _Loss

class BoundedLogitLossFixedRef(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLossFixedRef, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits.data.detach() - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)

class UAP(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=0.3,
                 alpha=1/255, steps=40, random_start=True):
        super().__init__("UAP", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = BoundedLogitLossFixedRef((num_classes=10, confidence=10, use_cuda=True))

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()-images.detach()

        return adv_images

def gen_uap(inputs_adv, labels, ori_inputs):
    UAP = torch.zeros(size=(10,3,32,32))
    for idx, image in enumerate(inputs_adv):
      UAP[labels[idx]] += image[idx]
      UAP[labels[idx]] = np.sign(UAP[labels[idx]])*np.minimum(abs(UAP[labels[idx]]), 8) #这个8是eps
    for idx, image in enumerate(ori_inputs):
      image = torch.clamp(images + UAP[labels[idx]], min=0, max=255) 
      ori_inputs[idx] = image
    return ori_inputs
