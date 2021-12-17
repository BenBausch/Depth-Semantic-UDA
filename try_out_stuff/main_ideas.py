import torch
import torch.nn as nn

from losses.losses import BootstrappedCrossEntropy

if __name__ == '__main__':
    # 2x2 image with 3 classes
    prediction = torch.tensor([[[0.3, 0.7, 0.0], [0.8, 0.2, 0.0]],
                               [[0.1, 0.9, 0.0], [0.6, 0.4, 0.0]]], requires_grad=True).transpose(0,2).transpose(1,2)

    target = torch.tensor([[0, 0],
                           [1, 2]], dtype=torch.long)

    # add batch dimensions
    prediction = prediction.unsqueeze(0)
    target = target.unsqueeze(0)
    print(f'Prediction shape: {prediction.shape}')
    print(f'Target shape: {target.shape}')

    bce = nn.CrossEntropyLoss(reduction='none')#ignore_index=250)
    print(bce(prediction, target).view(-1))


