# Inception V3

Inception v3, presented in the 2015 paper "Rethinking the Inception Architecture," refines the original GoogLeNet with several key innovations: factorized convolutions that decompose large filters into smaller asymmetric ones, label smoothing regularization, and auxiliary classifiers for training stability. These improvements together yield a more efficient and accurate architecture that became a standard baseline for image classification research.

## Code

```python
#!/usr/bin/env python3
'''
Inception v3 - Improved Inception Architecture
Paper: "Rethinking the Inception Architecture" (2015)
Key: Factorized convolutions (nx1 and 1xn), label smoothing, auxiliary classifiers
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    model = InceptionV3()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Discussion

The most significant architectural contribution of Inception v3 is the factorization of convolutions. A $5 \times 5$ convolution is replaced by two stacked $3 \times 3$ convolutions, reducing parameters from $25C^2$ to $18C^2$. Going further, an $n \times n$ convolution is factorized into an $n \times 1$ followed by a $1 \times n$ convolution, which for $n=7$ reduces parameters from $49C^2$ to $14C^2$. These factorizations maintain the receptive field while dramatically reducing computation.

Label smoothing is another important contribution. Instead of using hard one-hot targets, Inception v3 replaces the target distribution with a mixture: $(1-\epsilon) \cdot \delta_{k,y} + \epsilon / K$, where $\epsilon$ is typically 0.1 and $K$ is the number of classes. This prevents the model from becoming overconfident and improves generalization. The technique has since become standard practice in training deep networks.

Auxiliary classifiers, attached to intermediate layers during training, provide additional gradient signals to combat the vanishing gradient problem in very deep networks. They are weighted by a small factor (0.3) and removed at inference time. While their contribution to preventing vanishing gradients has been debated, they serve as effective regularizers by encouraging discriminative features at intermediate representations.

## Exercises

**Exercise 1.**
Calculate the parameter savings from factorizing a $7 \times 7$ convolution with 256 input and 256 output channels into a $7 \times 1$ followed by $1 \times 7$ convolution.

??? success "Solution to Exercise 1"
    Original $7 \times 7$ convolution: $256 \times 256 \times 7 \times 7 = 3,211,264$ parameters. Factorized: $7 \times 1$ convolution has $256 \times 256 \times 7 \times 1 = 458,752$ parameters, and $1 \times 7$ convolution has $256 \times 256 \times 1 \times 7 = 458,752$ parameters. Total factorized: $917,504$ parameters. Savings: $3,211,264 / 917,504 \approx 3.5\times$ fewer parameters, or about 71% reduction.

---

**Exercise 2.**
Explain why label smoothing helps prevent overfitting. How does it interact with the cross-entropy loss function?

??? success "Solution to Exercise 2"
    With hard one-hot targets, the cross-entropy loss drives the model to output arbitrarily large logits for the correct class relative to others, as $-\log(\text{softmax})$ approaches zero only as the logit difference approaches infinity. This leads to overconfident predictions and poor calibration. Label smoothing changes the target distribution so the correct class has probability $1-\epsilon+\epsilon/K$ instead of 1, and incorrect classes have probability $\epsilon/K$ instead of 0. The loss now has a finite optimum where the model assigns high but bounded probability to the correct class. This acts as a regularizer that encourages the model to maintain uncertainty, improves calibration, and reduces the gap between training and validation performance.

---

**Exercise 3.**
Implement an Inception module with three parallel branches: a $1 \times 1$ convolution, a factorized $3 \times 3$ convolution (using $3 \times 1$ and $1 \times 3$), and a max-pooling branch.

??? success "Solution to Exercise 3"
    ```python
    class InceptionModule(nn.Module):
        def __init__(self, in_ch, out_1x1, out_3x3, pool_proj):
            super().__init__()
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, out_1x1, 1, bias=False),
                nn.BatchNorm2d(out_1x1), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_ch, out_3x3, (3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
                nn.Conv2d(out_3x3, out_3x3, (1, 3), padding=(0, 1), bias=False),
                nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
            )
            self.branch3 = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                nn.Conv2d(in_ch, pool_proj, 1, bias=False),
                nn.BatchNorm2d(pool_proj), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
    ```
