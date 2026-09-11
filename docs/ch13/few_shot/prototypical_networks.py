"""prototypical_networks — prototypical_networks 페이지의 코드를 모듈로 쓸 수 있게 옮겨 놓은 것이다.

이 파일은 ch13/few_shot/prototypical_networks.md 에서 자동으로 뽑아낸 것이므로, 고칠 일이 있으면
그 페이지를 고치고 다시 뽑아내는 편이 낫다.
"""

"""
소수 예시 학습을 위한 원형 망

참고: Snell 외, "Prototypical Networks for Few-shot Learning" (2017)

핵심 생각: 받침 보기의 묻힘을 평균 내어 부류마다 원형 표현을 셈한 다음,
그 원형까지의 거리를 바탕으로
물음을 가려낸다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class ConvEncoder(nn.Module):
    """
    그림을 묻는 단순한 4층 합성곱 부호기.
    소수 예시 학습 논문에서 흔히 쓴다.
    """
    def __init__(self, input_channels=1, hidden_dim=64, output_dim=64):
        super(ConvEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            self._conv_block(input_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, output_dim),
        )
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class PrototypicalNetwork(nn.Module):
    """
    N-갈래 K-예시 분류를 위한 원형 망.
    
    인수:
        encoder: 입력을 특징 공간에 묻는 신경망
    """
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, support, support_labels, query):
        """
        인수:
            support: (n_support, *input_shape) - 받침 집합 보기
            support_labels: (n_support,) - 받침 집합의 이름표
            query: (n_query, *input_shape) - 가려낼 물음 보기
        
        반환값:
            logits: (n_query, n_classes) - 가려내기 로짓
        """
        # 모든 보기를 묻는다
        n_classes = len(torch.unique(support_labels))
        n_support = support.shape[0]
        n_query = query.shape[0]
        
        # 효율적인 부호화를 위해 받침과 물음을 이어 붙인다
        all_examples = torch.cat([support, query], dim=0)
        embeddings = self.encoder(all_examples)
        
        # 다시 받침과 물음으로 쪼갠다
        support_embeddings = embeddings[:n_support]
        query_embeddings = embeddings[n_support:]
        
        # 부류마다 원형을 셈한다
        prototypes = self._compute_prototypes(support_embeddings, support_labels, n_classes)
        
        # 물음에서 원형까지의 거리를 셈한다
        logits = self._compute_logits(query_embeddings, prototypes)
        
        return logits
    
    def _compute_prototypes(self, embeddings, labels, n_classes):
        """
        받침 묻힘의 평균으로 부류마다 원형을 셈한다.
        """
        prototypes = []
        for c in range(n_classes):
            # 부류 c의 받침 보기를 모두 찾는다
            class_mask = (labels == c)
            class_embeddings = embeddings[class_mask]
            # 평균(원형)을 셈한다
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def _compute_logits(self, query_embeddings, prototypes):
        """
        물음에서 원형까지의 유클리드 거리 제곱의 음수를 셈한다.
        거리의 음수가 로짓 노릇을 한다(가까울수록 확률이 높다).
        """
        # 퍼뜨리기를 위해 차원을 늘린다
        # query: (n_query, 1, embedding_dim)
        # prototypes: (1, n_classes, embedding_dim)
        query_expanded = query_embeddings.unsqueeze(1)
        prototypes_expanded = prototypes.unsqueeze(0)
        
        # 유클리드 거리의 제곱을 셈한다
        distances = torch.sum((query_expanded - prototypes_expanded) ** 2, dim=2)
        
        # 거리의 음수를 로짓으로 낸다
        return -distances


def train_step(model, support, support_labels, query, query_labels, optimizer):
    """
    원형 망의 학습 걸음 하나.
    """
    model.train()
    optimizer.zero_grad()
    
    # 순전파
    logits = model(support, support_labels, query)
    
    # 손실을 계산한다
    loss = F.cross_entropy(logits, query_labels)
    
    # 역전파
    loss.backward()
    optimizer.step()
    
    # 정확도를 계산한다
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


def evaluate(model, support, support_labels, query, query_labels):
    """
    소수 예시 과제에서 모델을 평가한다.
    """
    model.eval()
    with torch.no_grad():
        logits = model(support, support_labels, query)
        loss = F.cross_entropy(logits, query_labels)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == query_labels).float().mean()
    
    return loss.item(), accuracy.item()


# 사용 예
if __name__ == "__main__":
    # 모델 생성
    encoder = ConvEncoder(input_channels=1, hidden_dim=64, output_dim=64)
    model = PrototypicalNetwork(encoder)
    
    # 5-갈래 1-예시 과제 보기
    n_way = 5
    k_shot = 1
    n_query = 15
    
    # 흉내 데이터(batch_size, channels, height, width)
    support = torch.randn(n_way * k_shot, 1, 28, 28)
    support_labels = torch.arange(n_way).repeat_interleave(k_shot)
    query = torch.randn(n_query, 1, 28, 28)
    query_labels = torch.randint(0, n_way, (n_query,))
    
    # 순전파
    logits = model(support, support_labels, query)
    print(f"Logits shape: {logits.shape}")  # (15, 5)이어야 한다
    
    # 학습 보기
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train_step(model, support, support_labels, query, query_labels, optimizer)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
