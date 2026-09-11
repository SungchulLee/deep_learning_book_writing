"""N-그램 말 모델 — 묻힘 절의 예제들이 함께 쓰는 바탕.

앞선 낱말 몇 개(문맥)를 보고 다음 낱말을 맞히는 가장 단순한 신경 말 모델이다.
묻힘 층 하나와 완전 연결 층 둘이 전부이며, 손실 함수를 견주는 예제들이
이 모델을 공통 바탕으로 쓴다.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ARGS", "TEST_SENTENCE", "vocab", "word_to_ix", "ngrams",
           "NGramLanguageModeler", "train"]

# 예제들이 같은 설정을 쓰도록 한곳에 모아 둔다
ARGS = SimpleNamespace(
    context_size=2,
    embedding_dim=10,
    hidden_dim=128,
    lr=0.001,
    epochs=10,
    seed=42,
)

# 셰익스피어 소네트 2번의 앞부분. 작지만 되풀이가 있어 학습이 눈에 보인다.
TEST_SENTENCE = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a tattered weed of small worth held:
Then being asked where all thy beauty lies,
Where all the treasure of thy lusty days;
To say within thine own deep sunken eyes,
Were an all-eating shame and thriftless praise.""".split()

# (문맥 낱말들, 다음 낱말) 짝
ngrams = [
    ([TEST_SENTENCE[i - j - 1] for j in range(ARGS.context_size)], TEST_SENTENCE[i])
    for i in range(ARGS.context_size, len(TEST_SENTENCE))
]

vocab = sorted(set(TEST_SENTENCE))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ARGS.vocab_size = len(vocab)


class NGramLanguageModeler(nn.Module):
    """문맥 낱말 context_size개로 다음 낱말을 맞힌다.

    내놓는 값은 로짓이다. 소프트맥스를 걸지 않으므로 CrossEntropyLoss에
    그대로 넣으면 되고, NLLLoss를 쓰려면 log_softmax를 따로 걸어야 한다.
    """

    def __init__(self, vocab_size=None, embedding_dim=None, context_size=None):
        super().__init__()
        vocab_size = vocab_size or ARGS.vocab_size
        embedding_dim = embedding_dim or ARGS.embedding_dim
        context_size = context_size or ARGS.context_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, ARGS.hidden_dim)
        self.linear2 = nn.Linear(ARGS.hidden_dim, vocab_size)

    def forward(self, inputs):
        # (묶음, 문맥) -> (묶음, 문맥 * 묻힘) 으로 편다
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        return self.linear2(out)


def make_batch(device="cpu"):
    """모든 n-그램을 텐서 한 쌍으로 만든다. 자료가 작아 묶음을 나누지 않는다."""
    contexts = torch.tensor(
        [[word_to_ix[w] for w in ctx] for ctx, _ in ngrams], dtype=torch.long, device=device
    )
    targets = torch.tensor(
        [word_to_ix[tgt] for _, tgt in ngrams], dtype=torch.long, device=device
    )
    return contexts, targets


def train(model, loss_fn, optimizer, epochs=None, verbose=True, device="cpu"):
    """모델을 익히고 에포크마다의 손실을 돌려준다.

    loss_fn은 nn.CrossEntropyLoss()처럼 부를 수 있는 것이면 무엇이든 좋다.
    F.cross_entropy를 그대로 넘겨도 된다.
    """
    epochs = epochs or ARGS.epochs
    contexts, targets = make_batch(device)
    model.to(device).train()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(contexts)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose:
            print(f"  epoch {epoch + 1:>3}/{epochs}  loss {loss.item():.4f}")
    return losses


if __name__ == "__main__":
    torch.manual_seed(ARGS.seed)
    model = NGramLanguageModeler()
    optimizer = torch.optim.SGD(model.parameters(), lr=ARGS.lr)
    losses = train(model, nn.CrossEntropyLoss(), optimizer, epochs=5)
    print(f"vocab {ARGS.vocab_size}, n-grams {len(ngrams)}, last loss {losses[-1]:.4f}")
