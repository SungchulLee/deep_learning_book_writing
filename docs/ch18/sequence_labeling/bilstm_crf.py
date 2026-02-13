"""
BiLSTM-CRF for Named Entity Recognition
=========================================

Implements a Bidirectional LSTM with a Conditional Random Field (CRF) layer
for sequence labeling (NER with BIO tagging).

Architecture:
    word indices -> Embedding
                 -> BiLSTM (bidirectional)
                 -> Linear (emission scores)
                 -> CRF layer (transition scores + Viterbi decoding)

Key components:
    - Forward algorithm:  computes partition function Z(x) in log-space
    - Viterbi algorithm:  finds best tag sequence at inference
    - Loss:              -log P(y|x) = forward_score - gold_score

Training data:
    "the wall street journal reported today that apple corporation made money"
      B   I    I      I       O       O    O    B      I           O    O

    "georgia tech is a university in georgia"
      B       I   O  O    O      O    B

Source: https://github.com/pytorch/tutorials/blob/main/beginner_source/nlp/advanced_tutorial.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"


# =============================================================================
# Helper functions
# =============================================================================
def argmax(vec: torch.Tensor) -> int:
    """Return argmax as a Python int."""
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq: list[str], to_ix: dict[str, int]) -> torch.Tensor:
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)


def log_sum_exp(vec: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-sum-exp for 1×n vector."""
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size(1))
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# =============================================================================
# Model
# =============================================================================
class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_to_ix: dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Transition parameters: transitions[i, j] = score of j -> i
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # Never transition *to* START or *from* STOP
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self._init_hidden()

    def _init_hidden(self):
        return (
            torch.randn(2, 1, self.hidden_dim // 2),
            torch.randn(2, 1, self.hidden_dim // 2),
        )

    # ----- BiLSTM emission features -----
    def _get_lstm_features(self, sentence: torch.Tensor) -> torch.Tensor:
        self.hidden = self._init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # ----- Forward algorithm (partition function) -----
    def _forward_alg(self, feats: torch.Tensor) -> torch.Tensor:
        init_alphas = torch.full((1, self.tagset_size), -10000.0)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.0
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    # ----- Score of a specific tag sequence -----
    def _score_sentence(self, feats: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags]
        )
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # ----- Viterbi decoding -----
    def _viterbi_decode(self, feats: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.0)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Trace back
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    # ----- Loss: negative log-likelihood -----
    def neg_log_likelihood(
        self, sentence: torch.Tensor, tags: torch.Tensor
    ) -> torch.Tensor:
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    # ----- Inference -----
    def forward(self, sentence: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# =============================================================================
# Data
# =============================================================================
training_data = [
    (
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split(),
    ),
    (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split(),
    ),
]

# Build vocabularies
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}


# =============================================================================
# Configuration
# =============================================================================
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
EPOCHS = 300
LR = 0.01
WEIGHT_DECAY = 1e-4


# =============================================================================
# Training
# =============================================================================
def train(model, optimizer):
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}  Loss: {total_loss:.4f}")


# =============================================================================
# Main
# =============================================================================
def main():
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Before training
    print("=== Before training ===")
    with torch.no_grad():
        for sent, gold in training_data:
            sent_in = prepare_sequence(sent, word_to_ix)
            score, pred_ix = model(sent_in)
            pred_tags = [ix_to_tag[i] for i in pred_ix]
            print(f"  {' '.join(sent)}")
            print(f"    Gold: {gold}")
            print(f"    Pred: {pred_tags}  (score={score.item():.4f})")

    # Train
    print("\n=== Training ===")
    train(model, optimizer)

    # After training
    print("\n=== After training ===")
    with torch.no_grad():
        for sent, gold in training_data:
            sent_in = prepare_sequence(sent, word_to_ix)
            score, pred_ix = model(sent_in)
            pred_tags = [ix_to_tag[i] for i in pred_ix]
            print(f"  {' '.join(sent)}")
            print(f"    Gold: {gold}")
            print(f"    Pred: {pred_tags}  (score={score.item():.4f})")


if __name__ == "__main__":
    main()
