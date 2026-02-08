"""Chapter 29.7.2: Drug Discovery - Drug-target interaction prediction."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class DrugTargetPredictor(nn.Module):
    def __init__(self, drug_feat=10, target_feat=8, hidden=32):
        super().__init__()
        self.drug_enc = nn.Sequential(nn.Linear(drug_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.target_enc = nn.Sequential(nn.Linear(target_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.predictor = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, drug_graph, target_feat):
        x, ei = drug_graph
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n); sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        h = self.drug_enc[0](x)
        out = torch.zeros(n, h.shape[1]); out.scatter_add_(0, da.unsqueeze(1).expand_as(h[sa]), h[sa])
        drug_emb = F.relu(self.drug_enc[2](F.relu(out))).mean(dim=0, keepdim=True)
        target_emb = self.target_enc(target_feat.unsqueeze(0))
        return torch.sigmoid(self.predictor(torch.cat([drug_emb, target_emb], dim=-1)))

def demo():
    print("=" * 60); print("Drug-Target Interaction Prediction"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_drugs, n_targets = 50, 10
    drugs = [(torch.randn(np.random.randint(5,12), 10),
              torch.tensor([[i,i+1,i+1,i] for i in range(np.random.randint(4,11))], dtype=torch.long).T.contiguous()
              if np.random.randint(4,11) > 0 else torch.tensor([[0,1],[1,0]], dtype=torch.long))
             for _ in range(n_drugs)]
    # Fix edge indices
    drugs_clean = []
    for x, _ in drugs:
        n = x.shape[0]; edges = [(i,(i+1)%n) for i in range(n-1)]
        src = [e[0] for e in edges]+[e[1] for e in edges]
        dst = [e[1] for e in edges]+[e[0] for e in edges]
        drugs_clean.append((x, torch.tensor([src, dst], dtype=torch.long)))
    targets = [torch.randn(8) for _ in range(n_targets)]
    interactions = [(np.random.randint(n_drugs), np.random.randint(n_targets), float(np.random.random()>0.5)) for _ in range(200)]

    model = DrugTargetPredictor(); opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(30):
        total_loss = 0
        for d_idx, t_idx, label in interactions[:160]:
            opt.zero_grad()
            pred = model(drugs_clean[d_idx], targets[t_idx])
            loss = F.binary_cross_entropy(pred.squeeze(), torch.tensor(label))
            loss.backward(); opt.step(); total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {total_loss/160:.4f}")

if __name__ == "__main__":
    demo()
