"""
Chapter 29.4.3: Jumping Knowledge Networks
"""
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx

class JKNet(nn.Module):
    """Jumping Knowledge Network with configurable aggregation."""
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=4, jk_mode='cat'):
        super().__init__()
        self.jk_mode = jk_mode
        self.input_lin = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList([nn.Linear(hidden_ch, hidden_ch) for _ in range(num_layers)])
        if jk_mode == 'cat':
            self.out_lin = nn.Linear(hidden_ch * num_layers, out_ch)
        elif jk_mode == 'lstm':
            self.lstm = nn.LSTM(hidden_ch, hidden_ch, batch_first=True)
            self.att_lin = nn.Linear(hidden_ch, 1)
            self.out_lin = nn.Linear(hidden_ch, out_ch)
        else:
            self.out_lin = nn.Linear(hidden_ch, out_ch)

    def forward(self, x, edge_index):
        n = x.shape[0]; x = F.relu(self.input_lin(x))
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a]*deg[dst_a]).pow(-0.5); norm[norm==float('inf')]=0

        layer_outputs = []
        for lin in self.convs:
            h = lin(x); msg = h[src_a]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
            layer_outputs.append(x)

        if self.jk_mode == 'cat':
            h = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == 'max':
            h = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.jk_mode == 'lstm':
            stacked = torch.stack(layer_outputs, dim=1)  # [n, L, d]
            lstm_out, _ = self.lstm(stacked)
            att = torch.softmax(self.att_lin(lstm_out).squeeze(-1), dim=-1)
            h = (lstm_out * att.unsqueeze(-1)).sum(dim=1)
        return self.out_lin(h)

def demo_jk():
    print("=" * 60); print("Jumping Knowledge Networks"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)])
    tm = torch.zeros(n, dtype=torch.bool); tm[::2] = True

    for mode in ['cat', 'max', 'lstm']:
        torch.manual_seed(42)
        model = JKNet(n, 16, 2, num_layers=4, jk_mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(200):
            opt.zero_grad(); F.cross_entropy(model(x, ei)[tm], y[tm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, ei).argmax(1)[~tm] == y[~tm]).float().mean()
        print(f"  JK-{mode:5s}: Test Acc = {acc:.4f}")

if __name__ == "__main__":
    demo_jk()
