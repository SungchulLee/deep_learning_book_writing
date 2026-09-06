# 건너뛰는 앎 신경망

건너뛰는 앎(JK) 신경망은 여느 그래프 신경망의 근본 한계를 다룬다. 마디마다 필요한 이웃 자리의 크기가 다를 수 있다는 것이다. 마지막 층의 나타냄만 쓰는 대신 건너뛰는 앎 신경망은 중간 층의 나타냄을 모두 모아 마디마다 가장 좋은 받아들이는 자리를 스스로 고르게 한다. 이 얼개는 그 자리의 얼개가 들쭉날쭉한 그래프에서 특히 잘 듣는다.

## 코드

```python
"""
29.4.3: 뛰어넘는 앎 신경망
"""
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx

# ========================================================================
# 메인
# ========================================================================

class JKNet(nn.Module):
    """모으기를 고를 수 있는 뛰어넘는 앎 신경망."""
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
```

## 논의

여느 $L$층 그래프 신경망에서 마디마다의 마지막 나타냄은 꼭 그 $L$뜀 이웃을 비춘다. 이 붙박인 받아들이는 자리는 문제가 된다. 그래프의 자리마다 마디에 이로운 앎의 잣수가 다를 수 있기 때문이다. 빽빽이 뭉친 무리 안의 마디는 1~2뜀이면 되지만 무리를 잇는 다리 마디는 더 먼 곳의 앎이 필요할 수 있다. 건너뛰는 앎 신경망은 층마다 나타냄을 모아 마지막에 합쳐 이를 푼다.

모으기 방식 셋이 저마다 다른 맞바꿈을 준다. 이어 붙이기(`cat`)는 층의 앎을 모두 지키지만 매개변수 개수가 깊이에 선형으로 늘어난다. 최대 모으기(`max`)는 층에 걸쳐 차원마다 가장 도드라진 특징을 고르며, 층마다 서로 보완하는 무늬를 담을 때 잘 듣는 매개변수 없는 방식이다. 장단기 기억망 눈길 방식(`lstm`)이 가장 너그러우며 마디마다 달라질 수 있는 눈길 얼개로 층 나타냄의 무게 있는 결합을 배운다.

가라테 클럽 그래프의 실험 결과는 세 방식 모두 그럴듯한 정확도를 이루지만 서로 견준 성능은 그래프 얼개에 따라 달라짐을 보인다. 그 자리의 위상이 더 들쭉날쭉한 그래프에서는 마디마다의 층 무게를 배울 수 있어 장단기 기억망 방식이 보통 뛰어나다. 건너뛰는 앎 틀은 그래프 신경망 층을 어떻게 고르든 상관없이 어떤 쪽지 건네기 얼개와도 합칠 수 있어 더 깊고 잘 듣는 그래프 신경망을 세우는 두루 쓰는 도구가 된다.

## 연습문제

**연습문제 1.**
보여 주기를 돌려 가라테 클럽 그래프에서 건너뛰는 앎의 세 방식(cat, max, lstm)을 견주어라. 어느 방식이 시험 정확도가 가장 높은가? `hidden_ch=16`이고 `num_layers=4`일 때 방식마다 매개변수 개수를 셈하라.

??? success "연습문제 1 풀이"
    매개변수 개수가 크게 다르다. `cat` 방식: 내놓기 선형 층의 들임 특징이 $16 \times 4 = 64$개이므로 내놓기 층만 $64 \times 2 + 2 = 130$개이다. `max` 방식: 내놓기 층이 $16 \times 2 + 2 = 34$개이다. `lstm` 방식: 장단기 기억망이 $4 \times (16 + 16 + 1) \times 16 = 2112$개를 더하고 눈길 층($16 + 1 = 17$)과 내놓기 층($34$)이 붙는다. 성능은 씨앗에 따라 달라지지만 나타냄이 풍성해 보통 `cat`과 `lstm`이 가장 좋고 `max`은 매개변수를 아낀다.

---

**연습문제 2.**
최대 모으기를 쓴 건너뛰는 앎 신경망이 층의 차례에 대해 불변인 까닭을 밝혀라. 최대 함수의 어떤 성질이 이를 가능하게 하며 이것이 늘 바람직한가?

??? success "연습문제 2 풀이"
    최대 모으기는 층에 걸쳐 원소마다 $h_i = \max_{l=1}^{L} h_i^{(l)}$을 셈한다. 최대는 자리 바꿈과 묶음 바꿈이 되는 연산이므로 층 차례의 자리바꿈에 불변이다. 층 사이에 자연스러운 중요도 차례가 없을 때는 바람직하지만, 그 자리에서 온 자리로 나아가는 흐름(1층에서 $L$층으로)이 뜻있는 얼개를 지닐 때는 한계가 된다. 그런 경우 차례에 민감한 장단기 기억망 모으기가 더 풍성한 무늬를 담을 수 있다.

---

**연습문제 3.**
층마다 내놓기에 배울 수 있는 낱값 무게 $\alpha_l$(소프트맥스로 고르게 맞춘다)을 곱하고 마지막 나타냄을 $h = \sum_l \alpha_l h^{(l)}$으로 하는 "무게 있는 합" 건너뛰는 앎 방식을 짜라. 이를 기존 방식과 견주어라.

??? success "연습문제 3 풀이"
    ```python
    # JKNet.__init__에 더한다:
    elif jk_mode == 'weighted':
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.out_lin = nn.Linear(hidden_ch, out_ch)

    # JKNet.forward에 더한다:
    elif self.jk_mode == 'weighted':
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(layer_outputs, dim=0)  # [L, n, d]
        h = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # [n, d]
    ```
    이 방식은 (마디마다가 아니라) 온 자리의 층 무게를 배운다. 장단기 기억망 눈길보다 단순하지만 최대 모으기보다 나타냄 힘이 크다. 보통 최대와 장단기 기억망 사이의 성능을 내며 모델의 복잡함과 나타냄 힘의 맞바꿈이 좋다.
