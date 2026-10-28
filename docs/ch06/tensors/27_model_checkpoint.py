"""튜토리얼 27: 되짚음 저장 — 무엇이 저장되고 무엇이 저장되지 않는가."""
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

torch.manual_seed(0)
TMP = Path(tempfile.mkdtemp())


def header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def make_model():
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))


def main():
    header("1. state_dict 에 들어 있는 것")
    m = make_model()
    m(torch.randn(8, 4))                      # 배치 정규화 통계를 움직여 둔다
    for k, v in m.state_dict().items():
        print(f"  {k:22s} {tuple(v.shape)}")
    print("웨이트와 바이어스, 그리고 **버퍼**(running_mean 따위)가 들어간다.")

    header("2. 들어 있지 않은 것 — 모드")
    m.eval()
    print(f"저장하기 직전 m.training = {m.training}")
    torch.save(m.state_dict(), TMP / "sd.pt")
    print(f"state_dict 에 'training' 이라는 열쇠가 있는가: "
          f"{any('training' in k for k in m.state_dict())}")

    m2 = make_model()
    m2.load_state_dict(torch.load(TMP / "sd.pt", weights_only=True))
    print(f"불러온 뒤 m2.training = {m2.training}")
    print("eval 로 두고 저장해도 불러오면 train 이다. 모드는 저장되지 않는다.")
    print(f"버퍼는 따라왔는가: "
          f"{torch.equal(m.state_dict()['1.running_mean'], m2.state_dict()['1.running_mean'])}")

    header("3. 학습을 이어 가려면 네 가지를 함께 저장한다")
    model = nn.Linear(4, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):                        # Adam 의 모멘텀을 쌓아 둔다
        opt.zero_grad()
        nn.MSELoss()(model(torch.randn(8, 4)), torch.randn(8, 1)).backward()
        opt.step()

    ckpt = {"epoch": 3,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "loss": 0.1234}
    torch.save(ckpt, TMP / "ck.pt")
    print(f"저장한 열쇠: {list(ckpt)}")
    exp_avg = opt.state_dict()["state"][0]["exp_avg"]
    print(f"옵티마이저가 들고 있는 상태: step={opt.state_dict()['state'][0]['step'].item():.0f}, "
          f"exp_avg 노름={exp_avg.norm():.4f}")
    print("옵티마이저를 빼고 저장하면 Adam 의 모멘텀이 0 부터 다시 쌓인다.")

    header("4. 옵티마이저를 빼먹으면 생기는 일")
    def first_step_after_resume(restore_optim):
        torch.manual_seed(1)
        m = nn.Linear(4, 1)
        o = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(3):
            o.zero_grad()
            nn.MSELoss()(m(torch.randn(8, 4)), torch.randn(8, 1)).backward()
            o.step()
        saved_m, saved_o = m.state_dict(), o.state_dict()

        m3 = nn.Linear(4, 1); m3.load_state_dict(saved_m)
        o3 = torch.optim.Adam(m3.parameters(), lr=1e-3)
        if restore_optim:
            o3.load_state_dict(saved_o)
        before = m3.weight.detach().clone()
        torch.manual_seed(99)
        o3.zero_grad()
        nn.MSELoss()(m3(torch.randn(8, 4)), torch.randn(8, 1)).backward()
        o3.step()
        return (m3.weight.detach() - before).norm().item()

    a = first_step_after_resume(True)
    b = first_step_after_resume(False)
    print(f"옵티마이저까지 되살린 첫 걸음의 크기 {a:.6f}")
    print(f"모델만 되살린 첫 걸음의 크기       {b:.6f}   {b/a:.1f}배")
    print("Adam 은 처음 몇 걸음을 보정하므로, 상태를 잃으면 첫 걸음이 커진다.")

    header("5. 통째로 저장하면 클래스에 묶인다")
    print("torch.save(model) 은 클래스를 가리키는 이름을 함께 절인다.")
    print("불러오는 쪽에 그 클래스가 같은 경로로 있어야 하고, 없으면")
    print("  AttributeError: Can't get attribute 'MyNet' on <module '__main__'>")
    print("이 난다. state_dict 는 숫자만 담으므로 그 묶임이 없다.")

    header("6. 저장한 뒤에는 되읽어 맞추어 본다")
    m4 = make_model()
    m4.load_state_dict(torch.load(TMP / "sd.pt", weights_only=True))
    same = all(torch.equal(a, b) for a, b in
               zip(m.state_dict().values(), m4.state_dict().values()))
    print(f"모든 텐서가 같은가: {same}")
    print("weights_only=True 로 읽는다 — 숫자만 받아들이므로 절인 코드가 돌지 않는다.")


if __name__ == "__main__":
    main()
