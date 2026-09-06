# MAE 비전 트랜스포머

가린 자동 부호기(MAE)는 자기 지도 시각 학습의 큰 전환으로, 대조 목표에서 가린 이미지 모형화로 옮겨 간다. BERT 같은 가린 언어 모형에서 영감을 얻어 MAE는 그림 조각의 큰 몫(대개 75%)을 무작위로 가리고 빠진 화소를 되살리도록 비전 트랜스포머를 학습시킨다. 이 방식은 놀랍도록 간단하고 잘 커지며 계산이 효율적이다. 부호기가 보이는 조각만 처리하여 그림 전체를 처리할 때보다 계산이 대략 4배 준다.

## 코드

```python
"""
MAE: 가린 자동 부호기는 잘 커지는 시각 학습기이다
그림의 자기 지도 학습을 위한 가린 자동 부호기 구현.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class PatchEmbed(nn.Module):
    """
    그림에서 조각 임베딩으로
    그림을 조각 임베딩의 수열로 바꾼다
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """다중 머리 자기 어텐션"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """순전파 신경망"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """트랜스포머 블록"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(nn.Module):
    """
    비전 트랜스포머 등뼈를 쓰는 가린 자동 부호기

    인수:
        img_size: 입력 그림 크기
        patch_size: 조각 크기
        in_chans: 입력 통로의 수
        embed_dim: 부호기의 임베딩 차원
        depth: 부호기의 깊이
        num_heads: 어텐션 머리의 수
        decoder_embed_dim: 복호기의 임베딩 차원
        decoder_depth: 복호기의 깊이
        decoder_num_heads: 복호기의 어텐션 머리 수
        mlp_ratio: 임베딩 차원에 대한 다층 퍼셉트론 숨은 차원의 비
        mask_ratio: 가릴 조각의 비율
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        mask_ratio=0.75
    ):
        super().__init__()

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 부호기
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 복호기
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # 예측 머리
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        # 위치 임베딩을 시작한다
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # patch_embed를 nn.Linear처럼 초기화한다
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # cls_token과 mask_token을 초기화한다
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        MAE 논문을 따르는 무작위 가리기

        인수:
            x: [N, L, D], 수열
            mask_ratio: 가릴 조각의 비율

        반환값:
            x_masked: [N, L_kept, D], 보이는 조각
            mask: [N, L], 0은 남기기, 1은 없애기
            ids_restore: [N, L], 본디 순서를 되살릴 색인
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # 무작위 잡음을 만든다
        noise = torch.rand(N, L, device=x.device)

        # 표본마다 잡음을 정렬한다
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 앞쪽 부분만 남긴다
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 이진 가림을 만든다: 0은 남기기, 1은 없애기
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """부호기의 앞먹임"""
        # 조각을 임베딩한다
        x = self.patch_embed(x)

        # 위치 임베딩을 더한다 (cls 토큰은 빼고)
        x = x + self.pos_embed[:, 1:, :]

        # 가리기
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # cls 토큰을 덧붙인다
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 트랜스포머 블록을 적용한다
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """복호기의 앞먹임"""
        # 토큰 임베딩
        x = self.decoder_embed(x)

        # 수열에 가림 토큰을 덧붙인다
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # cls 토큰 없음
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # 뒤섞은 것을 되돌린다
        x = torch.cat([x[:, :1, :], x_], dim=1)  # cls 토큰을 덧붙인다

        # 위치 임베딩을 더한다
        x = x + self.decoder_pos_embed

        # 트랜스포머 블록을 적용한다
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 예측기 사영
        x = self.decoder_pred(x)

        # cls 토큰을 없앤다
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        되살리기 손실을 셈한다

        인수:
            imgs: [N, 3, H, W]
            pred: [N, L, p*p*3]
            mask: [N, L], 0은 남기기, 1은 없애기
        """
        target = self.patchify(imgs)

        # 목표를 조각마다 정규화한다
        if True:  # 조각마다 정규화한다
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], 조각마다의 평균 손실

        loss = (loss * mask).sum() / mask.sum()  # 없앤 조각에 대한 평균 손실
        return loss

    def patchify(self, imgs):
        """
        그림을 조각으로 바꾼다
        imgs: [N, 3, H, W]
        x: [N, L, patch_size**2 *3]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        조각을 도로 그림으로 바꾼다
        x: [N, L, patch_size**2 *3]
        imgs: [N, 3, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, imgs, mask_ratio=None):
        """
        앞먹임

        인수:
            imgs: [N, 3, H, W]
            mask_ratio: 가릴 조각의 비율
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


def visualize_reconstruction(model, img, device):
    """
    가린 그림과 되살린 그림을 그려 본다
    """
    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        loss, pred, mask = model(img)

        # 조각을 되돌려 그림으로 만든다
        pred_img = model.unpatchify(pred)

        # 본디 그림에 가림을 적용한다
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
        mask = model.unpatchify(mask)

        masked_img = img * (1 - mask)

        return img, masked_img, pred_img, loss.item()


if __name__ == "__main__":
    # 사용 예
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MAE를 시작한다
    model = MAE(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mask_ratio=0.75
    ).to(device)

    print("MAE Model initialized successfully!")
    print(f"Image size: 224x224")
    print(f"Patch size: 16x16")
    print(f"Number of patches: {model.patch_embed.num_patches}")
    print(f"Mask ratio: {model.mask_ratio}")
    print(f"Encoder depth: 12 blocks")
    print(f"Decoder depth: 8 blocks")

    # 순전파 시험
    dummy_img = torch.randn(4, 3, 224, 224).to(device)
    loss, pred, mask = model(dummy_img)

    print(f"\nTest forward pass successful!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of masked patches: {mask.sum(dim=1).mean().item():.0f} / {model.patch_embed.num_patches}")
```

## 논의

MAE의 구조는 비대칭 부호기-복호기 설계이다. **부호기**는 보이는(가리지 않은) 조각에서만 도는 표준 비전 트랜스포머로, 가림 비율이 75%이면 조각이 $16 \times 16$인 $224 \times 224$ 그림에서 196개 가운데 49개만 처리한다는 뜻이다. 이것이 핵심 계산 통찰이다. 가린 조각을 무거운 부호기에 넣지 않아 MAE는 모든 조각을 처리할 때보다 학습이 약 $3$배 빨라지고 기억도 크게 아낀다. 보이는 조각은 부호기에 들어가기 전에 위치 임베딩을 받으므로 공간 정보가 지켜진다.

**복호기**는 일부러 가볍게 두고(이를테면 부호기의 12개에 견주어 블록 8개), 토큰 전체, 곧 부호화된 보이는 조각에 가린 자리의 자리 지킴이 노릇을 하는 학습되는 가림 토큰을 더한 것에서 돈다. 복호기가 할 일은 가린 조각의 화소 값을 되살리는 것이며, 가림 토큰에 제 공간 위치를 알려 주려고 제 나름의 위치 임베딩 묶음을 쓴다. 사전 학습 뒤에 복호기는 버리고 부호기만 아래쪽 과제에 쓴다. 되살릴 목표를 조각마다 정규화하면 모형이 쉽게 맞힐 수 있는 평균 화소 값에 휘둘리지 않고 국소 짜임에 집중하게 된다.

75%라는 가림 비율은 놀랍도록 높고 MAE가 잘 통하는 데 매우 중요하다. 가림 비율이 낮으면(이를테면 글에 쓰는 BERT처럼 25%이면) 과제가 너무 쉬워진다. 모형이 넉넉한 뜻 표현을 배우지 않고 가까운 조각에서 사이를 채우는 것만으로 풀 수 있다. 75%에서는 남은 조각이 충분히 성겨서 모형이 잘하려면 물체와 결과 공간 관계를 참으로 이해해야 한다. 이 높은 가림 비율은 또한 과제를 가장 참된 뜻에서 자기 지도로 만든다. 그림의 대부분이 지도 신호 노릇을 하고 모형은 시각 장면을 통째로 헤아려야 한다.

## 연습문제

**연습문제 1.**
`img_size=224`, `patch_size=16`, `embed_dim=768`, `depth=12`, `mask_ratio=0.75`인 MAE 모형에서 (가) 전체 조각 수, (나) 부호기가 처리하는 보이는 조각 수, (다) (자기 어텐션이 지배하며 $O(N^2)$으로 는다고 할 때) 모든 조각을 처리할 때에 견준 FLOP 절약을 셈하라.

??? success "연습문제 1 풀이"
    (가) 전체 조각: $(224 / 16)^2 = 14 \times 14 = 196$개.

    (나) 보이는 조각: $196 \times (1 - 0.75) = 49$개 (CLS 토큰 1개를 더해 토큰 50개).

    (다) 자기 어텐션의 FLOP은 수열 길이를 $N$이라 할 때 $O(N^2)$으로 는다. 가리지 않으면 부호기가 토큰 $N = 197$개(조각 196개와 CLS)를 처리한다. 가리면 토큰 $N = 50$개를 처리한다.

    FLOP 비: $(197^2) / (50^2) = 38{,}809 / 2{,}500 \approx 15.5$.

    따라서 75% 가림에서 부호기의 어텐션 계산이 대략 $15.5$배 싸다. (어텐션과 다층 퍼셉트론을 아우른) 트랜스포머 전체로 보면 다층 퍼셉트론 층이 $N$에 대해 제곱이 아니라 일차로 늘므로 전체 속도 향상은 약 $3$~$4$배이다.

---

**연습문제 2.**
MAE가 되살릴 목표를 조각마다 정규화하는(조각의 평균을 빼고 표준편차로 나누는) 까닭을 설명하라. 모형이 대신 날화소 값을 맞히면 어떻게 되는가?

??? success "연습문제 2 풀이"
    조각마다 정규화하지 않으면 되살리기 손실이 조각마다의 평균 색을 맞히는 데 휘둘리는데, 이는 이웃 조각에서 사이 채우기만으로 알아낼 수 있는 낮은 진동수의 신호이다. 모형이 세밀한 시각 특징을 배우지 않고 평균 색만 맞혀도 손실을 낮출 수 있다.

    조각마다의 정규화는 이 쉽게 맞힐 수 있는 성분을 없애고 모형이 조각 안의 **국소 짜임**, 곧 모서리와 결과 기울기 같은 높은 진동수의 세부를 되살리는 데 집중하게 만든다. 이는 더 어렵고 모형이 더 넉넉한 속 표현을 갖추게 한다. 정규화된 목표는 절댓값보다 상대적인 화소 변화를 도드라지게 하여, 모형이 조각의 평균 모습만 맞히는 대신 무엇이 조각을 시각적으로 남다르게 하는지 이해하도록 민다.

---

**연습문제 3.**
가리지 않고 모든 조각을 처리하는 `encode_full` 메서드를 더하여 추론 때 가림 비율을 바꿀 수 있도록 `MAE` 클래스를 고쳐라. 같은 입력 그림에 대해 (가림을 쓰는) `forward_encoder`와 (가림을 쓰지 않는) `encode_full`에서 얻은 표현을 견주어라.

??? success "연습문제 3 풀이"
    ```python
    def encode_full(self, x):
        """가리지 않고 모든 조각을 부호화한다 (아래쪽 과제용)."""
        # 조각을 임베딩한다
        x = self.patch_embed(x)

        # 위치 임베딩을 더한다 (cls 토큰은 빼고)
        x = x + self.pos_embed[:, 1:, :]

        # cls 토큰을 덧붙인다
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 트랜스포머 블록을 적용한다
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x  # 꼴: [N, num_patches + 1, embed_dim]
    ```

    `encode_full` 메서드는 196 + 1개 토큰을 모두 부호기에 넣어 온전한 공간 표현을 준다. 가림을 쓰는 사전 학습 중에 부호기는 조각의 25%만 보므로 일부 정보에서 뜻있는 표현을 내는 법을 배운다. 추론 때 모든 조각을 처리하면 자기 어텐션이 모든 공간 자리에 주의할 수 있어 더 넉넉한 표현이 나온다. `encode_full`의 CLS 토큰은 대개 분류 과제에 쓰고, 조각 토큰 수열 전체는 분할 같은 빽빽한 예측 과제에 쓴다.
