# MAE 비전 트랜스포머

MAE(마스크 오토인코더)는 일부 패치만 인코딩하고 디코더에서 가려진 픽셀을 복원하는 자기 지도 학습 방식이다. 대표 마스크 비율 75%는 인코더 토큰 수를 줄이지만 전체 학습 속도가 일정 배율로 빨라진다는 뜻은 아니다. 대조 학습과 다른 목표의 선택지다.

## 1. 코드

??? note "코드 (388줄)"

    ```python
    """
    MAE: 가린 자동 인코더는 잘 커지는 시각 학습기이다
    그림의 자기 지도 학습을 위한 가린 자동 인코더 구현.
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
        비전 트랜스포머 등뼈를 쓰는 가린 자동 인코더

        인수:
            img_size: 입력 그림 크기
            patch_size: 조각 크기
            in_chans: 입력 통로의 수
            embed_dim: 인코더의 임베딩 차원
            depth: 인코더의 깊이
            num_heads: 어텐션 머리의 수
            decoder_embed_dim: 디코더의 임베딩 차원
            decoder_depth: 디코더의 깊이
            decoder_num_heads: 디코더의 어텐션 머리 수
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

            # 인코더
            self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

            self.encoder_blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
                for _ in range(depth)
            ])
            self.encoder_norm = nn.LayerNorm(embed_dim)

            # 디코더
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
            """인코더의 앞먹임"""
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
            """디코더의 앞먹임"""
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


**출력:**

```
MAE Model initialized successfully!
Image size: 224x224
Patch size: 16x16
Number of patches: 196
Mask ratio: 0.75
Encoder depth: 12 blocks
Decoder depth: 8 blocks

Test forward pass successful!
Loss: 1.3435
Prediction shape: torch.Size([4, 196, 768])
Mask shape: torch.Size([4, 196])
Number of masked patches: 147 / 196
```

## 2. 논의

224×224 RGB, 패치 16에서 패치는 196개다. 인코더는 위치를 더한 뒤 무작위로 49개를 남기고 CLS를 붙여 50토큰을 처리한다. 디코더는 보이는 특징을 다른 폭으로 투영하고 mask_token을 끼워 원래 순서를 복원한다. 전체 패치에 대한 예측은 (B,196,768), mask는 (B,196)이며 1이 손실 대상으로 가린 위치다.

손실은 패치별로 정규화한 목표와 예측의 제곱 오차를 가린 위치에서만 평균한다. 이 코드는 if True로 정규화를 항상 적용한다. 모델 출력이 정규화된 패치 값이므로 unpatchify만 적용한 pred_img는 원래 픽셀 스케일의 복원이 아니다. 입력 정규화와 패치 평균·분산을 어떻게 되돌릴지 시각화의 해석을 정해야 한다. 여기서는 그림을 재생성하지 않는다.

이 구현은 위치 표를 학습 가능한 Parameter로 초기화해 원래 MAE의 고정 사인·코사인 위치 표현과 다르다. img_size에 맞는 고정 길이를 가정하고 patchify·unpatchify는 채널 3을 하드코딩하므로 생성자의 in_chans를 다른 값으로 바꾸는 것만으로 전체 모델이 지원하지 않는다. 정사각·패치 배수 입력과 헤드별 특징 차원도 확인한다.

0≤mask_ratio≤1 범위와 실제 정수 반올림 결과를 확인한다. 마스크가 하나도 없으면 mask.sum=0으로 나누어 손실이 NaN이 된다. 전체 특징 추출은 별도 encode_full을 사용하는 편이 적절하다. eval에서도 random_masking은 새 난수를 뽑으므로 같은 입력의 마스크·복원 결과가 매번 달라질 수 있다.

75%는 원 연구의 효과적인 설정이지만 데이터·패치·목표에 따라 달라진다. BERT의 대표 예측 대상은 15%이며 25%가 아니다. 높은 마스킹이 자동으로 물체의 의미·공간 구조를 완전히 이해하게 한다거나 MAE가 자기 지도 학습의 다른 방법을 대체했다고 단정할 수 없다. main은 독립 난수 이미지의 순전파로 실제 표현 학습을 수행하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
196패치의 75% 마스킹에서 인코더의 토큰 수와 점수 비용 비율을 계산하고 전체 속도와 구분하라.

</div>

??? success "연습문제 1 풀이"
    보이는 패치 49개와 CLS로 50토큰이며 전부 처리하면 197토큰이다. 점수·가중합의 토큰 제곱 항은 197²/50²≈15.52배 차이다. 투영·MLP는 대략 토큰 수에 선형이고 전체 디코더·패치 투영·데이터 처리도 있어 전체 FLOP·메모리·실행 시간이 같은 비율로 줄지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
패치별 목표 정규화의 역할과 원시 픽셀 복원과의 차이를 설명하라.

</div>

??? success "연습문제 2 풀이"
    패치 평균·스케일의 영향을 줄이고 상대 픽셀 변화에 맞춘 손실을 만든다. 원시 목표와 다른 정보·오차를 강조하며 더 좋은 표현을 보장하지는 않는다. 정규화한 예측을 이미지로 표시하려면 픽셀 공간의 의미와 역변환 규칙을 정해야 한다. 코드에는 목표 정규화를 끄는 생성자 옵션이 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 encode_full과 mask_ratio=0인 일반 forward의 차이를 설명하라.

</div>

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

    encode_full은 전체 패치와 CLS를 원래 순서로 인코딩해 (B,197,E) 특징을 반환하며 복원 손실을 계산하지 않는다. 반면 일반 forward(mask_ratio=0)은 무작위 순서를 사용하고 가린 위치 수가 0이라 손실 분모도 0이 된다. CLS·패치 특징은 새 과제의 헤드와 평가가 필요하며 전체 입력을 본다고 더 좋은 예측이 자동 보장되지는 않는다. 이 함수 조각은 MAE 메서드로 연결해야 한다.

## 정리하며

MAE의 보이는 토큰 인코더, 순서를 복원하는 디코더와 가린 위치 손실을 연결한다. 학습 가능한 위치 표·RGB 고정·빈 마스크 분모와 정규화된 복원 이미지의 해석을 확인하고, 인코더 점수 비용과 전체 학습 속도를 구분한다.
