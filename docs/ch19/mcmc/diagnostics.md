# MCMC 진단

MCMC 진단은 유한한 체인의 비정상성·탐색 문제와 관심량의 몬테카를로 오차를 살핀다. 어떤 단일 지표나 임계값도 목표 전체에 대한 수렴을 증명하지는 않는다.

## 1. 근본 문제

적절한 에르고드 조건 아래 표본 평균은 목표 기댓값에 수렴한다. 유한한 실행에서는 초기 상태의 영향, 모드 간 탐색, 특정 평균·분위수·꼬리확률의 정밀도를 따로 확인한다. 모든 체인이 같은 영역을 놓치면 체인 간 일치도 진단도 실패를 감지하지 못할 수 있다.

체인 길이를 늘리는 것이 도움이 되는 경우와 모형·기하·구현을 고쳐야 하는 경우를 구분한다. 수렴하지 않는 코드나 접근 불가능한 커널은 단순한 장기 실행으로 해결되지 않는다.

## 2. 체인 간 일치도: R-hat

### Split R-hat과 순위 정규화

각 체인을 반으로 나눠 체인 안·사이 변동을 비교한다. 길이 n의 분할 체인 m개에서 W를 체인 안 표본분산 평균, B를 n배한 체인 평균들의 표본분산이라 두면 다음 기본식을 얻는다.

$$\widehat V^+=\frac{n-1}{n}W+\frac Bn,\qquad \hat R=\sqrt{\widehat V^+/W}.$$

현대적인 개선은 순위 정규화와 중앙값에서 접은 값의 진단을 결합해 위치·스케일 차이에 더 민감하게 한다. 아래 split_rhat은 원자료 분산을 사용하는 기본 분할식이며 순위 정규화·folding은 구현하지 않는다. [개선된 R-hat 설명](https://mc-stan.org/rstan/reference/Rhat.html)을 참고한다.

입력은 체인×추출×파라메터다. 홀수 길이면 마지막 추출을 버리고, 반쪽 길이가 2 미만이거나 체인 안 분산이 0이면 유효한 진단이 안 될 수 있다. 상수 체인에서 NaN이 나온다고 좋은 수렴으로 해석하지 않는다. 1.01보다 큰 값은 조사할 신호이고 작은 값은 수렴의 충분조건이 아니다.

```python
import torch

def split_rhat(chains: torch.Tensor) -> torch.Tensor:
    """
    모임을 살피려고 쪼갠 R-hat 셈하기.
    
    파라메터
    ----------
    chains : Tensor of shape (M, N, d)
        사슬 M개, 사슬마다 표본 N개, 파라메터 d개
    
    반환값
    -------
    rhat : Tensor of shape (d,)
    """
    M, N, d = chains.shape
    
    # 사슬마다 반으로 쪼개기
    N_half = N // 2
    split_chains = torch.cat([
        chains[:, :N_half, :],
        chains[:, N_half:2*N_half, :]
    ], dim=0)  # (2M, N_half, d)
    
    m = split_chains.shape[0]  # 2M
    n = N_half
    
    # 사슬 안 흩어짐
    chain_vars = split_chains.var(dim=1)  # (2M, d)
    W = chain_vars.mean(dim=0)  # (d,)
    
    # 사슬 사이 흩어짐
    chain_means = split_chains.mean(dim=1)  # (2M, d)
    B = n * chain_means.var(dim=0)  # (d,)
    
    # 모아 어림한 흩어짐
    var_hat = (n - 1) / n * W + B / n
    
    rhat = torch.sqrt(var_hat / W)
    return rhat
```

## 3. 유효 표본 크기(ESS)

정상 함수열의 적절한 중심극한정리 아래 평균 분산은 Var_π(f)τ_f/N, τ_f=1+2∑ρ_f(k)로 근사한다. ESS_f=N/τ_f는 함수별 점근 분산에 연결된 값이다. 음의 상관이면 N보다 클 수 있으며 웨이트 ESS의 [1,N] 범위와 다르다.

bulk ESS는 순위 정규화된 값의 중심부 탐색을, tail ESS는 꼬리 분위수의 지시함수 변동을 진단한다. 원자료 평균·표준편차·임의 함수의 정확한 MCSE와 모두 같지는 않다. 필요한 ESS는 허용오차와 관심량에 따라 정해야 한다.

아래 effective_sample_size는 1차원 단일 체인의 FFT 자기상관을 인접한 양의 쌍까지 더하는 단순 구현이다. 다중 체인 결합·순위 정규화·초기 단조 수열 보정이 없고 일반적인 antithetic ESS>N을 충분히 반영하지 못한다. 상수열은 acf[0]=0으로 실패하고 max_lag가 N을 넘으면 인덱스 범위를 벗어날 수 있다. 이를 현대적인 bulk·tail ESS 전체 구현으로 소개하지 않는다.

```python
def effective_sample_size(samples: torch.Tensor, max_lag: int = None) -> float:
    """
    처음의 양수 늘어놓음 어림꼴로 ESS 어림하기.
    
    파라메터
    ----------
    samples : 꼴 (N,)의 텐서 — 사슬 하나, 파라메터 하나
    """
    N = len(samples)
    if max_lag is None:
        max_lag = N // 2
    
    # FFT로 자기상관 셈하기
    x = samples - samples.mean()
    fft_result = torch.fft.rfft(x, n=2 * N)
    acf = torch.fft.irfft(fft_result * fft_result.conj())[:N]
    acf = acf / acf[0]
    
    # 처음의 양수 늘어놓음으로 자기상관 더하기
    tau = 1.0
    for k in range(1, max_lag, 2):
        rho_pair = acf[k] + acf[k + 1] if k + 1 < max_lag else acf[k]
        if rho_pair < 0:
            break
        tau += 2 * rho_pair
    
    return N / tau
```

## 4. 트레이스·랭크·변수 쌍 그림

트레이스 플롯에서 장기 추세·서로 다른 체인 위치·정체 구간을 살핀다. 겉보기에 안정적인 띠라도 같은 모드 안에 갇혔을 수 있다. 자기상관은 관심 함수에 따라 평가하고, 특정 지연 20 안에 0이 되어야 한다는 보편적인 규칙은 없다.

랭크 플롯은 체인들을 합친 표본의 순위가 각 체인에 어떻게 분포하는지 비교한다. 체인 내부 순위를 각각 따로 매기면 위치 차이를 놓친다. 변수 쌍 그림과 발산 위치 표시는 상관·깔때기·분리 모드의 구조를 찾는 데 도움이 된다.

## 5. HMC와 NUTS의 추가 진단

발산은 HMC의 수치 적분이 큰 에너지 오차 등으로 어려움을 겪은 신호이며 NUTS에만 한정되지 않는다. 소수라는 이유로 무시하면 바이어스를 놓칠 수 있지만 하나의 발산이 모든 요약의 오류를 논리적으로 증명하는 것도 아니다. 위치·재현성·설정 변화와 결과 민감도를 조사한다.

목표 수락률을 높여 스텝 크기를 줄이거나 동등한 리파라메리제이션을 검토할 수 있다. 사전을 더 강하게 바꾸면 추론 목표도 바뀌므로 단순한 수치 설정 수정과 구분한다. 비중심화는 항상 더 좋은 방법이 아니다.

최대 트리 깊이 도달은 계산 제한으로 궤적이 끝났다는 효율 신호다. 발산과 같은 오류가 아니고 깊이를 무작정 높이면 비용이 커질 수 있다. E-BFMI는 연속 에너지 변화의 제곱평균을 에너지 분산으로 나눈 진단이며 낮은 값은 에너지 탐색 문제를 의심하게 한다. 0.3 등의 기준은 경고 휴리스틱이다. [HMC 진단 안내](https://mc-stan.org/docs/cmdstan-guide/diagnose_utility.html)를 참고한다.

## 6. 진단을 함께 사용하는 순서

아래 목록은 점검 순서의 예다. 여러 체인·R-hat·ESS·발산·그림을 함께 확인하되 모든 항목을 통과했다는 사실을 수렴 증명으로 해석하지 않는다. 문제가 생기면 실행 연장과 설정·리파라메리제이션·구현 검토 중 원인에 맞는 조치를 선택한다.

```
1. Run 4+ chains from dispersed starting points
2. Check split-R̂ < 1.01 for ALL parameters
3. Check bulk ESS > 400 and tail ESS > 400
4. Check 0 divergences (for NUTS/HMC)
5. Inspect trace plots for any pathology
6. If diagnostics fail:
   a. Reparameterize model
   b. Run chains longer
   c. Adjust sampler settings
   d. Simplify model if necessary
7. Only report results after all diagnostics pass
```

## 7. 시계열 검정과 MCSE

Geweke 진단은 체인의 앞·뒤 구간 평균 차이를 각 평균의 시계열 분산 추정으로 표준화한다. 일반적인 독립 표본 분산을 대입하면 자기상관을 무시한다. 정규 근사 아래 큰 |Z|는 차이를 의심하게 하지만 작은 값은 비정상성이 없다는 증명이 아니다.

Heidelberger–Welch 방식은 정상성 점검과 추정 평균의 몬테카를로 신뢰구간 반폭을 사용한다. 이 구간은 파라메터의 사후 신용구간이 아니며 평균이 0 근처이면 상대 반폭 기준이 불안정하다.

평균의 적절한 ESS_f가 있을 때 MCSE≈SD_π(f)/√ESS_f다. 사후 표준편차는 통계적 불확실성, MCSE는 유한한 시뮬레이션으로 기댓값을 계산한 수치 오차다. 분위수에는 해당 밀도와 지시함수의 상관을 고려한 별도 오차 추정이 필요하다.

## 8. 필요한 표본 크기

평균의 MCSE를 사후 표준편차의 r배 이하로 원하면 해당 평균 ESS가 대략 1/r² 이상이어야 한다. r=0.05이면 400, r=0.01이면 10,000이다. 이는 유효한 정상 CLT·분산 추정 아래의 근사이며 모드 누락을 진단하지 않는다.

꼬리확률이나 분위수의 요구량을 평균과 같은 고정 ESS 표로 결정하지 않는다. 보고할 각 관심량의 허용 절대·상대 오차와 MCSE를 명시하고 추정의 안정성을 확인한다.

## 9. Thinning과 저장 비용

추출을 일정 간격으로 버리는 thinning은 저장·후처리 비용을 줄일 때 유용할 수 있다. 그러나 같은 생성 비용에서 일반적으로 전체 추출을 보존하고 자기상관을 반영해 분석하는 편이 정보를 더 활용한다. 간격을 늘려도 완전한 독립성이 보장되지는 않는다.

Thinning으로 표본당 ESS가 높아져도 총 ESS나 계산시간당 정확도가 좋아졌다는 뜻은 아니다. 음의 상관 같은 구조에서는 오히려 효율을 크게 잃을 수 있다. 메모리 제약과 실제 비용을 기준으로 판단한다.

## 10. 문제에 맞는 조치

높은 R-hat은 체인 위치·스케일·시간 변화가 맞지 않는다는 신호일 수 있다. 낮은 ESS에는 자기상관, 나쁜 기하, 짧은 실행 등이 원인일 수 있다. 여러 모드에서는 모든 체인이 같은 모드를 놓쳤는지도 살핀다.

리파라메리제이션·블록 갱신·템퍼링·다른 제안 등을 원인에 맞춰 선택한다. HMC로 바꾸거나 더 오래 실행하면 자동으로 해결된다는 보장은 없다. 설정 변경 전후에는 동일한 관심량의 요약·MCSE·계산비용을 비교한다.

## 11. 워밍업과 초기 상태

워밍업은 초기 상태의 영향을 줄이고 설정을 적응하는 단계이며, 해당 적응 단계 표본은 보통 최종 고정 커널 요약에서 제외한다. 처음 50%를 버리는 규칙은 엄밀한 수렴 기준이 아니다.

필요한 길이는 모형과 시작점에 따라 다르며 정상성을 확정하는 순간을 진단만으로 정확히 알 수는 없다. 워밍업 후에도 여러 체인의 탐색과 함수별 오차를 확인한다.

## 12. 소프트웨어 도구

다음은 진단 라이브러리를 호출하는 짧은 조각이다. samples의 체인·추출 차원과 자료 컨테이너, 실제 설치한 인터페이스의 summary·diagnose 메서드를 확인해야 한다. stan.build만으로 샘플링이 수행되는 것은 아니며, 아래 서로 다른 인터페이스 관례가 섞인 조각을 완결된 실행 예제로 간주하지 않는다. ArviZ의 HDI와 등꼬리 신용구간도 구분한다.

```python
import arviz as az

# 두루 간추리기
az.summary(samples, hdi_prob=0.95)
# 담긴 것: mean, sd, hdi, ess_bulk, ess_tail, r_hat
```
```python
fit = stan.build(model, data=data)
fit.summary()  # R-hat과 ESS를 담음
fit.diagnose()  # 말썽에 대한 경고
```

## 13. 결과를 보고할 때

보고할 관심량, 여러 체인의 시작점과 워밍업·수집 길이, R-hat·bulk/tail ESS, MCSE, 트레이스·랭크 그림과 HMC 경고를 함께 검토한다.

임계값을 통과했는지뿐 아니라 모드 누락 가능성, 설정·시드 변화에 대한 안정성과 남은 제한을 기록한다. 이상 신호가 있으면 그 원인을 조사하며, 경고가 없다는 사실만으로 모든 사후 추론이 정확하다고 단정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
R-hat이 작으면 모든 모드를 탐색했다는 뜻인가?

</div>

??? success "풀이"
    모든 체인이 같은 모드에 갇혀도 서로 비슷해 보일 수 있다. R-hat은 관측한 체인 간 일치도 진단이며 다양한 시작점·그림·모형 구조와 함께 해석한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
사후 표준편차와 MCSE를 구분하라.

</div>

??? success "풀이"
    전자는 주어진 자료·모형의 파라메터 불확실성, 후자는 유한 시뮬레이션으로 특정 요약을 계산한 수치 오차다. 표본 수를 늘리면 적절한 조건에서 MCSE가 줄지만 목표 사후 표준편차 자체가 줄지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 FFT ESS 함수가 현대적인 bulk·tail ESS와 같은가?

</div>

??? success "풀이"
    단일 1차원 원자료의 자기상관을 단순 절단한다. 순위 정규화·다중 체인·꼬리 지시함수 등의 처리가 없으므로 같지 않다. 상수열과 max_lag 범위에도 별도 처리가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
워밍업 후 발산이 적으면 무시해도 되는가?

</div>

??? success "풀이"
    비율만으로 허용 여부를 정하지 않는다. 발산 위치와 원인·민감도를 조사하고 스텝 크기·동등한 리파라메리제이션 등을 검토한다. 하나의 경고가 모든 요약의 오류를 증명하는 것도 아니지만 수치 바이어스의 신호를 설명 없이 넘기면 안 된다.

## 정리하며

R-hat은 체인 일치도, 함수별 ESS·MCSE는 정밀도, 그림과 HMC 경고는 탐색·기하 문제를 살핀다. 모두 유한 실행의 진단이며 고정 임계값의 통과가 수렴 증명을 대신하지 않는다.

**참고 문헌**

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved $\hat{R}$ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. 11장.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.

---

**더 읽을거리**

- Gelman & Rubin (1992): $\hat{R}$ 통계량
- Geweke (1992): 시계열 진단
- Vehtari et al. (2021): 차례로 고르게 한 $\hat{R}$, ESS 권고
- Gelman et al. (2013): Bayesian Data Analysis(12장)
