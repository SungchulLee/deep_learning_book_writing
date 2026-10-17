# 개체 유형과 주석 체계

개체명 인식(NER)의 유형 체계는 무엇을 개체로 보고 어떤 이름으로 분류할지 정한다. 사람·조직·장소처럼 익숙한 범주도 데이터셋에 따라 경계와 처리 규칙이 다르다. 유형 목록만 가져오기보다 언급 범위·중첩·모호한 표현·정규화와 주석 지침을 함께 정의해야 한다.

## 1. 대표 데이터셋의 유형

[CoNLL-2003](https://aclanthology.org/W03-0419/)은 PER·ORG·LOC·MISC의 네 유형을 사용한다. MISC는 세계의 모든 ‘나머지 대상’을 빠짐없이 담는 범주가 아니라 해당 주석 지침의 범주다. 국적 형용사 등 데이터셋의 처리 예를 확인하며 행사·수상 이름을 다른 데이터의 EVENT·WORK_OF_ART와 기계적으로 대응시키지 않는다.

[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)의 흔한 NER 체계는 PERSON·NORP·FAC·ORG·GPE·LOC·PRODUCT·EVENT·WORK_OF_ART·LAW·LANGUAGE와 DATE·TIME·PERCENT·MONEY·QUANTITY·ORDINAL·CARDINAL의18유형이다. 앞의11개와 수량·시간 표현7개를 구분할 수 있다. GPE는 국가·도시 등 지정학적 실체로 자연 지형 등을 포함하는 LOC와 다르지만 문맥·지침에 따른 판단이 필요하다.

ACE의 정식 명칭은 Automatic Content Extraction이며 PER·ORG·GPE·LOC·FAC·WEA·VEH와 세부 유형을 사용한다. NER와 이름·보통명사·대명사 언급을 포괄하는 entity mention 탐지의 범위는 다를 수 있다. 서로 다른 벤치마크의 클래스 수·성능을 주석 범위가 같은 것처럼 비교하지 않는다.

## 2. 분야별 체계와 개체 연결

생의학에서는 유전자·단백질·질병·화학물질·종·세포 등 목적에 맞는 유형을 사용할 수 있다. 같은 텍스트 범위가 유전자와 단백질 중 무엇을 가리키는지, 명명 변형과 중첩을 어떻게 처리할지는 데이터 지침에 달렸다. 데이터셋 이름만으로 모든 범주가 포함된다고 가정하지 않는다.

금융 문서에는 회사·종목 코드·금융상품·금액·경제지표·보고 기간 등 별도의 설계가 가능하다. AAPL이라는 종목 코드와 회사 언급, S&P 500이라는 지수와 금융상품을 같은 범주로 둘지 결정해야 한다. 회사 이름을 식별하는 NER와 특정 회사·종목 ID에 연결하는 entity linking은 다른 단계다. 규제기관도 ORG의 세부 유형으로 둘 수 있으며 반드시 배타적인 독립 최상위 클래스로 만들 필요는 없다.

법률 문서의 법원·법령·사건·당사자·인용도 목적에 맞게 정의한다. 이들 예시는 과제 설계의 후보이지 모든 금융·의료·법률 자료에 적용되는 단일 표준은 아니다.

## 3. 계층과 다중 레이블

계층 체계는 세부 유형과 상위 유형의 관계를 표현한다. 예를 들어 하위 유형을 선택하면 모든 조상 유형도 선택하도록 하는 폴리시를 둘 수 있다.

```
Entity
├── Person
│   ├── Politician
│   ├── Athlete
│   ├── Scientist
│   └── Artist
├── Organization
│   ├── Company
│   │   ├── Tech Company
│   │   └── Financial Institution
│   ├── Government Agency
│   └── Educational Institution
└── Location
    ├── Country
    ├── City
    └── Natural Feature
```

언급 m과 문맥 c에서 각 유형 t의 포함 확률을 q_t로 두고 이진 포함 여부 z_t를 독립 Bernoulli로 근사한다면 완전한 레이블 집합의 확률은 다음과 같다.

$$p(z\mid m,c)=\prod_{t\in\mathcal T}q_t^{z_t}(1-q_t)^{1-z_t}.$$

양성 유형의 확률만 곱하면 ‘이 유형들은 모두 포함’의 독립 근사이지, 다른 유형이 제외된 정확한 집합의 확률은 아니다. 또한 독립 sigmoid만으로 하위 선택→상위 선택의 일관성이 자동 보장되지 않는다. 계층적 조건부 모델·제약·조상 추가 등의 폴리시를 명시해야 한다.

같은 수준의 유형을 배타적으로 둘 수도, 여러 레이블을 허용할 수도 있다. 실체의 유형 계층과 한 문장에서 겹치거나 중첩되는 언급 범위는 별개의 구조다. 평면 BIO 하나로 모든 중첩·불연속 개체를 표현할 수 있는 것은 아니다.

## 4. 주석 지침과 일치도

유형은 후속 과제에 유용하면서 사람이 일관되게 판단할 수 있어야 한다. 이름의 접미사·직함·구두점 포함 여부, 약어·대명사·일반명사, 모호한 문맥·중첩·미정 유형을 예문으로 명시한다. 작은 표본을 독립적으로 주석하고 불일치의 원인을 지침과 데이터에 반영한다. 희귀 유형을 너무 세분화하면 학습·평가의 불확실성이 커질 수 있다.

Cohen의 κ=(p_o−p_e)/(1−p_e)는 두 주석자의 고정된 판단 단위에서 관찰·우연 일치를 비교한다. p_e=1이면 분모가0이며 빈도 분포·주석 단위에 민감하다. 자유로운 span 선택에는 먼저 판단 단위와 대응 규칙을 정해야 한다. O가 많은 토큰 일치도나 임의의 κ>0.8 하나만으로 경계·유형의 신뢰성을 보장할 수 없다.

개체 span의 정확 일치 F1·경계만의 일치·유형 혼동과 어려운 사례를 함께 살펴본다. 일치도가 높아도 지침 자체가 목표 업무의 의미를 잘못 반영할 수 있으므로 타당성과 일치도를 구분한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
CoNLL의 LOC·MISC를 OntoNotes의 GPE·LOC·EVENT로 단순 매핑하기 어려운 이유를 설명하라.

</div>

??? success "풀이"

    범주 수뿐 아니라 언급의 범위·정의·주석 지침이 다르다. MISC는 보편적인 나머지 전체가 아니고 LOC의 포함 범위도 체계에 따라 다르다. 원문·지침을 확인한 재주석이나 명시적인 정보 로스 매핑이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
독립 sigmoid 유형 예측이 계층 일관성을 자동으로 만족하는가?

</div>

??? success "풀이"

    하위 유형의 확률은 높고 상위는 낮을 수 있어 자동 보장되지 않는다. 조상 레이블의 추가·제약 또는 계층 조건부 모델을 정의해야 한다. 유형의 다중 레이블과 span 중첩은 별개다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정확한 다중 레이블 집합 확률에서 음성 유형 항이 필요한 이유를 설명하라.

</div>

??? success "풀이"

    선택된 유형뿐 아니라 제외된 유형의 사건도 지정하는 확률이기 때문이다. 독립 Bernoulli 근사에서는 q_t의 양성 항과1−q_t의 음성 항을 모두 곱한다. 양성 항만 곱하면 다른 유형의 제외를 표현하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
κ가 높으면 NER 주석 체계가 검증되었다고 말할 수 있는가?

</div>

??? success "풀이"

    판단 단위·유형 빈도·O 비중·경계 처리에 따라 값이 달라 단독으로 충분하지 않다. span 대응·정확 일치·유형 혼동·희귀 사례와 실제 업무의 타당성을 함께 검토한다. 보편적인0.8임계값이 신뢰성을 보장하지 않는다.



## 정리하며

개체 유형은 데이터셋의 지침·언급 범위·분야 목표와 함께 정의한다. 계층·다중 레이블·중첩 span·개체 연결을 구분하고, 독립 확률과 계층 제약 및 주석 일치도의 가정을 명시한다.

**참고 문헌**

1. Tjong Kim Sang, E. F., & De Meulder, F. (2003). [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](https://aclanthology.org/W03-0419/).
2. Weischedel, R., et al. (2013). [OntoNotes Release 5.0](https://catalog.ldc.upenn.edu/LDC2013T19). LDC.
3. Ling, X., & Weld, D. S. (2012). Fine-Grained Entity Recognition. AAAI.
