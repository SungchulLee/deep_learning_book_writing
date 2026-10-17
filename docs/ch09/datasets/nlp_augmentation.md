# 자연어 데이터 어그멘테이션

문자·단어 수준의 변형으로 텍스트 어그멘테이션을 시연한다. 어그멘테이션된 문장이 원래 의미와 과제 레이블을 유지하는지 확인하는 것이 우선이다.

## 1. 코드

??? note "코드 (437줄)"

    ```python
    """자연어 데이터 어그멘테이션."""
    # ---
    # title: "NLP Data Augmentation"
    # description: "문자 수준, 낱말 수준, 문장 수준의 텍스트 어그멘테이션
    #               기법. nlpaug와 직접 구현을 함께 쓴다"
    # ---
    #
    # 자연어 처리의 데이터 어그멘테이션은 이미지보다 어렵다. 텍스트는 그냥
    # 뒤집거나 돌릴 수 없다. 대신 뜻을 지키면서 다양한 학습 예를 만드는
    # 언어적 흔들기를 쓴다.
    #
    #   1부 – 문자 수준 어그멘테이션 (OCR 오류, 자판 오타)
    #   2부 – 낱말 수준 어그멘테이션 (유의어, 임베딩 기반)
    #   3부 – 역번역 어그멘테이션
    #   4부 – 가려진 언어 모델을 쓰는 문맥 어그멘테이션
    #   5부 – 금융 텍스트를 위한 사용자 정의 어그멘테이션 파이프라인
    #   6부 – 어그멘테이션의 효과 재기
    #
    # 출처: O'Reilly "Practical NLP" 2장에서 고쳐 씀

    import numpy as np
    import random
    import re
    from typing import List, Tuple


    # =====================================================================
    # 1부 – 문자 수준 어그멘테이션
    # =====================================================================
    print("=" * 60)
    print("Part 1: Character-Level Augmentation")
    print("=" * 60)

    # 문자 수준의 흔들기는 실제 잡음을 흉내 낸다:
    # - OCR 오류 (문서 스캔)
    # - 자판 오타 (소셜 미디어 / 대화 데이터)
    # - 맞춤법 실수

    # OCR 오류 모의실험
    OCR_CONFUSIONS = {
        'o': '0', '0': 'o', 'l': '1', '1': 'l',
        'S': '5', '5': 'S', 'g': '9', '9': 'g',
        'B': '8', '8': 'B', 'Z': '2', '2': 'Z',
        'O': '0', 'I': '1', 'i': '!', 'rn': 'm',
    }

    # 자판 인접 지도 (QWERTY 배열)
    KEYBOARD_NEIGHBORS = {
        'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf',
        't': 'ryfg', 'y': 'tugh', 'u': 'yijh', 'i': 'uokj',
        'o': 'iplk', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz',
        'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujbn',
        'j': 'huiknm', 'k': 'jiolm', 'l': 'kop', 'z': 'asx',
        'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
        'n': 'bhjm', 'm': 'njk',
    }


    def augment_ocr(text: str, prob: float = 0.1) -> str:
        """모양이 비슷한 문자로 바꾸어 OCR 오류를 흉내 낸다."""
        result = []
        for char in text:
            if random.random() < prob and char in OCR_CONFUSIONS:
                result.append(OCR_CONFUSIONS[char])
            else:
                result.append(char)
        return "".join(result)


    def augment_keyboard(text: str, prob: float = 0.05) -> str:
        """이웃한 자판으로 바꾸어 오타를 흉내 낸다."""
        result = []
        for char in text:
            if random.random() < prob and char.lower() in KEYBOARD_NEIGHBORS:
                neighbors = KEYBOARD_NEIGHBORS[char.lower()]
                replacement = random.choice(neighbors)
                result.append(replacement if char.islower() else replacement.upper())
            else:
                result.append(char)
        return "".join(result)


    def augment_spelling(text: str, prob: float = 0.1) -> str:
        """흔한 맞춤법 실수를 흉내 낸다 (맞바꾸기, 삭제, 삽입, 반복)."""
        words = text.split()
        result = []
        for word in words:
            if random.random() < prob and len(word) > 3:
                op = random.choice(["swap", "delete", "repeat"])
                chars = list(word)
                pos = random.randint(1, len(chars) - 2)
                if op == "swap":
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                elif op == "delete":
                    chars.pop(pos)
                elif op == "repeat":
                    chars.insert(pos, chars[pos])
                result.append("".join(chars))
            else:
                result.append(word)
        return " ".join(result)


    # 시연
    random.seed(42)
    sample = "The quick brown fox jumps over the lazy dog"
    print(f"  Original:  {sample}")
    print(f"  OCR:       {augment_ocr(sample, prob=0.15)}")
    print(f"  Keyboard:  {augment_keyboard(sample, prob=0.1)}")
    print(f"  Spelling:  {augment_spelling(sample, prob=0.3)}")
    print()

    # nlpaug 라이브러리 사용법
    print("  Using nlpaug library (recommended for production):")
    print("""
        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw

        # OCR 어그멘터
        aug = nac.OcrAug()
        augmented = aug.augment(text, n=3)  # 변형 3개

        # 자판 어그멘터
        aug = nac.KeyboardAug()
        augmented = aug.augment(text, n=3)
    """)


    # =====================================================================
    # 2부 – 낱말 수준 어그멘테이션
    # =====================================================================
    print("=" * 60)
    print("Part 2: Word-Level Augmentation")
    print("=" * 60)

    # 낱말 수준 어그멘테이션은 문법은 지키면서 어휘를 바꾼다:
    # - 유의어 치환 (WordNet)
    # - 임베딩 기반 치환 (Word2Vec, GloVe)
    # - 무작위 삽입 / 삭제 / 맞바꾸기

    # 작은 사전을 쓰는 간단한 유의어 치환
    FINANCIAL_SYNONYMS = {
        "increase": ["rise", "growth", "gain", "surge", "uptick"],
        "decrease": ["decline", "drop", "fall", "downturn", "slump"],
        "profit": ["earnings", "income", "gains", "returns"],
        "loss": ["deficit", "shortfall", "writedown"],
        "revenue": ["sales", "turnover", "income"],
        "strong": ["robust", "solid", "healthy", "impressive"],
        "weak": ["soft", "disappointing", "lackluster", "tepid"],
        "buy": ["acquire", "purchase", "accumulate"],
        "sell": ["divest", "offload", "liquidate"],
        "stock": ["shares", "equity", "securities"],
    }


    def augment_synonym(text: str, prob: float = 0.2) -> str:
        """분야 사전의 유의어로 낱말을 바꾼다."""
        words = text.lower().split()
        result = []
        for word in words:
            if random.random() < prob and word in FINANCIAL_SYNONYMS:
                result.append(random.choice(FINANCIAL_SYNONYMS[word]))
            else:
                result.append(word)
        return " ".join(result)


    def augment_random_swap(text: str, n_swaps: int = 1) -> str:
        """이웃한 낱말을 무작위로 맞바꾼다."""
        words = text.split()
        for _ in range(n_swaps):
            if len(words) > 2:
                i = random.randint(0, len(words) - 2)
                words[i], words[i + 1] = words[i + 1], words[i]
        return " ".join(words)


    def augment_random_delete(text: str, prob: float = 0.1) -> str:
        """낱말을 무작위로 지운다 (적어도 절반은 남긴다)."""
        words = text.split()
        if len(words) <= 2:
            return text
        result = [w for w in words if random.random() > prob]
        return " ".join(result) if result else text


    sample_fin = "Revenue increase was strong driven by stock buyback"
    random.seed(42)
    print(f"  Original:  {sample_fin}")
    print(f"  Synonym:   {augment_synonym(sample_fin, prob=0.5)}")
    print(f"  Swap:      {augment_random_swap(sample_fin, n_swaps=2)}")
    print(f"  Delete:    {augment_random_delete(sample_fin, prob=0.2)}")
    print()

    # 임베딩 기반 어그멘테이션
    print("  Embedding-based augmentation (nlpaug + Word2Vec/GloVe):")
    print("""
        import nlpaug.augmenter.word as naw

        # 뜻이 비슷한 낱말 넣기
        aug = naw.WordEmbsAug(
            model_type='word2vec',
            model_path='GoogleNews-vectors-negative300.bin',
            action="insert",
        )
        augmented = aug.augment("Revenue grew 15% this quarter")
        # → "Revenue grew substantially 15% this profitable quarter"

        # 비슷한 낱말로 바꾸기
        aug = naw.WordEmbsAug(
            model_type='word2vec',
            model_path='GoogleNews-vectors-negative300.bin',
            action="substitute",
        )
        augmented = aug.augment("Revenue grew 15% this quarter")
        # → "Sales increased 15% this period"
    """)


    # =====================================================================
    # 3부 – 역번역 어그멘테이션
    # =====================================================================
    print("=" * 60)
    print("Part 3: Back-Translation Augmentation")
    print("=" * 60)

    # 역번역: 영어 → 프랑스어 → 영어
    # 뜻을 지키면서 문장을 자연스럽게 바꾸어 쓴다.

    print("""
      HuggingFace를 쓰는 역번역 파이프라인:

        from transformers import MarianMTModel, MarianTokenizer

        # 영어 → 프랑스어
        en_fr_model = "Helsinki-NLP/opus-mt-en-fr"
        en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model)
        en_fr = MarianMTModel.from_pretrained(en_fr_model)

        # 프랑스어 → 영어
        fr_en_model = "Helsinki-NLP/opus-mt-fr-en"
        fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model)
        fr_en = MarianMTModel.from_pretrained(fr_en_model)

        def back_translate(text, src_model, src_tok, tgt_model, tgt_tok):
            # 순방향 번역
            inputs = src_tok(text, return_tensors="pt", truncation=True)
            translated = src_model.generate(**inputs)
            intermediate = src_tok.decode(translated[0], skip_special_tokens=True)

            # 역방향 번역
            inputs = tgt_tok(intermediate, return_tensors="pt", truncation=True)
            back = tgt_model.generate(**inputs)
            return tgt_tok.decode(back[0], skip_special_tokens=True)

        original = "The company reported strong quarterly earnings."
        augmented = back_translate(
            original, en_fr, en_fr_tokenizer, fr_en, fr_en_tokenizer
        )
        # → "The company announced solid quarterly results."
    """)


    # =====================================================================
    # 4부 – 가려진 언어 모델을 쓰는 문맥 어그멘테이션
    # =====================================================================
    print("=" * 60)
    print("Part 4: Contextual Augmentation with Masked LM")
    print("=" * 60)

    # BERT의 가려진 언어 모델로 낱말을 문맥에 알맞은
    # 다른 낱말로 바꾼다.

    print("""
      from transformers import pipeline

      fill_mask = pipeline("fill-mask", model="bert-base-uncased")

      text = "The company reported [MASK] quarterly earnings."
      suggestions = fill_mask(text)
      for s in suggestions[:5]:
          print(f"  {s['token_str']:>10}: {s['score']:.3f}")
      # →    strong: 0.142
      #       solid: 0.098
      #       good: 0.087
      #     record: 0.064
      #     better: 0.052

      # 금융 텍스트에는 FinBERT를 쓴다:
      fill_mask_fin = pipeline("fill-mask", model="ProsusAI/finbert")
      # 분야에 알맞은 치환을 얻는다
    """)


    # =====================================================================
    # 5부 – 금융 텍스트를 위한 사용자 정의 파이프라인
    # =====================================================================
    print("=" * 60)
    print("Part 5: Custom Augmentation Pipeline for Financial Text")
    print("=" * 60)


    class FinancialTextAugmenter:
        """금융 자연어 처리를 위한 다중 전략 텍스트 어그멘테이션 파이프라인.

        여러 어그멘테이션 전략을 설정 가능한 확률로 결합한다. 분야 특유의 어휘를 지켜야 하는
        금융 텍스트를 위해 설계되었다.


        인수:
            synonym_prob: 낱말마다 유의어로 바꿀 확률
            typo_prob:    낱말마다 문자 수준 오타가 날 확률
            delete_prob:  낱말을 무작위로 지울 확률
            swap_prob:    이웃한 낱말을 맞바꿀 확률
        """

        def __init__(
            self,
            synonym_prob: float = 0.15,
            typo_prob: float = 0.05,
            delete_prob: float = 0.1,
            swap_prob: float = 0.1,
        ):
            self.synonym_prob = synonym_prob
            self.typo_prob = typo_prob
            self.delete_prob = delete_prob
            self.swap_prob = swap_prob

            # 절대 바꾸면 안 되는 낱말 (종목 기호, 숫자, 핵심 용어)
            self.protected_pattern = re.compile(
                r'^\$[A-Z]+$|^\d+\.?\d*%?$|^Q[1-4]$|^FY\d{2,4}$'
            )

        def _is_protected(self, word: str) -> bool:
            """어그멘테이션하면 안 되는 낱말인지 확인한다 (종목 기호, 숫자 등)."""
            return bool(self.protected_pattern.match(word))

        def augment(self, text: str, n: int = 1) -> List[str]:
            """입력 텍스트의 어그멘테이션된 판본을 n개 만든다."""
            augmented = []
            for _ in range(n):
                # 어그멘테이션 전략을 무작위로 고르기
                strategy = random.choice([
                    "synonym", "typo", "delete", "swap", "combined"
                ])
                if strategy == "synonym":
                    aug_text = augment_synonym(text, self.synonym_prob)
                elif strategy == "typo":
                    aug_text = augment_keyboard(text, self.typo_prob)
                elif strategy == "delete":
                    aug_text = augment_random_delete(text, self.delete_prob)
                elif strategy == "swap":
                    aug_text = augment_random_swap(text, n_swaps=1)
                else:  # 합침
                    aug_text = augment_synonym(text, self.synonym_prob * 0.5)
                    aug_text = augment_keyboard(aug_text, self.typo_prob * 0.5)
                augmented.append(aug_text)
            return augmented


    # 시연
    augmenter = FinancialTextAugmenter()
    sample = "$AAPL revenue increase of 15% was strong in Q3 FY2024"
    random.seed(42)
    print(f"  Original: {sample}")
    print(f"  Augmented versions:")
    for i, aug in enumerate(augmenter.augment(sample, n=5)):
        print(f"    [{i+1}] {aug}")
    print()


    # =====================================================================
    # 6부 – 어그멘테이션의 효과 재기
    # =====================================================================
    print("=" * 60)
    print("Part 6: Measuring Augmentation Effectiveness")
    print("=" * 60)

    # 어그멘테이션은 다음을 해야 한다:
    #   1. 모델의 성능을 높인다 (특히 작은 데이터셋에서)
    #   2. 레이블의 올바름을 지킨다
    #   3. 잡음을 들이지 않으면서 다양성을 늘린다
    #
    # 핵심 지표:
    #   - 정확도 차이 (어그멘테이션 학습과 원래 학습)
    #   - 레이블 보존율
    #   - 어휘 다양성 (서로 다른 n-그램 / 전체 n-그램)

    def lexical_diversity(texts: List[str], n: int = 2) -> float:
        """텍스트 모음에 걸쳐 n-그램의 유형-토큰 비율을 계산한다."""
        all_ngrams = []
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - n + 1):
                all_ngrams.append(tuple(words[i:i + n]))
        if not all_ngrams:
            return 0.0
        return len(set(all_ngrams)) / len(all_ngrams)


    original_texts = [
        "Revenue increased by 10% this quarter",
        "Profit margins declined sharply",
        "Strong earnings beat analyst expectations",
    ] * 5

    augmented_texts = []
    for t in original_texts:
        augmented_texts.extend(augmenter.augment(t, n=2))

    print(f"  Original texts: {len(original_texts)}")
    print(f"  Augmented texts: {len(augmented_texts)}")
    print(f"  Original lexical diversity (bigrams):  {lexical_diversity(original_texts):.3f}")
    print(f"  Augmented lexical diversity (bigrams): {lexical_diversity(augmented_texts):.3f}")
    print()

    print("""
      어그멘테이션으로 얻는 일반적인 개선:

      ┌──────────────────┬──────────┬───────────────────┐
      │ 데이터셋 크기    │ 어그멘테이션 없음│ 어그멘테이션 적용(2~5배)  │
      ├──────────────────┼──────────┼───────────────────┤
      │ 100 examples     │  0.62    │  0.71 (+9%)       │
      │ 500 examples     │  0.74    │  0.79 (+5%)       │
      │ 2000 examples    │  0.82    │  0.84 (+2%)       │
      │ 10000 examples   │  0.87    │  0.88 (+1%)       │
      └──────────────────┴──────────┴───────────────────┘

      핵심: 어그멘테이션은 데이터셋이 작을 때 가장 큰 도움이 된다.
      데이터가 충분하면(1만 개 이상) 개선 폭이 줄어든다.
    """)

    print("Done.")


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (143줄)"

    ```
    ============================================================
    Part 1: Character-Level Augmentation
    ============================================================
      Original:  The quick brown fox jumps over the lazy dog
      OCR:       The quick br0wn fox jumps 0ver the lazy d0g
      Keyboard:  Tye quick btown fox jympx over the lazy dig
      Spelling:  The quiick brown fox jumps over the lazy dog

      Using nlpaug library (recommended for production):

        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw

        # OCR 어그멘터
        aug = nac.OcrAug()
        augmented = aug.augment(text, n=3)  # 변형 3개

        # 자판 어그멘터
        aug = nac.KeyboardAug()
        augmented = aug.augment(text, n=3)

    ============================================================
    Part 2: Word-Level Augmentation
    ============================================================
      Original:  Revenue increase was strong driven by stock buyback
      Synonym:   revenue gain was robust driven by equity buyback
      Swap:      increase was Revenue strong driven by stock buyback
      Delete:    Revenue increase was strong driven by stock buyback

      Embedding-based augmentation (nlpaug + Word2Vec/GloVe):

        import nlpaug.augmenter.word as naw

        # 뜻이 비슷한 낱말 넣기
        aug = naw.WordEmbsAug(
            model_type='word2vec',
            model_path='GoogleNews-vectors-negative300.bin',
            action="insert",
        )
        augmented = aug.augment("Revenue grew 15% this quarter")
        # → "Revenue grew substantially 15% this profitable quarter"

        # 비슷한 낱말로 바꾸기
        aug = naw.WordEmbsAug(
            model_type='word2vec',
            model_path='GoogleNews-vectors-negative300.bin',
            action="substitute",
        )
        augmented = aug.augment("Revenue grew 15% this quarter")
        # → "Sales increased 15% this period"

    ============================================================
    Part 3: Back-Translation Augmentation
    ============================================================

      HuggingFace를 쓰는 역번역 파이프라인:

        from transformers import MarianMTModel, MarianTokenizer

        # 영어 → 프랑스어
        en_fr_model = "Helsinki-NLP/opus-mt-en-fr"
        en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model)
        en_fr = MarianMTModel.from_pretrained(en_fr_model)

        # 프랑스어 → 영어
        fr_en_model = "Helsinki-NLP/opus-mt-fr-en"
        fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model)
        fr_en = MarianMTModel.from_pretrained(fr_en_model)

        def back_translate(text, src_model, src_tok, tgt_model, tgt_tok):
            # 순방향 번역
            inputs = src_tok(text, return_tensors="pt", truncation=True)
            translated = src_model.generate(**inputs)
            intermediate = src_tok.decode(translated[0], skip_special_tokens=True)

            # 역방향 번역
            inputs = tgt_tok(intermediate, return_tensors="pt", truncation=True)
            back = tgt_model.generate(**inputs)
            return tgt_tok.decode(back[0], skip_special_tokens=True)

        original = "The company reported strong quarterly earnings."
        augmented = back_translate(
            original, en_fr, en_fr_tokenizer, fr_en, fr_en_tokenizer
        )
        # → "The company announced solid quarterly results."

    ============================================================
    Part 4: Contextual Augmentation with Masked LM
    ============================================================

      from transformers import pipeline

      fill_mask = pipeline("fill-mask", model="bert-base-uncased")

      text = "The company reported [MASK] quarterly earnings."
      suggestions = fill_mask(text)
      for s in suggestions[:5]:
          print(f"  {s['token_str']:>10}: {s['score']:.3f}")
      # →    strong: 0.142
      #       solid: 0.098
      #       good: 0.087
      #     record: 0.064
      #     better: 0.052

      # 금융 텍스트에는 FinBERT를 쓴다:
      fill_mask_fin = pipeline("fill-mask", model="ProsusAI/finbert")
      # 분야에 알맞은 치환을 얻는다

    ============================================================
    Part 5: Custom Augmentation Pipeline for Financial Text
    ============================================================
      Original: $AAPL revenue increase of 15% was strong in Q3 FY2024
      Augmented versions:
        [1] $aapl revenue increase of 15% was impressive in q3 fy2024
        [2] $aapl revenue increase of 15% was strong in q3 fy2024
        [3] $AAPL revenue increase of 15% was strong Q3 in FY2024
        [4] $AAPL revenye increase of 15% aas strong in Q3 FY2024
        [5] $aapl revenue uptick of 15% was strong in q3 fy2024

    ============================================================
    Part 6: Measuring Augmentation Effectiveness
    ============================================================
      Original texts: 15
      Augmented texts: 30
      Original lexical diversity (bigrams):  0.200
      Augmented lexical diversity (bigrams): 0.256


      어그멘테이션으로 얻는 일반적인 개선:

      ┌──────────────────┬──────────┬───────────────────┐
      │ 데이터셋 크기    │ 어그멘테이션 없음│ 어그멘테이션 적용(2~5배)  │
      ├──────────────────┼──────────┼───────────────────┤
      │ 100 examples     │  0.62    │  0.71 (+9%)       │
      │ 500 examples     │  0.74    │  0.79 (+5%)       │
      │ 2000 examples    │  0.82    │  0.84 (+2%)       │
      │ 10000 examples   │  0.87    │  0.88 (+1%)       │
      └──────────────────┴──────────┴───────────────────┘

      핵심: 어그멘테이션은 데이터셋이 작을 때 가장 큰 도움이 된다.
      데이터가 충분하면(1만 개 이상) 개선 폭이 줄어든다.

    Done.
    ```
## 2. 논의

오타와 OCR 변형은 예상 입력 잡음을 흉내 낼 수 있지만 유의어 치환·삭제·단어 교환은 문법이나 의미를 바꿀 수 있다. 특히 부정어와 수치가 바뀌면 원래 목표가 더 이상 맞지 않을 수 있다.

FinancialTextAugmenter는 보호 패턴과 _is_protected를 정의하지만 실제 augment 경로에서 이를 호출하지 않는다. 따라서 코드에 있는 보호 규칙이 종목 기호나 숫자를 보존한다고 설명할 수 없다. 각 어그멘테이션 함수가 보호 범위를 반영하도록 추가 연결이 필요하다.

n-그램 다양성은 어휘 변화의 지표이며 레이블 정확성이나 학습 효과를 보장하지 않는다. 원본 분할을 먼저 만든 뒤 학습 부분만 어그멘테이션하고, 어그멘테이션 없는 기준과 같은 검증 조건에서 비교한다. 특정 표본 수 이하에서 일정한 정확도 향상이 난다는 규칙은 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
보호 패턴을 정의한 것만으로 보호가 동작하지 않는 이유를 설명하라.

</div>

??? success "풀이"
    실제 문장을 변환하는 경로에서 패턴을 확인하고 해당 토큰을 제외해야 한다. 이 클래스의 augment는 함수들을 직접 호출하여 _is_protected를 사용하지 않는다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
무작위 삭제로 레이블이 바뀔 수 있는 예를 들어라.

</div>

??? success "풀이"
    “수익이 증가하지 않았다”에서 부정 표현을 지우면 의미가 반대가 될 수 있다. 감성·관계·수치 목표를 유지하는지 표본을 검사하고 변환 범위를 제한해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
분할 전에 어그멘테이션하면 어떤 누출이 생길 수 있는가?

</div>

??? success "풀이"
    같은 원문에서 만든 유사 문장이 학습과 검증·테스트에 나뉘어 들어갈 수 있다. 원문 단위로 먼저 분할하고 학습 쪽만 어그멘테이션한다.

## 정리하며

텍스트 어그멘테이션은 의미 보존이 자동으로 성립하지 않는다. 보호 규칙이 실제 호출되는지 확인하고 원문 단위 분할과 레이블 검수를 유지한다.
