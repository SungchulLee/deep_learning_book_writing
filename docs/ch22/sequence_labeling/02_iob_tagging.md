# IOB2·BIOES 태그의 검사와 변환 예제

이 예제는 미리 주어진 토큰·태그 열에서 경계 규칙을 검사하고 IOB2와 BIOES를 변환해 개체 span을 추출한다. 뉴럴 넷트웍을 학습하거나 텍스트를 생성하는 코드는 아니다. 문자열 변환·유효성 검사·오류 복구는 서로 다른 기능이므로 어느 함수를 호출하는지 구분해야 한다.

## 1. 코드와 기존 출력

??? note "코드"

    ```python
    """
    이름 알아보기의 IOB와 BIOES 이름표 방식
    ======================================

    이 단원은 이름 알아보기에 쓰는 여러 이름표 방식을 자세히 밝히고 짠다:
    이름 알아보기에 쓰이는 방식:
    - IOB(안-바깥-시작)
    - IOB2(더 빡빡한 판)
    - BIOES(시작-안-바깥-끝-홑)

    이 방식은 이름 알아보기의 차례 이름표 붙이기에 결정적이다.

    학습 목표:
    - 여러 이름표 방식을 이해한다
    - 이름표 방식 사이를 바꾼다
    - 이름표 확인 짜기
    - 모서리 경우 다루기

    지은이: 러닝 목적
    날짜: 2025
    """

    from typing import List, Tuple, Dict
    from enum import Enum

    # ========================================================================
    # 메인
    # ========================================================================


    class TagScheme(Enum):
        """
        받치는 이름표 방식의 열거.
        """
        IOB = "IOB"      # 안-바깥-시작(처음의 IOB)
        IOB2 = "IOB2"    # IOB 두 번째 판(더 널리 쓰임)
        BIOES = "BIOES"  # 시작-안-바깥-끝-홑


    class TagValidator:
        """
        여러 이름표 방식에 따라 이름표 차례를 확인한다.

        이름표 방식마다 올바른 이름표 넘어가기에 대한 규칙이 있다.
        이 클래스는 이름표 차례가 규칙을 따르는지 살핀다.
        """

        @staticmethod
        def validate_iob2(tags: List[str]) -> Tuple[bool, str]:
            """
            IOB2 이름표 차례 확인하기.

            IOB2 규칙:
            1. 것의 첫 이름표는 늘 B-TYPE이다
            2. 이어짐 이름표는 I-TYPE이다
            3. B-TYPE 뒤에는 (같은 클래스의) I-TYPE이나 O나 B-*만 올 수 있다
            4. I-TYPE 뒤에는 (같은 클래스의) I-TYPE이나 O나 B-*만 올 수 있다
            5. 같은 클래스의 B-TYPE 없이 I-TYPE이 올 수 없다

            인수:
                tags: 이름표의 목록(보기로 ["B-PER", "I-PER", "O", "B-LOC"])

            반환값:
                (is_valid, error_message) 튜플

            보기:
                >>> validator = TagValidator()
                >>> tags = ["B-PER", "I-PER", "O", "B-LOC"]
                >>> is_valid, msg = validator.validate_iob2(tags)
                >>> print(is_valid)  # 참
            """
            for i in range(len(tags)):
                current_tag = tags[i]

                # O 이름표 건너뛰기
                if current_tag == "O":
                    continue

                # 이름표 뜯어 읽기
                try:
                    prefix, entity_type = current_tag.split("-")
                except ValueError:
                    return False, f"Invalid tag format at position {i}: {current_tag}"

                # 올바른 앞가지인지 살피기
                if prefix not in ["B", "I"]:
                    return False, f"Invalid prefix at position {i}: {prefix}"

                # I-TYPE 뒤에 같은 클래스의 B-TYPE이나 I-TYPE이 오는지 살피기
                if prefix == "I":
                    if i == 0:
                        return False, f"I-{entity_type} cannot be first tag"

                    prev_tag = tags[i-1]
                    if prev_tag == "O":
                        return False, f"I-{entity_type} at position {i} follows O"

                    try:
                        prev_prefix, prev_type = prev_tag.split("-")
                        if prev_type != entity_type:
                            return False, (f"I-{entity_type} at position {i} "
                                         f"follows {prev_prefix}-{prev_type}")
                    except ValueError:
                        return False, f"Invalid previous tag at position {i-1}: {prev_tag}"

            return True, "Valid IOB2 sequence"

        @staticmethod
        def validate_bioes(tags: List[str]) -> Tuple[bool, str]:
            """
            BIOES 이름표 차례 확인하기.

            BIOES 규칙:
            1. S-TYPE은 토막 하나짜리 것에 쓴다
            2. B-TYPE이 토막 여럿짜리 것을 시작한다
            3. I-TYPE이 것을 이어 간다(같은 클래스의 B나 I 뒤에 와야 한다)
            4. E-TYPE이 것을 끝낸다(같은 클래스의 B나 I 뒤에 와야 한다)
            5. O는 어느 것에도 들지 않는다

            인수:
                tags: 이름표의 목록(보기로 ["B-PER", "I-PER", "E-PER", "O", "S-LOC"])

            반환값:
                (is_valid, error_message) 튜플
            """
            for i in range(len(tags)):
                current_tag = tags[i]

                # O 이름표 건너뛰기
                if current_tag == "O":
                    continue

                # 이름표 뜯어 읽기
                try:
                    prefix, entity_type = current_tag.split("-")
                except ValueError:
                    return False, f"Invalid tag format at position {i}: {current_tag}"

                # 올바른 앞가지인지 살피기
                if prefix not in ["B", "I", "E", "S"]:
                    return False, f"Invalid prefix at position {i}: {prefix}"

                # 홑 것 뒤에 이어짐이 오면 안 된다
                if prefix == "S":
                    if i < len(tags) - 1:
                        next_tag = tags[i+1]
                        if next_tag != "O" and not next_tag.startswith("B-") and not next_tag.startswith("S-"):
                            return False, f"S-{entity_type} at position {i} followed by {next_tag}"

                # I나 E 앞에는 같은 클래스의 B나 I가 와야 한다
                if prefix in ["I", "E"]:
                    if i == 0:
                        return False, f"{prefix}-{entity_type} cannot be first tag"

                    prev_tag = tags[i-1]
                    if prev_tag == "O":
                        return False, f"{prefix}-{entity_type} at position {i} follows O"

                    try:
                        prev_prefix, prev_type = prev_tag.split("-")
                        if prev_type != entity_type:
                            return False, (f"{prefix}-{entity_type} at position {i} "
                                         f"follows {prev_prefix}-{prev_type}")
                        if prev_prefix not in ["B", "I"]:
                            return False, (f"{prefix}-{entity_type} at position {i} "
                                         f"follows invalid prefix {prev_prefix}")
                    except ValueError:
                        return False, f"Invalid previous tag at position {i-1}: {prev_tag}"

                # B나 I 뒤에는 같은 클래스의 I나 E가 와야 한다(또는 차례의 끝)
                if prefix in ["B", "I"]:
                    if i < len(tags) - 1:
                        next_tag = tags[i+1]
                        if next_tag == "O":
                            return False, f"{prefix}-{entity_type} at position {i} followed by O (should end with E)"
                        if next_tag not in ["O"] and not next_tag.startswith("I-") and not next_tag.startswith("E-"):
                            return False, f"{prefix}-{entity_type} at position {i} not followed by I or E"
                    else:
                        # B나 I 상태의 마지막 이름표는 E여야 한다
                        return False, f"{prefix}-{entity_type} at position {i} is last tag (should be E or S)"

            return True, "Valid BIOES sequence"


    class TagConverter:
        """
        서로 다른 이름표 방식 사이를 바꾸기.

        다음과 같을 때 쓸모 있다:
        1. 익힘 자료의 꼴과 모델이 바라는 꼴이 다르다
        2. 서로 다른 체계의 결과 견주기
        3. 특정 모델에 맞춰 자료 앞손질하기
        """

        @staticmethod
        def iob2_to_bioes(tags: List[str]) -> List[str]:
            """
            IOB2 이름표를 BIOES 이름표로 바꾸기.

            바꾸기 규칙:
            - 토막 하나짜리 것: B-TYPE → S-TYPE
            - 토막 여럿짜리 것: B-TYPE → B-TYPE, I-TYPE → I-TYPE, 마지막 I-TYPE → E-TYPE
            - O는 O로 둔다

            인수:
                tags: IOB2 이름표의 목록

            반환값:
                BIOES 이름표의 목록

            보기:
                >>> converter = TagConverter()
                >>> iob2_tags = ["B-PER", "I-PER", "O", "B-LOC"]
                >>> bioes_tags = converter.iob2_to_bioes(iob2_tags)
                >>> print(bioes_tags)
                ['B-PER', 'E-PER', 'O', 'S-LOC']
            """
            bioes_tags = []

            for i in range(len(tags)):
                current_tag = tags[i]

                # O 이름표는 그대로 둔다
                if current_tag == "O":
                    bioes_tags.append("O")
                    continue

                # 지금 이름표 뜯어 읽기
                prefix, entity_type = current_tag.split("-")

                # 다음 이름표를 미리 보기
                is_last = (i == len(tags) - 1)
                if not is_last:
                    next_tag = tags[i+1]
                    next_continues = (next_tag != "O" and 
                                    next_tag.startswith("I-") and 
                                    next_tag.split("-")[1] == entity_type)
                else:
                    next_continues = False

                # BIOES 이름표 정하기
                if prefix == "B":
                    if next_continues:
                        # 토막 여럿짜리 것의 시작
                        bioes_tags.append(f"B-{entity_type}")
                    else:
                        # 토막 하나짜리 것
                        bioes_tags.append(f"S-{entity_type}")

                elif prefix == "I":
                    if next_continues:
                        # 것의 가운데
                        bioes_tags.append(f"I-{entity_type}")
                    else:
                        # 것의 끝
                        bioes_tags.append(f"E-{entity_type}")

            return bioes_tags

        @staticmethod
        def bioes_to_iob2(tags: List[str]) -> List[str]:
            """
            BIOES 이름표를 IOB2 이름표로 바꾸기.

            바꾸기 규칙:
            - S-TYPE → B-TYPE
            - B-TYPE → B-TYPE
            - I-TYPE → I-TYPE
            - E-TYPE → I-TYPE
            - O → O

            인수:
                tags: BIOES 이름표의 목록

            반환값:
                IOB2 이름표의 목록
            """
            iob2_tags = []

            for tag in tags:
                if tag == "O":
                    iob2_tags.append("O")
                else:
                    prefix, entity_type = tag.split("-")

                    if prefix == "S":
                        iob2_tags.append(f"B-{entity_type}")
                    elif prefix == "B":
                        iob2_tags.append(f"B-{entity_type}")
                    elif prefix in ["I", "E"]:
                        iob2_tags.append(f"I-{entity_type}")

            return iob2_tags

        @staticmethod
        def tags_to_entities(tokens: List[str], tags: List[str], 
                            scheme: TagScheme = TagScheme.IOB2) -> List[Tuple[str, str, int, int]]:
            """
            토막-이름표 짝을 것의 구간으로 바꾸기.

            이름표 붙인 차례에서 실제 것을 뽑아낸다.

            인수:
                tokens: 토막 글자열의 목록
                tags: 그에 맞는 이름표의 목록
                scheme: 쓴 이름표 방식(IOB2 또는 BIOES)

            반환값:
                튜플의 목록: (entity_text, entity_type, start_idx, end_idx)

            보기:
                >>> tokens = ["Steve", "Jobs", "founded", "Apple"]
                >>> tags = ["B-PER", "I-PER", "O", "B-ORG"]
                >>> entities = TagConverter.tags_to_entities(tokens, tags)
                >>> print(entities)
                [('Steve Jobs', 'PER', 0, 2), ('Apple', 'ORG', 3, 4)]
            """
            entities = []
            current_entity = None
            current_tokens = []
            current_start = None

            for i, (token, tag) in enumerate(zip(tokens, tags)):
                if tag == "O":
                    # 있으면 지금 것을 끝내기
                    if current_entity:
                        entity_text = " ".join(current_tokens)
                        entities.append((entity_text, current_entity, current_start, i))
                        current_entity = None
                        current_tokens = []
                        current_start = None

                else:
                    prefix, entity_type = tag.split("-")

                    if scheme == TagScheme.IOB2:
                        if prefix == "B":
                            # 있으면 앞 것 갈무리
                            if current_entity:
                                entity_text = " ".join(current_tokens)
                                entities.append((entity_text, current_entity, current_start, i))

                            # 새 것 시작하기
                            current_entity = entity_type
                            current_tokens = [token]
                            current_start = i

                        elif prefix == "I":
                            if current_entity == entity_type:
                                # 지금 것을 이어 가기
                                current_tokens.append(token)
                            else:
                                # 올바른 IOB2에서는 일어나지 않아야 한다
                                # 그래도 새 것을 시작해 다룬다
                                if current_entity:
                                    entity_text = " ".join(current_tokens)
                                    entities.append((entity_text, current_entity, current_start, i))

                                current_entity = entity_type
                                current_tokens = [token]
                                current_start = i

                    elif scheme == TagScheme.BIOES:
                        if prefix == "S":
                            # 토막 하나짜리 것
                            if current_entity:
                                entity_text = " ".join(current_tokens)
                                entities.append((entity_text, current_entity, current_start, i))

                            entities.append((token, entity_type, i, i+1))
                            current_entity = None
                            current_tokens = []
                            current_start = None

                        elif prefix == "B":
                            # 새 것 시작하기
                            if current_entity:
                                entity_text = " ".join(current_tokens)
                                entities.append((entity_text, current_entity, current_start, i))

                            current_entity = entity_type
                            current_tokens = [token]
                            current_start = i

                        elif prefix in ["I", "E"]:
                            if current_entity == entity_type:
                                current_tokens.append(token)

                                # E 이름표면 것을 끝내기
                                if prefix == "E":
                                    entity_text = " ".join(current_tokens)
                                    entities.append((entity_text, current_entity, current_start, i+1))
                                    current_entity = None
                                    current_tokens = []
                                    current_start = None

            # 차례가 것 도중에 끝나면 마지막 것 다루기
            if current_entity:
                entity_text = " ".join(current_tokens)
                entities.append((entity_text, current_entity, current_start, len(tokens)))

            return entities


    def demonstrate_tagging_schemes():
        """
        여러 이름표 방식을 두루 보이기.
        """
        print("="*70)
        print("IOB and BIOES Tagging Schemes Demonstration")
        print("="*70)

        # 보기 1: IOB2 이름표 붙이기
        print("\n1. IOB2 Tagging Scheme")
        print("-" * 70)

        text = "Steve Jobs founded Apple Inc. in Cupertino"
        tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", ".", "in", "Cupertino"]
        iob2_tags = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "O", "B-LOC"]

        print(f"Text: {text}")
        print(f"\nTokens and IOB2 tags:")
        print(f"{'Token':<15} {'IOB2 Tag':<10}")
        print("-" * 30)
        for token, tag in zip(tokens, iob2_tags):
            print(f"{token:<15} {tag:<10}")

        # IOB2 차례 확인하기
        is_valid, message = TagValidator.validate_iob2(iob2_tags)
        print(f"\nValidation: {message}")

        # 보기 2: BIOES 이름표 붙이기
        print("\n\n2. BIOES Tagging Scheme")
        print("-" * 70)

        # BIOES로 바꾸기
        bioes_tags = TagConverter.iob2_to_bioes(iob2_tags)

        print(f"{'Token':<15} {'IOB2':<10} {'BIOES':<10}")
        print("-" * 40)
        for token, iob2, bioes in zip(tokens, iob2_tags, bioes_tags):
            print(f"{token:<15} {iob2:<10} {bioes:<10}")

        # BIOES 차례 확인하기
        is_valid, message = TagValidator.validate_bioes(bioes_tags)
        print(f"\nValidation: {message}")

        # 보기 3: 이름표 방식 견줌
        print("\n\n3. Comparing IOB2 and BIOES")
        print("-" * 70)

        examples = [
            {
                'text': "IBM",
                'tokens': ["IBM"],
                'iob2': ["B-ORG"],
                'bioes': ["S-ORG"],
                'note': "Single-token entity: B-ORG vs S-ORG"
            },
            {
                'text': "New York",
                'tokens': ["New", "York"],
                'iob2': ["B-LOC", "I-LOC"],
                'bioes': ["B-LOC", "E-LOC"],
                'note': "Two-token entity: B-I vs B-E"
            },
            {
                'text': "University of California",
                'tokens': ["University", "of", "California"],
                'iob2': ["B-ORG", "I-ORG", "I-ORG"],
                'bioes': ["B-ORG", "I-ORG", "E-ORG"],
                'note': "Multi-token entity: last I vs E"
            }
        ]

        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}: {example['text']}")
            print(f"{'Token':<20} {'IOB2':<10} {'BIOES':<10}")
            print("-" * 40)
            for token, iob2, bioes in zip(example['tokens'], example['iob2'], example['bioes']):
                print(f"{token:<20} {iob2:<10} {bioes:<10}")
            print(f"Note: {example['note']}")

        # 보기 4: 이름표에서 것 뽑기
        print("\n\n4. Extracting Entities from Tags")
        print("-" * 70)

        tokens_complex = ["Barack", "Obama", "visited", "New", "York", "City", "and", "Microsoft"]
        iob2_tags_complex = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O", "B-ORG"]

        print(f"Tokens: {' '.join(tokens_complex)}")
        print(f"\nTags: {' '.join(iob2_tags_complex)}")

        entities = TagConverter.tags_to_entities(tokens_complex, iob2_tags_complex)

        print(f"\nExtracted entities:")
        for entity_text, entity_type, start, end in entities:
            print(f"  - '{entity_text}' ({entity_type}) at tokens [{start}:{end}]")

        # 보기 5: 올바르지 않은 이름표 차례
        print("\n\n5. Detecting Invalid Tag Sequences")
        print("-" * 70)

        invalid_examples = [
            {
                'tags': ["I-PER", "I-PER", "O"],
                'issue': "Starts with I- instead of B-"
            },
            {
                'tags': ["B-PER", "I-LOC", "O"],
                'issue': "Entity type mismatch (PER→LOC)"
            },
            {
                'tags': ["B-PER", "O", "I-PER"],
                'issue': "I-PER after O (should be B-PER)"
            }
        ]

        for i, example in enumerate(invalid_examples, 1):
            print(f"\nInvalid Example {i}:")
            print(f"Tags: {example['tags']}")
            is_valid, message = TagValidator.validate_iob2(example['tags'])
            print(f"Valid: {is_valid}")
            print(f"Issue: {example['issue']}")
            print(f"Error: {message}")

        # 보기 6: 방식마다의 이점
        print("\n\n6. When to Use Each Scheme")
        print("-" * 70)

        print("\nIOB2 (Inside-Outside-Beginning):")
        print("  Advantages:")
        print("    - Simpler: Only 3 tag types (B, I, O)")
        print("    - More compact representation")
        print("    - Widely used in research")
        print("  Disadvantages:")
        print("    - Cannot explicitly mark entity boundaries")
        print("    - Harder for model to learn where entities end")

        print("\nBIOES (Beginning-Inside-Outside-End-Single):")
        print("  Advantages:")
        print("    - Explicit entity boundaries (E tag)")
        print("    - Distinguishes single-token entities (S tag)")
        print("    - Better for models: clearer structure")
        print("    - Potentially better performance")
        print("  Disadvantages:")
        print("    - More complex: 5 tag types")
        print("    - Larger label space")
        print("    - More training data needed")

        # 보기 7: 실전 바꾸기 보기
        print("\n\n7. Practical Conversion Example")
        print("-" * 70)

        # 것이 여럿인 월
        sentence = "Microsoft CEO Satya Nadella spoke at Stanford"
        tokens_ex = ["Microsoft", "CEO", "Satya", "Nadella", "spoke", "at", "Stanford"]
        iob2_ex = ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "B-ORG"]

        print(f"Sentence: {sentence}")
        print(f"\nOriginal IOB2 tags:")
        for token, tag in zip(tokens_ex, iob2_ex):
            print(f"  {token:<15} {tag}")

        # BIOES로 바꾸기
        bioes_ex = TagConverter.iob2_to_bioes(iob2_ex)
        print(f"\nConverted to BIOES:")
        for token, tag in zip(tokens_ex, bioes_ex):
            print(f"  {token:<15} {tag}")

        # 것 뽑기
        entities_iob2 = TagConverter.tags_to_entities(tokens_ex, iob2_ex, TagScheme.IOB2)
        entities_bioes = TagConverter.tags_to_entities(tokens_ex, bioes_ex, TagScheme.BIOES)

        print(f"\nExtracted entities (both schemes give same result):")
        for entity_text, entity_type, start, end in entities_iob2:
            print(f"  - '{entity_text}' ({entity_type})")


    def interactive_tag_converter():
        """
        이름표 바꾸기를 익히는 주고받는 도구.
        """
        print("\n\n" + "="*70)
        print("Interactive Tag Converter")
        print("="*70)

        # 익히기용 보기
        print("\nPractice Example:")
        print("Tokens: ['Barack', 'Obama', 'visited', 'Google']")
        print("IOB2 tags: ['B-PER', 'I-PER', 'O', 'B-ORG']")

        tokens = ['Barack', 'Obama', 'visited', 'Google']
        iob2_tags = ['B-PER', 'I-PER', 'O', 'B-ORG']

        # 바꾸기 보이기
        bioes_tags = TagConverter.iob2_to_bioes(iob2_tags)

        print("\nConverted to BIOES:")
        for token, iob2, bioes in zip(tokens, iob2_tags, bioes_tags):
            print(f"  {token:<15} {iob2:<10} → {bioes:<10}")

        # 둘 다 확인하기
        print("\nValidation:")
        valid_iob2, msg_iob2 = TagValidator.validate_iob2(iob2_tags)
        valid_bioes, msg_bioes = TagValidator.validate_bioes(bioes_tags)
        print(f"  IOB2: {msg_iob2}")
        print(f"  BIOES: {msg_bioes}")

        # 것 뽑기
        entities = TagConverter.tags_to_entities(tokens, iob2_tags)
        print("\nExtracted entities:")
        for text, etype, start, end in entities:
            print(f"  - '{text}' ({etype})")


    if __name__ == "__main__":
        # 시연 실행
        demonstrate_tagging_schemes()

        # 주고받는 바꾸개
        interactive_tag_converter()

        print("\n" + "="*70)
        print("Demonstration complete!")
        print("="*70)
        print("\nKey takeaways:")
        print("1. IOB2 uses B- and I- prefixes, simpler but less explicit")
        print("2. BIOES adds E- and S- for explicit boundaries")
        print("3. Both schemes can represent the same entities")
        print("4. BIOES often performs better in deep learning models")
        print("5. Always validate tag sequences for consistency")
    ```

??? note "기존 전체 출력"

    ```
    ======================================================================
    IOB and BIOES Tagging Schemes Demonstration
    ======================================================================

    1. IOB2 Tagging Scheme
    ----------------------------------------------------------------------
    Text: Steve Jobs founded Apple Inc. in Cupertino

    Tokens and IOB2 tags:
    Token           IOB2 Tag  
    ------------------------------
    Steve           B-PER     
    Jobs            I-PER     
    founded         O         
    Apple           B-ORG     
    Inc             I-ORG     
    .               O         
    in              O         
    Cupertino       B-LOC     

    Validation: Valid IOB2 sequence


    2. BIOES Tagging Scheme
    ----------------------------------------------------------------------
    Token           IOB2       BIOES     
    ----------------------------------------
    Steve           B-PER      B-PER     
    Jobs            I-PER      E-PER     
    founded         O          O         
    Apple           B-ORG      B-ORG     
    Inc             I-ORG      E-ORG     
    .               O          O         
    in              O          O         
    Cupertino       B-LOC      S-LOC     

    Validation: Valid BIOES sequence


    3. Comparing IOB2 and BIOES
    ----------------------------------------------------------------------

    Example 1: IBM
    Token                IOB2       BIOES     
    ----------------------------------------
    IBM                  B-ORG      S-ORG     
    Note: Single-token entity: B-ORG vs S-ORG

    Example 2: New York
    Token                IOB2       BIOES     
    ----------------------------------------
    New                  B-LOC      B-LOC     
    York                 I-LOC      E-LOC     
    Note: Two-token entity: B-I vs B-E

    Example 3: University of California
    Token                IOB2       BIOES     
    ----------------------------------------
    University           B-ORG      B-ORG     
    of                   I-ORG      I-ORG     
    California           I-ORG      E-ORG     
    Note: Multi-token entity: last I vs E


    4. Extracting Entities from Tags
    ----------------------------------------------------------------------
    Tokens: Barack Obama visited New York City and Microsoft

    Tags: B-PER I-PER O B-LOC I-LOC I-LOC O B-ORG

    Extracted entities:
      - 'Barack Obama' (PER) at tokens [0:2]
      - 'New York City' (LOC) at tokens [3:6]
      - 'Microsoft' (ORG) at tokens [7:8]


    5. Detecting Invalid Tag Sequences
    ----------------------------------------------------------------------

    Invalid Example 1:
    Tags: ['I-PER', 'I-PER', 'O']
    Valid: False
    Issue: Starts with I- instead of B-
    Error: I-PER cannot be first tag

    Invalid Example 2:
    Tags: ['B-PER', 'I-LOC', 'O']
    Valid: False
    Issue: Entity type mismatch (PER→LOC)
    Error: I-LOC at position 1 follows B-PER

    Invalid Example 3:
    Tags: ['B-PER', 'O', 'I-PER']
    Valid: False
    Issue: I-PER after O (should be B-PER)
    Error: I-PER at position 2 follows O


    6. When to Use Each Scheme
    ----------------------------------------------------------------------

    IOB2 (Inside-Outside-Beginning):
      Advantages:
        - Simpler: Only 3 tag types (B, I, O)
        - More compact representation
        - Widely used in research
      Disadvantages:
        - Cannot explicitly mark entity boundaries
        - Harder for model to learn where entities end

    BIOES (Beginning-Inside-Outside-End-Single):
      Advantages:
        - Explicit entity boundaries (E tag)
        - Distinguishes single-token entities (S tag)
        - Better for models: clearer structure
        - Potentially better performance
      Disadvantages:
        - More complex: 5 tag types
        - Larger label space
        - More training data needed


    7. Practical Conversion Example
    ----------------------------------------------------------------------
    Sentence: Microsoft CEO Satya Nadella spoke at Stanford

    Original IOB2 tags:
      Microsoft       B-ORG
      CEO             O
      Satya           B-PER
      Nadella         I-PER
      spoke           O
      at              O
      Stanford        B-ORG

    Converted to BIOES:
      Microsoft       S-ORG
      CEO             O
      Satya           B-PER
      Nadella         E-PER
      spoke           O
      at              O
      Stanford        S-ORG

    Extracted entities (both schemes give same result):
      - 'Microsoft' (ORG)
      - 'Satya Nadella' (PER)
      - 'Stanford' (ORG)


    ======================================================================
    Interactive Tag Converter
    ======================================================================

    Practice Example:
    Tokens: ['Barack', 'Obama', 'visited', 'Google']
    IOB2 tags: ['B-PER', 'I-PER', 'O', 'B-ORG']

    Converted to BIOES:
      Barack          B-PER      → B-PER     
      Obama           I-PER      → E-PER     
      visited         O          → O         
      Google          B-ORG      → S-ORG     

    Validation:
      IOB2: Valid IOB2 sequence
      BIOES: Valid BIOES sequence

    Extracted entities:
      - 'Barack Obama' (PER)
      - 'Google' (ORG)

    ======================================================================
    Demonstration complete!
    ======================================================================

    Key takeaways:
    1. IOB2 uses B- and I- prefixes, simpler but less explicit
    2. BIOES adds E- and S- for explicit boundaries
    3. Both schemes can represent the same entities
    4. BIOES often performs better in deep learning models
    5. Always validate tag sequences for consistency
    ```

## 2. 레이블 체계와 경계 정보

IOB2에서 개체의 첫 토큰은 B-X, 이어지는 토큰은 I-X, 바깥은 O다. BIOES는 단일 토큰을 S-X, 여러 토큰의 마지막을 E-X로 더 세분화한다. IBM은 B-ORG↔S-ORG, New York은 B-LOC I-LOC↔B-LOC E-LOC다. 같은 유효 span을 나타내므로 IOB2가 개체 경계를 표현할 수 없다는 기존 출력 문구는 정확하지 않다. 종료 표지가 명시적인지의 차이다.

TagScheme.IOB라는 enum은 선언되어 있지만 검사·변환·추출의 일반 IOB1 지원은 없다. IOB1의 첫 I-X가 가능한 규칙과 IOB2를 혼동하지 않는다. 유형 K개에서 IOB2는2K+1, BIOES는4K+1개 레이블을 사용한다. BIOES가 반드시 더 많은 자료를 요구하거나 항상 더 높은 성능을 주는 것은 아니다.

## 3. 검사와 변환의 전제

validate_iob2는 초기 I·O 뒤 I·유형 불일치 I를 검사한다. BIOES는 시작·종료와 같은 유형의 B/I→I/E 연결을 검사한다. 다만 빈 유형 `B-`를 허용하고 등록된 유형 목록을 검증하지 않는다. `split('-')`로 나누므로 유형 이름 자체에 하이픈이 들어가면 형식 오류로 처리한다. 빈 열은 현재 유효하다고 반환하며 실제 학습 모델이 빈 열을 처리하는지는 별도 문제다.

변환기는 유효한 입력을 가정한다. IOB2→BIOES는 뒤의 같은 유형 I가 이어지는지 확인해 B→S 또는 I→E를 정한다. BIOES→IOB2는 S→B·E→I로 바꾼다. 미지원 prefix를 자동으로 올바른 개체로 바꾸는 것은 아니며 변환 중 일부 태그가 누락될 수도 있다. 먼저 체계·형식·유형을 검사하고 변환 후 길이·span이 보존되는지 확인해야 한다.

## 4. 추출기의 관대한 처리와 offset

IOB2 추출기는 시작 I나 다른 유형 I를 만나면 새 개체로 여는 복구 폴리시를 사용한다. 같은 태그를 validator는 무효로 판정할 수 있어 검사와 추출을 같은 의미로 취급하면 안 된다. BIOES에서도 닫히지 않은 B/I를 끝에서 개체로 내보낼 수 있으므로 추출 성공이 유효한 태그 열의 증거는 아니다.

입력 tokens·tags를 zip하므로 길이가 다르면 짧은 쪽까지만 처리한다. 끝에서 열린 개체를 닫을 때는 len(tokens)를 사용해 실제 처리 범위와 어긋날 수도 있다. 반환 start·end는 토큰 단위 반열린 구간이며 문자 offset이 아니다. 공백으로 합친 문자열도 원문의 공백·구두점을 그대로 복원하지 않는다. scheme=IOB에서는 개체를 여는 분기가 없어 결과가 비어 버릴 수 있다.

## 5. 시연과 실제 평가의 구분

시연은 수동 태그의 검사·변환 결과를 표시한다. `interactive_tag_converter`도 사용자 입력을 받는 인터페이스가 아니라 고정된 예제를 출력하는 함수다. 두 체계의 추출 결과를 각각 계산하지만 마지막 표시는 IOB2 결과만 반복하며 둘의 동일성을 assert하지 않는다. 출력 메시지만으로 일반적인 변환 보존성이 검증되었다고 해석하지 않는다.

IOB2·BIOES의 실제 모델 성능 비교에는 같은 데이터·분할·토큰화·인코더·제약·span 평가와 여러 초기값이 필요하다. 이 파일에는 모델·로스·최적화가 없으므로 성능 향상·경사 안정성·과적합을 실험한 코드라고 설명하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
IOB2의 개체 종료를 알아낼 수 없는가? BIOES와의 차이를 설명하라.

</div>

??? success "풀이"

    유효한 IOB2에서 다음 O·새 B·문장 끝으로 현재 개체의 끝을 알 수 있다. BIOES는 E·S로 종료·단일 개체를 명시한다. 두 체계는 같은 평면 연속 span을 표현하며 종료 정보가 없는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
초기 I-PER을 validator와 tags_to_entities가 어떻게 다르게 다루는가?

</div>

??? success "풀이"

    validator는 IOB2에서 무효로 판정하지만 추출기는 새 PER 개체를 여는 복구를 할 수 있다. 엄격 평가와 오류 교정 평가의 폴리시를 명시하고 변환 전에 검사해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
서로 다른 길이의 tokens·tags와 scheme=IOB의 한계를 설명하라.

</div>

??? success "풀이"

    zip이 남은 항목을 버리고 최종 end는 len(tokens)로 정해 잘못된 범위가 될 수 있다. IOB enum은 선언만 되어 일반 IOB1 추출 분기가 없다. 길이·체계의 지원 여부를 먼저 검증해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 TagScheme 시험 조각과 달리 이 문자열 처리 예제에 맞는 확인 항목을 제안하라.

</div>

??? success "풀이"

    TagScheme은 enum이며 뉴럴 넷트웍이 아니므로 forward·parameters·backward 검사는 맞지 않는다. 아래 미정의 입력도 실행 가능한 시험이 아니다. 유효 변환의 왕복·길이·span 보존, 인접 동일 유형·빈 열·무효 I·잘못된 prefix·하이픈 유형과 불일치 길이를 명세로 확인한다.

    ```python
    def test_tagscheme():
        model = TagScheme(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

IOB2와 BIOES는 같은 평면 span을 다른 경계 표지로 표현한다. 이 예제의 validator·변환기·추출기는 입력 전제와 오류 처리 폴리시가 다르다. 지원 체계·토큰 길이·유형·offset을 확인하고 출력 시연을 일반 성능 검증과 구분한다.
