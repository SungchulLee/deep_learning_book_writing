# IOB 이름표 붙이기

이름 알아보기의 IOB와 BIOES 이름표 방식. 이 단원은 여러 이름표 방식을 자세히 밝히고 짠다.

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

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

지은이: 배움 목적
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
        3. B-TYPE 뒤에는 (같은 갈래의) I-TYPE이나 O나 B-*만 올 수 있다
        4. I-TYPE 뒤에는 (같은 갈래의) I-TYPE이나 O나 B-*만 올 수 있다
        5. 같은 갈래의 B-TYPE 없이 I-TYPE이 올 수 없다
        
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
            
            # I-TYPE 뒤에 같은 갈래의 B-TYPE이나 I-TYPE이 오는지 살피기
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
        3. I-TYPE이 것을 이어 간다(같은 갈래의 B나 I 뒤에 와야 한다)
        4. E-TYPE이 것을 끝낸다(같은 갈래의 B나 I 뒤에 와야 한다)
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
            
            # I나 E 앞에는 같은 갈래의 B나 I가 와야 한다
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
            
            # B나 I 뒤에는 같은 갈래의 I나 E가 와야 한다(또는 차례의 끝)
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

## 2. 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 차례 이름표 붙이기에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
IOB 이름표 붙이기의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.

## 정리하며

**다룬 것** — IOB 이름표 붙이기

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

고갱이 갈래는 `TagScheme`, `TagValidator`, `TagConverter`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
