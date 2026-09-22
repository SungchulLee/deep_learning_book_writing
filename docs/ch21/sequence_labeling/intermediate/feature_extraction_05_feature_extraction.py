"""feature_extraction_05_feature_extraction — ch20/sequence_labeling/05_feature_extraction.md 의 코드를
모듈로 쓸 수 있게 옮겨 놓은 것이다.
"""

"""
예로부터의 이름 알아보기를 위한 특징 뽑기
=======================================

예로부터의 기계 배움 바탕 이름 알아보기(CRF, MaxEnt 등)를 위한 특징 빚기.

뽑은 특징:
- 낱말 수준 특징
- 글자 수준 특징
- 철자 특징
- 맥락 특징
- 지명록 특징

지은이: 배움 목적
날짜: 2025
"""

from typing import List, Dict, Set
import re

# ========================================================================
# 메인
# ========================================================================


class FeatureExtractor:
    """예로부터의 기계 배움 바탕 이름 알아보기를 위한 특징 뽑기."""
    
    def __init__(self):
        """특징 뽑개 첫자리매김."""
        self.gazetteers = {}
    
    def extract_token_features(self, tokens: List[str], index: int, 
                              window_size: int = 2) -> Dict:
        """
        토막의 특징을 두루 뽑기.
        
        인수:
            tokens: 월 속 토막의 목록
            index: 지금 토막의 번호
            window_size: 맥락 창의 크기
            
        반환값:
            특징의 사전
        """
        token = tokens[index]
        features = {}
        
        # 기본 낱말 특징
        features['word'] = token.lower()
        features['word_length'] = len(token)
        
        # 철자 특징
        features['is_capitalized'] = token[0].isupper() if token else False
        features['is_all_caps'] = token.isupper()
        features['is_all_lower'] = token.islower()
        features['is_title'] = token.istitle()
        features['is_alphanumeric'] = token.isalnum()
        features['is_alpha'] = token.isalpha()
        features['is_digit'] = token.isdigit()
        
        # 낱말 꼴 특징
        features['word_shape'] = self.get_word_shape(token)
        features['short_word_shape'] = self.get_word_shape(token, short=True)
        
        # 앞가지와 뒷가지 특징
        for n in range(1, min(5, len(token) + 1)):
            features[f'prefix_{n}'] = token[:n].lower()
            features[f'suffix_{n}'] = token[-n:].lower()
        
        # 글자 수준 특징
        features['contains_hyphen'] = '-' in token
        features['contains_digit'] = any(c.isdigit() for c in token)
        features['contains_upper'] = any(c.isupper() for c in token)
        
        # 맥락 특징(앞 토막들)
        for i in range(1, window_size + 1):
            if index - i >= 0:
                prev_token = tokens[index - i]
                features[f'prev_{i}_word'] = prev_token.lower()
                features[f'prev_{i}_is_cap'] = prev_token[0].isupper() if prev_token else False
        
        # 맥락 특징(다음 토막들)
        for i in range(1, window_size + 1):
            if index + i < len(tokens):
                next_token = tokens[index + i]
                features[f'next_{i}_word'] = next_token.lower()
                features[f'next_{i}_is_cap'] = next_token[0].isupper() if next_token else False
        
        # 자리 특징
        features['is_first'] = (index == 0)
        features['is_last'] = (index == len(tokens) - 1)
        
        return features
    
    @staticmethod
    def get_word_shape(word: str, short: bool = False) -> str:
        """
        낱말 꼴 나타냄 얻기.
        
        글자를 꼴 부호에 대응시킨다:
        - 대문자: 'X'
        - 소문자: 'x'
        - 숫자: 'd'
        - 그 밖: 'c'
        
        short=True이면 잇달아 같은 글자를 하나로 뭉갠다.
        
        보기:
            "iPhone5" -> "xXxxxxd"(긴 꼴) 또는 "xXxd"(짧은 꼴)
        """
        shape = []
        for char in word:
            if char.isupper():
                shape.append('X')
            elif char.islower():
                shape.append('x')
            elif char.isdigit():
                shape.append('d')
            else:
                shape.append('c')
        
        shape_str = ''.join(shape)
        
        if short:
            # 잇달아 같은 글자를 하나로 뭉개기
            if not shape_str:
                return shape_str
            compressed = [shape_str[0]]
            for char in shape_str[1:]:
                if char != compressed[-1]:
                    compressed.append(char)
            return ''.join(compressed)
        
        return shape_str


if __name__ == "__main__":
    # 예
    extractor = FeatureExtractor()
    tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", "."]
    
    for i, token in enumerate(tokens):
        features = extractor.extract_token_features(tokens, i)
        print(f"\nToken: {token}")
        print(f"Features: {len(features)} features extracted")
        print(f"Word shape: {features['word_shape']}")
        print(f"Is capitalized: {features['is_capitalized']}")
