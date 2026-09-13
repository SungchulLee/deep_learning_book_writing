"""23장 익히기 예제가 쓰는 모델 꾸러미.

모델의 뜻매김 자체는 23.1 절의 `자기 부호기 모듈` 페이지에 있고, 여기서는
익히기 스크립트가 `from models.autoencoder import SimpleAutoencoder`처럼
들여올 수 있게 모듈로 묶어 둔다.
"""

from .autoencoder import SimpleAutoencoder

__all__ = ["SimpleAutoencoder"]
