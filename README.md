<div align="center">

# ❄️ Snowflake Arctic Embed v2.0 ONNX Quantized

**불경 온디바이스 검색 엔진** — Snowflake Arctic Embed v2.0 모바일 최적화 ONNX 모델

[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com)

</div>

---

## 🌟 개요

Snowflake Arctic Embed M v2.0(768차원)을 모바일 온디바이스 불경 검색에 맞게 최적화한 ONNX 모델입니다. 1.23GB → **188MB(85% 압축)**를 달성하여 Android 앱에 내장합니다.

## 🛠 최적화 내역

| 항목 | 상세 | 비고 |
|------|------|------|
| **원본 모델** | Snowflake/snowflake-arctic-embed-m-v2.0 | 768차원 |
| **어휘 프루닝** | 250,048 → 99,435 토큰 | 한/영/산스크리트 보존 |
| **양자화** | Dynamic INT8 | 속도·배터리 효율 향상 |
| **최종 용량** | 1.23GB → **188MB** | **85% 압축** |

## 🔍 핵심 기술 상세

### 어휘 가지치기 (Vocabulary Pruning)
한국어, 영어, 산스크리트어, 팔리어에 필요한 99,435개 토큰만 남기고 불필요한 어휘를 제거합니다. 임베딩 행렬 크기가 줄어 모델 용량이 대폭 감소합니다.

### Matryoshka 임베딩
768차원의 앞부분 N차원만으로도 의미 있는 표현 구성. 모바일 환경에서 차원을 줄여 검색 속도와 메모리 사용량을 조절합니다.

### 비대칭 검색 최적화
검색 쿼리에 전용 prefix를 자동 추가하여 쿼리-문서 비대칭 검색 성능을 향상시킵니다.

### Android 온디바이스 통합
서버 없이 Android 앱 내에서 직접 ONNX Runtime으로 임베딩을 생성하여 프라이버시와 레이턴시를 동시에 확보합니다.