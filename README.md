<div align="center">

# Snowflake Arctic Embed M v2.0 — ONNX INT8 Quantized

온디바이스 불경 검색을 위한 경량화 임베딩 모델 (768차원, 188MB)

![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow?style=for-the-badge)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green?style=for-the-badge)
![iOS](https://img.shields.io/badge/iOS-On--Device-000000?style=for-the-badge&logo=apple&logoColor=white)
![Android](https://img.shields.io/badge/Android-On--Device-3DDC84?style=for-the-badge&logo=android&logoColor=white)

![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Compression](https://img.shields.io/badge/compression-85%25-orange?style=flat-square)

**[HuggingFace 모델 보기 →](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)**

</div>

---

## 목차

1. [소개](#소개)
2. [주요 기능](#주요-기능)
3. [기술 스택 / 최적화 내역](#기술-스택--최적화-내역)
4. [아키텍처 / 구현 원리](#아키텍처--구현-원리)
5. [데이터 흐름](#데이터-흐름)
6. [설치 및 사용](#설치-및-사용)
7. [Flutter 통합](#flutter-통합)
8. [Roadmap](#roadmap)
9. [라이선스](#라이선스)

---

## 소개

Snowflake Arctic Embed M v2.0을 iOS·Android 온디바이스 불경(팔만대장경) 검색에 최적화한 ONNX INT8 양자화 모델입니다. 원본 1.23GB FP32 PyTorch 모델을 어휘 프루닝과 Dynamic INT8 양자화를 통해 188MB로 압축하여 서버 없이 iOS·Android 기기에서 직접 실행할 수 있습니다. 768차원 Matryoshka 임베딩을 지원하므로 사용 목적에 따라 앞 N차원만 잘라 속도와 메모리를 추가로 절감할 수 있습니다. 비대칭 검색 최적화가 적용되어 쿼리 프리픽스가 자동으로 부가됩니다.

> 이 모델은 팔만대장경 한국어·영어·산스크리트 텍스트를 모두 아우르는 어휘를 보존하면서 85%의 용량 절감을 달성하였습니다.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| 온디바이스 추론 | 서버 없이 iOS·Android에서 ONNX Runtime으로 직접 실행 |
| Matryoshka 임베딩 | 768차원 전체 또는 앞 N차원만 사용하여 유연한 속도·정확도 트레이드오프 |
| 비대칭 검색 최적화 | 쿼리 프리픽스 자동 삽입으로 문서-쿼리 비대칭 검색 지원 |
| 어휘 프루닝 | 한국어·영어·산스크리트 관련 토큰만 보존하여 모델 크기 축소 |
| Dynamic INT8 양자화 | 칼리브레이션 데이터 없이 런타임에 양자화 수행 |
| 크로스플랫폼 ONNX | ONNX Runtime 지원 플랫폼 어디에서나 실행 가능 |

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 기술 스택 / 최적화 내역

| 항목 | 내용 |
|---|---|
| 베이스 모델 | Snowflake Arctic Embed M v2.0 |
| 원본 크기 | 1.23GB (FP32 PyTorch) |
| 최종 크기 | 188MB (85% 압축) |
| 양자화 방식 | Dynamic INT8 (onnxruntime.quantization) |
| 변환 도구 | Hugging Face optimum 라이브러리 |
| 어휘 크기 변화 | 250,048 → 99,435 토큰 |
| 보존 어휘 | 한국어 / 영어 / 산스크리트 |
| 임베딩 차원 | 768d (Matryoshka, 앞 N차원 사용 가능) |
| 추론 런타임 | ONNX Runtime (iOS·Android) |

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 아키텍처 / 구현 원리

```
[Snowflake Arctic Embed M v2.0 아키텍처]

입력 텍스트
    │
    ▼
┌─────────────────────────────────────┐
│  토크나이저 (어휘 99,435 토큰)       │
│  - 한/영/산스크리트 어휘 보존         │
│  - 비대칭 쿼리 프리픽스 자동 삽입     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ONNX 트랜스포머 인코더 (INT8)       │
│  - Dynamic INT8 양자화 가중치         │
│  - 멀티헤드 어텐션                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Matryoshka 임베딩 출력 (768d)       │
│  - 앞 N차원만 사용 가능               │
│  - L2 정규화                          │
└─────────────────────────────────────┘
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 데이터 흐름

```
원본 PyTorch 모델 (1.23GB, 250k 어휘)
      │
      ▼
어휘 프루닝 (한/영/산스크리트 99,435토큰 보존)
      │
      ▼
ONNX 변환 (optimum 라이브러리)
      │
      ▼
Dynamic INT8 양자화 (onnxruntime quantization)
      │
      ▼
최종 모델 188MB → iOS·Android 기기 배포
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 설치 및 사용

**요구사항**

- Python 3.8+
- `onnxruntime >= 1.16`
- `transformers >= 4.35`
- `optimum[onnxruntime]`

```bash
# HuggingFace Hub에서 모델 다운로드
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Snowflake/snowflake-arctic-embed-m-v2.0",
    local_dir="./model"
)
EOF
```

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
session = ort.InferenceSession("./model/model.onnx")

query = "Represent this sentence for searching relevant passages: 반야바라밀다심경"
inputs = tokenizer(query, return_tensors="np", padding=True, truncation=True, max_length=512)

outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
})

embedding = outputs[0][0]           # 768차원 전체
embedding_256 = embedding[:256]     # 앞 256차원만 사용
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## Flutter 통합

Flutter iOS·Android 앱에서 ONNX Runtime으로 온디바이스 추론하는 방법입니다.

```yaml
# pubspec.yaml
dependencies:
  onnxruntime: ^1.x.x
```

```dart
import 'package:onnxruntime/onnxruntime.dart';

// 모델 로드
final session = OrtSession.fromFile(
  'assets/model.onnx',
  OrtSessionOptions(),
);

// 토큰화된 입력 준비
final List<int> tokenIds = /* 토크나이저 출력 */;
final int seqLen = tokenIds.length;

final inputs = {
  'input_ids': OrtValueTensor.createTensorWithDataList(
    tokenIds,
    [1, seqLen],
  ),
};

// 추론 실행
final outputs = session.run(null, inputs);
final embedding = outputs[0]?.value as List<List<double>>;

// Matryoshka: 앞 N차원만 사용
final embedding256 = embedding[0].sublist(0, 256);
```

> iOS·Android 모두 `assets/` 폴더에 `model.onnx`와 토크나이저 파일을 포함시키고, `pubspec.yaml`의 `flutter.assets`에 경로를 등록하세요.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## Roadmap

- [x] 어휘 프루닝 (250k → 99k 토큰)
- [x] Dynamic INT8 양자화
- [x] ONNX 변환
- [x] iOS·Android 온디바이스 통합
- [x] Matryoshka 임베딩 지원
- [ ] iOS CoreML 변환 버전 추가
- [ ] Static INT8 양자화 버전 비교
- [ ] 벤치마크 문서 (속도·정확도)
- [ ] FP16 버전 추가

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 라이선스

MIT License

Copyright (c) 2026 kimdzhekhon

본 저장소의 최적화 코드 및 변환 스크립트는 MIT 라이선스로 배포됩니다. 베이스 모델(Snowflake Arctic Embed M v2.0)의 라이선스는 [HuggingFace 모델 카드](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)를 확인하세요.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

