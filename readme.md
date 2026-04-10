# Buddhist Scripture Search Engine Optimization

이 레포지토리는 Snowflake Arctic Embed v2.0 모델을 불교 경전 검색 엔진에 최적화하여 온디바이스용으로 변환한 프로젝트의 소스 코드를 담고 있습니다.

## 📁 주요 내용
- **Model Optimization**: 어휘 가지치기(Pruning) 및 Int8 양자화 적용 코드
- **Embedding Strategy**: 마트료시카 임베딩(MRL)을 활용한 데이터 추출 스크립트
- **Deployment**: Android 앱용 ONNX 모델 배포 정보

## 🚀 Hugging Face Model
최적화된 모델 파일은 아래 주소에서 확인하실 수 있습니다:
[jaehyun-kim/snowflake-arctic-embed-v2.0-onnx-quantized](https://huggingface.co/jaehyun-kim/snowflake-arctic-embed-v2.0-onnx-quantized)

## 🛠️ 최적화 결과
- **용량**: 1.23GB -> 188MB (85% 압축)
- **포맷**: ONNX (Int8 Dynamic Quantization)
