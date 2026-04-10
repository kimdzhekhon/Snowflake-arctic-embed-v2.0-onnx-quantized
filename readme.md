# ☸️ 검색 엔진 최적화 프로젝트 (Buddhist Search Engine)

본 프로젝트는 **Snowflake Arctic Embed v2.0** 모델을 활용하여 데이터를 모바일 기기 내에서 실시간으로 검색할 수 있도록 최적화하고, 온디바이스(On-device) 검색 엔진을 구축하는 프로젝트입니다.

---

## 💎 1. 모델 최적화 결과 (Model Optimization)

기존 1.2GB의 대형 모델을 모바일 환경에 맞게 경량화하여 성능은 유지하되 용량은 획기적으로 줄였습니다.

| 항목 | 상세 내용 | 비고 |
| :--- | :--- | :--- |
| **원본 모델** | `Snowflake/snowflake-arctic-embed-m-v2.0` | 768차원 |
| **어휘 가지치기** | 250,048개 → **99,435개** 토큰 | 한/중/영/산스크리트 보존 |
| **양자화** | **Dynamic Int8** 적용 | 연산 속도 및 배터리 효율 향상 |
| **최종 용량** | 1.23 GB → **188 MB** | **85% 압축 성공** |

---

## 📱 2. Android 앱 구현 및 배포

최적화된 ONNX 모델을 Android 앱 엔진에 탑재하여 서버 없는 로컬 검색을 구현했습니다.

- **엔진 교체**: 기존 E5-small(384차원)에서 최신 Snowflake(768차원) 모델로 업그레이드하여 검색 품질 향상.
- **비대칭 검색 최적화**: 검색 쿼리 시 전용 접두사(Prefix) 자동 추가 로직 구현.

---

## 🎡 3. 데이터 임베딩 전략 (Matryoshka Embedding)

Snowflake v2.0의 핵심 기능인 **마트료시카 임베딩(MRL)**을 적용하여 효율적인 데이터 구조를 설계했습니다.

- **이중 임베딩 추출**: 768차원(고정밀)과 256차원(고성능) 벡터를 동시에 추출하여 CSV 저장.
- **로컬 적재 전략**: 정제된 데이터를 Supabase의 `tripitaka_documents` 테이블에 Bulk Insert 하거나 기기 로컬 DB에 적재하여 활용.

---

## 🛠️ 4. 기술 스택 (Tech Stack)

- **Model Framework**: ONNX Runtime, Hugging Face Hub
- **Optimization**: Python, Transformers, Optimum
- **Database**: Supabase (pgvector), SQLite
- **Mobile**: Flutter (Android)

---

## 📂 관련 리소스 (Links)

- **Hugging Face Model**: [jaehyun-kim/snowflake-arctic-embed-v2.0-onnx-quantized](https://huggingface.co/jaehyun-kim/snowflake-arctic-embed-v2.0-onnx-quantized)
- **Base Model**: [Snowflake Arctic Embed v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0)
