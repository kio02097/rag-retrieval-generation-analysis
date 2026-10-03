# RAG 검색 성능과 생성 성능의 관계 분석

**검색을 잘하는 RAG가 답변도 잘 생성할까?** BioASQ 생의학 질의응답에서 임베딩 모델, 풀링 방식, 검색 문서 수에 따른 검색·생성 성능의 관계를 비교한 연구 자료입니다.

박지원 · 양치복의 「RAG 프레임워크에서 검색과 생성 성능의 관계 분석」 수정 원고와 실험 노트북을 바탕으로 정리했습니다. 제공된 원고의 권·호·발행일은 자리표시자이므로 게재 완료 논문으로 표기하지 않습니다.

## 핵심 결과

- 검색 단계의 최적 설정이 생성 단계의 최적 설정과 항상 일치하지 않았습니다.
- 질문 유형에 따라 유리한 모델, 풀링 방식, Top-k가 달랐습니다.
- 원고에서 보고한 생성 성능 최고 조합은 factoid에서 **BGE / Mean / k=20**, list에서 **Solon / Mean / k=10**, yesno에서 **S-BERT / CLS / k=5**였습니다.

| 질문 유형 | 검색 최고 조합 | 원고 표 4 검색 점수¹ | 생성 최고 조합 | 생성 지표 | 생성 점수 |
|---|---|---:|---|---|---:|
| factoid | BGE · CLS · 20 | 0.6304² | BGE · Mean · 20 | MRR | 0.1131 |
| list | Solon · Mean · 20 | 0.6377 | Solon · Mean · 10 | F1 | 0.1975 |
| yesno | Solon · Mean · 20 | 0.6356 | S-BERT · CLS · 5 | Macro-F1 | 0.7698 |

¹ 표 4의 `score` 열을 전사했습니다. 해당 열만으로 Recall/nDCG 또는 집계 점수 여부를 확정할 수 없어 특정 검색 지표 이름을 붙이지 않았습니다. 검색과 생성의 점수, 서로 다른 질문 유형의 생성 점수는 동일한 척도로 비교하면 안 됩니다.

² 같은 페이지 본문에는 0.6390으로 표기되어 있습니다. 이 저장소는 표의 0.6304를 사용하며, 수치 불일치를 임의로 수정하지 않았습니다. 이 결과는 원고에 보고된 값이며 이번 정리 과정에서 실험을 재실행한 결과가 아닙니다.

## 실험 설계

| 항목 | 원고의 설정 |
|---|---|
| 데이터 | BioASQ4 (2016), PubMed 제목·초록 |
| 평가 질문 | 1,022개: factoid 327 / list 327 / yesno 368 |
| 문헌 | 정제 후 13,007개 |
| 임베딩 모델 | all-MiniLM-L6-v2, BGE-M3, Solon-embeddings-large-0.1, embeddinggemma-300m |
| 풀링 | CLS, Mean, Max |
| 검색 Top-k | 5, 10, 20 |
| 청크 크기 / 중첩 | 500 / 100 |
| 생성 모델 | Ollama `gemma3:4b` |
| 문맥 제한 | 문서당 1,000자, 전체 40,000자 |
| 검색 평가 | Recall@k, nDCG@k |

```mermaid
flowchart LR
    A[BioASQ 질문과 PMID] --> B[PubMed 제목·초록]
    B --> C[청크 분할 500 / 100]
    C --> D[임베딩 모델 + 풀링]
    D --> E[Chroma 검색 Top-k]
    A --> E
    E --> F[검색 평가]
    E --> G[Gemma 3 4B 답변 생성]
    G --> H[질문 유형별 생성 평가]
```

## 저장소 구성

- [논문 상세 요약](docs/paper-summary.md): 연구 질문, 방법, 결과, 한계
- [재현 안내 및 확인 사항](docs/reproducibility.md): 노트북 실행 범위와 원고·코드 차이
- [임베딩 실험 노트북](notebooks/rag_embeddings.ipynb): `rag_논문.ipynb`의 정리본
- [RAG 평가 노트북](notebooks/rag_evaluation.ipynb): 데이터 전처리, 검색·생성, 결과 집계
- [생성 프롬프트](prompt/rag_prompt2.yaml): 질문 유형에 따른 JSON 답변 형식
- [데이터 안내](data/README.md): 로컬 데이터 구성과 준비 방법
- [원고 표 4 결과](results/paper_table4.csv): 원고에서 옮긴 요약 수치

## 실행 안내

평가 셀은 `rag_pipeline/` 패키지로도 분리했습니다. `embeddings.py`는 임베딩·풀링, `prompts.py`는 프롬프트 로딩, `evaluator.py`는 평가, `__main__.py`는 실행 인자를 담당합니다. 기존 Chroma 인덱스를 사용하는 평가 진입점이며 인덱스 구축은 아래 노트북 흐름을 사용합니다.

```bash
pip install -r requirements.txt
python -m rag_pipeline --help
python -m rag_pipeline --dataset data/bioasq_factoid.csv --chroma-dir 500-100/embeddinggemma/chroma_db --collections gemma_C gemma_M gemma_X --k 5 10 20 --dry-run
```

입력 경로를 확인한 후 `--dry-run`을 제거하면 실제 평가를 실행합니다. 실행 설정은 출력 폴더에 JSON으로 남습니다. `--dry-run`은 파일·CSV 열·인자만 검사하며 모델 호환성이나 Ollama 연결을 확인하지 않습니다. 핵심 평가 계산은 원본을 유지했습니다. 의존성 버전 고정과 전체 실험 재현은 검증되지 않았습니다.

빠른 입력 검증 테스트: `python -m unittest rag_pipeline.test_cli -v`

두 노트북은 **선택 실행용 연구 기록**입니다. 서로 다른 모델 실험, Colab 명령, 중간 점검 셀이 함께 있으므로 `Run All`로 완전 재현되는 패키지는 아닙니다.

1. [데이터 안내](data/README.md)에 따라 BioASQ JSON 및 PubMed CSV를 로컬에 준비합니다.
2. Jupyter 환경에 노트북의 설치 셀에 명시된 패키지를 준비합니다. 원래 실험의 정확한 버전 잠금 파일은 제공되지 않았습니다.
3. 필요한 경우 `HF_TOKEN`, PubMed 수집 시 `NCBI_EMAIL` 환경변수를 설정합니다. 인증정보를 노트북에 직접 저장하지 마세요.
4. 저장소 루트에서 Jupyter를 시작합니다. 첫 설정 셀은 노트북 폴더에서 시작한 경우에도 루트를 찾습니다.
5. 임베딩 노트북에서 모델과 풀링을 선택하고 컬렉션을 구축합니다. 평가 노트북의 `CHROMA_DIR`, `collection_list`를 그 컬렉션과 일치시킵니다.
6. 로컬 Ollama에 `gemma3:4b`를 준비하고, 평가·집계 셀을 선택 실행합니다.

세부 실행 순서와 알려진 제한은 [재현 안내](docs/reproducibility.md)를 참고하세요.

## 공개 범위

코드, 프롬프트, 논문 요약과 원고의 집계 수치를 포함합니다. 노트북 출력·실행 메타데이터·하드코딩된 인증정보는 제거했습니다. 대용량 Chroma DB, 원문 데이터, 실험별 원시 응답, 논문 PDF 원본은 이 공개본에 포함하지 않습니다. 원본 파일은 별도로 보존되어 있습니다.

코드에 대한 별도 공개 라이선스는 아직 지정하지 않았습니다. 데이터와 모델의 이용 조건은 각 제공처의 조건을 따릅니다.
