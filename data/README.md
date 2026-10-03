# 데이터 준비

원문 데이터와 벡터 DB는 이 저장소에 포함하지 않습니다. BioASQ 데이터는 제공처에서 이용 조건을 확인한 후 준비하세요. 원본 데이터 안내의 인용 정보는 [SOURCE_README.txt](SOURCE_README.txt)에 보존했습니다.

| 로컬 경로 | 내용 |
|---|---|
| `data/BioASQ-trainingDataset4b.json` | BioASQ 질문 및 정답 |
| `data/bioasq_factoid.csv` | factoid 질문 |
| `data/bioasq_list.csv` | list 질문 |
| `data/bioasq_yesno.csv` | yesno 질문 |
| `pubmed_title_abstract.csv` | PubMed 문헌, 저장소 루트에 배치 |

제공된 PubMed CSV는 13,007행이며 `pmid`, `title`, `abstract`, `text` 열을 갖습니다. 이는 로컬 파일을 검사한 결과입니다. 질문 CSV는 평가 노트북의 전처리 셀에서 생성할 수 있습니다.

벡터 DB는 임베딩 노트북으로 재구축하거나 기존 로컬 DB 경로를 지정합니다. 데이터의 다운로드 시점과 정제 규칙이 달라지면 원고 결과와 달라질 수 있습니다.
