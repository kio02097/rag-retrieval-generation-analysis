"""Original evaluation algorithms, isolated from notebook setup and execution.

Metric definitions are preserved for comparison with the research notebook.
See docs/reproducibility.md before interpreting these as official BioASQ scores.
"""
import os
import time
import ast
import json
import re
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from .prompts import load_prompt_text_from_yaml, split_system_user

class RAGEvaluator:
    def __init__(
        self,
        dataset_csv: str,
        collection_name: str,
        k: int,
        embedding_function: Embeddings,
        chroma_dir: str,
        llm_model: str = "gemma3:4b",
        prompt_path: str = "prompt/rag_prompt2.yaml",
        do_llm: bool = True,
        save_contexts: bool = True,
        results_dir: str = "outputs/results",
        summary_dir: str = "outputs/summary",
        max_doc_chars: int = 1000,
        max_total_chars: int = 40000,
        debug_save_first_n: int = 50,
        evidence_fallback: bool = True,
    ):
        self.max_doc_chars = max_doc_chars
        self.max_total_chars = max_total_chars
        self.debug_save_first_n = debug_save_first_n
        self.evidence_fallback = evidence_fallback
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        self.dataset_csv = dataset_csv
        self.collection_name = collection_name
        self.k = k
        self.embedding_function = embedding_function
        self.chroma_dir = chroma_dir
        self.llm_model = llm_model
        self.prompt_path = prompt_path
        self.do_llm = do_llm
        self.save_contexts = save_contexts

        dataset_name = os.path.splitext(os.path.basename(dataset_csv))[0]
        self.run_name = f"{dataset_name}__{self.collection_name}_k{self.k}"

        self.result_csv_path = os.path.join(results_dir, self.run_name + ".csv")
        self.eval_csv_path = os.path.join(summary_dir, self.run_name + "_retrieval.csv")
        self.debug_csv_path = os.path.join(summary_dir, self.run_name + "_debug_retrieval.csv")

        self.df_questions = None
        self.df_results = None
        self.df_eval = None

        self._init_vectorstore()
        if self.do_llm:
            self._init_llm_and_prompt()

    # ---------- utils ----------
    @staticmethod
    def _maybe_parse(x):
        if isinstance(x, str):
            s = x.strip()
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return x
        return x

    @staticmethod
    def _normalize(text):
        if text is None:
            return ""
        return unicodedata.normalize("NFC", str(text)).strip()

    @staticmethod
    def _unwrap_singleton(x, max_depth=3):
        # [[...]] 같은 “중첩 singleton list” 풀기
        for _ in range(max_depth):
            if isinstance(x, list) and len(x) == 1:
                x = x[0]
            else:
                break
        return x

    @staticmethod
    def _extract_pmid(value):
        if value is None:
            return None
        if isinstance(value, int):
            return str(value)
        if isinstance(value, dict):
            if "pmid" in value:
                return RAGEvaluator._extract_pmid(value["pmid"])
            if "document" in value:
                return RAGEvaluator._extract_pmid(value["document"])

        s = str(value).strip()
        if s.isdigit():
            return s

        patterns = [
            r"ncbi\.nlm\.nih\.gov/pubmed/(\d+)",
            r"www\.ncbi\.nlm\.nih\.gov/pubmed/(\d+)",
            r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)",
            r"/pubmed/(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                return m.group(1)

        m = re.search(r"(\d+)$", s)
        if m and ("pubmed" in s or "ncbi" in s):
            return m.group(1)

        return None

    def _norm_pmid(self, pmid, source=None):
        p = self._extract_pmid(pmid)
        if not p and source:
            p = self._extract_pmid(source)
        return p

    @staticmethod
    def _safe_json(text: str):
        if text is None:
            return None
        t = text.strip()
        try:
            return json.loads(t)
        except Exception:
            pass
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _find_offsets(full_text: str, span_text: str):
        if not full_text or not span_text:
            return None, None
        idx = full_text.find(span_text)
        if idx == -1:
            return None, None
        return idx, idx + len(span_text)

    # ✅ pred exact를 "gold와 같은 형태"로 맞춰 저장
    @staticmethod
    def _to_gold_format(qtype: str, pred_exact):
        qtype = (qtype or "").lower().strip()

        if qtype == "factoid":
            p = "" if pred_exact is None else str(pred_exact).strip()
            return [[p]] if p else []

        if qtype == "list":
            if not pred_exact:
                return []
            return [[str(x).strip()] for x in pred_exact if str(x).strip()]

        if qtype == "yesno":
            p = "" if pred_exact is None else str(pred_exact).strip().lower()
            return p if p in ["yes", "no"] else ""

        return "" if pred_exact is None else str(pred_exact).strip()

    # ---------- gold/pred helpers (BioASQ exact_answer 형태 대응) ----------
    @staticmethod
    def _gold_groups(exact_answer_field):
        x = RAGEvaluator._maybe_parse(exact_answer_field)
        groups = []

        if isinstance(x, list):
            for item in x:
                if isinstance(item, list):
                    s = {
                        RAGEvaluator._normalize(v).lower()
                        for v in item
                        if v is not None and RAGEvaluator._normalize(v)
                    }
                    if s:
                        groups.append(s)
                else:
                    v = RAGEvaluator._normalize(item).lower()
                    if v:
                        groups.append({v})
        elif isinstance(x, str):
            v = RAGEvaluator._normalize(x).lower()
            if v:
                groups.append({v})

        return groups

    @staticmethod
    def _pred_items(pred_answer):
        if pred_answer is None:
            return []
        if isinstance(pred_answer, list):
            return [RAGEvaluator._normalize(v).lower() for v in pred_answer if RAGEvaluator._normalize(v)]
        if isinstance(pred_answer, str):
            s = RAGEvaluator._normalize(pred_answer)
            if not s:
                return []
            parts = re.split(r"[;\n,]\s*", s)
            return [RAGEvaluator._normalize(p).lower() for p in parts if RAGEvaluator._normalize(p)]
        return []

    # list: group EM/F1
    def _list_group_f1(self, pred_answer, exact_answer_field):
        gold_groups = self._gold_groups(exact_answer_field)
        pred_items = self._pred_items(pred_answer)

        if not gold_groups and not pred_items:
            return 1.0
        if not gold_groups or not pred_items:
            return 0.0

        matched = 0
        used_pred = set()

        for g in gold_groups:
            hit = False
            for i, p in enumerate(pred_items):
                if i in used_pred:
                    continue
                if p in g:
                    hit = True
                    used_pred.add(i)
                    break
            if hit:
                matched += 1

        tp = matched
        precision = tp / len(pred_items)
        recall = tp / len(gold_groups)
        return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    def _list_group_em(self, pred_answer, exact_answer_field):
        gold_groups = self._gold_groups(exact_answer_field)
        pred_items = self._pred_items(pred_answer)

        if not gold_groups and not pred_items:
            return 1
        if not gold_groups or not pred_items:
            return 0

        matched = 0
        used_pred = set()

        for g in gold_groups:
            hit = False
            for i, p in enumerate(pred_items):
                if i in used_pred:
                    continue
                if p in g:
                    hit = True
                    used_pred.add(i)
                    break
            if hit:
                matched += 1

        if matched == len(gold_groups) and len(used_pred) == len(pred_items):
            return 1
        return 0

    # ---------- init vectorstore ----------
    def _init_vectorstore(self):
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.chroma_dir,
            embedding_function=self.embedding_function,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
        print(f"✅ Chroma 로드 완료: {self.collection_name} (k={self.k})")

    # ---------- init llm + prompt ----------
    def _init_llm_and_prompt(self):
        print(f"--- LLM 로드: ChatOllama({self.llm_model}) ---")
        self.llm = ChatOllama(model=self.llm_model)
        print("✅ LLM 로드 완료")

        print(f"--- Prompt YAML 로드: {self.prompt_path} ---")
        self.prompt_text = load_prompt_text_from_yaml(self.prompt_path)
        if not isinstance(self.prompt_text, str) or not self.prompt_text.strip():
            raise ValueError("Prompt YAML에서 유효한 프롬프트 텍스트를 찾지 못했습니다.")

        # ✅ [SYSTEM]/[User] 분리 (없으면 user로 취급)
        self.system_tmpl, self.user_tmpl = split_system_user(self.prompt_text)

        # user 파트가 아예 없으면, 최소 user 템플릿을 깔아줌
        if not self.user_tmpl.strip():
            self.user_tmpl = "[Question Type] question: {question}\nAnswer:"

        print("✅ Prompt 로드 완료 (system/user split)")

    # ---------- load ----------
    def load_questions(self):
        self.df_questions = pd.read_csv(self.dataset_csv)

        # ✅ snippets는 안 쓰므로 파싱/필수 컬럼에서 제거
        for col in ["documents", "exact_answer"]:
            if col in self.df_questions.columns:
                self.df_questions[col] = self.df_questions[col].apply(self._maybe_parse)

        need_cols = ["body", "documents", "type", "id", "exact_answer"]
        missing = [c for c in need_cols if c not in self.df_questions.columns]
        if missing:
            raise ValueError(f"질문 CSV에 필요한 컬럼이 없음: {missing}")

        if "ideal_answer" not in self.df_questions.columns:
            self.df_questions["ideal_answer"] = ""

        print(f"✅ 질문 로드 완료: {len(self.df_questions)} rows")

    # ---------- context builder ----------
    def _build_context(self, retrieved_docs):
        blocks = []
        docid_to_text = {}
        docid_to_pmid = {}
        total = 0

        for rank, doc in enumerate(retrieved_docs):
            pmid = self._norm_pmid(doc.metadata.get("pmid"), doc.metadata.get("source")) or "UNKNOWN"

            chunk_text = (doc.page_content or "").strip()
            if len(chunk_text) > self.max_doc_chars:
                chunk_text = chunk_text[:self.max_doc_chars]

            doc_id = f"{pmid}#{rank}"
            block = f"[DOCID={doc_id}][PMID={pmid}]\n{chunk_text}\n"

            if total + len(block) > self.max_total_chars:
                break

            blocks.append(block)
            total += len(block)

            docid_to_text[doc_id] = chunk_text
            docid_to_pmid[doc_id] = pmid

        return "\n".join(blocks), docid_to_text, docid_to_pmid

    # ✅ system/user 각각에 치환
    def _format_tmpl(self, tmpl: str, context_str: str, question: str, qtype: str) -> str:
        p = tmpl or ""
        if "{context}" in p:
            p = p.replace("{context}", context_str)
        if "{question}" in p:
            p = p.replace("{question}", question)
        if "{type}" in p:
            p = p.replace("{type}", qtype)
        return p

    # ---------- parse LLM output ----------
    def _parse_llm_output(self, raw_text: str):
        parsed = self._safe_json(raw_text)
        pred_exact = ""
        pred_ideal = ""
        sources = []

        if isinstance(parsed, dict):
            pred_exact = parsed.get("exact_answer", "")
            if pred_exact == "" and "answer" in parsed:
                pred_exact = parsed.get("answer", "")

            pred_ideal = parsed.get("ideal_answer", "")

            src = parsed.get("sources", []) or []
            if isinstance(src, list):
                sources = src

            return pred_exact, pred_ideal, sources

        t = (raw_text or "").strip()
        return (t if t else ""), "", []

    # ✅ 타입별 exact sanitize (factoid/list에서 yes/no 방지)
    def _sanitize_pred_exact_by_type(self, qtype: str, pred_exact):
        qtype = (qtype or "").lower().strip()

        # 중첩 singleton 풀기
        pred_exact = self._unwrap_singleton(pred_exact)

        # 공통: 문자열이면 strip
        if isinstance(pred_exact, str):
            s = pred_exact.strip()
        else:
            s = None

        # ✅ factoid인데 yes/no면 무조건 빈값
        if qtype == "factoid":
            if isinstance(pred_exact, list):
                pred_exact = pred_exact[0] if pred_exact else ""
                pred_exact = "" if pred_exact is None else str(pred_exact).strip()
            else:
                pred_exact = "" if pred_exact is None else str(pred_exact).strip()

            if pred_exact.lower() in ["yes", "no"]:
                pred_exact = ""  # 핵심 가드
            return pred_exact

        # ✅ list인데 yes/no면 []
        if qtype == "list":
            if pred_exact is None:
                return []
            if isinstance(pred_exact, str):
                if pred_exact.strip().lower() in ["yes", "no"]:
                    return []
            # 아래에서 list 형태로 강제
            if isinstance(pred_exact, str):
                s = pred_exact.strip()
                if s == "":
                    return []
                if s.startswith("[") and s.endswith("]"):
                    try:
                        tmp = json.loads(s)
                        if isinstance(tmp, list):
                            return tmp
                    except Exception:
                        pass
                return [a for a in re.split(r"[;\n,]\s*", s) if a]
            if isinstance(pred_exact, list):
                # 리스트 내부도 정리
                out = []
                for v in pred_exact:
                    vv = "" if v is None else str(v).strip()
                    if vv:
                        out.append(vv)
                return out
            return [str(pred_exact).strip()] if str(pred_exact).strip() else []

        # yesno
        if qtype == "yesno":
            if isinstance(pred_exact, list):
                pred_exact = pred_exact[0] if pred_exact else ""
            pred_exact = "" if pred_exact is None else str(pred_exact).strip().lower()
            return pred_exact if pred_exact in ["yes", "no"] else ""

        # 기타
        if isinstance(pred_exact, list):
            pred_exact = pred_exact[0] if pred_exact else ""
        return "" if pred_exact is None else str(pred_exact).strip()

    # ---------- generation ----------
    def run_generation(self):
        if self.df_questions is None:
            self.load_questions()

        results = []
        debug_rows = []
        start = time.time()

        for idx, (_, row) in enumerate(tqdm(self.df_questions.iterrows(), total=len(self.df_questions), desc="Retrieve+Generate")):
            question = self._normalize(row.get("body", ""))
            qtype = self._normalize(row.get("type", "")).lower()

            gold_exact = row.get("exact_answer")
            gold_ideal = row.get("ideal_answer", "")

            # 1) retrieve
            retrieved_docs = self.retriever.invoke(question)

            retrieved_pmids = []
            retrieved_docids = []
            contexts = []

            for rank, doc in enumerate(retrieved_docs):
                pmid = self._norm_pmid(doc.metadata.get("pmid"), doc.metadata.get("source")) or "UNKNOWN"
                retrieved_pmids.append(pmid)
                retrieved_docids.append(f"{pmid}#{rank}")
                if self.save_contexts:
                    contexts.append(doc.page_content)

            # 2) generation
            pred_exact_answer_goldfmt = []
            pred_ideal_answer = ""
            sources_out = []
            raw = ""

            # 내부 평가용
            generated_answer = None
            ideal_answer = ""

            if self.do_llm:
                context_str, docid_to_text, docid_to_pmid = self._build_context(retrieved_docs)

                # ✅ system/user 각각 치환 후 messages로 invoke
                sys_msg = self._format_tmpl(self.system_tmpl, context_str, question, qtype)
                usr_msg = self._format_tmpl(self.user_tmpl, context_str, question, qtype)

                try:
                    msg = self.llm.invoke([
                        SystemMessage(content=sys_msg),
                        HumanMessage(content=usr_msg),
                    ])
                    raw = msg.content if hasattr(msg, "content") else str(msg)
                except Exception as e:
                    raw = ""
                    print(f"\n[WARN] LLM invoke error: {repr(e)}")

                pred_exact, pred_ideal, sources_in = self._parse_llm_output(raw)

                # ✅ 타입별 sanitize (factoid에서 yes/no 차단 포함)
                pred_exact = self._sanitize_pred_exact_by_type(qtype, pred_exact)
                pred_ideal = "" if pred_ideal is None else str(pred_ideal).strip()

                generated_answer = pred_exact
                ideal_answer = pred_ideal

                # ✅ 저장용: gold 형태로
                pred_exact_answer_goldfmt = self._to_gold_format(qtype, pred_exact)
                pred_ideal_answer = pred_ideal

                # ---------- sources 검증(있을 때만) ----------
                if isinstance(sources_in, list):
                    for s in sources_in:
                        if not isinstance(s, dict):
                            continue

                        doc_id = s.get("doc_id")
                        pmid = s.get("pmid")
                        quote = s.get("quote")

                        if quote is None and isinstance(s.get("text"), str):
                            quote = s.get("text")

                        doc_id = str(doc_id) if doc_id is not None else None
                        pmid = str(pmid) if pmid is not None else None
                        quote = quote if isinstance(quote, str) else None

                        if not doc_id or doc_id not in docid_to_text:
                            continue

                        chunk_text = docid_to_text.get(doc_id, "")
                        pmid_expected = docid_to_pmid.get(doc_id)

                        pmid_norm = self._extract_pmid(pmid) if pmid else None
                        pmid_expected_norm = self._extract_pmid(pmid_expected) if pmid_expected else None

                        if not pmid_norm:
                            pmid_norm = pmid_expected_norm
                        if pmid_expected_norm and pmid_norm != pmid_expected_norm:
                            continue

                        if quote is None:
                            continue

                        b, e = self._find_offsets(chunk_text, quote)
                        if b is None or e is None:
                            continue

                        sources_out.append({
                            "doc_id": doc_id,
                            "pmid": pmid_norm,
                            "text": quote,
                            "offsetInBeginSection": b,
                            "offsetInEndSection": e,
                        })

                if self.evidence_fallback and not sources_out:
                    sources_out = [
                        {
                            "doc_id": f"{p}#{i}",
                            "pmid": p,
                            "text": None,
                            "offsetInBeginSection": None,
                            "offsetInEndSection": None,
                        }
                        for i, p in enumerate(retrieved_pmids)
                        if p and p != "UNKNOWN"
                    ]

            # 3) answer metrics (임시/참고용)
            if qtype == "list":
                answer_em = float(self._list_group_em(generated_answer, gold_exact))
                answer_f1 = float(self._list_group_f1(generated_answer, gold_exact))

            elif qtype == "factoid":
                gold_groups = self._gold_groups(gold_exact)
                pred = self._normalize(generated_answer).lower()
                hit = 0.0
                if pred and gold_groups:
                    for g in gold_groups:
                        if pred in g:
                            hit = 1.0
                            break
                answer_em = hit
                answer_f1 = hit

            elif qtype == "yesno":
                gold = self._normalize(gold_exact).lower()
                pred = self._normalize(generated_answer).lower()
                hit = 1.0 if (gold in ["yes", "no"] and pred == gold) else 0.0
                answer_em = hit
                answer_f1 = hit

            else:
                answer_em = np.nan
                answer_f1 = np.nan

            out = {
                "body": row.get("body"),
                "documents": row.get("documents"),
                "type": row.get("type"),
                "id": row.get("id"),

                # gold
                "gold_exact_answer": gold_exact,
                "gold_ideal_answer": gold_ideal,

                # retrieval
                "retrieved_pmids": retrieved_pmids,
                "retrieved_docids": retrieved_docids,

                # pred
                "pred_exact_answer": pred_exact_answer_goldfmt,  # ✅ 하나만 저장(골드형태)
                "pred_ideal_answer": pred_ideal_answer,

                "raw_llm": raw,
                "sources": sources_out,

                "answer_em": float(answer_em) if answer_em is not np.nan else np.nan,
                "answer_f1": float(answer_f1) if answer_f1 is not np.nan else np.nan,
            }
            if self.save_contexts:
                out["contexts"] = contexts

            results.append(out)

            if self.debug_save_first_n and idx < self.debug_save_first_n:
                gold_docs = row.get("documents", []) or []
                gold_pmids = [self._extract_pmid(d) for d in gold_docs]
                gold_pmids = [p for p in gold_pmids if p]

                overlap = set(gold_pmids) & set([p for p in retrieved_pmids if p and p != "UNKNOWN"])
                unk_ratio = sum(p == "UNKNOWN" for p in retrieved_pmids) / max(len(retrieved_pmids), 1)

                debug_rows.append({
                    "qid": row.get("id"),
                    "qtype": qtype,
                    "question_head": question[:120],
                    "gold_count": len(set(gold_pmids)),
                    "retrieved_count": len(set(retrieved_pmids)),
                    "overlap_count": len(overlap),
                    "overlap_example": ";".join(list(overlap)[:5]),
                    "gold_pmids_head": ";".join(gold_pmids[:10]),
                    "retrieved_pmids_head": ";".join([p for p in retrieved_pmids[:10]]),
                    "unknown_ratio": float(unk_ratio),
                })

        self.df_results = pd.DataFrame(results)
        self.df_results.to_csv(self.result_csv_path, index=False, encoding="utf-8-sig")
        print(f"📁 결과 저장: {self.result_csv_path} (time={time.time()-start:.1f}s)")

        if debug_rows:
            pd.DataFrame(debug_rows).to_csv(self.debug_csv_path, index=False, encoding="utf-8-sig")
            print(f"📁 debug 저장: {self.debug_csv_path}")

    # ---------- evidence doc-level metrics ----------
    def _evidence_doc_scores(self, gold_documents, pred_sources):
        gold = set()
        for d in (gold_documents or []):
            pmid = self._extract_pmid(d)
            if pmid:
                gold.add(pmid)

        pred = set()
        for s in (pred_sources or []):
            if isinstance(s, dict) and s.get("pmid"):
                p = self._extract_pmid(s.get("pmid"))
                if p:
                    pred.add(p)

        if not gold:
            return np.nan, np.nan, np.nan

        inter = gold & pred
        prec = len(inter) / max(len(pred), 1)
        rec = len(inter) / len(gold)
        hit = 1.0 if len(inter) > 0 else 0.0
        return prec, rec, hit

    # ---------- retrieval evaluation ----------
    def run_retrieval_eval(self):
        if self.df_results is None:
            self.df_results = pd.read_csv(self.result_csv_path)

        df = self.df_results.copy()

        for c in ["documents", "retrieved_pmids", "retrieved_docids",
                  "gold_exact_answer", "gold_ideal_answer",
                  "pred_exact_answer", "pred_ideal_answer",
                  "sources", "contexts"]:
            if c in df.columns:
                df[c] = df[c].apply(self._maybe_parse)

        eval_rows = []
        for _, row in df.iterrows():
            gold_set = set()
            docs = row.get("documents", None)
            if isinstance(docs, list):
                for d in docs:
                    pmid = self._extract_pmid(d)
                    if pmid:
                        gold_set.add(pmid)

            pred = row.get("retrieved_pmids", [])
            pred_norm = []
            if isinstance(pred, list):
                seen = set()
                for p in pred:
                    pm = self._extract_pmid(p) or (str(p).strip() if p is not None else None)
                    if not pm or pm == "UNKNOWN":
                        continue
                    if pm not in seen:
                        pred_norm.append(pm)
                        seen.add(pm)

            L = min(self.k, len(pred_norm))
            pred_top = pred_norm[:L]

            if len(gold_set) == 0 or L == 0:
                recall_k = np.nan
                precision_k = np.nan
                hit_k = np.nan
                ndcg_k = np.nan
            else:
                hit_count = sum([1 for p in pred_top if p in gold_set])
                recall_k = hit_count / len(gold_set)
                precision_k = hit_count / L
                hit_k = 1.0 if hit_count > 0 else 0.0

                rel = [1 if (p in gold_set) else 0 for p in pred_top]
                if sum(rel) == 0:
                    ndcg_k = 0.0
                else:
                    L2 = len(rel)
                    K2 = max(self.k, 2)
                    rel_pad = (rel + [0.0] * (K2 - L2))[:K2]
                    score_pad = (list(range(L2, 0, -1)) + [0.0] * (K2 - L2))[:K2]
                    y_true  = np.asarray([rel_pad], dtype=float)
                    y_score = np.asarray([score_pad], dtype=float)
                    ndcg_k = float(ndcg_score(y_true, y_score, k=min(self.k, K2)))

            ev_prec, ev_rec, ev_hit = self._evidence_doc_scores(
                row.get("documents", []),
                row.get("sources", []),
            )

            eval_rows.append({
                "id": row.get("id"),
                "type": row.get("type"),
                "recall@k": recall_k,
                "precision@k": precision_k,
                "ndcg@k": ndcg_k,
                "hit@k": hit_k,

                "answer_em": row.get("answer_em", np.nan),
                "answer_f1": row.get("answer_f1", np.nan),

                "evidence_precision": ev_prec,
                "evidence_recall": ev_rec,
                "evidence_hit": ev_hit,

                "gold_doc_count": len(gold_set),
            })

        self.df_eval = pd.DataFrame(eval_rows)
        self.df_eval.to_csv(self.eval_csv_path, index=False, encoding="utf-8-sig")
        print(f"📁 retrieval 평가 저장: {self.eval_csv_path}")

        overall_recall = float(self.df_eval["recall@k"].mean(skipna=True))
        overall_precision = float(self.df_eval["precision@k"].mean(skipna=True))
        overall_ndcg = float(self.df_eval["ndcg@k"].mean(skipna=True))
        overall_hit = float(self.df_eval["hit@k"].mean(skipna=True))

        overall_answer_em = float(self.df_eval["answer_em"].mean(skipna=True))
        overall_answer_f1 = float(self.df_eval["answer_f1"].mean(skipna=True))

        overall_ev_precision = float(self.df_eval["evidence_precision"].mean(skipna=True))
        overall_ev_recall = float(self.df_eval["evidence_recall"].mean(skipna=True))
        overall_ev_hit = float(self.df_eval["evidence_hit"].mean(skipna=True))

        print(f"- Recall@K          : {overall_recall:.6f}")
        print(f"- Precision@K       : {overall_precision:.6f}")
        print(f"- NDCG@K            : {overall_ndcg:.6f}")
        print(f"- Hit@K             : {overall_hit:.6f}")
        print(f"- AnswerEM          : {overall_answer_em:.6f}")
        print(f"- AnswerF1          : {overall_answer_f1:.6f}")
        print(f"- EvidencePrecision : {overall_ev_precision:.6f}")
        print(f"- EvidenceRecall    : {overall_ev_recall:.6f}")
        print(f"- EvidenceHit       : {overall_ev_hit:.6f}")

    def run_all(self):
        self.run_generation()
        self.run_retrieval_eval()
