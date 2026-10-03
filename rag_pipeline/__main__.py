"""Run the extracted evaluator against an existing Chroma index."""
import argparse
import csv
import json
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chroma-dir", type=Path, required=True)
    parser.add_argument("--collections", nargs="+", required=True)
    parser.add_argument("--k", nargs="+", type=positive_int, default=[5, 10, 20])
    parser.add_argument("--prompt", type=Path, default=Path("prompt/rag_prompt2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--llm-model", default="gemma3:4b")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-doc-chars", type=positive_int, default=1000)
    parser.add_argument("--max-total-chars", type=positive_int, default=40000)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs without loading models or calling Ollama")
    args = parser.parse_args(argv)
    for path in (args.dataset, args.prompt):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if not (args.chroma_dir / "chroma.sqlite3").is_file():
        parser.error(f"existing Chroma database not found: {args.chroma_dir}")
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])
    required = {"id", "body", "type", "documents", "exact_answer"}
    if required - columns:
        parser.error(f"missing dataset columns: {sorted(required - columns)}")
    # The CLI targets the paper's four models and three individual pooling modes.
    for collection in args.collections:
        model, separator, pooling = collection.rpartition("_")
        if not separator or model not in {"SBERT", "bge", "solon", "gemma"} or pooling not in {"C", "M", "X"}:
            parser.error(f"unsupported paper collection: {collection}")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("Input check only: collection existence, model compatibility and Ollama are not checked.")
        return

    # Import model dependencies only after CLI validation, so --help/dry-run stay lightweight.
    import random
    import numpy as np
    import torch
    from chromadb import PersistentClient
    from .embeddings import make_wrapper_for_collection
    from .evaluator import RAGEvaluator

    client = PersistentClient(path=str(args.chroma_dir))
    for collection in args.collections:
        existing = client.get_collection(collection)
        if existing.count() == 0:
            raise ValueError(f"empty Chroma collection: {collection}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    config_path = args.output_dir / ("run_config_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for collection in args.collections:
        wrapper = make_wrapper_for_collection(collection, device=device)
        for k in args.k:
            evaluator = RAGEvaluator(
                dataset_csv=str(args.dataset), collection_name=collection, k=k,
                embedding_function=wrapper, chroma_dir=str(args.chroma_dir),
                llm_model=args.llm_model, prompt_path=str(args.prompt),
                results_dir=str(args.output_dir / "results"),
                summary_dir=str(args.output_dir / "summary"),
                max_doc_chars=args.max_doc_chars, max_total_chars=args.max_total_chars,
            )
            evaluator.run_all()


if __name__ == "__main__":
    main()
