import subprocess
import yaml
from pathlib import Path
import json
import time
import sys  # 新增

# 1. 要实验的 chunk 配置列表（可以按需增删）
EXPERIMENTS = [
    {"exp_id": "Base", "chunk_size": 1000, "chunk_overlap": 200},
    {"exp_id": "A_500_150", "chunk_size": 500, "chunk_overlap": 150},
    {"exp_id": "B_800_200", "chunk_size": 800, "chunk_overlap": 200},
    {"exp_id": "C_1200_250", "chunk_size": 1200, "chunk_overlap": 250},
]

CONFIG_PATH = Path("config.yaml")
RESULTS_PATH = Path("phase1_chunking_results.jsonl")
QUESTIONS_PATH = Path("questions_phase1.txt")

# 使用当前解释器，保证和你激活的 venv 一致
PYTHON_CMD = sys.executable


def load_questions():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_PATH}")
    questions = []
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                questions.append(q)
    return questions


def update_chunking_in_config(chunk_size: int, chunk_overlap: int):
    print(f"[CONFIG] Updating chunking: chunk_size={chunk_size}, overlap={chunk_overlap}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "chunking" not in config:
        config["chunking"] = {}

    config["chunking"]["chunk_size"] = int(chunk_size)
    config["chunking"]["chunk_overlap"] = int(chunk_overlap)

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_ingest_reset():
    print("[INGEST] Rebuilding vector database with --reset ...")
    cmd = [PYTHON_CMD, "ingest_documents.py", "--reset"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("ingest_documents.py failed")


def ask_single_question(question: str):
    """
    使用单问题模式调用 main.py: python main.py --question "..."
    返回整个 stdout 方便后面分析。
    """
    cmd = [PYTHON_CMD, "main.py", "--question", question]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print("[ERROR] main.py failed")
        print(result.stderr)
        raise RuntimeError("main.py failed")

    return result.stdout, elapsed


def main():
    questions = load_questions()
    print(f"[INFO] Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    # 如果之前有结果文件，先备份一下（带编号，防止重名）
    if RESULTS_PATH.exists():
        i = 1
        while True:
            backup = RESULTS_PATH.with_suffix(f".bak.{i}")
            if not backup.exists():
                break
            i += 1
        RESULTS_PATH.rename(backup)
        print(f"[INFO] Existing results backed up to {backup}")

    with RESULTS_PATH.open("w", encoding="utf-8") as f_out:
        for exp in EXPERIMENTS:
            exp_id = exp["exp_id"]
            chunk_size = exp["chunk_size"]
            chunk_overlap = exp["chunk_overlap"]

            print(
                f"\n===== Running experiment {exp_id} "
                f"(chunk_size={chunk_size}, overlap={chunk_overlap}) ====="
            )

            # 1) 更新 config.yaml
            update_chunking_in_config(chunk_size, chunk_overlap)

            # 2) 重新 build 向量库
            run_ingest_reset()

            # 3) 对每个问题调用 main.py --question
            for q_idx, q in enumerate(questions, start=1):
                print(f"\n[Q{q_idx}] ({exp_id}) {q}")
                stdout, elapsed = ask_single_question(q)

                # 把结果存成一行 JSON，方便后续打分
                record = {
                    "exp_id": exp_id,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "question_index": q_idx,
                    "question": q,
                    "answer_raw": stdout,
                    "response_time_sec": elapsed,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()  # 防止中途崩了数据没写进去

    print(f"\n[DONE] All experiments finished. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
