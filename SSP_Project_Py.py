import argparse
import os
import re
import sys
from typing import List, Dict

import pandas as pd

def _lazy_import_datasets():
    try:
        import datasets  # type: ignore
        return datasets
    except Exception as e:
        print("ERROR: The 'datasets' package is required. Install with: pip install -r requirements.txt", file=sys.stderr)
        raise

TASK1_CSV = "task1_pull_requests.csv"
TASK2_CSV = "task2_repositories.csv"
TASK3_CSV = "task3_pr_task_type.csv"
TASK4_CSV = "task4_pr_commit_details.csv"
TASK5_CSV = "task5_joined_security_flags.csv"

SECURITY_KEYWORDS = [
    "race", "racy", "buffer", "overflow", "stack", "integer", "signedness", "underflow",
    "improper", "unauthenticated", "gain access", "permission", "cross site", "css", "xss",
    "denial service", "dos", "crash", "deadlock", "injection", "request forgery", "csrf",
    "xsrf", "forged", "security", "vulnerability", "vulnerable", "exploit", "attack",
    "bypass", "backdoor", "threat", "expose", "breach", "violate", "fatal", "blacklist",
    "overrun", "insecure"
]


def load_table(dataset_dict, table_name: str) -> pd.DataFrame:
    if table_name not in dataset_dict:
        raise KeyError(f"Table '{table_name}' not found in dataset; available: {list(dataset_dict.keys())}")
    return dataset_dict[table_name].to_pandas()

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clean_diff_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def to_bool01(flag: bool) -> int:
    return 1 if bool(flag) else 0

def contains_security_keyword(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(k in lower for k in keywords)


def task1_all_pull_request(df_pr: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "TITLE": "title",
        "ID": "id",
        "AGENTNAME": "agent",
        "BODYSTRING": "body",
        "REPOID": "repo_id",
        "REPOURL": "repo_url",
    }
    missing = [src for src in cols.values() if src not in df_pr.columns]
    if missing:
        raise KeyError(f"Task-1: missing expected columns in all_pull_request: {missing}")
    out = df_pr[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    return out

def task2_all_repository(df_repo: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "REPOID": "id",
        "LANG": "language",
        "STARS": "stars",
        "REPOURL": "url",
    }
    missing = [src for src in cols.values() if src not in df_repo.columns]
    if missing:
        raise KeyError(f"Task-2: missing expected columns in all_repository: {missing}")
    out = df_repo[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    return out

def task3_pr_task_type(df_type: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "PRID": "id",
        "PRTITLE": "title",
        "PRREASON": "reason",
        "PRTYPE": "type",
        "CONFIDENCE": "confidence",
    }
    missing = [src for src in cols.values() if src not in df_type.columns]
    if missing:
        raise KeyError(f"Task-3: missing expected columns in pr_task_type: {missing}")
    out = df_type[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    return out

def task4_pr_commit_details(df_commit: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "PRID": "pr_id",
        "PRSHA": "sha",
        "PRCOMMITMESSAGE": "message",
        "PRFILE": "filename",
        "PRSTATUS": "status",
        "PRADDS": "additions",
        "PRDELSS": "deletions",
        "PRCHANGECOUNT": "changes",
        "PRDIFF": "patch",
    }
    missing = [src for src in cols.values() if src not in df_commit.columns]
    if missing:
        raise KeyError(f"Task-4: missing expected columns in pr_commit_details: {missing}")
    out = df_commit[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
    out["PRDIFF"] = out["PRDIFF"].map(clean_diff_text)
    return out

def task5_join_security(task1_df: pd.DataFrame, task3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Task-1 (pull requests) with Task-3 (types) to produce:
      ID, AGENT, TYPE, CONFIDENCE, SECURITY
    SECURITY = 1 iff any keyword appears in PR title or body; else 0
    """
    pr_cols_needed = ["ID", "AGENTNAME", "TITLE", "BODYSTRING"]
    type_cols_needed = ["PRID", "PRTYPE", "CONFIDENCE"]
    for c in pr_cols_needed:
        if c not in task1_df.columns:
            raise KeyError(f"Task-5: missing column in Task-1 CSV: {c}")
    for c in type_cols_needed:
        if c not in task3_df.columns:
            raise KeyError(f"Task-5: missing column in Task-3 CSV: {c}")

    # Join on ID == PRID
    merged = task1_df.merge(task3_df, left_on="ID", right_on="PRID", how="left")

    def sec_flag(row) -> int:
        title = row.get("TITLE", "")
        body = row.get("BODYSTRING", "")
        return to_bool01(contains_security_keyword(f"{title}\n{body}", SECURITY_KEYWORDS))

    out = pd.DataFrame({
        "ID": merged["ID"],
        "AGENT": merged["AGENTNAME"],
        "TYPE": merged["PRTYPE"],
        "CONFIDENCE": merged["CONFIDENCE"],
        "SECURITY": merged.apply(sec_flag, axis=1)
    })
    return out

def load_cfg_as_df(datasets, cfg: str) -> pd.DataFrame:
    ds_cfg = datasets.load_dataset("hao-li/AIDev", cfg, split="train")
    return ds_cfg.to_pandas()

def main():
    parser = argparse.ArgumentParser(description="Produce CSVs for AIDev Tasks 1–5.")
    parser.add_argument("--outdir", default="outputs", help="Directory to write CSV outputs (default: outputs)")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    datasets = _lazy_import_datasets()

    # Load each table by config name
    print("Loading AIDev tables from Hugging Face configs ...")
    df_pr     = load_cfg_as_df(datasets, "all_pull_request")
    df_repo   = load_cfg_as_df(datasets, "all_repository")
    df_type   = load_cfg_as_df(datasets, "pr_task_type")
    df_commit = load_cfg_as_df(datasets, "pr_commit_details")

    # Task 1
    print("Running Task-1 ...")
    t1 = task1_all_pull_request(df_pr)
    t1_path = os.path.join(args.outdir, TASK1_CSV)
    t1.to_csv(t1_path, index=False)
    print(f"Wrote {t1_path}")

    # Task 2
    print("Running Task-2 ...")
    t2 = task2_all_repository(df_repo)
    t2_path = os.path.join(args.outdir, TASK2_CSV)
    t2.to_csv(t2_path, index=False)
    print(f"Wrote {t2_path}")

    # Task 3
    print("Running Task-3 ...")
    t3 = task3_pr_task_type(df_type)
    t3_path = os.path.join(args.outdir, TASK3_CSV)
    t3.to_csv(t3_path, index=False)
    print(f"Wrote {t3_path}")

    # Task 4
    print("Running Task-4 ...")
    t4 = task4_pr_commit_details(df_commit)
    t4_path = os.path.join(args.outdir, TASK4_CSV)
    t4.to_csv(t4_path, index=False)
    print(f"Wrote {t4_path}")

    # Task 5
    print("Running Task-5 ...")
    t5 = task5_join_security(t1, t3)
    t5_path = os.path.join(args.outdir, TASK5_CSV)
    t5.to_csv(t5_path, index=False)
    print(f"Wrote {t5_path}")

    print("All tasks complete.")

if __name__ == "__main__":
    main()
