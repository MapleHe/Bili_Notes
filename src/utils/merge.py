"""
Text merging utility for combining multiple transcripts or summaries.
"""
from pathlib import Path
from typing import List


def merge_texts(texts: List[str], separator: str = "\n\n---\n\n") -> str:
    """Merge multiple text strings with a separator."""
    return separator.join(t.strip() for t in texts if t.strip())


def merge_files(file_paths: List[str], separator: str = "\n\n---\n\n") -> str:
    """Read and merge multiple text files."""
    texts = []
    for path in file_paths:
        p = Path(path)
        if p.exists():
            texts.append(p.read_text(encoding="utf-8"))
    return merge_texts(texts, separator)


def find_userdata_files(userdata_dir: str, suffix: str) -> List[str]:
    """
    Find all files in userdata_dir matching the given suffix.
    suffix examples: "-transcript.txt", "-summary.txt"
    Returns sorted list of file paths.
    """
    d = Path(userdata_dir)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob(f"*{suffix}"))
