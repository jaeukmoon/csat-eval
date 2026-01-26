"""데이터셋 빌더 CLI - JSONL 데이터셋 자동 생성"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .huggingface import download_from_huggingface, is_available_on_huggingface
from .math_pdf import build_math_jsonl
from .english_pdf import build_english_jsonl
from .korean_pdf import build_korean_jsonl


def find_pdf_files(pdf_dir: Path, year: str, subject: str) -> tuple[Optional[Path], Optional[Path]]:
    """PDF 파일 경로 찾기 (새 네이밍 규칙: {year}_{subject}_problem.pdf)"""
    problem_pdf = pdf_dir / f"{year}_{subject}_problem.pdf"
    answer_pdf = pdf_dir / f"{year}_{subject}_answer.pdf"
    
    if problem_pdf.exists() and answer_pdf.exists():
        return problem_pdf, answer_pdf
    
    return None, None


def build_if_missing(
    split: str,
    data_dir: Path,
    pdf_dir: Path,
    force: bool = False,
    vision_model: str = "gpt-5.2",
    azure_vision_model: str = "gpt-4o",
) -> bool:
    """
    split에 해당하는 JSONL이 없으면 생성
    
    우선순위:
    1. 이미 존재하면 스킵
    2. HuggingFace에서 다운로드 가능하면 다운로드
    3. PDF가 있으면 변환
    
    Returns:
        True if dataset exists or was created successfully
    """
    out_path = data_dir / f"{split}.jsonl"
    
    # 1. 이미 존재하면 스킵
    if out_path.exists() and not force:
        print(f"[build] Already exists: {out_path}")
        return True
    
    # split 파싱 (예: "2026_math" -> year="2026", subject="math")
    if "_" not in split:
        print(f"[build] Invalid split format: {split}")
        return False
    
    year, subject = split.split("_", 1)
    
    # 2. HuggingFace에서 다운로드 시도
    if is_available_on_huggingface(split):
        result = download_from_huggingface(split, out_path, force=force)
        if result:
            return True
    
    # 3. PDF에서 변환 시도
    problem_pdf, answer_pdf = find_pdf_files(pdf_dir, year, subject)
    if problem_pdf and answer_pdf:
        print(f"[build] Building from PDF: {problem_pdf}")
        try:
            if subject == "math":
                build_math_jsonl(
                    problem_pdf=problem_pdf,
                    answer_pdf=answer_pdf,
                    out_path=out_path,
                    vision_model=vision_model,
                    force=force,
                )
            elif subject == "english":
                build_english_jsonl(
                    problem_pdf=problem_pdf,
                    answer_pdf=answer_pdf,
                    out_path=out_path,
                    vision_model=vision_model,
                    force=force,
                )
            elif subject == "korean":
                build_korean_jsonl(
                    problem_pdf=problem_pdf,
                    answer_pdf=answer_pdf,
                    out_path=out_path,
                    vision_model=vision_model,
                    force=force,
                )
            else:
                print(f"[build] Unknown subject: {subject}")
                return False
            return True
        except Exception as e:
            print(f"[build] PDF conversion failed: {e}")
            return False
    
    print(f"[build] No source found for: {split}")
    print(f"  - HuggingFace: {'available' if is_available_on_huggingface(split) else 'not available'}")
    print(f"  - PDF: {pdf_dir / f'{year}_{subject}_problem.pdf'} (not found)")
    return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="JSONL 데이터셋 자동 생성 (HuggingFace 또는 PDF에서)"
    )
    ap.add_argument(
        "--split",
        required=True,
        help="데이터셋 split 이름 (예: 2026_math, 2026_english, 2026_korean)"
    )
    ap.add_argument(
        "--data_dir",
        default="./data",
        help="JSONL 출력 디렉토리 (기본값: ./data)"
    )
    ap.add_argument(
        "--pdf_dir",
        default="./pdf_csat_data",
        help="PDF 파일 디렉토리 (기본값: ./pdf_csat_data)"
    )
    ap.add_argument(
        "--vision_model",
        default="gpt-5.2",
        help="PDF 변환에 사용할 OpenAI Vision 모델 - 수학/영어용 (기본값: gpt-5.2)"
    )
    ap.add_argument(
        "--azure_vision_model",
        default="gpt-5.1",
        help="PDF 변환에 사용할 Azure OpenAI Vision 모델 - 국어용 (기본값: gpt-5.1)"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="기존 파일이 있어도 강제 재생성"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    pdf_dir = Path(args.pdf_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    success = build_if_missing(
        split=args.split,
        data_dir=data_dir,
        pdf_dir=pdf_dir,
        force=args.force,
        vision_model=args.vision_model,
        azure_vision_model=args.azure_vision_model,
    )
    
    if success:
        print(f"[build] Success: {data_dir / f'{args.split}.jsonl'}")
    else:
        print(f"[build] Failed: {args.split}")
        exit(1)


if __name__ == "__main__":
    main()
