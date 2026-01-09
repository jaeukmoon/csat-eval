from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

import pdfplumber
from tqdm import tqdm

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
MARK_ODD = "홀수"
MARK_EVEN = "짝수형"

Q_START = 18
Q_END = 45
EXPECTED_N = Q_END - Q_START + 1


@dataclass(frozen=True)
class AnswerInfo:
    answer: int
    score: int


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _to_int_answer(tok: str) -> int:
    t = _norm(tok)
    if t in CIRCLED:
        return CIRCLED[t]
    t = re.sub(r"\s+", "", t)
    return int(t)

def load_dotenv_local() -> None:
    """
    로컬용 .env(.env.local 포함)를 읽어 환경변수로 로드한다.
    - repo에는 키를 커밋하지 않도록 .env.local 을 사용하세요(.gitignore에 포함되어 있음).
    - 단순 KEY=VALUE 형식만 지원.
    """
    # 사용자 환경에 따라 env.local(점 없음)로 만들기도 해서 같이 지원
    for fname in (".env.local", "env.local", ".env"):
        p = Path(fname)
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for line in txt.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v

def _strip_leading_qnum(qn: int, question: str) -> str:
    """
    모델이 question 안에 이미 '18.' 같은 접두를 넣는 경우가 있어 중복을 제거한다.
    """
    s = (question or "").lstrip()
    # e.g. "18." / "18 )" / "18 )" / "18"
    s = re.sub(rf"^\s*{qn}\s*[\.\)]\s*", "", s)
    # 간혹 "18. 18. ..." 형태로 중복되는 케이스도 처리
    s = re.sub(rf"^\s*{qn}\s*[\.\)]\s*", "", s)
    return s.strip()

def _wanted_set(wanted_numbers: Optional[Iterable[int]]) -> Optional[Set[int]]:
    if wanted_numbers is None:
        return None
    return {int(x) for x in wanted_numbers}

def parse_answer_pdf_odd(answer_pdf: Path) -> Dict[int, AnswerInfo]:
    """영어 정답 PDF 파싱 (홀수형)"""
    with pdfplumber.open(str(answer_pdf)) as pdf:
        odd_page = None
        for p in pdf.pages:
            txt = p.extract_text() or ""
            if MARK_ODD in txt:
                odd_page = p
                break
        if odd_page is None:
            odd_page = pdf.pages[0]
        table = odd_page.extract_table()
        if not table:
            raise RuntimeError("정답 PDF에서 표 추출 실패")

    header_idx = None
    for i, row in enumerate(table):
        if row and any("문항" in _norm(c) for c in row if c):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("정답 PDF에서 헤더 행 탐색 실패")

    data_rows = table[header_idx + 1 :]

    out: Dict[int, AnswerInfo] = {}

    def put(q: str, a: str, sc: str):
        if not _norm(q) or not _norm(a) or not _norm(sc):
            return
        try:
            qn = int(_norm(q))
            # 18-45번만 처리 (듣기 평가 제외)
            if qn < Q_START or qn > Q_END:
                return
            ans = _to_int_answer(a)
            score = int(_norm(sc))
            out[qn] = AnswerInfo(answer=ans, score=score)
        except (ValueError, TypeError):
            return

    # 영어 정답 PDF 구조 파싱 (여러 열 구조 가능)
    for row in data_rows:
        if not row:
            continue
        
        # 빈 셀 제거 후 처리
        row = [c for c in row if c]
        if len(row) < 3:
            continue
        
        # 여러 열 구조 시도: (문항, 정답, 배점) 또는 다른 순서
        # 일반적으로 3열 구조이지만, 여러 문항이 한 행에 있을 수 있음
        # 각 열을 순회하면서 문항 번호를 찾고, 그 다음 열이 정답, 그 다음이 배점
        for i in range(len(row) - 2):
            try:
                qn_str = _norm(row[i])
                if not qn_str:
                    continue
                # 숫자로 시작하는지 확인
                qn = int(re.sub(r"[^\d]", "", qn_str)[:2]) if re.search(r"\d", qn_str) else 0
                if 18 <= qn <= 45:
                    ans_str = _norm(row[i + 1])
                    score_str = _norm(row[i + 2])
                    if ans_str and score_str:
                        put(str(qn), ans_str, score_str)
            except (ValueError, IndexError):
                continue
        
        # 단순 3열 구조도 시도
        if len(row) >= 3:
            put(row[0], row[1], row[2])

    # 영어 문제는 18번부터 45번까지 (듣기 평가 제외)
    missing = [q for q in range(Q_START, Q_END + 1) if q not in out]
    if missing:
        print(f"경고: 다음 문제 번호의 정답이 없습니다: {missing}")
        print(f"추출된 문제 번호: {sorted(out.keys())}")
        # 일부 누락은 허용하되, 너무 많이 누락되면 에러
        if len(missing) > 5:
            raise RuntimeError(f"너무 많은 정답 누락: {len(missing)}개")

    return out


def pdf_page_png_bytes(page, resolution: int, pdf_path: Path) -> bytes:
    """PDF 페이지를 PNG 이미지로 변환 (PyMuPDF 사용)"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF가 필요합니다. 설치: pip install PyMuPDF\n"
            "또는 pdf2image를 사용하려면: pip install pdf2image"
        )
    
    page_num = page.page_number - 1  # PyMuPDF는 0-based index 사용
    
    doc = fitz.open(str(pdf_path))
    pdf_page = doc[page_num]
    
    zoom = resolution / 72.0  # 기본 DPI는 72
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)
    
    png_bytes = pix.tobytes("png")
    doc.close()
    
    return png_bytes


def b64_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _merge_problem(out: Dict[int, str], qn: int, problem_text: str) -> None:
    """
    중복 번호가 들어오면 더 긴(정보가 많은) 쪽을 남긴다.
    """
    if not problem_text:
        return
    prev = out.get(qn, "")
    if len(problem_text) > len(prev):
        out[qn] = problem_text

def _find_candidate_pages_for_numbers(problem_pdf: Path, wanted: Set[int]) -> Dict[int, List[int]]:
    """
    pdfplumber의 텍스트 추출로 각 문항 번호가 등장하는 페이지 후보를 찾는다.
    (정확도가 완벽하진 않지만, 누락 문항만 재시도할 때 비용을 줄여준다.)
    """
    cand: Dict[int, List[int]] = {q: [] for q in wanted}
    num_res = {q: re.compile(rf"(^|\\s){q}\\s*[\\.|\\)]", re.MULTILINE) for q in wanted}
    with pdfplumber.open(str(problem_pdf)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            if not txt.strip():
                continue
            for q, rx in num_res.items():
                if rx.search(txt):
                    cand[q].append(page_idx)
    return cand

def _extract_from_page_image(
    client,
    data_url: str,
    model: str,
    detail: str,
    max_output_tokens: int,
    wanted_numbers: Optional[Set[int]] = None,
) -> List[dict]:
    if wanted_numbers:
        wanted_str = ", ".join(str(x) for x in sorted(wanted_numbers))
        scope_line = f"중요: 18~45번 중에서도 반드시 다음 문항만 추출하세요: {wanted_str}.\n"
    else:
        scope_line = "중요: 듣기 평가(1~17번)는 제외하고, 18~45번만 추출하세요.\n"

    base_prompt = (
        "다음 이미지는 2026학년도 수능 영어 영역 문제지(홀수형)의 한 페이지입니다.\n"
        "페이지에 있는 문항들을 추출해서 JSON 배열만 출력하세요.\n"
        + scope_line +
        "각 원소는 다음 키를 가진 객체입니다:\n"
        "- number: 문항 번호 (18-45)\n"
        "- question: 문제 본문 텍스트 (문항 번호/선택지 번호는 제외한 본문)\n"
        "- option_1: 첫 번째 선택지\n"
        "- option_2: 두 번째 선택지\n"
        "- option_3: 세 번째 선택지\n"
        "- option_4: 네 번째 선택지\n"
        "- option_5: 다섯 번째 선택지\n"
        "규칙:\n"
        "1) 정답/해설/배점은 절대 포함하지 마세요.\n"
        "2) 반드시 JSON 배열만 출력하세요(코드블록 금지).\n"
        "3) 선택지(option_1~5)는 가능한 한 빠짐없이 추출하세요(없다면 빈 문자열로).\n"
        "4) 문항이 이 페이지에 없으면 빈 배열 []을 출력하세요.\n"
        "5) JSON은 반드시 파싱 가능해야 합니다.\n"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": base_prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )
    text_out = (resp.output_text or "").strip()

    i = text_out.find("[")
    j = text_out.rfind("]")
    if i < 0 or j < 0 or j <= i:
        return []
    return json.loads(text_out[i : j + 1])

def extract_problems_with_openai_vision(
    problem_pdf: Path,
    model: str,
    detail: str,
    max_output_tokens: int,
    resolution: int,
    wanted_numbers: Optional[List[int]] = None,
    max_retries_per_page: int = 2,
) -> Dict[int, str]:
    """영어 문제 PDF에서 vision 모델로 문제 추출 (선택지 포함)"""
    from openai import OpenAI

    load_dotenv_local()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

    client = OpenAI(api_key=api_key)
    out: Dict[int, str] = {}
    wanted = _wanted_set(wanted_numbers)

    with pdfplumber.open(str(problem_pdf)) as pdf:
        for page_idx, page in enumerate(tqdm(pdf.pages, desc="vision_extract")):
            png = pdf_page_png_bytes(page, resolution=resolution, pdf_path=problem_pdf)
            data_url = b64_data_url(png)
            for _attempt in range(max(1, int(max_retries_per_page))):
                try:
                    items = _extract_from_page_image(
                        client=client,
                        data_url=data_url,
                        model=model,
                        detail=detail,
                        max_output_tokens=max_output_tokens,
                        wanted_numbers=wanted,
                    )
                except Exception:
                    items = []

                got_any = False
                for it in items:
                    qn = int(it.get("number", 0) or 0)
                    if qn < Q_START or qn > Q_END:
                        continue
                    if wanted is not None and qn not in wanted:
                        continue

                    question = _strip_leading_qnum(qn, _norm(it.get("question", "")))
                    option_1 = _norm(it.get("option_1", ""))
                    option_2 = _norm(it.get("option_2", ""))
                    option_3 = _norm(it.get("option_3", ""))
                    option_4 = _norm(it.get("option_4", ""))
                    option_5 = _norm(it.get("option_5", ""))

                    if not question and not any([option_1, option_2, option_3, option_4, option_5]):
                        continue

                    # 문제 텍스트에 선택지 포함하여 포맷팅 (2025_english.jsonl 형식)
                    problem_parts = [f"{qn}. {question}".strip()]

                    problem_parts.append("\n\n\\begin{itemize}")
                    problem_parts.append(f" \\item[1] {option_1}")
                    problem_parts.append(f" \\item[2] {option_2}")
                    problem_parts.append(f" \\item[3] {option_3}")
                    problem_parts.append(f" \\item[4] {option_4}")
                    problem_parts.append(f" \\item[5] {option_5}")
                    problem_parts.append(" \\end{itemize}")

                    _merge_problem(out, qn, "".join(problem_parts))
                    got_any = True

                # 목표 번호 추출 모드에서는 1개라도 건지면 바로 다음 페이지로
                if wanted is not None and got_any:
                    break
    
    # 18-45번 문제 추출 상태 확인 (경고만 출력)
    missing = [q for q in range(Q_START, Q_END + 1) if q not in out]
    if missing:
        print(f"경고: 다음 문제 번호가 추출되지 않았습니다: {missing}")
        print(f"추출된 문제 번호: {sorted(out.keys())}")
        print(f"추출된 문제 수: {len(out)} / {EXPECTED_N}")

    return out

def load_existing_jsonl(out_path: Path) -> Dict[int, str]:
    if not out_path.exists():
        return {}
    out: Dict[int, str] = {}
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            qn = int(it.get("id"))
            prob = _norm(it.get("problem", ""))
            if Q_START <= qn <= Q_END and prob:
                out[qn] = prob
    return out


def build_jsonl(
    problem_pdf: Path,
    answer_pdf: Path,
    out_path: Path,
    vision_model: str,
    vision_detail: str,
    max_output_tokens: int,
    resolution: int,
    force: bool,
    resume: bool,
    allow_partial: bool,
) -> Path:
    """영어 PDF를 JSONL 형식으로 변환"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ans_map = parse_answer_pdf_odd(answer_pdf)
    existing = load_existing_jsonl(out_path) if (resume and out_path.exists()) else {}

    # 1) 전체(또는 누락만) 추출
    need = [q for q in range(Q_START, Q_END + 1) if q not in existing]
    prob_map = dict(existing)
    if need:
        # 먼저 전체 페이지를 훑되, 필요한 번호만 추출하도록 제한(비용 절감 + 누락 보완)
        partial = extract_problems_with_openai_vision(
            problem_pdf=problem_pdf,
            model=vision_model,
            detail=vision_detail,
            max_output_tokens=max_output_tokens,
            resolution=resolution,
            wanted_numbers=need,
            max_retries_per_page=2,
        )
        for k, v in partial.items():
            _merge_problem(prob_map, k, v)

    # 2) 그래도 누락이면: 텍스트 기반 페이지 후보를 찾고, 누락 번호만 페이지 단위로 재시도
    still_missing = [q for q in range(Q_START, Q_END + 1) if q not in prob_map]
    if still_missing:
        print(f"누락 문항 재시도 대상: {still_missing}")
        cand = _find_candidate_pages_for_numbers(problem_pdf, set(still_missing))
        # 후보 페이지가 없으면 전체 페이지 대상으로라도 다시 시도
        for qn in list(still_missing):
            pages = cand.get(qn) or []
            if not pages:
                pages = []  # 빈 경우 아래에서 전체 페이지 루프를 타게 하지 않고, 바로 전체 추출로 fallback

        # 페이지 후보가 제대로 나오지 않는 PDF도 있어서, fallback은 전체 페이지 재시도로 처리
        if any(not (cand.get(q) or []) for q in still_missing):
            fallback = extract_problems_with_openai_vision(
                problem_pdf=problem_pdf,
                model=vision_model,
                detail=vision_detail,
                max_output_tokens=max_output_tokens,
                resolution=resolution,
                wanted_numbers=still_missing,
                max_retries_per_page=3,
            )
            for k, v in fallback.items():
                _merge_problem(prob_map, k, v)

    final_missing = [q for q in range(Q_START, Q_END + 1) if q not in prob_map]
    if final_missing:
        msg = f"최종 누락 문항: {final_missing} (추출 {len(prob_map)}/{EXPECTED_N})"
        if not allow_partial:
            raise RuntimeError(msg)
        print(f"경고(부분 저장): {msg}")

    rows: List[dict] = []

    # 정답 맵의 모든 문제 번호에 대해 처리
    for qn in sorted(ans_map.keys()):
        if qn not in prob_map:
            if allow_partial:
                print(f"경고: 문제 {qn}번이 추출되지 않아 건너뜁니다.")
                continue
            raise RuntimeError(f"문제 {qn}번이 추출되지 않았습니다.")
        
        ai = ans_map[qn]
        rows.append(
            {
                "id": qn,
                "name": str(qn),
                "problem": prob_map[qn],
                "answer": ai.answer,
                "score": ai.score,
                "review": None,
            }
        )

    rows.sort(key=lambda r: int(r["id"]))
    
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"총 {len(rows)}개 문항 변환 완료")
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem_pdf", default="pdf_csat_data/2026수능영어문제.pdf")
    ap.add_argument("--answer_pdf", default="pdf_csat_data/2026수능영어정답.pdf")
    ap.add_argument("--out_jsonl", default="data/2026_english.jsonl")
    ap.add_argument("--vision_model", default="gpt-5.2")
    ap.add_argument("--vision_detail", choices=["low", "high"], default="high")
    ap.add_argument("--max_output_tokens", type=int, default=6000)
    ap.add_argument("--resolution", type=int, default=220)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true", help="기존 out_jsonl이 있으면 누락 문항만 채움")
    ap.add_argument("--allow_partial", action="store_true", help="누락이 있어도 부분 JSONL 저장 허용")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = build_jsonl(
        problem_pdf=Path(args.problem_pdf),
        answer_pdf=Path(args.answer_pdf),
        out_path=Path(args.out_jsonl),
        vision_model=args.vision_model,
        vision_detail=args.vision_detail,
        max_output_tokens=int(args.max_output_tokens),
        resolution=int(args.resolution),
        force=bool(args.force),
        resume=bool(args.resume),
        allow_partial=bool(args.allow_partial),
    )
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
