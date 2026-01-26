"""국어 PDF에서 JSONL 데이터셋 생성"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pdfplumber
from tqdm import tqdm

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}

# 국어 문항 범위 (1-45)
Q_START = 1
Q_END = 45
EXPECTED_N = Q_END - Q_START + 1

MARK_EVEN = "짝수형"
MARK_ODD = "홀수"


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
    """로컬용 .env를 읽어 환경변수로 로드"""
    # 현재 파일 기준 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).parent.parent
    search_dirs = [Path("."), current_dir]
    
    for base_dir in search_dirs:
        for fname in (".env.local", "env.local", ".env"):
            p = base_dir / fname
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


def parse_answer_pdf_odd(answer_pdf: Path) -> Dict[int, AnswerInfo]:
    """국어 정답 PDF 파싱 (홀수형)"""
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

    data_rows = table[header_idx + 1:]

    out: Dict[int, AnswerInfo] = {}

    def put(q: str, a: str, sc: str):
        if not _norm(q) or not _norm(a) or not _norm(sc):
            return
        try:
            qn = int(_norm(q))
            if qn < Q_START or qn > Q_END:
                return
            ans = _to_int_answer(a)
            score = int(_norm(sc))
            out[qn] = AnswerInfo(answer=ans, score=score)
        except (ValueError, TypeError):
            return

    for row in data_rows:
        if not row:
            continue
        
        # Non-None 셀만 추출
        row = [c for c in row if c]
        if len(row) < 3:
            continue
        
        # 여러 열 그룹 처리 (문항번호, 정답, 배점 순서)
        for i in range(len(row) - 2):
            try:
                qn_str = _norm(row[i])
                if not qn_str:
                    continue
                qn = int(re.sub(r"[^\d]", "", qn_str)[:2]) if re.search(r"\d", qn_str) else 0
                if Q_START <= qn <= Q_END:
                    ans_str = _norm(row[i + 1])
                    score_str = _norm(row[i + 2])
                    if ans_str and score_str:
                        put(str(qn), ans_str, score_str)
            except (ValueError, IndexError):
                continue
        
        # 첫 3개 열도 시도
        if len(row) >= 3:
            put(row[0], row[1], row[2])

    missing = [q for q in range(Q_START, Q_END + 1) if q not in out]
    if missing:
        print(f"경고: 다음 문제 번호의 정답이 없습니다: {missing}")
        if len(missing) > 5:
            raise RuntimeError(f"너무 많은 정답 누락: {len(missing)}개")

    return out


def pdf_page_png_bytes(page, resolution: int, pdf_path: Path) -> bytes:
    """PDF 페이지를 PNG 이미지로 변환 (PyMuPDF 사용)"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF가 필요합니다. 설치: pip install PyMuPDF")
    
    page_num = page.page_number - 1
    
    doc = fitz.open(str(pdf_path))
    pdf_page = doc[page_num]
    
    zoom = resolution / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)
    
    png_bytes = pix.tobytes("png")
    doc.close()
    
    return png_bytes


def b64_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _merge_problem(out: Dict[int, str], qn: int, problem_text: str) -> None:
    if not problem_text:
        return
    prev = out.get(qn, "")
    if len(problem_text) > len(prev):
        out[qn] = problem_text


def _strip_leading_qnum(qn: int, question: str) -> str:
    s = (question or "").lstrip()
    s = re.sub(rf"^\s*{qn}\s*[\.\\)]\s*", "", s)
    s = re.sub(rf"^\s*{qn}\s*[\.\\)]\s*", "", s)
    return s.strip()


def _extract_from_page_image_openai(
    client,
    data_url: str,
    model: str,
    detail: str,
    max_output_tokens: int,
    wanted_numbers: Optional[Set[int]] = None,
) -> List[dict]:
    """OpenAI Vision API를 사용하여 페이지 이미지에서 국어 문제 추출"""
    if wanted_numbers:
        wanted_str = ", ".join(str(x) for x in sorted(wanted_numbers))
        scope_line = f"중요: 1~45번 중에서도 반드시 다음 문항만 추출하세요: {wanted_str}.\n"
    else:
        scope_line = "중요: 1~45번 문항을 모두 추출하세요.\n"

    base_prompt = (
        "다음 이미지는 수능 국어 영역 문제지(홀수형)의 한 페이지입니다.\n"
        "페이지에 있는 문항들을 추출해서 JSON 배열만 출력하세요.\n"
        + scope_line +
        "각 원소는 다음 키를 가진 객체입니다:\n"
        "- number: 문항 번호 (1-45)\n"
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
        "6) 지문이 있는 경우 지문 내용도 question에 포함하세요.\n"
    )

    # OpenAI Responses API 사용
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
    model: str = "gpt-5.2",
    detail: str = "high",
    max_output_tokens: int = 6000,
    resolution: int = 220,
    wanted_numbers: Optional[List[int]] = None,
    max_retries_per_page: int = 2,
) -> Dict[int, str]:
    """OpenAI Vision 모델로 국어 문제 PDF에서 문제 추출
    
    Args:
        problem_pdf: 문제 PDF 파일 경로
        model: OpenAI Vision 모델 이름 (기본값: gpt-5.2)
        detail: 이미지 분석 상세도 (high/low)
        max_output_tokens: 최대 출력 토큰 수
        resolution: PDF 렌더링 해상도
        wanted_numbers: 추출할 문제 번호 목록 (None이면 전체)
        max_retries_per_page: 페이지당 최대 재시도 횟수
    
    Returns:
        문제 번호 -> 문제 텍스트 딕셔너리
    """
    from openai import OpenAI

    load_dotenv_local()
    
    # OpenAI API 키 읽기
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

    client = OpenAI(api_key=api_key)
    
    out: Dict[int, str] = {}
    wanted = set(wanted_numbers) if wanted_numbers else None

    with pdfplumber.open(str(problem_pdf)) as pdf:
        total_pages = len(pdf.pages)
        found_problems = set()
        
        for page_idx, page in enumerate(tqdm(pdf.pages, desc="korean_vision_extract")):
            # 짝수형 페이지는 스킵 (단, 첫 페이지는 제외하고, "짝수형"이 명확히 표시된 경우만)
            txt = page.extract_text() or ""
            
            # 짝수형 페이지 감지: "짝수형"이 있고, 첫 페이지가 아니며, 이미 문제를 찾았을 때만 중단
            if MARK_EVEN in txt and page_idx > 0 and len(found_problems) > 0:
                # 현재 페이지에서 찾은 문제 번호 확인
                page_problems = []
                try:
                    png = pdf_page_png_bytes(page, resolution=resolution, pdf_path=problem_pdf)
                    data_url = b64_data_url(png)
                    test_items = _extract_from_page_image_openai(
                        client=client,
                        data_url=data_url,
                        model=model,
                        detail=detail,
                        max_output_tokens=max_output_tokens,
                        wanted_numbers=wanted,
                    )
                    page_problems = [int(it.get("number", 0)) for it in test_items if it.get("number")]
                except:
                    pass
                
                # 현재 페이지에 홀수형 문제가 없고, 이미 충분히 찾았으면 중단
                if not page_problems and len(found_problems) >= (len(wanted) if wanted else Q_END - Q_START + 1):
                    print(f"[korean_pdf] 짝수형 페이지 감지 (페이지 {page_idx + 1}), 처리 중단")
                    break
                
            png = pdf_page_png_bytes(page, resolution=resolution, pdf_path=problem_pdf)
            data_url = b64_data_url(png)
            for _attempt in range(max(1, int(max_retries_per_page))):
                try:
                    items = _extract_from_page_image_openai(
                        client=client,
                        data_url=data_url,
                        model=model,
                        detail=detail,
                        max_output_tokens=max_output_tokens,
                        wanted_numbers=wanted,
                    )
                except Exception as e:
                    print(f"페이지 {page_idx + 1} 처리 중 오류: {e}")
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

                    problem_parts = [f"{qn}. {question}".strip()]
                    problem_parts.append("\n\n\\begin{itemize}")
                    problem_parts.append(f" \\item[1] {option_1}")
                    problem_parts.append(f" \\item[2] {option_2}")
                    problem_parts.append(f" \\item[3] {option_3}")
                    problem_parts.append(f" \\item[4] {option_4}")
                    problem_parts.append(f" \\item[5] {option_5}")
                    problem_parts.append(" \\end{itemize}")

                    _merge_problem(out, qn, "".join(problem_parts))
                    found_problems.add(qn)
                    got_any = True

                # wanted_numbers가 지정된 경우, 모든 문제를 찾았으면 중단
                if wanted is not None:
                    found_all = all(qn in out for qn in wanted)
                    if found_all:
                        print(f"[korean_pdf] 모든 요청된 문제를 찾았습니다. (페이지 {page_idx + 1}/{total_pages})")
                        break
                    if got_any:
                        break
    
    missing = [q for q in range(Q_START, Q_END + 1) if q not in out]
    if missing:
        print(f"경고: 다음 문제 번호가 추출되지 않았습니다: {missing}")

    return out


def build_korean_jsonl(
    problem_pdf: Path,
    answer_pdf: Path,
    out_path: Path,
    vision_model: str = "gpt-5.2",
    vision_detail: str = "high",
    max_output_tokens: int = 6000,
    resolution: int = 220,
    force: bool = False,
    allow_partial: bool = True,
    wanted_numbers: Optional[List[int]] = None,
) -> Path:
    """국어 PDF에서 JSONL 데이터셋 생성
    
    Args:
        problem_pdf: 문제 PDF 파일 경로
        answer_pdf: 정답 PDF 파일 경로
        out_path: 출력 JSONL 파일 경로
        vision_model: OpenAI Vision 모델 이름 (기본값: gpt-5.2)
        vision_detail: 이미지 분석 상세도 (high/low)
        max_output_tokens: 최대 출력 토큰 수
        resolution: PDF 렌더링 해상도
        force: 기존 파일 덮어쓰기 여부
        allow_partial: 일부 문제 누락 허용 여부
        wanted_numbers: 추출할 문제 번호 목록 (None이면 전체)
    
    Returns:
        생성된 JSONL 파일 경로
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[korean_pdf] Already exists: {out_path}")
        return out_path

    ans_map = parse_answer_pdf_odd(answer_pdf)
    
    # wanted_numbers가 지정된 경우 해당 문제만 필터링
    if wanted_numbers:
        ans_map = {qn: ans_map[qn] for qn in wanted_numbers if qn in ans_map}
    
    prob_map = extract_problems_with_openai_vision(
        problem_pdf=problem_pdf,
        model=vision_model,
        detail=vision_detail,
        max_output_tokens=max_output_tokens,
        resolution=resolution,
        wanted_numbers=wanted_numbers,
    )

    rows: List[dict] = []

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

    print(f"[korean_pdf] Created: {out_path} ({len(rows)} items)")
    return out_path
