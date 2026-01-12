"""수학 PDF에서 JSONL 데이터셋 생성"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pdfplumber
from tqdm import tqdm

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
SUBJECTS = ("common", "probstat", "calculus", "geometry")

MARK_PROB = "선택 과목(확률과 통계)"
MARK_CALC = "선택 과목(미적분)"
MARK_GEOM = "선택 과목(기하)"
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


def parse_answer_pdf_odd(answer_pdf: Path) -> Dict[Tuple[str, int], AnswerInfo]:
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

    out: Dict[Tuple[str, int], AnswerInfo] = {}

    def put(subject: str, q: str, a: str, sc: str):
        if not _norm(q) or not _norm(a) or not _norm(sc):
            return
        qn = int(_norm(q))
        ans = _to_int_answer(a)
        score = int(_norm(sc))
        out[(subject, qn)] = AnswerInfo(answer=ans, score=score)

    for row in data_rows:
        if not row:
            continue
        row = (row + [None] * 15)[:15]
        put("common", row[0], row[1], row[2])
        put("common", row[3], row[4], row[5])
        put("probstat", row[6], row[7], row[8])
        put("calculus", row[9], row[10], row[11])
        put("geometry", row[12], row[13], row[14])

    for q in range(1, 23):
        if ("common", q) not in out:
            raise RuntimeError(f"공통 정답 누락: {q}")
    for subj in ("probstat", "calculus", "geometry"):
        for q in range(23, 31):
            if (subj, q) not in out:
                raise RuntimeError(f"선택({subj}) 정답 누락: {q}")

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


def extract_problems_with_openai_vision(
    problem_pdf: Path,
    model: str,
    detail: str,
    max_output_tokens: int,
    resolution: int,
) -> Dict[Tuple[str, int], str]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

    client = OpenAI(api_key=api_key)
    out: Dict[Tuple[str, int], str] = {}

    base_prompt = (
        "다음 이미지는 수능 수학 영역 문제지(홀수형)의 한 페이지입니다.\n"
        "페이지에 있는 문항들을 모두 추출해서 JSON 배열만 출력하세요.\n"
        "각 원소는 다음 키를 가진 객체입니다:\n"
        "- subject: common | probstat | calculus | geometry\n"
        "- orig_id: 시험지에 인쇄된 문항 번호 (공통 1-22, 선택 23-30)\n"
        "- problem: 문항 전체 텍스트(선택지 포함), 보기/그림 설명 포함. 수식은 LaTeX로 써도 됩니다.\n"
        "규칙:\n"
        "1) 정답/해설/배점은 절대 포함하지 마세요.\n"
        "2) 반드시 JSON 배열만 출력하세요(코드블록 금지).\n"
        "3) subject는 페이지 상단의 '선택 과목(확률과 통계/미적분/기하)' 표기를 기준으로 결정하세요. 해당 표기가 없으면 common.\n"
    )

    with pdfplumber.open(str(problem_pdf)) as pdf:
        for page_idx, page in enumerate(tqdm(pdf.pages, desc="vision_extract")):
            txt = page.extract_text() or ""
            if MARK_EVEN in txt and page_idx > 0:
                break

            png = pdf_page_png_bytes(page, resolution=resolution, pdf_path=problem_pdf)
            data_url = b64_data_url(png)

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
                continue
            items = json.loads(text_out[i : j + 1])

            for it in items:
                subj = _norm(it.get("subject"))
                if subj not in SUBJECTS:
                    continue
                q = int(it.get("orig_id"))
                prob = _norm(it.get("problem"))
                if not prob:
                    continue
                if subj == "common" and not (1 <= q <= 22):
                    continue
                if subj != "common" and not (23 <= q <= 30):
                    continue
                out[(subj, q)] = prob

    for q in range(1, 23):
        if ("common", q) not in out:
            raise RuntimeError(f"문제 추출 누락(공통): {q}")
    for subj in ("probstat", "calculus", "geometry"):
        for q in range(23, 31):
            if (subj, q) not in out:
                raise RuntimeError(f"문제 추출 누락(선택 {subj}): {q}")

    return out


def unified_id(subject: str, orig_id: int) -> int:
    if subject == "common":
        return orig_id
    if subject == "probstat":
        return orig_id
    if subject == "calculus":
        return orig_id + 8
    if subject == "geometry":
        return orig_id + 16
    raise ValueError(subject)


def row_name(uid: int) -> str:
    return str(uid)


def build_math_jsonl(
    problem_pdf: Path,
    answer_pdf: Path,
    out_path: Path,
    vision_model: str = "gpt-5.2",
    vision_detail: str = "high",
    max_output_tokens: int = 6000,
    resolution: int = 220,
    force: bool = False,
) -> Path:
    """수학 PDF에서 JSONL 데이터셋 생성"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[math_pdf] Already exists: {out_path}")
        return out_path

    ans_map = parse_answer_pdf_odd(answer_pdf)
    prob_map = extract_problems_with_openai_vision(
        problem_pdf=problem_pdf,
        model=vision_model,
        detail=vision_detail,
        max_output_tokens=max_output_tokens,
        resolution=resolution,
    )

    rows: List[dict] = []

    for q in range(1, 23):
        subj = "common"
        uid = unified_id(subj, q)
        ai = ans_map[(subj, q)]
        rows.append(
            {
                "id": uid,
                "name": row_name(uid),
                "problem": prob_map[(subj, q)],
                "answer": ai.answer,
                "score": ai.score,
                "review": None,
            }
        )

    for subj in ("probstat", "calculus", "geometry"):
        for q in range(23, 31):
            uid = unified_id(subj, q)
            ai = ans_map[(subj, q)]
            rows.append(
                {
                    "id": uid,
                    "name": row_name(uid),
                    "problem": prob_map[(subj, q)],
                    "answer": ai.answer,
                    "score": ai.score,
                    "review": None,
                }
            )

    rows.sort(key=lambda r: int(r["id"]))
    if len(rows) != 46:
        raise RuntimeError(f"문항 수 오류: {len(rows)}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[math_pdf] Created: {out_path}")
    return out_path
