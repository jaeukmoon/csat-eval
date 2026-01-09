#!/usr/bin/env python
"""테스트용: 소량 샘플로 빠르게 테스트하는 스크립트

로컬에서만 사용할 것:
- `env.local.example`을 복사해서 `.env.local`을 만들고 OPENAI_API_KEY를 넣으세요.
- `.env.local`은 `.gitignore`에 포함되어 있어 커밋되지 않습니다.
"""
import os
import subprocess
import sys

def load_dotenv_local() -> None:
    for fname in (".env.local", "env.local", ".env"):
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            continue


load_dotenv_local()
if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY가 설정되지 않았습니다. `.env.local`에 키를 넣어주세요.")
    sys.exit(1)

# 테스트 설정
MODEL = "gpt-5.1"
DATASET = "2025_english"
MAX_SAMPLES = 3

print("=" * 60)
print(f"테스트 평가: 모델={MODEL}, 데이터셋={DATASET}")
print(f"샘플 수: {MAX_SAMPLES}")
print("=" * 60)

# 평가 실행
result = subprocess.run(
    [
        sys.executable,
        "main.py",
        "--split", DATASET,
        "--model", MODEL,
        "--max_samples", str(MAX_SAMPLES),
    ],
    env=os.environ,
)

if result.returncode == 0:
    print("[SUCCESS] 테스트 완료!")
else:
    print("[FAILED] 테스트 실패!")
    sys.exit(result.returncode)
