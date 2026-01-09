#!/usr/bin/env python
"""테스트용: 소량 샘플로 빠르게 테스트하는 스크립트"""
import os
import subprocess
import sys

# API 키 확인 및 입력
if not os.getenv("OPENAI_API_KEY"):
    print("OPENAI_API_KEY가 설정되지 않았습니다.")
    api_key = input("OpenAI API 키를 입력하세요: ").strip()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("[ERROR] API 키가 입력되지 않았습니다.")
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
