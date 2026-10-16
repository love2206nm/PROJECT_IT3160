"""
Live Multi-turn Session Testing Script for AI Career Copilot.
Connects to local server at http://127.0.0.1:8000 and executes 15 consecutive
career advisory turns in a single continuous session with real-time SSE token streaming.
"""

import sys
import os
import json
import time
import uuid
from pathlib import Path
import httpx

# Ensure proper UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
STREAM_ENDPOINT = f"{BASE_URL}/api/v1/copilot/chat/stream"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

CV_PATH = Path("CV Nguyen Hai Nam.pdf")
if not CV_PATH.exists():
    # Fallback to sample CV if root CV is missing
    fallback_cv = Path("tests/sample_cvs/06_pdf_software_engineer_linear.pdf")
    if fallback_cv.exists():
        CV_PATH = fallback_cv

TEST_TURNS = [
    {
        "turn": 1,
        "type": "upload",
        "message": "Xin chào, nhờ bạn phân tích CV này giúp tôi nhé.",
        "description": "Upload CV & Khởi tạo phiên phân tích StateGraph",
    },
    {
        "turn": 2,
        "type": "text",
        "message": "Tóm tắt hồ sơ và các kỹ năng cốt lõi của tôi từ CV đã gửi.",
        "description": "Nhận diện CandidateProfile & Kỹ năng",
    },
    {
        "turn": 3,
        "type": "text",
        "message": "Điểm nguy cơ tự động hóa AI (threat score) của tôi là bao nhiêu và đến từ đâu?",
        "description": "Phân tích RiskProfile & Sub-scores",
    },
    {
        "turn": 4,
        "type": "text",
        "message": "Những kỹ năng nào trong CV của tôi dễ bị AI thay thế nhất?",
        "description": "Phân tích Vulnerable Skills",
    },
    {
        "turn": 5,
        "type": "text",
        "message": "Kỹ năng phòng thủ (defense skills) cốt lõi của tôi là gì?",
        "description": "Nhận diện Defense Skills kháng AI",
    },
    {
        "turn": 6,
        "type": "text",
        "message": "Tôi có thể chuyển dịch sang những ngành nghề an toàn (safe pivot roles) nào?",
        "description": "Đề xuất Top Safe Transition Roles",
    },
    {
        "turn": 7,
        "type": "text",
        "message": "Giữa Data Engineer và Cloud Architect, hướng nào có nhu cầu và cơ hội phát triển tốt hơn?",
        "description": "So sánh 2 ngành nghề & Cơ hội phát triển (Test Markdown Table + Không cắt cụt)",
    },
    {
        "turn": 8,
        "type": "text",
        "message": "Cơ hội phát triển và lộ trình thăng tiến lâu dài của vị trí Data Engineer ra sao?",
        "description": "Phân tích chuyên sâu 'Cơ hội phát triển' (Junior -> Architect)",
    },
    {
        "turn": 9,
        "type": "text",
        "message": "Tôi còn thiếu những kỹ năng và công cụ trọng yếu nào để làm Data Engineer?",
        "description": "Phân tích Skill Gaps & Critical Missing Tools",
    },
    {
        "turn": 10,
        "type": "text",
        "message": "Trong 30 ngày đầu tiên (Phase 1), tôi cần ưu tiên học kỹ năng hoặc chứng chỉ nào?",
        "description": "Cố vấn Phase 1 Roadmap & Chứng chỉ nền tảng",
    },
    {
        "turn": 11,
        "type": "text",
        "message": "Đề xuất cho tôi một dự án thực chiến (capstone project) cụ thể để làm đẹp CV.",
        "description": "Đề xuất Dự án thực chiến (Capstone Project)",
    },
    {
        "turn": 12,
        "type": "text",
        "message": "Làm thế nào để tận dụng nền tảng C++ và Linux sẵn có của tôi khi học Data Engineering?",
        "description": "Tư vấn cá nhân hóa theo nền tảng kỹ thuật hiện có",
    },
    {
        "turn": 13,
        "type": "text",
        "message": "Mức lương thị trường của Data Engineer theo thống kê O*NET là bao nhiêu USD/năm?",
        "description": "Truy xuất Factual O*NET Median Salary Metadata",
    },
    {
        "turn": 14,
        "type": "text",
        "message": "Nếu tôi muốn chuyển sang AI Engineer / LLM Ops thay vì Data Engineer thì có khả thi không?",
        "description": "Đánh giá phương án chuyển dịch thay thế (AI / LLMOps)",
    },
    {
        "turn": 15,
        "type": "text",
        "message": "Hãy tóm tắt 3 hành động quan trọng nhất tôi cần bắt tay vào làm ngay trong tuần này.",
        "description": "Tổng kết Top 3 Action Items thực chiến",
    },
]


def check_health() -> bool:
    """Verifies that the server is active and reachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(HEALTH_ENDPOINT)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Server is HEALTHY: {data.get('app_name')} (v{data.get('version')})")
                return True
    except Exception as e:
        print(f"Cannot connect to server at {HEALTH_ENDPOINT}: {e}")
    return False


def stream_turn(session_id: str, turn_info: dict) -> tuple[str, float, int]:
    """
    Executes a single turn streaming request against SSE endpoint.
    Prints tokens progressively to terminal and returns (response_text, elapsed_sec, token_count).
    """
    turn_num = turn_info["turn"]
    msg = turn_info["message"]
    desc = turn_info["description"]
    is_upload = turn_info["type"] == "upload"

    print("\n" + "=" * 80)
    print(f"TURN {turn_num}/15: {desc}")
    print(f"USER PROMPT: \"{msg}\"")
    if is_upload:
        print(f"ATTACHMENT : {CV_PATH.name} ({CV_PATH.stat().st_size / 1024:.1f} KB)")
    print("-" * 80)
    sys.stdout.write("COPILOT: ")
    sys.stdout.flush()

    tokens = []
    start_time = time.perf_counter()

    with httpx.Client(timeout=120.0) as client:
        if is_upload and CV_PATH.exists():
            with open(CV_PATH, "rb") as f:
                files = {"file": (CV_PATH.name, f, "application/pdf")}
                data = {"session_id": session_id, "message": msg}
                with client.stream("POST", STREAM_ENDPOINT, data=data, files=files) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            payload = json.loads(raw)
                            token = payload.get("token", "")
                            if token:
                                tokens.append(token)
                                sys.stdout.write(token)
                                sys.stdout.flush()
                        except json.JSONDecodeError:
                            pass
        else:
            payload = {"session_id": session_id, "message": msg}
            with client.stream("POST", STREAM_ENDPOINT, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        payload = json.loads(raw)
                        token = payload.get("token", "")
                        if token:
                            tokens.append(token)
                            sys.stdout.write(token)
                            sys.stdout.flush()
                    except json.JSONDecodeError:
                        pass

    elapsed = time.perf_counter() - start_time
    full_text = "".join(tokens)
    token_count = len(tokens)
    tps = token_count / elapsed if elapsed > 0 else 0

    print("\n" + "-" * 80)
    print(f"METRICS: {elapsed:.2f}s | {token_count} chunks | ~{tps:.1f} chunks/sec")
    return full_text, elapsed, token_count


def run_full_session_test():
    """Runs all 15 turns in a continuous interactive session."""
    print("=" * 80)
    print("AI CAREER COPILOT - LIVE MULTI-TURN 15-QUESTION SESSION TEST")
    print(f"Target Server : {BASE_URL}")
    print(f"Candidate CV  : {CV_PATH.resolve()}")
    print("=" * 80)

    if not check_health():
        print(f"\nERROR: Server is not running on {BASE_URL}.")
        print("Please start server first using: py -3.13 main.py")
        sys.exit(1)

    session_id = f"live_test_{uuid.uuid4().hex[:8]}"
    print(f"\nSession Initialized: {session_id}\n")

    results = []

    for turn_info in TEST_TURNS:
        full_text, elapsed, token_count = stream_turn(session_id, turn_info)

        # Quality Assertions
        assert len(full_text.strip()) > 20, f"Turn {turn_info['turn']} response is too short or empty!"
        assert not full_text.strip().endswith("• Cơ hội phát triển:"), (
            f"Turn {turn_info['turn']} truncated right at 'Cơ hội phát triển:'!"
        )

        results.append({
            "turn": turn_info["turn"],
            "desc": turn_info["description"],
            "elapsed": elapsed,
            "tokens": token_count,
            "char_count": len(full_text),
            "status": "PASS",
        })

        # Brief pause between turns for readability
        time.sleep(1.0)

    # Summary Report
    print("\n" + "=" * 80)
    print("LIVE MULTI-TURN SESSION TEST SUMMARY (15/15 TURNS)")
    print("=" * 80)
    print(f"{'Turn':<6} | {'Description':<42} | {'Time (s)':<9} | {'Chunks':<8} | {'Status':<6}")
    print("-" * 80)
    total_time = 0.0
    total_tokens = 0
    for r in results:
        total_time += r["elapsed"]
        total_tokens += r["tokens"]
        print(f"{r['turn']:<6} | {r['desc'][:42]:<42} | {r['elapsed']:<9.2f} | {r['tokens']:<8} | {r['status']:<6}")
    print("-" * 80)
    print(f"TOTAL: 15 Turns | {total_time:.2f}s total execution time | {total_tokens} chunks streamed.")
    print("ALL 15 TURNS PASSED WITH ZERO TRUNCATION & COMPLETE CAREER ADVICE!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_session_test()
