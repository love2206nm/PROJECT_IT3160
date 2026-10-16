"""
Automated E2E Stress Test Suite & Latency / Pattern Audit for AI Career Copilot.
Executes 10 realistic scenarios against http://localhost:8000/api/v1/copilot/chat/stream.
Measures TTFT, Total Latency, Token Throughput, and structural completeness.
"""

import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")



BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{BASE_URL}/api/v1/copilot/chat/stream"
CV_PATH = Path("CV Nguyen Hai Nam.pdf")

SCENARIOS = [
    {
        "id": "Q1",
        "name": "Out-of-Domain Chit-chat",
        "type": "chat",
        "message": "Hôm nay thời tiết Hà Nội thế nào?",
        "expected_tier": "DIRECT_CHAT",
        "checks": [
            ("has_badge", r"Luồng xử lý:\*\* DIRECT_CHAT"),
            ("helpful_or_polite", r"(thời tiết|nghề nghiệp|chiếc ghim|CV)"),
        ],
    },
    {
        "id": "Q2",
        "name": "System Identity & Guidance",
        "type": "chat",
        "message": "Bạn là ai và có thể giúp gì cho tôi?",
        "expected_tier": "DIRECT_CHAT",
        "checks": [
            ("has_badge", r"Luồng xử lý:\*\* DIRECT_CHAT"),
            ("identity_clarity", r"(AI Career Copilot|Cố vấn)"),
            ("key_features", r"(Phân tích rủi ro|Lộ trình|chiếc ghim|📎)"),
        ],
    },
    {
        "id": "Q3",
        "name": "General Labor Market / AI Risk Trend",
        "type": "chat",
        "message": "Xu hướng tự động hóa của ngành công nghệ thông tin trong 3 năm tới thế nào?",
        "expected_tier": "KNOWLEDGE_RAG",
        "checks": [
            ("has_badge", r"Luồng xử lý:\*\* KNOWLEDGE_RAG"),
            ("grounding_content", r"(tự động hóa|AI|O\*NET|kỹ năng|phần mềm|công nghệ)"),
        ],
    },
    {
        "id": "Q4",
        "name": "Certification & Tool Trade-off",
        "type": "chat",
        "message": "Nên thi chứng chỉ AWS Solutions Architect hay CKA trước để làm Data Engineer?",
        "expected_tier": "KNOWLEDGE_RAG",
        "checks": [
            ("has_badge", r"Luồng xử lý:\*\* KNOWLEDGE_RAG"),
            ("domain_content", r"(AWS|CKA|Data Engineer|Kubernetes|Cloud)"),
        ],
    },
    {
        "id": "Q5",
        "name": "Core Resume Ingestion & Deep Analysis",
        "type": "upload",
        "file_path": CV_PATH,
        "message": "Hãy phân tích chi tiết hồ sơ CV của tôi.",
        "expected_tier": "WORKFLOW",
        "checks": [
            ("has_badge", r"Luồng xử lý:\*\* WORKFLOW"),
            ("has_candidate_name", r"Nguyễn Hải Nam"),
            ("fresher_mode", r"(0\.0 năm|FRESHER|SINH VIÊN)"),
            ("threat_score_17", r"17(\.0)?%"),
            ("sub_scores", r"(sub_task_automation|sub_tool_maturity|sub_agentic_exposure)"),
            ("safe_pivots", r"(Lựa chọn 1|Data Engineer|Quant|AI Engineer)"),
            ("skill_countermeasures", r"(MA TRẬN ĐỐI SÁCH|Thế mạnh chuyển giao)"),
            ("roadmap_30_60_90", r"(30 - 60 - 90|Ngày 1 - 30)"),
        ],
    },
    {
        "id": "Q6",
        "name": "Post-CV Explainability Deep-dive",
        "type": "chat",
        "message": "Tại sao điểm nguy cơ tự động hóa của tôi lại là 17%? Tác vụ nào có nguy cơ bị thay thế cao nhất?",
        "expected_tier": "WORKFLOW",
        "checks": [
            ("mentions_threat_or_onet", r"(17%|nguy cơ|tác vụ|O\*NET|tự động hóa|boilerplate)"),
            ("has_badge", r"⏱️ \*\*Thời gian phản hồi:\*\*"),
        ],
    },
    {
        "id": "Q7",
        "name": "Safe Pivot Trade-off Comparison",
        "type": "chat",
        "message": "So sánh giữa hướng làm Data Engineer và Quant Analyst, tôi nên chọn cái nào và đánh đổi những gì?",
        "expected_tier": "WORKFLOW",
        "checks": [
            ("mentions_roles", r"(Data Engineer|Quant|Analyst)"),
            ("trade_off_analysis", r"(kỹ năng|lương|toán|C\+\+|Python|thời gian)"),
            ("has_badge", r"⏱️ \*\*Thời gian phản hồi:\*\*"),
        ],
    },
    {
        "id": "Q8",
        "name": "Skill Gap & Countermeasures",
        "type": "chat",
        "message": "Tôi còn thiếu công cụ gì và làm thế nào để tận dụng C++/Python hiện tại bù đắp nhanh nhất?",
        "expected_tier": "WORKFLOW",
        "checks": [
            ("mentions_skills", r"(C\+\+|Python|Cloud|Docker|Kafka|Spark|lab|dự án)"),
            ("has_badge", r"⏱️ \*\*Thời gian phản hồi:\*\*"),
        ],
    },
    {
        "id": "Q9",
        "name": "Detailed 30-day Phase 1 Action Plan",
        "type": "chat",
        "message": "Trong 30 ngày đầu tôi cần làm dự án gì cụ thể và lấy chứng chỉ nào?",
        "expected_tier": "WORKFLOW",
        "checks": [
            ("actionable_phase1", r"(30 ngày|Phase 1|dự án|chứng chỉ|AWS|GitHub)"),
            ("has_badge", r"⏱️ \*\*Thời gian phản hồi:\*\*"),
        ],
    },
    {
        "id": "Q10",
        "name": "Strategic Boundary & Academic Decision",
        "type": "chat",
        "message": "Học thạc sĩ Khoa học Dữ liệu có giúp giảm nguy cơ bị AI thay thế hơn so với đi làm ngay không?",
        "expected_tier": "WORKFLOW",
        "checks": [
            ("balanced_strategy", r"(thạc sĩ|kinh nghiệm|AI|nghiên cứu|thực tế|thị trường)"),
            ("has_badge", r"⏱️ \*\*Thời gian phản hồi:\*\*"),
        ],
    },
]


def run_e2e_suite():
    session_id = f"test_e2e_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    print(f"================================================================================")
    print(f"🚀 INITIATING SENIOR DATA ENGINEER E2E AUTOMATED SUITE (10 SCENARIOS)")
    print(f"🎯 Target Endpoint: {STREAM_ENDPOINT}")
    print(f"🆔 Session ID: {session_id}")
    print(f"================================================================================\n")

    client = httpx.Client(timeout=120.0)
    suite_results: List[Dict[str, Any]] = []

    for idx, sc in enumerate(SCENARIOS, 1):
        print(f"--------------------------------------------------------------------------------")
        print(f"[{idx}/10] Testing {sc['id']}: {sc['name']} ({sc['type'].upper()})")
        print(f"Prompt: \"{sc['message']}\"")

        start_time = time.perf_counter()
        ttft_time: Optional[float] = None
        full_tokens: List[str] = []
        raw_sse_events: List[str] = []
        error_msg: Optional[str] = None

        try:
            if sc["type"] == "chat":
                payload = {"session_id": session_id, "message": sc["message"]}
                with client.stream("POST", STREAM_ENDPOINT, json=payload) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP Status {resp.status_code}: {resp.read().decode('utf-8', errors='ignore')}")

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        raw_sse_events.append(line)
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                obj = json.loads(data_str)
                                if "token" in obj:
                                    if ttft_time is None:
                                        ttft_time = time.perf_counter()
                                    full_tokens.append(obj["token"])
                                elif "error" in obj:
                                    error_msg = obj["error"]
                            except json.JSONDecodeError:
                                pass
            elif sc["type"] == "upload":
                cv_file = sc["file_path"]
                if not cv_file.exists():
                    raise FileNotFoundError(f"CV file not found at {cv_file}")
                
                with open(cv_file, "rb") as f:
                    file_bytes = f.read()

                files = {"file": (cv_file.name, file_bytes, "application/pdf")}
                data = {"session_id": session_id, "message": sc["message"]}

                with client.stream("POST", STREAM_ENDPOINT, data=data, files=files) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP Status {resp.status_code}: {resp.read().decode('utf-8', errors='ignore')}")

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        raw_sse_events.append(line)
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                obj = json.loads(data_str)
                                if "token" in obj:
                                    if ttft_time is None:
                                        ttft_time = time.perf_counter()
                                    full_tokens.append(obj["token"])
                                elif "error" in obj:
                                    error_msg = obj["error"]
                            except json.JSONDecodeError:
                                pass

        except Exception as e:
            error_msg = str(e)

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000
        ttft_ms = ((ttft_time - start_time) * 1000) if ttft_time is not None else total_latency_ms
        full_text = "".join(full_tokens)
        char_count = len(full_text)
        token_count = len(full_text.split()) # Approx tokens by words
        throughput_tokens_sec = (token_count / (total_latency_ms / 1000)) if total_latency_ms > 0 else 0

        # Pattern Checks
        check_results = {}
        all_passed = True if not error_msg else False
        for check_name, pattern in sc.get("checks", []):
            matched = bool(re.search(pattern, full_text, re.IGNORECASE))
            check_results[check_name] = matched
            if not matched:
                all_passed = False

        # Extract badge if present
        badge_match = re.search(r"⏱️ \*\*Thời gian phản hồi:\*\* ([\d\.]+)ms \| 🧭 \*\*Luồng xử lý:\*\* (.+)", full_text)
        reported_latency_ms = float(badge_match.group(1)) if badge_match else None
        reported_tier = badge_match.group(2).strip() if badge_match else None

        result_item = {
            "id": sc["id"],
            "name": sc["name"],
            "type": sc["type"],
            "message": sc["message"],
            "status": "PASS" if (all_passed and not error_msg) else "FAIL",
            "error": error_msg,
            "metrics": {
                "ttft_ms": round(ttft_ms, 1),
                "total_latency_ms": round(total_latency_ms, 1),
                "reported_latency_ms": reported_latency_ms,
                "reported_tier": reported_tier,
                "char_count": char_count,
                "token_count": token_count,
                "throughput_tokens_sec": round(throughput_tokens_sec, 1),
            },
            "pattern_checks": check_results,
            "response_preview": full_text[:250].replace("\n", " "),
            "full_response": full_text,
        }
        suite_results.append(result_item)

        status_icon = "✅ PASS" if result_item["status"] == "PASS" else "❌ FAIL"
        print(f"Status: {status_icon} | TTFT: {ttft_ms:.1f}ms | Total: {total_latency_ms:.1f}ms | Throughput: {throughput_tokens_sec:.1f} w/s")
        print(f"Reported Tier: {reported_tier} | Chars: {char_count}")
        print(f"Checks: {check_results}")
        if error_msg:
            print(f"⚠️ Error: {error_msg}")
        print(f"Preview: {result_item['response_preview']}...\n")

    # Save full JSON report
    report_path = Path("scratch/e2e_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"session_id": session_id, "timestamp": time.time(), "results": suite_results}, f, ensure_ascii=False, indent=2)

    print(f"\n================================================================================")
    print(f"📊 SUMMARY AUDIT TABLE ACROSS ALL 10 SCENARIOS")
    print(f"================================================================================")
    print(f"| ID  | Scenario Name                        | Tier             | TTFT (ms) | Total (ms) | Tokens | Status |")
    print(f"| :-- | :----------------------------------- | :--------------- | :-------- | :--------- | :----- | :----- |")
    for r in suite_results:
        m = r["metrics"]
        tier = m.get("reported_tier") or "N/A"
        print(f"| {r['id']:<3} | {r['name'][:36]:<36} | {tier[:16]:<16} | {m['ttft_ms']:>9.1f} | {m['total_latency_ms']:>10.1f} | {m['token_count']:>6} | {r['status']:<6} |")
    print(f"================================================================================")
    print(f"💾 Full audit report written to: {report_path.resolve()}\n")

    return suite_results


if __name__ == "__main__":
    run_e2e_suite()
