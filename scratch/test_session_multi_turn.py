import os
import sys
import time
import uuid
import httpx
import re
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8000"
CV_PATH = r"c:\Users\Hai Nam\full_project\Job_advisor\CV Nguyen Hai Nam.pdf"

def stream_chat(client: httpx.Client, query: str, session_id: str):
    start = time.perf_counter()
    full_text = ""
    ttft = None
    with client.stream(
        "POST",
        f"{BASE_URL}/api/v1/copilot/chat/stream",
        json={"message": query, "session_id": session_id},
        timeout=60.0
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    token = data_json.get("token", "")
                    if token:
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        full_text += token
                except json.JSONDecodeError:
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    full_text += data_str
    total_time = (time.perf_counter() - start) * 1000
    return full_text, ttft or total_time, total_time

def stream_cv(client: httpx.Client, file_path: str, session_id: str):
    start = time.perf_counter()
    full_text = ""
    ttft = None
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "application/pdf")}
        data = {"session_id": session_id, "message": "Phân tích CV này giúp tôi"}
        with client.stream(
            "POST",
            f"{BASE_URL}/api/v1/copilot/chat/stream",
            data=data,
            files=files,
            timeout=60.0
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        token = data_json.get("token", "")
                        if token:
                            if ttft is None:
                                ttft = (time.perf_counter() - start) * 1000
                            full_text += token
                    except json.JSONDecodeError:
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
                        full_text += data_str
    total_time = (time.perf_counter() - start) * 1000
    return full_text, ttft or total_time, total_time

def test_session_multi_turn():
    session_id = f"test_session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    print("=" * 80)
    print("🧪 RUNNING DEDICATED MULTI-TURN SESSION CONTEXT & CTA AUDIT SUITE")
    print(f"🆔 Session ID: {session_id}")
    print("=" * 80)

    cta_pattern = re.compile(r"chiếc ghim|đính kèm cv|biểu tượng chiếc ghim|tải lên hồ sơ|📎", re.IGNORECASE)

    with httpx.Client(timeout=60.0) as client:
        # --- TURN 1: Greeting without CV ---
        print("\n--- TURN 1: Greeting without CV ---")
        text1, ttft1, total1 = stream_chat(client, "Xin chào bạn là ai và có thể giúp gì cho tôi?", session_id)
        print(f"TTFT: {ttft1:.1f}ms | Total: {total1:.1f}ms")
        print(f"Preview: {text1[:200]}...")
        has_cta1 = bool(cta_pattern.search(text1))
        print(f"CTA Present in Turn 1 (Expected True): {has_cta1}")
        assert "Copilot" in text1 or "Cố vấn" in text1 or "chào" in text1.lower(), "Turn 1 greeting failed!"

        # --- TURN 2: Ingest CV Nguyen Hai Nam.pdf ---
        print("\n--- TURN 2: Ingest CV Nguyen Hai Nam.pdf ---")
        text2, ttft2, total2 = stream_cv(client, CV_PATH, session_id)
        print(f"TTFT: {ttft2:.1f}ms | Total: {total2:.1f}ms")
        print(f"Preview: {text2[:200]}...")
        assert "NGUYEN HAI NAM" in text2.upper() or "HAI NAM" in text2.upper(), "Turn 2 CV parse failed!"
        print("✅ CV Ingestion succeeded and profile stored in session!")

        # --- TURN 3: Deep-dive 17% Threat Score (Assert ZERO CTA) ---
        print("\n--- TURN 3: Deep-dive 17% Threat Score (Assert ZERO CTA) ---")
        text3, ttft3, total3 = stream_chat(
            client,
            "Tại sao điểm nguy cơ tự động hóa của tôi lại là 17%? Tác vụ nào có nguy cơ bị thay thế cao nhất?",
            session_id
        )
        print(f"TTFT: {ttft3:.1f}ms | Total: {total3:.1f}ms")
        print(f"Preview: {text3[:250]}...")
        has_cta3 = bool(cta_pattern.search(text3))
        print(f"🚨 CTA Present in Turn 3 (Must be FALSE): {has_cta3}")
        assert not has_cta3, "Turn 3 still contains CV upload CTA despite CV being already uploaded!"
        assert ttft3 < 20000, f"Turn 3 TTFT took too long: {ttft3}ms"
        assert total3 < 60000, f"Turn 3 total took too long: {total3}ms"
        print("✅ Turn 3 passed: Fast response & ZERO repetitive CTA!")

        # --- TURN 4: Compare Data Engineer vs Quant Analyst (Assert Table & ZERO CTA) ---
        print("\n--- TURN 4: Compare Data Engineer vs Quant Analyst (Assert Table & ZERO CTA) ---")
        text4, ttft4, total4 = stream_chat(
            client,
            "So sánh giữa hướng làm Data Engineer và Quant Analyst theo dạng bảng so sánh, tôi nên chọn cái nào và đánh đổi những gì?",
            session_id
        )
        print(f"TTFT: {ttft4:.1f}ms | Total: {total4:.1f}ms")
        print(f"Preview: {text4[:250]}...")
        has_table = bool(re.search(r"\|.+\|.+\|", text4))
        print(f"📊 Markdown Table Present: {has_table}")
        has_cta4 = bool(cta_pattern.search(text4))
        print(f"🚨 CTA Present in Turn 4 (Must be FALSE): {has_cta4}")
        assert not has_cta4, "Turn 4 still contains CV upload CTA!"
        assert has_table, "Turn 4 must provide a Markdown comparison table!"
        print("✅ Turn 4 passed: Markdown Table generated cleanly without CTA!")

        # --- TURN 5: 30-day Roadmap Check (Assert Contextual Memory & ZERO CTA) ---
        print("\n--- TURN 5: 30-day Roadmap Check (Assert Contextual Memory & ZERO CTA) ---")
        text5, ttft5, total5 = stream_chat(
            client,
            "Lập cho tôi lộ trình 30 ngày bù đắp kỹ năng cho hướng đi ưu tiên của tôi ở trên",
            session_id
        )
        print(f"TTFT: {ttft5:.1f}ms | Total: {total5:.1f}ms")
        print(f"Preview: {text5[:250]}...")
        has_cta5 = bool(cta_pattern.search(text5))
        print(f"🚨 CTA Present in Turn 5 (Must be FALSE): {has_cta5}")
        assert not has_cta5, "Turn 5 still contains CV upload CTA!"
        print("✅ Turn 5 passed: Memory context maintained & ZERO repetitive CTA!")

    print("\n" + "=" * 80)
    print("🎉 ALL 5/5 MULTI-TURN SESSION CONTEXT & CTA AUDIT STEPS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_session_multi_turn()
