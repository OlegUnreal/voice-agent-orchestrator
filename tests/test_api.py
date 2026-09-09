import json

from fastapi.testclient import TestClient

from voice_agent.api.main import create_app
from voice_agent.config import Settings


def test_health_metrics_and_websocket_text():
    app = create_app(Settings(provider_api_key=""))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert client.get("/metrics").status_code == 200
        with client.websocket_connect("/ws/voice?session_id=demo") as ws:
            ws.send_text(json.dumps({"type": "user_text", "text": "Say hello in one word."}))
            types = []
            for _ in range(8):
                payload = json.loads(ws.receive_text())
                types.append(payload["type"])
                if payload["type"] == "assistant.done":
                    break
            assert "assistant.start" in types
            assert "assistant.done" in types
        traces = client.get("/traces/demo")
        assert traces.status_code == 200
        assert traces.json()["turns"]
        demo = client.get("/demo/")
        assert demo.status_code == 200
        assert b"Voice agent" in demo.content


def test_websocket_passthrough_audio():
    app = create_app(Settings(provider_api_key=""))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/voice?session_id=audio") as ws:
            ws.send_text(
                json.dumps(
                    {
                        "type": "user_audio",
                        "audio_b64": "aGVsbG8=",  # "hello"
                        "mime": "text/plain",
                    }
                )
            )
            saw_transcript = False
            for _ in range(8):
                payload = json.loads(ws.receive_text())
                if payload["type"] == "transcript.final":
                    saw_transcript = True
                    assert payload["text"] == "hello"
                if payload["type"] == "assistant.done":
                    break
            assert saw_transcript
