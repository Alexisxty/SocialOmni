from __future__ import annotations

import os
import time
from typing import Optional

import requests

from config.settings import CONFIG
from models.pipeline.types import InferenceRequest, InferenceResult


class BaichuanOmni15Client:
    @property
    def model_name(self) -> str:
        return "baichuan_omni_1_5"

    def _extract_clean_answer(self, raw_answer: str) -> str:
        if not raw_answer:
            return ""

        lines = [line.strip() for line in raw_answer.split("\n") if line.strip()]
        if not lines:
            return raw_answer.strip()

        last_line = lines[-1]
        if last_line[:1].upper() in {"A", "B", "C", "D"} and len(last_line) == 1:
            return last_line.upper()

        return raw_answer.strip()

    def _call_api(
        self,
        server_url: str,
        video_path: str,
        question: str,
        use_video: bool = True,
        max_retries: int = 5,
        retry_delay: float = 0.0,
    ) -> Optional[str]:
        for attempt in range(1, max_retries + 1):
            try:
                with open(video_path, "rb") as video_file:
                    response = requests.post(
                        f"{server_url.rstrip('/')}/analyze",
                        files={"video": video_file},
                        data={
                            "question": question,
                            "use_video": str(use_video).lower(),
                            "use_audio": "false",
                        },
                        timeout=300,
                    )

                if response.status_code == 200:
                    return response.json().get("answer", "")
            except Exception:  # noqa: BLE001
                pass

            if attempt < max_retries:
                time.sleep(retry_delay)

        return None

    def _build_prompt(self, request: InferenceRequest) -> str:
        asr_content = ""
        if request.metadata:
            asr_content = request.metadata.get("asr_content") or ""
        options = request.options or []
        question = request.question or ""
        parts = []
        if asr_content:
            parts.append("ASR Transcript:\n" + asr_content.strip())
        if options:
            parts.append("Options:\n" + "\n".join(options))
        if question:
            parts.append(question.strip())
        parts.append("Answer ONLY with the option letter (A, B, C, or D).")
        return "\n\n".join(parts)

    def predict(self, request: InferenceRequest) -> InferenceResult:
        model_config = CONFIG.model("baichuan_omni_1_5")
        server_url = request.metadata.get("server_url") if request.metadata else None
        server_url = server_url or model_config.get("server_url") or os.getenv("BAICHUAN_OMNI_1_5_SERVER_URL")
        if not server_url:
            raise ValueError("Missing Baichuan-Omni-1.5 server_url. Please configure it in config/config.yaml or environment variables.")

        full_question = self._build_prompt(request)

        use_video = True
        if request.metadata:
            use_video = bool(request.metadata.get("use_video", True))

        raw_answer = self._call_api(
            server_url,
            request.video_path,
            full_question,
            use_video=use_video,
            max_retries=CONFIG.runtime("max_retries", 5),
            retry_delay=CONFIG.runtime("request_delay", 0.0),
        )
        clean_answer = self._extract_clean_answer(raw_answer or "")
        return InferenceResult(answer=clean_answer, raw_response=raw_answer)
