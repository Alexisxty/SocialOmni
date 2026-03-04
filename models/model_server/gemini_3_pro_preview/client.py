from __future__ import annotations

from models.pipeline.types import InferenceRequest, InferenceResult
from models.utils.openai_compat_tester import OpenAICompatTester
from config.settings import CONFIG


class Gemini3ProPreviewClient:
    @property
    def model_name(self) -> str:
        return "gemini_3_pro_preview"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        use_video = True
        if request.metadata:
            use_video = bool(request.metadata.get("use_video", True))

        model_config = CONFIG.model("gemini_3_pro_preview")
        model_name = model_config.get("model_name", "gemini-3-pro-preview")
        tester = OpenAICompatTester(model_name=model_name)
        raw_answer = tester.call(
            request.video_path,
            request.question,
            user_prompt=user_prompt,
            model_params=model_config,
            include_images=use_video,
        )
        return InferenceResult(answer=raw_answer or "", raw_response=raw_answer)
