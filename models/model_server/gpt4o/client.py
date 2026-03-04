from __future__ import annotations

from models.pipeline.types import InferenceRequest, InferenceResult
from models.utils.openai_compat_tester import OpenAICompatTester
from config.settings import CONFIG


class GPT4oClient:
    @property
    def model_name(self) -> str:
        return "gpt4o"

    def predict(self, request: InferenceRequest) -> InferenceResult:
        # level1_pipeline already assembles ASR/options/answer_format into user_prompt; avoid duplicate assembly here.
        user_prompt = request.metadata.get("user_prompt") if request.metadata else None
        use_video = True
        if request.metadata:
            use_video = bool(request.metadata.get("use_video", True))

        model_config = CONFIG.model("gpt4o")
        model_name = model_config.get("model_name", "gpt-4o")
        tester = OpenAICompatTester(model_name=model_name)
        raw_answer = tester.call(
            request.video_path,
            request.question,
            user_prompt=user_prompt,
            model_params=model_config,
            include_images=use_video,
        )
        return InferenceResult(answer=raw_answer or "", raw_response=raw_answer)
