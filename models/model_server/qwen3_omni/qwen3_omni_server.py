import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import argparse
import warnings
import logging
import traceback
from flask import Flask, request, jsonify

from config.settings import CONFIG
from config.paths import PATHS

# GPU configuration - use dual H100 cards (must be set before importing transformers).
SPECIFIED_GPUS = CONFIG.model("qwen3_omni").get("gpu_ids", []) or CONFIG.runtime("gpu_ids", []) or [4, 5]  # Two H100 80GB GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, SPECIFIED_GPUS))

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from models.model_server.local_common.transformers_compat import ensure_qwen3_omni_config_compat

warnings.filterwarnings('ignore')

app = Flask(__name__)
logger = logging.getLogger("qwen3_omni_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    model_log_dir = PATHS.results_logs / "qwen3_omni"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = model_log_dir / "server.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Global configuration
MODEL_PATH = CONFIG.model("qwen3_omni").get("model_path") or "/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct"
USE_AUDIO_IN_VIDEO = CONFIG.model("qwen3_omni").get("use_audio_in_video", True)
MAX_TOKENS = CONFIG.model("qwen3_omni").get("max_tokens", 50)

# Global variables
model = None
processor = None
model_loaded = False


def _parse_bool(value, default=True):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


def load_model():
    """Load model onto specified dual GPUs"""
    global model, processor, model_loaded

    if model_loaded:
        return

    try:
        # Use the two specified GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, SPECIFIED_GPUS))

        print(f"Loading model on GPU {SPECIFIED_GPUS}...")

        ensure_qwen3_omni_config_compat()
        # Load model with transformers (auto-optimized, distributed across two GPUs)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto",  # Automatically distribute across multiple GPUs
            dtype='auto'
        )

        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        model_loaded = True
        print(f"Model loaded successfully！Using GPU {SPECIFIED_GPUS}")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise


def build_conversation(video_path, question):
    """Build conversation format"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ],
        }
    ]


def process_video_analysis(video_path, question, use_video, use_audio):
    """Process video analysis"""
    global model, processor
    use_audio_in_video = USE_AUDIO_IN_VIDEO and use_audio

    # Build conversation
    messages = build_conversation(video_path, question)

    # Prepare inputs
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    if not use_audio:
        audios = None
    if not use_video:
        images = None
        videos = None

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference
    result = model.generate(
        **inputs,
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=MAX_TOKENS,
        thinker_do_sample=False,
        use_audio_in_video=use_audio_in_video,
        return_audio=False
    )

    sequences = None
    if hasattr(result, "sequences"):
        sequences = result.sequences
    elif isinstance(result, tuple) and result:
        sequences = result[0].sequences if hasattr(result[0], "sequences") else result[0]
    elif isinstance(result, str):
        return result
    else:
        sequences = result

    response = processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "gpus": SPECIFIED_GPUS
    })


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze video endpoint - file upload only"""
    global model, processor

    if not model_loaded:
        return jsonify({"error": "Model is not loaded"}), 500

    temp_dir = None
    temp_path = None

    try:
        # File upload only
        if 'video' not in request.files:
            return jsonify({"error": "Video file not uploaded"}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        question = request.form.get('question', '')
        use_video = _parse_bool(request.form.get("use_video"), True)
        use_audio = _parse_bool(request.form.get("use_audio"), USE_AUDIO_IN_VIDEO)
        if not question.strip():
            return jsonify({"error": "Question cannot be empty"}), 400

        # Save uploaded file to temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(temp_path)

        # Process video analysis
        answer = process_video_analysis(temp_path, question, use_video, use_audio)

        # Simplified response format
        return jsonify({
            "status": "success",
            "answer": answer.strip()
        })

    except Exception as e:
        logger.error("Analyze failed: %s", e)
        logger.error("Traceback:\n%s", traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    finally:
        # Clean temporary files
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temporary files: {str(e)}")


def parse_args():
    """Parse command-line arguments"""
    default_host = CONFIG.model("qwen3_omni").get("host") or "127.0.0.1"
    default_port = CONFIG.model("qwen3_omni").get("port") or 5090
    parser = argparse.ArgumentParser(description="Qwen3 Omni Video Analysis Server")
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Server port (default: 5090)"
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help="Server host address (default: 127.0.0.1)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Load model on startup
    load_model()

    # Start server
    print(f"Starting server: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
