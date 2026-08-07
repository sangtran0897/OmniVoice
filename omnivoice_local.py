# -*- coding: utf-8 -*-
"""OmniVoice TTS local voi giao dien Gradio.

Chay:
    python omnivoice_gradio.py

Sau khi khoi dong, mo dia chi duoc in trong Terminal, thuong la:
    http://127.0.0.1:7860
"""

import re
import shutil
import subprocess
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# Tuy cau truc repo, OmniVoice co the duoc export theo mot trong hai cach nay.
try:
    from omnivoice import OmniVoice
except ImportError:
    try:
        from OmniVoice import OmniVoice
    except ImportError as exc:
        raise ImportError(
            "Khong import duoc OmniVoice. Hay cai package hoac dat file nay "
            "trong moi truong cua repo OmniVoice."
        ) from exc


# =========================
# Cau hinh mac dinh
# =========================

MODEL_PATH = r"F:/MyProjects/GIT/sangtran0897/Project/Tools/Forked/OmniVoice/models/KhanhTTS-OmniVoice"
REF_AUDIO_PATH = r"F:/MyProjects/local/Youtube/One Piece Discovery/Voice/miennamchuan_5s.WAV"
OUTPUT_DIR = Path("outputs")
OUTPUT_FILENAME = "clone_out.wav"

SAMPLE_RATE = 24000
NGHI_CAU = 0.2
AUDIO_SPEED = 1.05

REF_TEXT = (
    "để có thể đắp chút ánh hào quang rẻ tiền lên một gia tộc vốn đang "
    "khao khát có được sự chú ý."
)

# Model chi duoc giu mot ban trong bo nho.
model = None
loaded_model_path = None


# =========================
# Ham xu ly audio
# =========================

def speedup_audio(input_path, output_path, speed=1.08):
    """Tang toc audio bang ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Khong tim thay ffmpeg. Hay cai ffmpeg va them ffmpeg vao PATH."
        )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-filter:a", f"atempo={speed}",
            "-vn",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def make_silence(seconds=0.18, sample_rate=24000):
    """Tao khoang lang giua cac cau."""
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def normalize_audio_array(audio_array):
    """Dam bao audio la numpy array mot chieu float32."""
    audio_array = np.asarray(audio_array)

    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    return audio_array.astype(np.float32)


# =========================
# Ham xu ly text
# =========================

def count_vietnamese_units(text):
    return len(re.findall(r"[A-Za-zÀ-ỹĐđ0-9]+", text))


def estimate_pause_seconds(text):
    comma = len(re.findall(r"[,，]", text)) * 0.12
    semi = len(re.findall(r"[;:；：]", text)) * 0.18
    end = len(re.findall(r"[.!?。！？]", text)) * 0.28
    newline = text.count("\n") * 0.35
    return comma + semi + end + newline


def estimate_duration_from_ref(text, ref_audio_path, ref_text, speed=1.0):
    ref_audio, sample_rate = sf.read(ref_audio_path)
    ref_duration = len(ref_audio) / sample_rate

    ref_units = max(count_vietnamese_units(ref_text), 1)
    target_units = max(count_vietnamese_units(text), 1)
    sec_per_unit = ref_duration / ref_units

    # Chan bien de tranh khoang lang trong audio mau lam duration bi lech.
    sec_per_unit = min(max(sec_per_unit, 0.22), 0.42)

    duration = target_units * sec_per_unit
    duration += estimate_pause_seconds(text)
    duration /= speed

    return round(max(duration, 1.2), 2)


def split_text_to_sentences(text):
    """Cat text theo dau ket cau va giu lai dau cau o cuoi."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]+|[^.!?。！？]+$", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# =========================
# Nap model va generate
# =========================

def load_model(model_path, device_choice="Tự động", progress=gr.Progress()):
    """Nap model tu duong dan local theo GPU, CPU hoac tu dong."""
    global model, loaded_model_path

    try:
        model_path = str(model_path).strip()
        device_choice = str(device_choice or "Tự động").strip()

        if not model_path:
            return "❌ Chưa nhập đường dẫn model."

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            return f"❌ Không tìm thấy model tại: `{path}`"

        if device_choice == "GPU (CUDA)" and not torch.cuda.is_available():
            return "❌ Đã chọn GPU nhưng PyTorch không phát hiện CUDA. Hãy chọn CPU hoặc kiểm tra lại CUDA."

        if device_choice == "CPU":
            device_map = "cpu"
            dtype = torch.float32
            device_key = "cpu"
            device_message = "CPU, float32"
        elif device_choice == "GPU (CUDA)":
            device_map = "cuda:0"
            dtype = torch.float16
            device_key = "cuda:0"
            device_message = f"GPU: {torch.cuda.get_device_name(0)}, float16"
        else:
            if torch.cuda.is_available():
                device_map = "cuda:0"
                dtype = torch.float16
                device_key = "cuda:0"
                device_message = f"Tự động chọn GPU: {torch.cuda.get_device_name(0)}, float16"
            else:
                device_map = "cpu"
                dtype = torch.float32
                device_key = "cpu"
                device_message = "Tự động chọn CPU, float32"

        current_load_key = f"{path}|{device_key}"
        if model is not None and loaded_model_path == current_load_key:
            return (
                "✅ Model đã được nạp sẵn.\n\n"
                f"- Model: `{path}`\n"
                f"- Thiết bị: `{device_message}`"
            )

        progress(0.1, desc="Đang kiểm tra thiết bị...")

        # Giai phong model cu truoc khi doi model hoac doi thiet bi.
        if model is not None:
            del model
            model = None
            loaded_model_path = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        progress(0.25, desc=f"Đang nạp OmniVoice bằng {device_key}...")
        new_model = OmniVoice.from_pretrained(
            str(path),
            device_map=device_map,
            dtype=dtype,
            load_asr=False,
        )

        model = new_model
        loaded_model_path = current_load_key
        progress(1.0, desc="Đã nạp model")

        return (
            "✅ **Đã nạp OmniVoice và audio tokenizer.**\n\n"
            f"- Model: `{path}`\n"
            f"- Thiết bị: `{device_message}`"
        )
    except Exception:
        return "❌ **Nạp model thất bại.**\n\n```text\n" + traceback.format_exc() + "\n```"


def generate_audio(
    input_text,
    ref_audio_upload,
    ref_audio_path,
    ref_text,
    pause_seconds,
    apply_speed,
    audio_speed,
    progress=gr.Progress(),
):
    """Generate tung cau, ghep audio va tra file cho trinh phat cung nut tai."""
    global model

    try:
        if model is None:
            raise gr.Error("Chưa nạp model. Hãy nhấn nút Nạp model trước.")

        input_text = str(input_text or "").strip()
        if not input_text:
            raise gr.Error("Hãy nhập nội dung cần đọc.")

        ref_text = str(ref_text or "").strip()
        if not ref_text:
            raise gr.Error("Hãy nhập nội dung tương ứng với audio mẫu.")

        # Uu tien file nguoi dung upload. Neu khong co thi dung duong dan local.
        if ref_audio_upload:
            selected_ref_audio = Path(ref_audio_upload)
        else:
            selected_ref_audio = Path(str(ref_audio_path or "").strip()).expanduser()

        if not selected_ref_audio.is_file():
            raise gr.Error(f"Không tìm thấy audio mẫu: {selected_ref_audio}")

        sentences = split_text_to_sentences(input_text)
        if not sentences:
            raise gr.Error("Không có câu nào để generate.")

        pause_seconds = max(float(pause_seconds), 0.0)
        audio_speed = float(audio_speed)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        original_output = (OUTPUT_DIR / OUTPUT_FILENAME).resolve()
        final_output = original_output
        all_audio_segments = []
        log_lines = [
            f"Tổng số câu: {len(sentences)}",
            f"Nghỉ giữa mỗi câu: {pause_seconds:.2f} giây",
        ]

        for index, sentence in enumerate(sentences, start=1):
            progress(
                (index - 1) / len(sentences),
                desc=f"Đang tạo câu {index}/{len(sentences)}",
            )

            duration = estimate_duration_from_ref(
                sentence,
                str(selected_ref_audio),
                ref_text,
                speed=1.2,
            )

            audio = model.generate(
                text=sentence,
                ref_audio=str(selected_ref_audio),
                ref_text=ref_text,
                duration=duration,
                language="vi",
            )

            segment = normalize_audio_array(audio[0])
            all_audio_segments.append(segment)
            log_lines.append(
                f"Câu {index}/{len(sentences)}: {duration:.2f} giây | {sentence}"
            )

            if index < len(sentences):
                all_audio_segments.append(make_silence(pause_seconds, SAMPLE_RATE))

        progress(0.92, desc="Đang ghép audio...")
        final_audio = np.concatenate(all_audio_segments)
        sf.write(str(original_output), final_audio, SAMPLE_RATE)
        log_lines.append(f"Đã lưu file gốc: {original_output}")

        if apply_speed:
            progress(0.96, desc="Đang điều chỉnh tốc độ...")
            fast_output = original_output.with_name(
                original_output.stem + "_fast" + original_output.suffix
            )
            speedup_audio(original_output, fast_output, speed=audio_speed)
            final_output = fast_output
            log_lines.append(f"Đã lưu file tăng tốc: {fast_output}")

        progress(1.0, desc="Hoàn tất")
        status = (
            "✅ **Generate hoàn tất.** Bạn có thể nghe thử ở trình phát bên dưới "
            "hoặc tải file bằng khu vực Tải audio."
        )

        # Audio hien player review. File hien rieng nut tai xuong.
        return str(final_output), str(final_output), status, "\n".join(log_lines)

    except gr.Error:
        raise
    except Exception:
        error_text = traceback.format_exc()
        return None, None, "❌ **Generate thất bại.**", error_text


def clear_outputs():
    return None, None, "Đã xóa kết quả khỏi giao diện.", ""


def disable_generate_button():
    """Khoa nut Generate ngay khi bat dau xu ly."""
    return gr.update(interactive=False, value="Đang generate...")


def enable_generate_button():
    """Mo lai nut Generate sau khi xu ly xong hoac gap loi."""
    return gr.update(interactive=True, value="Generate audio")


# =========================
# Giao dien Gradio
# =========================

CSS = """
/* Phu toan bo khung trinh duyet va bo gioi han chieu rong mac dinh. */
html, body {
    width: 100%;
    min-width: 100%;
    height: 100%;
    min-height: 100%;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

.gradio-container {
    width: 100vw !important;
    max-width: none !important;
    min-height: 100vh !important;
    min-height: 100dvh !important;
    margin: 0 !important;
    padding: clamp(8px, 1.2vw, 20px) !important;
    box-sizing: border-box !important;
}

#title {
    text-align: center;
    margin: 0 0 2px 0;
    font-size: clamp(1.35rem, 2.2vw, 2.25rem);
}

#subtitle {
    text-align: center;
    color: #6b7280;
    margin: 0 0 clamp(8px, 1vw, 16px) 0;
    font-size: clamp(0.85rem, 1vw, 1rem);
}

/* Hai cot chinh tu gian theo kich thuoc man hinh. */
#main_workspace {
    width: 100%;
    align-items: stretch;
    gap: clamp(8px, 1vw, 16px);
}

#settings_column,
#generation_column {
    min-width: 0 !important;
}

#tts_text_input textarea {
    min-height: clamp(220px, 38vh, 560px) !important;
    resize: vertical !important;
}

#review_row {
    width: 100%;
    align-items: stretch;
    gap: clamp(8px, 1vw, 16px);
}

#review_audio,
#download_audio {
    min-width: 0 !important;
    height: 100%;
}

#processing_log textarea {
    min-height: clamp(100px, 14vh, 220px) !important;
    resize: vertical !important;
}

/* Man hinh vua va nho: xep doc de khong bi tran hay ep noi dung. */
@media (max-width: 900px) {
    .gradio-container {
        width: 100% !important;
        padding: 8px !important;
    }

    #main_workspace,
    #review_row {
        flex-direction: column !important;
    }

    #settings_column,
    #generation_column,
    #review_audio,
    #download_audio {
        width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    #tts_text_input textarea {
        min-height: 260px !important;
    }
}
"""

with gr.Blocks(
    title="OmniVoice TTS Local",
    css=CSS,
    fill_width=True,
) as demo:
    gr.Markdown("# OmniVoice TTS Local", elem_id="title")
    gr.Markdown(
        "Nạp model local, tạo giọng nói tiếng Việt, nghe thử trực tiếp và tải audio.",
        elem_id="subtitle",
    )

    with gr.Row(elem_id="main_workspace"):
        with gr.Column(scale=1, min_width=300, elem_id="settings_column"):
            gr.Markdown("## 1. Model")
            model_path_input = gr.Textbox(
                label="Đường dẫn model local",
                value=MODEL_PATH,
                placeholder=r"Ví dụ: D:\OmniVoice\model hoặc /home/user/model",
            )
            device_choice = gr.Dropdown(
                choices=["Tự động", "GPU (CUDA)", "CPU"],
                value="Tự động",
                label="Thiết bị nạp model",
                info="Tự động sẽ ưu tiên GPU nếu CUDA khả dụng.",
                interactive=True,
            )
            load_model_button = gr.Button("Nạp lại model", variant="primary")
            model_status = gr.Markdown("Chưa nạp model.")

            gr.Markdown("## 2. Audio mẫu")
            ref_audio_upload = gr.Audio(
                label="Audio mẫu",
                value=REF_AUDIO_PATH if Path(REF_AUDIO_PATH).is_file() else None,
                type="filepath",
                sources=["upload", "microphone"],
                interactive=True,
            )
            ref_audio_path_input = gr.Textbox(
                label="Hoặc nhập đường dẫn audio local",
                value=REF_AUDIO_PATH,
                placeholder=r"Ví dụ: D:\audio\ref.wav",
            )
            ref_text_input = gr.Textbox(
                label="Nội dung của audio mẫu",
                value=REF_TEXT,
                lines=4,
            )

        with gr.Column(scale=2, min_width=420, elem_id="generation_column"):
            gr.Markdown("## 3. Nội dung cần tạo")
            text_input = gr.Textbox(
                label="Văn bản",
                placeholder="Dán đoạn văn cần đọc vào đây. Audio sẽ tự generate sau khi paste...",
                lines=14,
                elem_id="tts_text_input",
            )

            with gr.Row():
                pause_input = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=NGHI_CAU,
                    step=0.05,
                    label="Nghỉ giữa câu, tính bằng giây",
                )
                speed_input = gr.Slider(
                    minimum=0.5,
                    maximum=2,
                    value=AUDIO_SPEED,
                    step=0.05,
                    label="Tốc độ audio",
                )

            apply_speed_input = gr.Checkbox(
                label="Áp dụng tăng hoặc giảm tốc sau khi generate",
                value=False,
            )

            with gr.Row():
                generate_button = gr.Button(
                    "Generate audio",
                    variant="primary",
                    scale=3,
                    elem_id="generate_audio_button",
                )
                clear_button = gr.Button("Xóa kết quả", scale=1)

    gr.Markdown("## 4. Review và tải audio")
    with gr.Row(elem_id="review_row"):
        audio_output = gr.Audio(
            label="Nghe thử audio đã tạo",
            elem_id="review_audio",
            type="filepath",
            interactive=False,
        )
        file_output = gr.File(
            label="Tải audio",
            elem_id="download_audio",
            interactive=False,
        )

    generate_status = gr.Markdown()
    log_output = gr.Textbox(
        label="Nhật ký xử lý",
        lines=6,
        elem_id="processing_log",
        interactive=False,
    )

    load_model_button.click(
        fn=load_model,
        inputs=[model_path_input, device_choice],
        outputs=[model_status],
    )

    # Khoa nut ngay khi bat dau, generate, sau do mo lai khi da ket thuc.
    # Chuoi .then cuoi van chay de mo lai nut neu generate gap loi.
    generate_event = generate_button.click(
        fn=disable_generate_button,
        inputs=[],
        outputs=[generate_button],
        queue=False,
    ).then(
        fn=generate_audio,
        inputs=[
            text_input,
            ref_audio_upload,
            ref_audio_path_input,
            ref_text_input,
            pause_input,
            apply_speed_input,
            speed_input,
        ],
        outputs=[audio_output, file_output, generate_status, log_output],
    ).then(
        fn=enable_generate_button,
        inputs=[],
        outputs=[generate_button],
        queue=False,
    )

    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[audio_output, file_output, generate_status, log_output],
    )

    # Tu dong nap model ngay khi giao dien vua mo.
    demo.load(
        fn=load_model,
        inputs=[model_path_input, device_choice],
        outputs=[model_status],
    )

    # Chi tu dong generate khi nguoi dung PASTE text vao khung.
    # Go ban phim binh thuong se khong kich hoat generate.
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            const bindPasteEvent = () => {
                const textContainer = document.querySelector('#tts_text_input');
                const textarea = textContainer?.querySelector('textarea');
                const buttonContainer = document.querySelector('#generate_audio_button');
                const generateButton = buttonContainer?.querySelector('button');

                if (!textarea || !generateButton) {
                    window.setTimeout(bindPasteEvent, 300);
                    return [];
                }

                if (textarea.dataset.pasteAutoGenerate === 'bound') {
                    return [];
                }

                textarea.dataset.pasteAutoGenerate = 'bound';

                textarea.addEventListener('paste', () => {
                    // Su kien paste xay ra truoc khi noi dung moi duoc chen vao textarea.
                    // Cho trinh duyet va Gradio cap nhat state, sau do moi bam Generate.
                    window.setTimeout(() => {
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                        textarea.dispatchEvent(new Event('change', { bubbles: true }));

                        window.setTimeout(() => {
                            generateButton.click();
                        }, 200);
                    }, 100);
                });

                return [];
            };

            bindPasteEvent();
            return [];
        }
        """,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )
