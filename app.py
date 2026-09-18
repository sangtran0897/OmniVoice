import importlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import traceback

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice


# =========================
# Cấu hình
# =========================

SAMPLE_RATE = 24000

# Khoảng nghỉ được chèn giữa các câu.
# Khoảng nghỉ này không bị ảnh hưởng bởi tốc độ audio.
nghi_cau = 0.2

# Tăng tốc riêng từng đoạn audio trước khi ghép.
#
# Gợi ý:
# 1.05 = nhanh nhẹ
# 1.08 = nhanh vừa
# 1.10 = nhanh rõ hơn
# 1.12 = khá nhanh
segment_speed = 1.03

# Hệ số tính thời lượng khi model generate.
# Không nên tăng quá cao vì có thể làm câu bị thiếu thời lượng
# và khiến model bỏ chữ.
duration_speed = 1.2

ref_text = (
    "để có thể đắp chút ánh hào quang rẻ tiền lên một gia tộc "
    "vốn đang khao khát có được sự chú ý."
)

OUTPUT_PATH = os.path.abspath("clone_out.wav")
MODEL_PATH = r"F:/MyProjects/GIT/sangtran0897/Project/Tools/Forked/OmniVoice/models/KhanhTTS-OmniVoice"
REF_AUDIO_PATH = r"F:/MyProjects/local/Youtube/One Piece Discovery/Voice/miennamchuan_5s.WAV"

_model = None
_model_lock = threading.Lock()


# =========================
# Nạp model local
# =========================

def get_model():
    """
    Nạp model OmniVoice local một lần và giữ trong bộ nhớ.
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(
                f"Không tìm thấy thư mục model: {MODEL_PATH}"
            )

        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        _model = OmniVoice.from_pretrained(
            MODEL_PATH,
            device_map=device_map,
            dtype=dtype
        )

        return _model


# =========================
# Hàm xử lý audio
# =========================

def speedup_audio(input_path, output_path, speed=1.08):
    """
    Tăng tốc một file audio bằng ffmpeg.

    Hàm này được giữ lại nếu sau này cần xử lý file riêng.
    Trong quá trình generate, code sẽ dùng
    speedup_audio_segment() cho từng câu.
    """
    subprocess.run([
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-filter:a", f"atempo={speed}",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-vn",
        output_path
    ], check=True)


def build_atempo_filter(speed):
    """
    Tạo chuỗi bộ lọc atempo hợp lệ cho ffmpeg.

    atempo hỗ trợ từng mức từ 0.5 đến 2.0.
    Hàm này vẫn xử lý được nếu sau này đặt tốc độ
    lớn hơn 2.0 hoặc thấp hơn 0.5.
    """
    speed = float(speed)

    if speed <= 0:
        raise ValueError("Tốc độ audio phải lớn hơn 0.")

    filters = []

    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0

    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5

    filters.append(f"atempo={speed:.6f}")

    return ",".join(filters)


def speedup_audio_segment(
    audio_array,
    speed=1.10,
    sample_rate=24000
):
    """
    Tăng tốc riêng một đoạn audio sau khi model generate.

    Đoạn audio được tăng tốc trước khi chèn khoảng nghỉ,
    vì vậy nghi_cau vẫn giữ đúng số giây đã thiết lập.

    FFmpeg atempo giúp tăng tốc mà không làm thay đổi
    cao độ rõ rệt như cách co ngắn mảng âm thanh thông thường.
    """
    audio_array = normalize_audio_array(audio_array)

    if speed is None or abs(speed - 1.0) < 0.0001:
        return audio_array

    input_temp_path = None
    output_temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as input_temp:
            input_temp_path = input_temp.name

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as output_temp:
            output_temp_path = output_temp.name

        sf.write(
            input_temp_path,
            audio_array,
            sample_rate
        )

        atempo_filter = build_atempo_filter(speed)

        subprocess.run([
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", input_temp_path,
            "-filter:a", atempo_filter,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-vn",
            output_temp_path
        ], check=True)

        processed_audio, processed_sr = sf.read(
            output_temp_path,
            dtype="float32"
        )

        processed_audio = normalize_audio_array(
            processed_audio
        )

        # Trường hợp hiếm khi ffmpeg trả về sample rate khác,
        # báo lỗi để tránh ghép các đoạn không đồng nhất.
        if processed_sr != sample_rate:
            raise ValueError(
                f"Sample rate sau khi tăng tốc là "
                f"{processed_sr}, cần {sample_rate}."
            )

        return processed_audio

    finally:
        if (
            input_temp_path
            and os.path.exists(input_temp_path)
        ):
            os.remove(input_temp_path)

        if (
            output_temp_path
            and os.path.exists(output_temp_path)
        ):
            os.remove(output_temp_path)


def make_silence(
    seconds=0.18,
    sample_rate=24000
):
    """
    Tạo khoảng lặng giữa các câu.
    """
    return np.zeros(
        int(seconds * sample_rate),
        dtype=np.float32
    )


def normalize_audio_array(audio_array):
    """
    Đảm bảo audio là numpy array một chiều float32.
    """
    audio_array = np.asarray(audio_array)

    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    return audio_array.astype(np.float32)


# =========================
# Mẫu ký tự phân tích từ
# =========================

VIETNAMESE_MARK_PATTERN = (
    r"[àáảãạăằắẳẵặâầấẩẫậ"
    r"èéẻẽẹêềếểễệ"
    r"ìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữự"
    r"ỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    r"ÈÉẺẼẸÊỀẾỂỄỆ"
    r"ÌÍỈĨỊ"
    r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    r"ÙÚỦŨỤƯỪỨỬỮỰ"
    r"ỲÝỶỸỴĐ]"
)

VOWEL_PATTERN = (
    r"[aeiouy"
    r"àáảãạăằắẳẵặâầấẩẫậ"
    r"èéẻẽẹêềếểễệ"
    r"ìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữự"
    r"ỳýỷỹỵ]+"
)

CONSONANT_CLUSTER_PATTERN = (
    r"[^aeiouy"
    r"àáảãạăằắẳẵặâầấẩẫậ"
    r"èéẻẽẹêềếểễệ"
    r"ìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợ"
    r"ùúủũụưừứửữự"
    r"ỳýỷỹỵ0-9]{2,}"
)


# =========================
# Hàm xử lý text
# =========================

def extract_words(text):
    """
    Tách từ tiếng Việt, từ nước ngoài và chữ số.
    """
    return re.findall(
        r"[A-Za-zÀ-ỹĐđ0-9]+",
        text
    )


def count_vietnamese_units(text):
    """
    Đếm số từ thực tế trong câu.
    """
    return len(extract_words(text))


def normalize_word(word):
    """
    Chuyển từ về chữ thường để phân tích.
    """
    return word.strip().lower()


def contains_vietnamese_marks(word):
    """
    Kiểm tra từ có dấu hoặc chữ cái đặc trưng
    của tiếng Việt hay không.
    """
    return bool(
        re.search(
            VIETNAMESE_MARK_PATTERN,
            word
        )
    )


def count_vowel_groups(word):
    """
    Đếm số cụm nguyên âm trong một từ.
    """
    normalized = normalize_word(word)

    return len(
        re.findall(
            VOWEL_PATTERN,
            normalized
        )
    )


def count_consonant_clusters(word):
    """
    Đếm số cụm có từ hai phụ âm liền nhau.
    """
    normalized = normalize_word(word)

    return len(
        re.findall(
            CONSONANT_CLUSTER_PATTERN,
            normalized
        )
    )


def is_capitalized_word(word):
    """
    Kiểm tra từ có bắt đầu bằng chữ hoa hay không.
    """
    if not word:
        return False

    return (
        word[0].isalpha()
        and word[0].isupper()
    )


def estimate_word_complexity_score(
    word,
    word_index=0
):
    """
    Tính điểm phức tạp của từ mà không cần
    danh sách tên riêng.
    """
    normalized = normalize_word(word)

    if not normalized:
        return 0

    if normalized.isdigit():
        return 0

    if contains_vietnamese_marks(word):
        return 0

    word_length = len(normalized)
    vowel_group_count = count_vowel_groups(word)
    consonant_cluster_count = count_consonant_clusters(
        word
    )

    score = 0

    if word_length >= 4:
        score += 1

    if word_length >= 6:
        score += 1

    if word_length >= 8:
        score += 1

    # Từ viết hoa ở giữa câu có khả năng là tên riêng.
    if (
        word_index > 0
        and is_capitalized_word(word)
    ):
        score += 2

    # Từ đầu câu có thể viết hoa do quy tắc thông thường.
    if (
        word_index == 0
        and is_capitalized_word(word)
    ):
        score += 1

    if vowel_group_count >= 2:
        score += 1

    if vowel_group_count >= 3:
        score += 1

    if consonant_cluster_count >= 1:
        score += 1

    if consonant_cluster_count >= 2:
        score += 1

    return score


def is_likely_complex_word(
    word,
    word_index=0
):
    """
    Xác định tương đối một từ có cách phát âm phức tạp.
    """
    complexity_score = estimate_word_complexity_score(
        word,
        word_index
    )

    return complexity_score >= 2


def estimate_word_weight(
    word,
    word_index=0
):
    """
    Ước tính trọng số phát âm cho từng từ.

    Từ hoặc tên phát âm phức tạp được cộng thời lượng
    vừa phải để hạn chế lỗi mất chữ.
    """
    normalized = normalize_word(word)

    if not normalized:
        return 0.0

    if normalized.isdigit():
        if len(normalized) <= 2:
            return 1.0

        return min(
            1.0 + len(normalized) * 0.05,
            1.40
        )

    # Từ có dấu tiếng Việt dùng trọng số cơ bản.
    if contains_vietnamese_marks(word):
        return 1.0

    word_length = len(normalized)
    vowel_group_count = count_vowel_groups(word)
    consonant_cluster_count = count_consonant_clusters(
        word
    )

    weight = 1.0

    if word_length >= 4:
        weight += 0.06

    if word_length >= 6:
        weight += 0.10

    if word_length >= 8:
        weight += 0.10

    if vowel_group_count >= 2:
        weight += min(
            (vowel_group_count - 1) * 0.07,
            0.18
        )

    if consonant_cluster_count >= 1:
        weight += min(
            consonant_cluster_count * 0.08,
            0.20
        )

    if (
        word_index > 0
        and is_capitalized_word(word)
    ):
        weight += 0.12

    if (
        word_index == 0
        and is_capitalized_word(word)
    ):
        weight += 0.03

    complexity_score = estimate_word_complexity_score(
        word,
        word_index
    )

    if complexity_score >= 4:
        weight += 0.06

    if complexity_score >= 6:
        weight += 0.05

    return min(
        weight,
        1.60
    )


def calculate_speaking_units(text):
    """
    Tính tổng đơn vị phát âm của câu.
    """
    words = extract_words(text)

    if not words:
        return 1.0

    total_units = 0.0

    for word_index, word in enumerate(words):
        total_units += estimate_word_weight(
            word,
            word_index
        )

    return max(
        total_units,
        1.0
    )


def count_likely_complex_words(text):
    """
    Đếm số từ có khả năng cần thêm thời gian phát âm.
    """
    words = extract_words(text)

    complex_word_count = 0

    for word_index, word in enumerate(words):
        if is_likely_complex_word(
            word,
            word_index
        ):
            complex_word_count += 1

    return complex_word_count


def estimate_pause_seconds(text):
    """
    Tính thời gian bổ sung dựa theo dấu câu.
    """
    comma = len(
        re.findall(
            r"[,，]",
            text
        )
    ) * 0.09

    semi = len(
        re.findall(
            r"[;:；：]",
            text
        )
    ) * 0.14

    end = len(
        re.findall(
            r"[.!?。！？]",
            text
        )
    ) * 0.20

    newline = text.count("\n") * 0.25

    return comma + semi + end + newline


def estimate_short_pause_seconds(text):
    """
    Tính thời gian dấu câu cho câu ngắn
    từ ba chữ trở xuống.
    """
    comma = len(
        re.findall(
            r"[,，]",
            text
        )
    ) * 0.03

    semi = len(
        re.findall(
            r"[;:；：]",
            text
        )
    ) * 0.05

    has_end_punctuation = bool(
        re.search(
            r"[.!?。！？]",
            text
        )
    )

    end = 0.06 if has_end_punctuation else 0.0

    return comma + semi + end


def estimate_complex_word_buffer(text):
    """
    Cộng khoảng đệm nhỏ cho câu có từ phát âm phức tạp.

    Khoảng đệm chỉ dùng trong lúc model tạo câu.
    Sau đó toàn bộ đoạn câu mới được tăng tốc.
    """
    complex_word_count = count_likely_complex_words(text)

    if complex_word_count == 0:
        return 0.0

    return min(
        complex_word_count * 0.06,
        0.24
    )


def estimate_duration_from_ref(
    text,
    ref_audio_path,
    ref_text,
    speed=1.0
):
    """
    Ước tính thời lượng model cần để tạo câu.

    Câu ngắn được xử lý riêng để tránh kéo dài.

    Câu có tên hoặc từ phát âm phức tạp vẫn được cấp
    đủ thời lượng nhằm hạn chế lỗi mất chữ.
    """
    ref_audio, sr = sf.read(ref_audio_path)

    ref_duration = len(ref_audio) / sr

    ref_units = max(
        count_vietnamese_units(ref_text),
        1
    )

    target_word_count = max(
        count_vietnamese_units(text),
        1
    )

    target_speaking_units = max(
        calculate_speaking_units(text),
        1.0
    )

    complex_word_count = count_likely_complex_words(text)

    sec_per_unit = ref_duration / ref_units

    # Chặn biên để khoảng lặng trong audio mẫu
    # không làm lệch thời gian quá nhiều.
    sec_per_unit = min(
        max(sec_per_unit, 0.22),
        0.40
    )

    # =========================
    # Câu ngắn từ một đến ba chữ
    # không chứa từ phức tạp
    # =========================

    if (
        target_word_count <= 3
        and complex_word_count == 0
    ):
        short_sec_per_unit = min(
            sec_per_unit,
            0.23
        )

        duration = (
            target_word_count
            * short_sec_per_unit
        )

        duration += estimate_short_pause_seconds(text)

        duration /= speed

        short_min_duration = {
            1: 0.36,
            2: 0.46,
            3: 0.60
        }

        short_max_duration = {
            1: 0.54,
            2: 0.68,
            3: 0.82
        }

        duration = max(
            duration,
            short_min_duration[target_word_count]
        )

        duration = min(
            duration,
            short_max_duration[target_word_count]
        )

        return round(
            duration,
            2
        )

    # =========================
    # Câu từ bốn chữ trở lên
    # hoặc có từ phát âm phức tạp
    # =========================

    duration = (
        target_speaking_units
        * sec_per_unit
    )

    duration += estimate_pause_seconds(text)

    duration += estimate_complex_word_buffer(text)

    duration /= speed

    base_minimum_duration = (
        target_word_count * 0.20
    )

    if complex_word_count > 0:
        complex_minimum_duration = (
            target_word_count * 0.21
            + complex_word_count * 0.05
        )

        minimum_duration = max(
            1.05,
            base_minimum_duration,
            complex_minimum_duration
        )
    else:
        minimum_duration = max(
            1.05,
            base_minimum_duration
        )

    return round(
        max(
            duration,
            minimum_duration
        ),
        2
    )


def split_text_to_sentences(text):
    """
    Cắt text thành từng câu nhỏ.

    Cắt tại dấu chấm, chấm than và chấm hỏi.
    Giữ lại dấu câu ở cuối câu để TTS đọc tự nhiên hơn.
    """
    text = text.strip()

    # Gom nhiều khoảng trắng và nhiều dòng
    # thành một khoảng trắng.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    sentences = re.findall(
        r"[^.!?。！？]+[.!?。！？]+|[^.!?。！？]+$",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# =========================
# Generate
# =========================

def generate_audio(
    input_text,
    ref_audio_path,
    input_ref_text
):
    """
    Hàm generate dùng cho giao diện Gradio local.
    """
    input_text = (input_text or "").strip()
    input_ref_text = (input_ref_text or "").strip()

    if not input_text:
        raise gr.Error("Vui lòng nhập nội dung cần tạo giọng.")

    if not ref_audio_path:
        raise gr.Error("Vui lòng chọn file audio giọng mẫu.")

    if not os.path.isfile(ref_audio_path):
        raise gr.Error("Không tìm thấy file audio giọng mẫu.")

    if not input_ref_text:
        raise gr.Error("Vui lòng nhập nội dung của audio giọng mẫu.")

    if shutil.which("ffmpeg") is None:
        raise gr.Error(
            "Không tìm thấy FFmpeg. Hãy cài FFmpeg và thêm vào PATH."
        )

    sentences = split_text_to_sentences(input_text)

    if not sentences:
        raise gr.Error("Không có câu nào để generate.")

    logs = [
        "Đang generate...",
        f"Tổng số câu: {len(sentences)}",
        f"Nghỉ giữa mỗi câu: {nghi_cau}s",
        f"Tốc độ riêng của từng đoạn: {segment_speed}x"
    ]

    all_audio_segments = []
    model = get_model()

    # Khóa model để tránh hai lượt generate chạy đồng thời
    # trên cùng một model/GPU.
    with _model_lock:
        for idx, sentence in enumerate(
            sentences,
            start=1
        ):
            duration = estimate_duration_from_ref(
                sentence,
                ref_audio_path,
                input_ref_text,
                speed=duration_speed
            )

            # Bỏ dấu # ở các dòng dưới nếu muốn
            # theo dõi cách tính của từng câu.

            # print(f"\nCâu {idx}/{len(sentences)}:")
            # print(sentence)
            # print(
            #     "Số từ:",
            #     count_vietnamese_units(sentence)
            # )
            # print(
            #     "Đơn vị phát âm:",
            #     round(
            #         calculate_speaking_units(sentence),
            #         2
            #     )
            # )
            # print(
            #     "Số từ phức tạp:",
            #     count_likely_complex_words(sentence)
            # )
            # print(
            #     f"Thời lượng generate: {duration}s"
            # )

            audio = model.generate(
                text=sentence,
                ref_audio=ref_audio_path,
                ref_text=input_ref_text,
                duration=duration,
                language="vi"
            )

            segment = normalize_audio_array(
                audio[0]
            )

            # Tăng tốc riêng đoạn audio vừa tạo.
            # Việc này được thực hiện trước khi chèn nghỉ_cau.
            segment = speedup_audio_segment(
                segment,
                speed=segment_speed,
                sample_rate=SAMPLE_RATE
            )

            all_audio_segments.append(segment)

            # Chỉ chèn khoảng nghỉ sau khi đoạn câu
            # đã được tăng tốc hoàn chỉnh.
            #
            # Vì khoảng nghỉ được tạo sau bước tăng tốc,
            # nó vẫn giữ đúng giá trị nghi_cau.
            if idx < len(sentences):
                all_audio_segments.append(
                    make_silence(
                        nghi_cau,
                        SAMPLE_RATE
                    )
                )

    if not all_audio_segments:
        raise gr.Error("Không có audio để ghép.")

    final_audio = np.concatenate(
        all_audio_segments
    )

    # File này đã bao gồm:
    # - Các câu được tăng tốc riêng.
    # - Khoảng nghỉ nguyên vẹn giữa các câu.
    sf.write(
        OUTPUT_PATH,
        final_audio,
        SAMPLE_RATE
    )

    logs.extend([
        "",
        "Xong.",
        f"File hoàn chỉnh: {OUTPUT_PATH}",
        f"Mỗi đoạn đã tăng tốc: {segment_speed}x",
        f"Khoảng nghỉ giữ nguyên: {nghi_cau}s"
    ])

    return OUTPUT_PATH, "\n".join(logs), OUTPUT_PATH


def safe_generate_audio(
    input_text,
    ref_audio_path,
    input_ref_text
):
    """
    Giữ thông báo lỗi rõ ràng trên giao diện local.
    """
    try:
        return generate_audio(
            input_text,
            ref_audio_path,
            input_ref_text
        )
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"Generate thất bại: {exc}") from exc


# =========================
# UI local bằng Gradio
# =========================

with gr.Blocks(title="Tạo giọng nói local") as demo:
    gr.Markdown("# Tạo giọng nói local")

    text_box = gr.Textbox(
        value="",
        placeholder="Paste đoạn text vào đây...",
        label="Text",
        lines=12
    )

    with gr.Row():
        ref_audio_input = gr.Audio(
            value=REF_AUDIO_PATH if os.path.isfile(REF_AUDIO_PATH) else None,
            sources=["upload", "microphone"],
            type="filepath",
            label="Audio giọng mẫu"
        )

        ref_text_input = gr.Textbox(
            value=ref_text,
            label="Nội dung audio giọng mẫu",
            lines=5
        )

    generate_button = gr.Button(
        "Generate",
        variant="primary"
    )

    status_output = gr.Textbox(
        label="Trạng thái",
        lines=7,
        interactive=False
    )

    audio_output = gr.Audio(
        label="Audio hoàn chỉnh",
        type="filepath"
    )

    download_output = gr.File(
        label="Tải clone_out.wav"
    )

    generate_button.click(
        fn=safe_generate_audio,
        inputs=[
            text_box,
            ref_audio_input,
            ref_text_input
        ],
        outputs=[
            audio_output,
            status_output,
            download_output
        ],
        show_progress="full"
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )
