import re
import numpy as np
import soundfile as sf
from IPython.display import Audio, display


def split_sentences(text: str) -> list[str]:
    """
    Tách full text thành các câu dựa trên dấu . ! ? và dấu ... (ellipsis).
    Loại bỏ câu rỗng, strip khoảng trắng.
    """
    # Tách theo . ! ? nhưng giữ lại dấu ... không bị tách nhầm
    # Chiến lược: thay ... thành placeholder, tách, rồi restore
    text = text.strip()
    text = re.sub(r'\.{3,}', '…', text)  # gộp "..." thành 1 ký tự

    # Tách theo . ! ? — nhưng chỉ khi theo sau là khoảng trắng hoặc cuối chuỗi
    parts = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    for part in parts:
        part = part.replace('…', '...').strip()  # restore ellipsis
        if part:
            sentences.append(part)

    return sentences


def generate_with_pauses(
    model,
    text: str,
    ref_audio: str,
    ref_text: str = None,
    pause_between: float = 0.4,   # giây nghỉ giữa các câu
    pause_end: float = 0.0,       # giây nghỉ cuối cùng (thường 0)
    sample_rate: int = 24000,
    output_path: str = "output.wav",
    speed: float = 1.0,
    language: str = "vi",
    verbose: bool = True,
    **generate_kwargs
) -> np.ndarray:
    """
    Nhận full text, tự cắt thành câu, generate từng câu, ghép lại với khoảng nghỉ.

    Args:
        model:           OmniVoice model đã load
        text:            Full text đầu vào
        ref_audio:       Đường dẫn file audio tham chiếu
        ref_text:        Transcript của ref_audio (optional, dùng Whisper nếu None)
        pause_between:   Khoảng nghỉ giữa các câu (giây), mặc định 0.4s
        pause_end:       Khoảng nghỉ cuối audio (giây)
        sample_rate:     Sample rate output, mặc định 24000
        output_path:     Đường dẫn lưu file wav
        speed:           Tốc độ đọc (>1 nhanh hơn, <1 chậm hơn)
        language:        Mã ngôn ngữ, mặc định "vi"
        verbose:         In tiến trình hay không
        **generate_kwargs: Các tham số bổ sung truyền vào model.generate()

    Returns:
        np.ndarray: toàn bộ audio đã ghép
    """
    sentences = split_sentences(text)

    if verbose:
        print(f"📝 Tổng số câu: {len(sentences)}")
        for i, s in enumerate(sentences):
            print(f"  [{i+1}] {s[:80]}{'...' if len(s) > 80 else ''}")
        print()

    silence_between = np.zeros(int(pause_between * sample_rate), dtype=np.float32)
    silence_end     = np.zeros(int(pause_end * sample_rate),     dtype=np.float32)

    audio_parts = []

    for i, sentence in enumerate(sentences):
        if verbose:
            print(f"🔊 Đang generate câu {i+1}/{len(sentences)}: {sentence[:60]}...")

        try:
            audio = model.generate(
                text=sentence,
                ref_audio=ref_audio,
                ref_text=ref_text,
                speed=speed,
                language=language,
                **generate_kwargs
            )
            audio_parts.append(audio[0].astype(np.float32))

            # Thêm khoảng nghỉ sau mỗi câu (trừ câu cuối nếu pause_end=0)
            if i < len(sentences) - 1:
                audio_parts.append(silence_between)

        except Exception as e:
            print(f"⚠️  Lỗi câu {i+1}: {e}")
            # Vẫn thêm silence để giữ cấu trúc
            audio_parts.append(silence_between)
            continue

    # Thêm khoảng nghỉ cuối nếu có
    if pause_end > 0:
        audio_parts.append(silence_end)

    # Ghép tất cả lại
    final_audio = np.concatenate(audio_parts)

    # Lưu file
    sf.write(output_path, final_audio, sample_rate)
    duration = len(final_audio) / sample_rate
    if verbose:
        print(f"\n✅ Xong! Tổng thời lượng: {duration:.1f}s — Đã lưu: {output_path}")

    return final_audio


# ─── Ví dụ sử dụng ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # (Chỉ chạy khi test độc lập — trên Colab bỏ phần này)

    # Kiểm tra split_sentences
    sample_text = """
    không ai trong one piece thực sự tự do. không phải imu, vị vua của thế giới.
    càng không phải râu đen, trên hòn đảo không luật pháp của hắn.
    thậm chí không phải luffy, chiến binh giải phóng đích thực!
    thế giới one piece tồn tại trong kỷ nguyên nô lệ.
    """.strip()

    sentences = split_sentences(sample_text)
    print(f"Số câu: {len(sentences)}")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")
