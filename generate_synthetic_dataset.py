"""
Generate a fully synthetic multilingual TTS dataset using NVIDIA MagpieTTS 357M.

This script creates speech samples across multiple speakers and languages,
producing paired audio files with a comprehensive metadata CSV.

Usage:
    python generate_synthetic_dataset.py \
        --output_dir ./output/synthetic_dataset \
        --languages en,fr,zh \
        --speakers Sofia,Aria \
        --text_file texts.json \
        --num_samples 10

    # Quick test with embedded sample texts:
    python generate_synthetic_dataset.py \
        --output_dir ./output/synthetic_dataset \
        --languages en,fr \
        --speakers Sofia \
        --num_samples 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tts_utils import (
    SPEAKER_NAMES,
    SUPPORTED_LANGUAGES,
    ensure_dir,
    generate_speech,
    load_magpie_model,
    make_audio_filename,
    save_audio,
)


# ─── Embedded Sample Texts ─────────────────────────────────────────────────────
# These are used when no --text_file is provided. They serve as quick test data
# and as a fallback for generating a small demo dataset.

SAMPLE_TEXTS = {
    "en": [
        "The sun sets behind the mountains, casting golden light across the valley.",
        "Technology continues to transform the way we communicate with each other.",
        "A gentle breeze carried the scent of flowers through the open window.",
        "Scientists have discovered new methods for generating renewable energy.",
        "The concert was spectacular, with musicians playing in perfect harmony.",
        "Education is the foundation for building a better and more equitable society.",
        "The architecture of the ancient city reflects centuries of cultural evolution.",
        "Artificial intelligence is reshaping industries and creating new possibilities.",
        "The children laughed as they chased butterflies through the meadow.",
        "Global cooperation is essential for addressing the challenges of climate change.",
    ],
    "es": [
        "El sol se pone detrás de las montañas, bañando el valle en luz dorada.",
        "La tecnología sigue transformando la forma en que nos comunicamos.",
        "Una brisa suave llevaba el aroma de las flores por la ventana abierta.",
        "Los científicos han descubierto nuevos métodos de energía renovable.",
        "El concierto fue espectacular, con músicos tocando en perfecta armonía.",
        "La educación es la base para construir una sociedad mejor y más equitativa.",
        "La arquitectura de la ciudad antigua refleja siglos de evolución cultural.",
        "La inteligencia artificial está transformando industrias y creando nuevas posibilidades.",
        "Los niños reían mientras perseguían mariposas por el prado.",
        "La cooperación global es esencial para enfrentar los desafíos del cambio climático.",
    ],
    "de": [
        "Die Sonne geht hinter den Bergen unter und taucht das Tal in goldenes Licht.",
        "Technologie verändert weiterhin die Art und Weise, wie wir kommunizieren.",
        "Eine sanfte Brise trug den Duft der Blumen durch das offene Fenster.",
        "Wissenschaftler haben neue Methoden zur Erzeugung erneuerbarer Energie entdeckt.",
        "Das Konzert war spektakulär, die Musiker spielten in perfekter Harmonie.",
        "Bildung ist die Grundlage für den Aufbau einer besseren Gesellschaft.",
        "Die Architektur der alten Stadt spiegelt Jahrhunderte kultureller Entwicklung wider.",
        "Künstliche Intelligenz verändert Branchen und schafft neue Möglichkeiten.",
        "Die Kinder lachten, als sie Schmetterlinge über die Wiese jagten.",
        "Globale Zusammenarbeit ist entscheidend für die Bewältigung des Klimawandels.",
    ],
    "fr": [
        "Le soleil se couche derrière les montagnes, baignant la vallée de lumière dorée.",
        "La technologie continue de transformer notre façon de communiquer.",
        "Une douce brise portait le parfum des fleurs par la fenêtre ouverte.",
        "Les scientifiques ont découvert de nouvelles méthodes d'énergie renouvelable.",
        "Le concert était spectaculaire, avec des musiciens jouant en parfaite harmonie.",
        "L'éducation est le fondement d'une société meilleure et plus équitable.",
        "L'architecture de la ville ancienne reflète des siècles d'évolution culturelle.",
        "L'intelligence artificielle transforme les industries et crée de nouvelles possibilités.",
        "Les enfants riaient en poursuivant des papillons dans la prairie.",
        "La coopération mondiale est essentielle pour relever les défis du changement climatique.",
    ],
    "it": [
        "Il sole tramonta dietro le montagne, bagnando la valle di luce dorata.",
        "La tecnologia continua a trasformare il modo in cui comunichiamo.",
        "Una brezza leggera portava il profumo dei fiori attraverso la finestra aperta.",
        "Gli scienziati hanno scoperto nuovi metodi per generare energia rinnovabile.",
        "Il concerto è stato spettacolare, con musicisti che suonavano in perfetta armonia.",
        "L'istruzione è la base per costruire una società migliore e più equa.",
        "L'architettura dell'antica città riflette secoli di evoluzione culturale.",
        "L'intelligenza artificiale sta ridefinendo le industrie e creando nuove possibilità.",
        "I bambini ridevano mentre inseguivano farfalle nel prato.",
        "La cooperazione globale è essenziale per affrontare le sfide del cambiamento climatico.",
    ],
    "vi": [
        "Mặt trời lặn sau dãy núi, tỏa ánh sáng vàng khắp thung lũng.",
        "Công nghệ tiếp tục thay đổi cách chúng ta giao tiếp với nhau.",
        "Một làn gió nhẹ mang hương hoa qua cửa sổ mở.",
        "Các nhà khoa học đã phát hiện phương pháp mới tạo năng lượng tái tạo.",
        "Buổi hòa nhạc thật tuyệt vời, các nhạc sĩ chơi hòa hợp hoàn hảo.",
        "Giáo dục là nền tảng để xây dựng một xã hội tốt đẹp hơn.",
        "Kiến trúc thành phố cổ phản ánh hàng thế kỷ phát triển văn hóa.",
        "Trí tuệ nhân tạo đang thay đổi các ngành công nghiệp.",
        "Bọn trẻ cười đùa khi đuổi bắt bướm trên đồng cỏ.",
        "Hợp tác toàn cầu là điều cần thiết cho thách thức khí hậu.",
    ],
    "zh": [
        "太阳在群山后落下，为山谷洒下金色的光芒。",
        "科技不断改变我们彼此交流的方式。",
        "一阵微风将花香从敞开的窗户中送进来。",
        "科学家们发现了产生可再生能源的新方法。",
        "音乐会精彩绝伦，音乐家们演奏得完美和谐。",
        "教育是建设更美好、更公平社会的基础。",
        "古城的建筑反映了几个世纪的文化演变。",
        "人工智能正在重塑各行各业，创造新的可能性。",
        "孩子们在草地上追逐蝴蝶，笑声不断。",
        "全球合作对于应对气候变化的挑战至关重要。",
    ],
    "hi": [
        "सूरज पहाड़ों के पीछे डूबता है, घाटी पर सुनहरी रोशनी बिखेरता है।",
        "प्रौद्योगिकी लगातार हमारे संवाद करने के तरीके को बदल रही है।",
        "एक हल्की हवा खिड़की से फूलों की खुशबू लेकर आई।",
        "वैज्ञानिकों ने नवीकरणीय ऊर्जा उत्पन्न करने के नए तरीके खोजे हैं।",
        "संगीत कार्यक्रम शानदार था, संगीतकार सही सामंजस्य में बजा रहे थे।",
        "शिक्षा एक बेहतर और अधिक न्यायसंगत समाज के निर्माण की नींव है।",
        "प्राचीन शहर की वास्तुकला सदियों के सांस्कृतिक विकास को दर्शाती है।",
        "कृत्रिम बुद्धिमत्ता उद्योगों को नया आकार दे रही है और नई संभावनाएं पैदा कर रही है।",
        "बच्चे मैदान में तितलियों का पीछा करते हुए हंस रहे थे।",
        "जलवायु परिवर्तन की चुनौतियों से निपटने के लिए वैश्विक सहयोग आवश्यक है।",
    ],
    "ja": [
        "太陽が山の向こうに沈み、谷に金色の光を投げかけている。",
        "テクノロジーは私たちのコミュニケーション方法を変え続けています。",
        "開いた窓から花の香りを運ぶそよ風が吹いてきました。",
        "科学者たちは再生可能エネルギーを生成する新しい方法を発見しました。",
        "コンサートは素晴らしく、ミュージシャンたちは完璧なハーモニーで演奏しました。",
        "教育はより良い、より公平な社会を築くための基盤です。",
        "古代都市の建築は何世紀にもわたる文化の進化を反映しています。",
        "人工知能は産業を変革し、新たな可能性を生み出しています。",
        "子どもたちは草原で蝶を追いかけながら笑っていました。",
        "気候変動の課題に取り組むためには、地球規模の協力が不可欠です。",
    ],
}


# ─── Main Logic ─────────────────────────────────────────────────────────────────


def load_text_data(text_file: str | None) -> dict:
    """
    Load text data from a JSON file or fall back to embedded samples.

    Expected JSON format:
    {
        "en": ["sentence 1", "sentence 2", ...],
        "fr": ["phrase 1", "phrase 2", ...],
        ...
    }
    """
    if text_file and os.path.exists(text_file):
        print(f"Loading texts from {text_file}")
        with open(text_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("Using embedded sample texts.")
        return SAMPLE_TEXTS


def generate_dataset(
    output_dir: str,
    languages: list[str],
    speakers: list[str],
    texts: dict,
    num_samples: int,
    device: str = "cuda",
):
    """
    Generate the synthetic multilingual TTS dataset.

    Creates audio files for each (speaker, language, text) combination
    and saves a metadata CSV.
    """
    audio_dir = ensure_dir(os.path.join(output_dir, "audio"))
    model = load_magpie_model(device=device)

    records = []
    total = len(speakers) * len(languages) * num_samples
    pbar = tqdm(total=total, desc="Generating speech", unit="sample")

    for speaker in speakers:
        for lang in languages:
            lang_texts = texts.get(lang, [])
            if not lang_texts:
                print(f"  ⚠ No texts available for language '{lang}', skipping.")
                pbar.update(num_samples)
                continue

            for i in range(min(num_samples, len(lang_texts))):
                text = lang_texts[i]
                filename = make_audio_filename(speaker, lang, i)
                filepath = os.path.join(audio_dir, lang, filename)

                try:
                    audio, length = generate_speech(
                        model, text, lang, speaker=speaker
                    )
                    abs_path = save_audio(audio, filepath)

                    records.append({
                        "speaker": speaker,
                        "language": lang,
                        "language_name": SUPPORTED_LANGUAGES[lang],
                        "text": text,
                        "audio_path": os.path.relpath(abs_path, output_dir),
                        "audio_length_samples": length,
                        "sample_index": i,
                    })
                except Exception as e:
                    print(f"\n  ✗ Failed: {speaker}/{lang}/{i}: {e}")

                pbar.update(1)

    pbar.close()

    # ── Save metadata ───────────────────────────────────────────────────────
    if records:
        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved {len(records)} records to {csv_path}")

        json_path = os.path.join(output_dir, "metadata.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"✓ Saved metadata JSON to {json_path}")

        # ── Summary ─────────────────────────────────────────────────────────
        print("\n── Dataset Summary ──")
        print(f"  Total samples:  {len(records)}")
        print(f"  Speakers:       {sorted(df['speaker'].unique())}")
        print(f"  Languages:      {sorted(df['language'].unique())}")
        print(f"  Output dir:     {os.path.abspath(output_dir)}")
    else:
        print("\n⚠ No samples were generated.")


# ─── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic multilingual TTS dataset using MagpieTTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/synthetic_dataset",
        help="Directory to save generated audio and metadata.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,fr,zh",
        help="Comma-separated language codes (e.g., 'en,fr,zh,es').",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        default=",".join(SPEAKER_NAMES),
        help="Comma-separated speaker names (e.g., 'Sofia,Aria,Jason').",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to a JSON file with texts per language. "
             "If not provided, uses embedded sample texts.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of text samples per (speaker, language) pair.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    languages = [l.strip() for l in args.languages.split(",")]
    speakers = [s.strip() for s in args.speakers.split(",")]

    # Validate languages
    for lang in languages:
        if lang not in SUPPORTED_LANGUAGES:
            print(f"Error: Language '{lang}' not supported. "
                  f"Available: {list(SUPPORTED_LANGUAGES.keys())}")
            sys.exit(1)

    # Validate speakers
    for speaker in speakers:
        if speaker not in SPEAKER_MAP:
            print(f"Error: Speaker '{speaker}' not supported. "
                  f"Available: {SPEAKER_NAMES}")
            sys.exit(1)

    texts = load_text_data(args.text_file)

    print("=" * 60)
    print("  MagpieTTS Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Languages:    {languages}")
    print(f"  Speakers:     {speakers}")
    print(f"  Samples/pair: {args.num_samples}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Device:       {args.device}")
    print("=" * 60)

    generate_dataset(
        output_dir=args.output_dir,
        languages=languages,
        speakers=speakers,
        texts=texts,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
