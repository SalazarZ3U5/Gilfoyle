import os
import json
import whisper
from collections import defaultdict

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"

def ensure_dirs():
    """Create necessary directories if they don't exist."""
    for d in [PROCESSED_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)

def parse_filename(filename):
    """Extract team and participant from filename format: Team_Participant.wav"""
    name, _ = os.path.splitext(filename)
    if "_" not in name:
        raise ValueError(f"Filename {filename} must follow format: Team_Participant.wav")
    team, participant = name.split("_", 1)
    return team, participant

def transcribe_audio():
    """Transcribe all audio files in RAW_DIR using Whisper Large with timestamps."""
    print("Loading Whisper model: large...")
    model = whisper.load_model("large")

    transcripts = defaultdict(list)   # team → [texts]
    segments_all = []                 # for merged timeline

    for file in os.listdir(RAW_DIR):
        if file.endswith((".wav", ".mp3")):
            audio_path = os.path.join(RAW_DIR, file)
            print(f"Transcribing {file}...")

            try:
                result = model.transcribe(audio_path, verbose=False)
                team, participant = parse_filename(file)

                # Save participant transcript (text only)
                text = result["text"].strip()
                out_file = f"{team}_{participant}.txt"
                out_path = os.path.join(PROCESSED_DIR, out_file)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

                # Save participant transcript with timestamps (JSON)
                json_out = out_path.replace(".txt", ".json")
                with open(json_out, "w", encoding="utf-8") as f:
                    json.dump(result["segments"], f, indent=2, ensure_ascii=False)

                # Collect for team merge
                transcripts[team].append(text)

                # Collect for global timeline
                for seg in result["segments"]:
                    segments_all.append({
                        "team": team,
                        "participant": participant,
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip()
                    })

                print(f"Saved transcripts for {team}-{participant}")

            except Exception as e:
                print(f"Error transcribing {file}: {e}")

    return transcripts, segments_all

def merge_teams(transcripts):
    """Merge transcripts per team and save outputs."""
    for team, texts in transcripts.items():
        team_text = "\n".join(texts)
        out_file = os.path.join(OUTPUTS_DIR, f"{team}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(team_text)
        print(f"Team transcript saved → {out_file}")

def merge_timeline(segments_all):
    """Merge all transcripts into a single time-sorted conversation."""
    # Sort by start time
    segments_all.sort(key=lambda x: x["start"])

    # Save JSON (structured)
    json_path = os.path.join(OUTPUTS_DIR, "conversation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments_all, f, indent=2, ensure_ascii=False)

    # Save TXT (readable format)
    txt_path = os.path.join(OUTPUTS_DIR, "conversation.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments_all:
            f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] "
                    f"({seg['team']} - {seg['participant']}): {seg['text']}\n")

    print(f"Global conversation saved → {json_path}, {txt_path}")

if __name__ == "__main__":
    ensure_dirs()
    transcripts, segments_all = transcribe_audio()
    merge_teams(transcripts)
    merge_timeline(segments_all)
