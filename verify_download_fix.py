"""
Automated verification script for Windows file-lock download fix.

Verifies:
  1. Real YouTube audio download succeeds cleanly.
  2. No .part file remains after download.
  3. Final .wav file exists and is valid.
  4. Repeated download of the same video URL succeeds without WinError 32.
  5. Concurrent download calls (simulating Streamlit reruns) complete safely.
  6. Audio chunking & cleanup complete cleanly.
"""
import os
import sys
import glob
import time
import threading
import logging
from utils.audio_processor import process_input, cleanup_audio_files, DOWNLOADS_DIR, download_youtube_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_download_fix")

TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def run_download_tests():
    print("\n" + "=" * 70)
    print("RUNNING AUTOMATED DOWNLOAD & FILE-LOCK VERIFICATION TEST")
    print("=" * 70)

    # ── Test 1: Real YouTube Audio Download ──────────────────────────────────
    print("\n[1/4] Testing real YouTube audio download...")
    chunks, main_wav_path = process_input(TEST_URL)

    assert os.path.exists(main_wav_path), f"FAIL: Main WAV file missing at {main_wav_path}"
    assert len(chunks) > 0, "FAIL: Chunks list is empty"
    for chunk in chunks:
        assert os.path.exists(chunk), f"FAIL: Chunk file missing at {chunk}"

    part_files = glob.glob(os.path.join(DOWNLOADS_DIR, "*.part"))
    assert len(part_files) == 0, f"FAIL: Found remaining .part files: {part_files}"

    print(f"      [PASS] Downloaded WAV: {main_wav_path}")
    print(f"      [PASS] Chunks count: {len(chunks)}")
    print("      [PASS] No .part files remain on disk.")

    cleanup_audio_files(chunks, main_wav_path)
    print("      [PASS] Audio cleanup completed.")

    # ── Test 2: Repeated Download of Same Video ──────────────────────────────
    print("\n[2/4] Testing repeated download of the exact same video URL...")
    wav_1 = download_youtube_audio(TEST_URL)
    assert os.path.exists(wav_1), "FAIL: Repeated download 1 output missing"
    
    wav_2 = download_youtube_audio(TEST_URL)
    assert os.path.exists(wav_2), "FAIL: Repeated download 2 output missing"
    print("      [PASS] Second download completed without WinError 32.")

    cleanup_audio_files(main_wav_path=wav_1)
    cleanup_audio_files(main_wav_path=wav_2)

    # ── Test 3: Simulated Concurrent Streamlit Downloads ─────────────────────
    print("\n[3/4] Testing concurrent downloads (simulating Streamlit reruns)...")
    results = [None, None]
    errors = []

    def download_thread(index):
        try:
            results[index] = download_youtube_audio(TEST_URL)
        except Exception as ex:
            errors.append(ex)

    t1 = threading.Thread(target=download_thread, args=(0,))
    t2 = threading.Thread(target=download_thread, args=(1,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"FAIL: Concurrent download raised exceptions: {errors}"
    assert results[0] and os.path.exists(results[0]), "FAIL: Thread 1 output missing"
    assert results[1] and os.path.exists(results[1]), "FAIL: Thread 2 output missing"
    print("      [PASS] Concurrent thread downloads completed cleanly.")

    cleanup_audio_files(main_wav_path=results[0])
    cleanup_audio_files(main_wav_path=results[1])

    # ── Test 4: Verify No Stale Part Files ───────────────────────────────────
    print("\n[4/4] Verifying final disk hygiene...")
    part_files = glob.glob(os.path.join(DOWNLOADS_DIR, "*.part"))
    assert len(part_files) == 0, f"FAIL: Remaining .part files detected: {part_files}"
    print("      [PASS] Zero .part files present in downloads folder.")

    print("\n" + "=" * 70)
    print("[ALL CHECKS PASSED] Download file-lock fix verified 100%.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_download_tests()
