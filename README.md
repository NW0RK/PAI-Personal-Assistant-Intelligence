# Gemma Live

Double-click **Start Gemma.cmd** to open the desktop app without keeping a terminal
open. On the first launch, choose a microphone and click **Start / Retry**. Later
launches reuse your selected microphone and start automatically. The app checks
dependencies and model files, starts the local server if needed, and waits for it
to be ready before listening.

The window includes **Pause / Resume**, **Cancel reply**, **Clear conversation**,
and **Quit**. It shows streamed replies, Copy reply buttons, server health, and a
microphone level meter. The microphone stops while Gemma thinks or speaks. Interrupted
answers are marked incomplete. There is no speech transcript; your input appears
as **Voice message**. Shortcuts: **Ctrl+Enter** starts/retries, **Ctrl+Space**
pauses/resumes, and **Esc** cancels a reply. Closing the window quits the app.

**Screen sharing starts off every time you open the app.** Enable **Share primary
screen with Gemma** to include a screenshot with subsequent voice messages. Changing
the toggle does not alter a turn already in progress. Earlier screenshots remain in
the conversation context until its ten-turn limit expires or you click **Clear
conversation**. Clearing also cancels the current reply and removes retained session
media. Your paused/listening state is preserved.

Choose a microphone, speech output, or floating captions and click **Apply setup**.
This cancels any active reply while applying the new configuration. **Refresh**
rescans microphones; a missing saved device requires selecting another one. Speech
sensitivity and silence duration are under **Advanced settings and diagnostics**.
Preferences are saved in `%LOCALAPPDATA%\GemmaLive\settings.json`; command-line
options override saved preferences. Invalid settings restore defaults with a warning.

Advanced settings also show dependency checks, free RAM, disk space, and recent
activity. Server paths, port, context size, GPU layers, and the draft model can be
changed before starting services. These advanced server changes apply to the current
session; use command-line options for a repeatable custom launch. To change an
already loaded server, quit and relaunch it with those options. Server diagnostics
are in `launcher-server.log`. Existing servers are reused and left running when you
quit; servers started by this app are stopped. No services are installed and no
downloads are performed automatically.
If the server becomes unavailable during use, **Start / Retry** reconnects services
and resets conversation context. A busy external server is left running.

## Requirements

Windows, Python with Pillow and Tk, FFmpeg/ffplay on PATH, llama-server and the Gemma
model files. Piper and its voice model enable speech; missing voice dependencies
fall back to captions. The supplied defaults match this machine's existing setup.
Install Python dependencies with `python -m pip install -r requirements.txt` if needed.

## Options

- `python launch_gemma.py --check`: inspect the system without starting services.
- `python launch_gemma.py --wait`: open the app without automatically starting services.
- `python launch_gemma.py --terminal`: use the original terminal dashboard. Shortcuts:
  **Enter** starts/retries, **P** pauses/resumes, **C** cancels, and **Q** quits.
  The terminal must remain open while using this mode.
- `python launch_gemma.py --port 9090`: use another port for both services.
- `python launch_gemma.py --device "Microphone name"`: select a microphone.
- `python launch_gemma.py --no-overlay`: disable floating captions.
- `python launch_gemma.py --no-tts`: disable speech; replies remain in the window.
- `--overlay`, `--tts`: explicitly enable captions or speech over saved preferences.
- `--silence 0.6`, `--threshold 0.02`: override speech detection settings.
- `--settings-file path`: use a different settings file.
- `python launch_gemma.py --speech-mode full`: wait for the complete answer before
  synthesizing speech. The default, `streamed`, speaks completed phrases while
  the model continues generating. Direct client use accepts the same option.
- `--server`, `--model`, `--mmproj`, `--mtp`, `--piper-path`: override local paths.
- `--ctx`, `--gpu-layers`, `--no-mtp`: tune the existing server configuration.

Direct use remains supported: run `run_server.ps1`, then `python live_gemma.py`.
The client also accepts `--server-url` and `--piper-path`. Direct client and terminal
mode use voice only unless you pass `--screen-sharing`. Desktop mode always starts
with screen sharing off; enable it using the visible toggle. `--captions-only` on
the direct client suppresses the separate floating control panel.

Both launchers explicitly use `--load-mode mmap` without memory locking (the
equivalent of disabling Keep Model in Memory). Windows can still cache model files
in RAM during loading; this setting does not guarantee zero RAM usage. Existing
servers must be restarted to apply the setting.

The microphone stops during generation and playback. Cancel returns to listening;
Pause cancels and stays paused. Partial captions are marked incomplete on interruption.
Only completed replies enter conversation history. Current-session media is removed
when its turn expires or the app exits normally; forcibly closing the terminal can
leave files behind. Model conversation history retains at most ten completed turns;
the desktop view retains up to 100 voice-message entries for reading and copying.
The desktop conversation view is not saved between sessions.

In streamed mode, one phrase can play while the next is synthesized. At most two
completed audio chunks wait for playback; remaining text is retained in order.
Speech splits at sentence boundaries or newlines, with a 180-character target for
long phrases (whole words are preserved). These limits live in `SpeechConfig`.
Short answers without a phrase boundary wait until generation completes.
Cancel/Pause stops both generation and all speech jobs. If generation fails after
speech has started, queued speech stops and captions are marked incomplete; words
already spoken cannot be retracted. An audio failure leaves generation running
and falls back to captions for the rest of that turn.

The controller owns conversation history and turn state. `speech_pipeline.py`
owns segmentation, bounded audio scheduling, and chunk cleanup through injectable
job factories. Generation and playback have independent states; Speaking takes
display priority while both run. The launcher reads JSON events for status,
microphone activity, screen sharing, reply progress, completion, failures, and
clearing. Service supervision and device discovery run outside Tk's main thread,
keeping window controls responsive. Floating captions run in their own process
without a second desktop control panel. Timing events contain only a turn ID,
monotonic timestamp, stage, and optional chunk/outcome fields.

To compare speech scheduling with the installed Piper/ffplay, run
`python benchmark_speech.py --runs 3`. This plays a fixed three-sentence replay in
both modes, excludes one warm-up per mode, and writes `speech-benchmark.json`.
For the full model path, start the local server and use
`python benchmark_speech.py --audio media/test.wav --runs 5`.
This sends the recording and current screen to the local model; it does not open
the microphone. Reports contain timing numbers only. They include median/p95
time to playback launch, total turn time, and the largest gap between playback
processes. Playback launch is not acoustic onset; listen for actual onset and
phrase naturalness, and use more runs for meaningful p95 comparisons. Keep the
model settings and desktop scene stable when comparing model runs.

Run checks with `python -m unittest discover -v`.
Optional desktop window checks, without starting the model or microphone:
`$env:GEMMA_DESKTOP_TESTS='1'; python -m unittest test_desktop_app.WindowTests -v`.
Local model checks (start an owned server if needed and release it afterward):
`$env:GEMMA_MODEL_TESTS='1'; python -m unittest test_desktop_model -v`.
These resample the existing `media/test_speech.wav` recording and test voice-only
and screen-shared requests; the latter captures your current primary screen.
Set `GEMMA_PLAYBACK_TESTS=1` as well to test actual streamed speech playback, or
`GEMMA_DESKTOP_SESSION_TESTS=1` to open the app and briefly test your microphone,
pause/resume, screen controls, and external-server ownership.
Optional desktop/model integration checks (with the server already running):
`$env:GEMMA_INTEGRATION='1'; python -m unittest test_integration -v` in PowerShell.
These briefly open the control panel and send the existing `media/test.wav` plus
a current screenshot to the local model.
