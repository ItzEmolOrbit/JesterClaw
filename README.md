# JesterClaw — Server Side
**Team Lapanic / EmolOrbit**

## Architecture

```
┌─────────────────────────────┐          WebSocket ws://IP:3839/agent
│   GPU Server  (Linux/Ubuntu)│  ◄──────────────────────────────────►  Windows Client
│                             │
│  • Qwen2.5-Omni-3B model    │   Client sends:  text / audio / image
│  • Runs inference           │   Server sends:  tokens / text / audio (TTS) / action JSON
│  • Safety filter            │
│  • Dispatches action JSON   │   Client EXECUTES actions locally on Windows
└─────────────────────────────┘   (pyautogui, pywinauto, playwright, mss)
```

The server **never runs any Windows OS automation**. It only:
1. Receives voice/text/image from client
2. Runs the AI model
3. Streams response tokens back
4. Sends structured `action` JSON to the client when needed
5. The **Windows client** executes every action locally

---

## Setup (on your Linux GPU server)

```bash
# 1. Install dependencies
bash dependency_installer.bash

# 2. Download the model (~7GB)
bash model_installer.bash

# 3. Configure
cp .env.example .env
nano .env  # set JESTERCLAW_TOKEN to something strong

# 4. Start
python Main.py
```

---

## Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | None | Server + VRAM status |
| `/agent` | WebSocket | `?token=TOKEN` | Main agent loop |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JESTERCLAW_HOST` | `0.0.0.0` | Bind address |
| `JESTERCLAW_PORT` | `3839` | Port |
| `JESTERCLAW_TOKEN` | `jesterclaw-secret-change-me` | Auth token (**change this!**) |
| `JESTERCLAW_TTS` | `1` | Enable TTS audio responses |
