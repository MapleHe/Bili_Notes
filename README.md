# Bili_Notes — A simple web service for summarizing Bilibili video transcriptions

Provide a BV ID and your preferred LLM API key (OpenAI-compatible providers only; DeepSeek by default).

The server is built with Python [Flask](https://github.com/pallets/flask). Please configure your firewall carefully before deploying on a public server. Use at your own risk.

This is a demo project for learning purposes only.

## Requirements

- Python 3.12+ managed via [`uv`](https://github.com/astral-sh/uv) (or any standard Python environment)
- `ffmpeg` installed on the system

## Installation

Download the archive and unarchive the files:

```bash
wget -O Bili_Notes-alpha.tar.gz https://codeload.github.com/MapleHe/Bili_Notes/tar.gz/refs/tags/alpha
tar -zxf Bili_Notes-alpha.tar.gz && mv Bili_Notes-alpha Bili_Notes && rm Bili_Notes-alpha.tar.gz
```

OR use git clone:

```bash
git clone https://github.com/MapleHe/Bili_Notes.git
```

Installation in termux (Android)

```bash
pkg install proot-distro
proot-distro install ubuntu
proot-distro login ubuntu

# in the ubuntu env
apt update && apt install python3 python3-pip curl build-essential ffmpeg -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -C - -o Bili_Notes-alpha.tar.gz https://codeload.github.com/MapleHe/Bili_Notes/tar.gz/refs/tags/alpha
tar -zxf Bili_Notes-alpha.tar.gz && mv Bili_Notes-alpha Bili_Notes && rm Bili_Notes-alpha.tar.gz
cd Bili_Notes;
uv venv
source .venv/bin/activate
uv pip install -r requirements
cd ..
```

## Usage

```bash
cd Bili_Notes/
chmod u+x ./start_server.sh
./start_server.sh
```

The startup script automatically detects your Python environment (`uv`, `.venv`, or system `pip3`, or `python -m pip`) and installs dependencies from `requirements.txt` before launching the server.

## Limitations

The current version of Bili_Notes is using a SenseVoice model which runs slow for long audio file. 

## Contributing

Implemented with Claude Sonnet 4.6.

## License

[MIT](https://github.com/MapleHe/Bili_Notes/blob/main/LICENSE)
