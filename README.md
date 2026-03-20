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
wget -O Bili_Note-alpha.tar.gz https://codeload.github.com/MapleHe/Bili_Note/tar.gz/refs/tags/alpha
tar -zxf Bili_Note-alpha && mv Bili_Note-alpha Bili_Note && rm Bili_Note-alpha.tar.gz
```

OR use git clone:

```bash
git clone https://github.com/MapleHe/Bili_Notes.git
```

## Usage

```bash
cd Bili_Note/
chmod u+x ./start_server.sh
./start_server.sh
```

The startup script automatically detects your Python environment (`uv`, `.venv`, or system `pip3`, or `python -m pip`) and installs dependencies from `requirements.txt` before launching the server.

## Contributing

Implemented with Claude Sonnet 4.6.

## License

[MIT](https://github.com/MapleHe/Bili_Notes/blob/main/LICENSE)
