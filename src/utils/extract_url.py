
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from curl_cffi import requests as cffi_requests


_MIRROR_HOST_RE = re.compile(
    r"^upos-([0-9a-z]+?)-mirror([0-9a-z]+)\.(bilivideo\.com|akamaized\.net)$",
    re.I,
)


def bvid_to_url(bvid: str) -> str:
    """Convert a BV ID like 'BV1ZDwWzpEgg' to a full Bilibili video URL."""
    bvid = bvid.strip()
    if not bvid.startswith("BV"):
        raise ValueError(f"Invalid BV ID: {bvid!r}")
    return f"https://www.bilibili.com/video/{bvid}"


def _safe_title(title: str) -> str:
    """Keep only alphanumeric and Chinese characters; strip everything else."""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", title, flags=re.ASCII).replace("_", "")


def _normalize_url(url: str) -> str:
    return url.replace("http://", "https://") if url else url


def _cookie_to_header(cookie: Optional[str | dict]) -> Optional[str]:
    if cookie is None:
        return None
    if isinstance(cookie, str):
        return cookie.strip()
    if isinstance(cookie, dict):
        return "; ".join(f"{k}={v}" for k, v in cookie.items())
    raise TypeError("cookie 必须是 str / dict / None")


def _extract_bvid_and_p(page_url: str) -> Tuple[Optional[str], int]:
    parsed = urlparse(page_url)
    m = re.search(r"/video/(BV[0-9A-Za-z]+)", parsed.path)
    bvid = m.group(1) if m else None
    qs = parse_qs(parsed.query)
    try:
        p = max(int(qs.get("p", ["1"])[0]), 1)
    except ValueError:
        p = 1
    return bvid, p


def _extract_initial_state(html: str) -> Optional[dict]:
    patterns = [
        r"window\.__INITIAL_STATE__\s*=\s*({.*?})\s*;\s*\(function",
        r"window\.__INITIAL_STATE__\s*=\s*({.*?})\s*;\s*</script>",
        r"window\.__INITIAL_STATE__\s*=\s*({.*?})\s*;",
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _classify_url_type(url: str) -> str:
    """
    对齐源码里的 q(e)：
    mirror / upos / bcache / mcdn / other
    """
    try:
        u = urlparse(url)
        host = (u.hostname or "").lower()
        port = u.port
        query = parse_qs(u.query)
        os_value = (query.get("os", [""])[0] or "").lower()

        if _MIRROR_HOST_RE.match(host):
            return "mirror"
        if os_value == "upos" or "upos" in host:
            return "upos"
        if os_value == "bcache":
            return "bcache"
        non_std_port = port is not None and port not in (80, 443)
        if os_value == "mcdn" or "mcdn" in host or non_std_port:
            return "mcdn"
        return "other"
    except Exception:
        return "other"


def _pick_preferred_url(all_urls: List[str], prefer_url_type: Optional[str]) -> str:
    all_urls = [u for u in all_urls if u]
    if not all_urls:
        raise ValueError("没有可用下载地址")

    if prefer_url_type:
        for u in all_urls:
            if _classify_url_type(u) == prefer_url_type:
                return u

    order = {"mirror": 0, "upos": 1, "bcache": 2, "mcdn": 3, "other": 4}
    return sorted(all_urls, key=lambda u: order.get(_classify_url_type(u), 999))[0]


def _build_session(page_url: str, cookie: Optional[str | dict]) -> cffi_requests.Session:
    sess = cffi_requests.Session(impersonate="chrome110")
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/110.0.0.0 Safari/537.36"
            ),
            "Referer": page_url,
            "Origin": "https://www.bilibili.com",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )
    cookie_header = _cookie_to_header(cookie)
    if cookie_header:
        sess.headers["Cookie"] = cookie_header
    return sess


def _fetch_page_meta(
    sess: cffi_requests.Session,
    page_url: str,
    timeout: int = 20,
) -> Dict[str, Any]:
    """
    尽量从页面里取 aid/cid，模拟源码里的 unsafeWindow.aid / unsafeWindow.cid。
    解析不到时，普通视频页回退到 x/web-interface/view?bvid=...
    """
    resp = sess.get(page_url, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    state = _extract_initial_state(html)
    parsed = urlparse(page_url)
    is_bangumi = "/bangumi/" in parsed.path or "/play/ep" in parsed.path or "/play/ss" in parsed.path
    bvid, p = _extract_bvid_and_p(page_url)

    if state:
        if is_bangumi:
            ep = state.get("epInfo") or {}
            aid = ep.get("aid") or state.get("aid")
            cid = ep.get("cid") or state.get("cid")
            title = ep.get("long_title") or ep.get("title") or state.get("h1Title") or "bilibili_audio"
            if aid and cid:
                return {
                    "aid": aid,
                    "cid": cid,
                    "bvid": bvid,
                    "title": title,
                    "is_bangumi": True,
                }

        video_data = state.get("videoData") or {}
        aid = video_data.get("aid") or state.get("aid")
        bvid = video_data.get("bvid") or state.get("bvid") or bvid
        title = video_data.get("title") or state.get("h1Title") or "bilibili_audio"
        cid = video_data.get("cid") or state.get("cid")

        pages = video_data.get("pages") or []
        if pages:
            idx = min(max(p - 1, 0), len(pages) - 1)
            cid = pages[idx].get("cid") or cid

        if aid and cid:
            return {
                "aid": aid,
                "cid": cid,
                "bvid": bvid,
                "title": title,
                "is_bangumi": False,
            }

    if not bvid:
        raise ValueError("无法从页面 URL 解析出 BV 号，也无法从页面提取 aid/cid")

    # 普通视频页兜底
    api = "https://api.bilibili.com/x/web-interface/view"
    r = sess.get(api, params={"bvid": bvid}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"view API 失败: {payload.get('message') or payload}")

    data = payload.get("data") or {}
    pages = data.get("pages") or []
    idx = min(max(p - 1, 0), len(pages) - 1) if pages else 0
    cid = pages[idx].get("cid") if pages else data.get("cid")
    aid = data.get("aid")

    if not aid or not cid:
        raise ValueError("view API 返回里没有 aid/cid")

    return {
        "aid": aid,
        "cid": cid,
        "bvid": bvid,
        "title": data.get("title") or "bilibili_audio",
        "is_bangumi": False,
    }


def extract_bilibili_dash_audio_url(
    page_url: str,
    cookie: Optional[str | dict] = None,
    qn: Optional[int] = None,
    prefer_url_type: str = "mirror",
    timeout: int = 20,
) -> Dict[str, Any]:
    """
    按 Bilibili-Evolved 的 video.dash.audio 思路：
    1) 页面 URL -> aid/cid
    2) 调 playurl 接口
    3) 收集 dash.audio / dolby.audio / flac.audio
    4) 选 bandwidth 最大的音频流
    5) 按 prefer_url_type 选最终 download URL

    返回:
    {
        "download_url": ...,
        "all_urls": [...],
        "url_type": "mirror|upos|bcache|mcdn|other",
        "audio_type": "audio|flacAudio",
        "bandwidth": 123456,
        "codecs": "...",
        "extension": ".m4a|.flac",
        "aid": ...,
        "cid": ...,
        "bvid": ...,
        "title": ...,
        "raw_playurl": {...},
    }
    """
    sess = _build_session(page_url, cookie)
    meta = _fetch_page_meta(sess, page_url, timeout=timeout)

    params = {
        "avid": meta["aid"],
        "cid": meta["cid"],
        "qn": "" if qn is None else str(qn),
        "otype": "json",
        "fourk": 1,
        "fnver": 0,
        "fnval": 4048,
    }

    if meta["is_bangumi"]:
        api = "https://api.bilibili.com/pgc/player/web/playurl"
    else:
        api = "https://api.bilibili.com/x/player/playurl"

    r = sess.get(api, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()

    if payload.get("code") not in (None, 0):
        raise RuntimeError(f"playurl API 失败: {payload.get('message') or payload}")

    if meta["is_bangumi"]:
        playurl = payload.get("result") or {}
    else:
        playurl = payload.get("data") or {}
    dash = playurl.get("dash")
    if not dash:
        raise ValueError("此视频没有 dash 格式，或者当前账号/清晰度下拿不到 dash 数据")

    candidates: List[Dict[str, Any]] = []

    def add_audio(item: dict, audio_type: str = "audio") -> None:
        if not item:
            return
        download_url = _normalize_url(item.get("baseUrl") or item.get("base_url") or "")
        backup_urls = [
            _normalize_url(u)
            for u in (item.get("backupUrl") or item.get("backup_url") or [])
        ]
        if not download_url:
            return
        candidates.append(
            {
                "audio_type": audio_type,
                "bandwidth": int(item.get("bandwidth") or 0),
                "codecs": item.get("codecs") or "",
                "codec_id": item.get("codecid") or 0,
                "download_url": download_url,
                "all_urls": [download_url, *backup_urls],
                "extension": ".flac" if audio_type == "flacAudio" else ".m4a",
            }
        )

    for x in dash.get("audio") or []:
        add_audio(x, "audio")

    dolby = dash.get("dolby") or {}
    for x in dolby.get("audio") or []:
        add_audio(x, "audio")

    flac = dash.get("flac") or {}
    if flac.get("audio"):
        add_audio(flac["audio"], "flacAudio")

    if not candidates:
        raise ValueError("dash 中没有可用音频流")

    # 对齐源码：音频按 bandwidth 降序取第一条
    candidates.sort(key=lambda x: x["bandwidth"], reverse=True)
    best = candidates[0]

    chosen_url = _pick_preferred_url(best["all_urls"], prefer_url_type)

    return {
        "download_url": chosen_url,
        "all_urls": best["all_urls"],
        "url_type": _classify_url_type(chosen_url),
        "audio_type": best["audio_type"],
        "bandwidth": best["bandwidth"],
        "codecs": best["codecs"],
        "extension": best["extension"],
        "aid": meta["aid"],
        "cid": meta["cid"],
        "bvid": meta["bvid"],
        "title": meta["title"],
        "raw_playurl": playurl,
    }


def download_audio(
    download_url: str,
    filename: str,
    page_url: str = "https://www.bilibili.com",
    cookie: Optional[str | dict] = None,
    chunk_size: int = 1024 * 1024,
    timeout: int = 60,
    show_progress: bool = True,
) -> str:
    """
    根据提取到的音频 URL 下载文件并保存到本地。

    参数：
        download_url  : 由 extract_bilibili_dash_audio_url 返回的 download_url
        filename      : 保存的本地文件路径（含扩展名，例如 "output.m4a"）
        page_url      : 视频页面 URL，用于构造正确的 Referer / Origin 请求头
        cookie        : 可选 Cookie（str 或 dict），需要登录态时传入
        chunk_size    : 流式读取的块大小（字节），默认 1 MB
        timeout       : 请求超时秒数，默认 60 秒
        show_progress : 是否在终端显示下载进度条，默认 True

    返回：
        实际写入的文件路径（与 filename 相同）

    异常：
        requests.HTTPError  : HTTP 状态码非 2xx
        OSError             : 文件写入失败
    """
    sess = _build_session(page_url, cookie)
    # Range 请求头：从头开始下载，同时让服务端返回 Content-Length
    sess.headers.update({"Range": "bytes=0-"})

    resp = sess.get(download_url, stream=True, timeout=timeout)
    # 206 Partial Content 也是正常响应
    if resp.status_code not in (200, 206):
        resp.raise_for_status()

    # 尝试从 Content-Range 或 Content-Length 获取总大小
    total: Optional[int] = None
    content_range = resp.headers.get("Content-Range", "")  # e.g. "bytes 0-1234/5678"
    if content_range:
        m = re.search(r"/(\d+)$", content_range)
        if m:
            total = int(m.group(1))
    if total is None:
        cl = resp.headers.get("Content-Length")
        if cl and cl.isdigit():
            total = int(cl)

    downloaded = 0

    def _fmt(n: int) -> str:
        """将字节数格式化为人类可读的字符串。"""
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    with open(filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if show_progress:
                if total:
                    pct = downloaded / total * 100
                    bar_len = 30
                    filled = int(bar_len * downloaded / total)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(
                        f"\r  [{bar}] {pct:5.1f}%  "
                        f"{_fmt(downloaded)} / {_fmt(total)}   ",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r  已下载 {_fmt(downloaded)} ...", end="", flush=True)
    if show_progress:
        print(f"\r  下载完成：{filename} ({_fmt(downloaded)}){'':30}")
    return filename


def download_bilibili_wav(
    bvid: str,
    output_dir: str,
    cookie: Optional[str | dict] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Download Bilibili audio by BV ID and convert to 16kHz mono WAV.

    Steps:
    1. Extract DASH audio URL from Bilibili.
    2. Stream-download the raw audio (m4a/flac) to a temp file.
    3. Convert to 16kHz mono PCM WAV via ffmpeg.
    4. Delete the temp file.

    Parameters:
        bvid        : Bilibili BV ID, e.g. "BV1ZDwWzpEgg"
        output_dir  : Directory where the final .wav file is saved.
        cookie      : Optional Bilibili cookie (str or dict).
        show_progress: Print download progress to stdout.

    Returns:
        {
            "bvid": str,
            "title": str,
            "safe_title": str,
            "wav_path": str,   # absolute path to the output WAV file
        }
    """
    page_url = bvid_to_url(bvid)

    # Resolve the temp directory relative to the project structure.
    # output_dir is typically <project>/data/userdata/; temp is <project>/data/temp/
    output_dir = os.path.abspath(output_dir)
    project_data_dir = os.path.join(os.path.dirname(output_dir), "..", "data")
    temp_dir = os.path.join(os.path.abspath(os.path.join(output_dir, "..", "..")), "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if show_progress:
        print(f"正在提取音频地址：{bvid}")

    info = extract_bilibili_dash_audio_url(
        page_url=page_url,
        cookie=cookie,
        prefer_url_type="mirror",
    )

    title = info["title"]
    safe = _safe_title(title) or bvid
    extension = info["extension"]  # ".m4a" or ".flac"

    temp_audio = os.path.join(temp_dir, f"{bvid}{extension}")
    wav_path = os.path.join(output_dir, f"{bvid}-{safe}.wav")

    if show_progress:
        print(f"标题：{title}")
        print(f"开始下载音频到临时文件：{temp_audio}")

    download_audio(
        download_url=info["download_url"],
        filename=temp_audio,
        page_url=page_url,
        cookie=cookie,
        show_progress=show_progress,
    )

    if show_progress:
        print(f"正在转换为 WAV：{wav_path}")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-threads", "2",
        "-i", temp_audio,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        wav_path,
    ]
    result = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 转换失败 (returncode={result.returncode}):\n"
            + result.stderr.decode(errors="replace")
        )

    # Clean up temp file
    try:
        os.remove(temp_audio)
    except OSError:
        pass

    if show_progress:
        print(f"WAV 文件已保存：{wav_path}")

    return {
        "bvid": bvid,
        "title": title,
        "safe_title": safe,
        "wav_path": wav_path,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_url.py <BVID>")
        print("示例: python extract_url.py BV1ZDwWzpEgg")
        sys.exit(1)

    bvid_arg = sys.argv[1].strip()
    output_directory = os.getcwd()

    print(f"BV ID: {bvid_arg}")
    print(f"输出目录: {output_directory}")

    result = download_bilibili_wav(
        bvid=bvid_arg,
        output_dir=output_directory,
        show_progress=True,
    )
    print(f"\n完成！")
    print(f"  标题     : {result['title']}")
    print(f"  安全标题 : {result['safe_title']}")
    print(f"  WAV 路径 : {result['wav_path']}")
