"""
LLM-based text completion and summarization using OpenAI-compatible API.
Default provider: DeepSeek (https://api.deepseek.com/v1)
"""

import argparse
import os
import sys
from typing import Optional

from openai import OpenAI


# Constants
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# Provider configurations
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    },
    "custom": {
        "name": "Custom",
        "base_url": "",
        "default_model": "",
        "models": [],
    },
}


def complete_transcription(
    text: str,
    bvid: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Complete and correct ASR-generated transcription text using LLM.

    Args:
        text: The original ASR transcription text
        bvid: Bilibili video ID
        api_key: API key for the LLM provider
        model: Model name to use (default: deepseek-chat)
        base_url: API base URL (default: DeepSeek)

    Returns:
        Completed text with BV ID header

    Raises:
        Exception: If API call fails
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = (
        "你是一个专业的文字编辑。请对以下自动语音识别（ASR）生成的文字进行校对和补全，"
        "保持原意的同时修正明显的识别错误，使文字更加通顺易读。"
        "直接输出校正后的文字，不要添加任何解释或前缀。"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        completed_text = response.choices[0].message.content.strip()
        return f"## BVID: {bvid}\n\n{completed_text}"
    except Exception as e:
        raise Exception(f"Failed to complete transcription: {str(e)}")


def summarize_text(
    text: str,
    bvid: str,
    summary_prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Summarize text using LLM based on user-provided prompt.

    Args:
        text: The text to summarize
        bvid: Bilibili video ID
        summary_prompt: User-defined summarization prompt
        api_key: API key for the LLM provider
        model: Model name to use (default: deepseek-chat)
        base_url: API base URL (default: DeepSeek)

    Returns:
        Summary with BV ID header

    Raises:
        Exception: If API call fails
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = "你是一个专业的内容总结助手。请根据用户的要求对提供的文字进行总结。"
    user_message = f"{summary_prompt}\n\n---\n\n{text}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        summary = response.choices[0].message.content.strip()
        return f"## BVID: {bvid}\n\n{summary}"
    except Exception as e:
        raise Exception(f"Failed to summarize text: {str(e)}")


def get_available_models(base_url: str, api_key: str) -> list[str]:
    """
    Get list of available models from the API.

    Args:
        base_url: API base URL
        api_key: API key for the provider

    Returns:
        List of available model names, or default models if API call fails
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:
        # Return default models on any error
        return ["deepseek-chat", "deepseek-reasoner"]


def main():
    """CLI entry point for text completion and summarization."""
    parser = argparse.ArgumentParser(
        description="LLM-based text completion and summarization"
    )
    parser.add_argument("input", help="Input text file path")
    parser.add_argument(
        "--output",
        help="Output file path (default: input with _completed or _summary suffix)",
    )
    parser.add_argument(
        "--mode",
        choices=["complete", "summarize"],
        default="complete",
        help="Operation mode (default: complete)",
    )
    parser.add_argument("--prompt", help="Summarization prompt (for summarize mode)")
    parser.add_argument(
        "--api-key",
        help="API key (or read from DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--bvid", default="UNKNOWN", help="Bilibili video ID")

    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: API key not provided. Set --api-key or DEEPSEEK_API_KEY env var")
        sys.exit(1)

    # Read input file
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        sys.exit(1)

    # Process based on mode
    try:
        if args.mode == "complete":
            result = complete_transcription(
                text, args.bvid, api_key, args.model, args.base_url
            )
            output_path = args.output or args.input.rsplit(".", 1)[0] + "_completed.txt"
        else:  # summarize
            if not args.prompt:
                print("Error: --prompt is required for summarize mode")
                sys.exit(1)
            result = summarize_text(
                text, args.bvid, args.prompt, api_key, args.model, args.base_url
            )
            output_path = args.output or args.input.rsplit(".", 1)[0] + "_summary.txt"

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)

        print(f"Success! Output saved to: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
