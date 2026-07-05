import argparse
import os
import site
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if VENDOR_DIR.exists():
    site.addsitedir(str(VENDOR_DIR))

import requests
from dotenv.main import load_dotenv

load_dotenv()

LITELLM_BASE_URL = os.getenv("LLM_API_BASE")
API_KEY = os.getenv("LLM_API_KEY")


def print_guide():
    guide_text = """
===========================================
LiteLLM Budget Checker - Usage Guide
===========================================

This tool helps you monitor API usage against an allocated budget when the
configured endpoint is a LiteLLM proxy exposing /key/info.

Setup:
1. Ensure your .env file contains:
   LLM_API_KEY=sk-...
   LLM_API_BASE=https://your-litellm-proxy

Commands:
- Check Status:  python budget_checker.py
- Show Guide:    python budget_checker.py --guide

Note:
- Provider-native OpenAI-compatible endpoints such as Groq generally do not
  expose /key/info, so this script cannot query spend from them directly.
===========================================
"""
    print(guide_text)


def get_key_info(api_key: str, base_url: str):
    if not api_key:
        print("[ERROR] LLM_API_KEY not found in environment variables.")
        print("        Please check your .env file.")
        sys.exit(1)

    if not base_url:
        print("[ERROR] LLM_API_BASE not found in environment variables.")
        print("        Please check your .env file.")
        sys.exit(1)

    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/key/info"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params={"key": api_key},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print(f"[INFO] {base_url} does not expose /key/info.")
            print("       This checker works with LiteLLM proxy deployments.")
            print("       The configured endpoint looks like a provider-native")
            print("       OpenAI-compatible API, so spend cannot be fetched here.")
            sys.exit(2)
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"[ERROR] Budget check failed with HTTP {status}.")
        print(f"        Details: {exc}")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Could not connect to {base_url}")
        print(f"        Details: {exc}")
        sys.exit(1)


def display_budget(info):
    info_data = info.get("info", {}) if isinstance(info, dict) else {}
    if not info_data:
        info_data = info

    max_budget = info_data.get("max_budget")
    spend = info_data.get("spend", 0.0)
    user_id = info_data.get("user_id", "Unknown User")

    print("\nAPI Budget Status")
    print("-------------------")
    print(f"User ID:      {user_id}")
    print(f"Total Spend:  ${spend:.4f}")

    if max_budget is None:
        print("Max Budget:   Unlimited")
    else:
        remaining_val = max_budget - spend
        print(f"Max Budget:   ${max_budget:.4f}")
        print(f"Remaining:    ${remaining_val:.4f}")

    print("-------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Check LiteLLM API Key Budget")
    parser.add_argument("--guide", action="store_true", help="Show usage guide and exit")
    args = parser.parse_args()

    if args.guide:
        print_guide()
        return

    masked_key = f"{API_KEY[:4]}...{API_KEY[-4:]}" if API_KEY else "(missing)"
    print(f"Checking budget for key: {masked_key}")
    key_info = get_key_info(API_KEY, LITELLM_BASE_URL)
    display_budget(key_info)


if __name__ == "__main__":
    main()
