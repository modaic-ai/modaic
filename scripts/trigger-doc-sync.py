#!/usr/bin/env python3
"""
Documentation Sync Trigger Script (Python version)

This script triggers the documentation sync workflow in the modaic repository.
Usage: python scripts/trigger-doc-sync.py
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    NC = "\033[0m"  # No Color


def print_status(message: str) -> None:
    """Print a status message in green."""
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def get_github_token() -> Optional[str]:
    """
    Get GitHub token from environment variable.

    Returns:
        GitHub token if found, None otherwise
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print_error("GITHUB_TOKEN environment variable is not set")
        print_error("Please set your GitHub personal access token:")
        print_error("export GITHUB_TOKEN='your_token_here'")
    return token


def trigger_doc_sync(repo: str, token: str) -> bool:
    """
    Trigger the documentation sync workflow.

    Params:
        repo: Repository name in format 'owner/repo'
        token: GitHub personal access token

    Returns:
        True if successful, False otherwise
    """
    url = f"https://api.github.com/repos/{repo}/dispatches"

    payload = {
        "event_type": "doc-sync",
        "client_payload": {
            "triggered_by": "external_python_script",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}",
            "User-Agent": "doc-sync-trigger-python-script",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            status_code = response.getcode()

            if status_code == 204:
                print_status("Documentation sync triggered successfully!")
                print_status(
                    f"Check the Actions tab at: https://github.com/{repo}/actions"
                )
                return True
            else:
                print_error(f"Unexpected status code: {status_code}")
                return False

    except urllib.error.HTTPError as e:
        status_code = e.code

        if status_code == 401:
            print_error("Authentication failed (HTTP 401)")
            print_error("Please check your GITHUB_TOKEN permissions")
        elif status_code == 403:
            print_error("Forbidden (HTTP 403)")
            print_error(
                "Your token may not have the required permissions for this repository"
            )
        elif status_code == 404:
            print_error("Repository not found (HTTP 404)")
            print_error(f"Please check the repository name: {repo}")
        elif status_code == 422:
            print_error("Unprocessable Entity (HTTP 422)")
            print_error("The repository may not have the doc-sync workflow configured")
        else:
            print_error(f"HTTP Error {status_code}: {e.reason}")

        return False

    except Exception as e:
        print_error(f"An error occurred: {e}")
        return False


def main() -> None:
    """Main function to trigger documentation sync."""
    repo = os.getenv("MODAIC_REPO", "modaic-ai/modaic")

    print_status(f"Triggering documentation sync for repository: {repo}")

    token = get_github_token()
    if not token:
        sys.exit(1)

    success = trigger_doc_sync(repo, token)

    if success:
        print_status("Script completed successfully")
        sys.exit(0)
    else:
        print_error("Script failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
