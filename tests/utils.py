import os
from typing import Optional

import requests


def delete_program_repo(
    username: str,
    program_name: str,
    base_url: str = None,
    bearer_token: Optional[str] = None,
    stytch_session: Optional[str] = None,
) -> requests.Response:
    """
    Delete an agent repository.

    Params:
        base_url (str): API base URL (e.g., http://localhost:8000).
        username (str): Owner username.
        program_name (str): Repository name.
        bearer_token (Optional[str]): Bearer token for Authorization header.
        stytch_session (Optional[str]): Session token for 'stytch_session' cookie.

    Returns:
        requests.Response: HTTP response object.
    """
    if base_url is None:
        base_url = os.getenv("MODAIC_API_URL")
    if bearer_token is None:
        bearer_token = os.getenv("MODAIC_TOKEN")
    url = f"{base_url}/api/v1/agents/delete/owner/{username}/agent/{program_name}"
    headers = {"Authorization": f"token {bearer_token}"}
    cookies = {"stytch_session": stytch_session} if stytch_session else {}
    resp = requests.delete(url, headers=headers, cookies=cookies, timeout=30)
    return resp
