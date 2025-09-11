from typing import Optional
import requests
import os


def delete_agent_repo(
    base_url: str,
    username: str,
    agent_name: str,
    bearer_token: Optional[str] = None,
    stytch_session: Optional[str] = None,
) -> requests.Response:
    """
    Delete an agent repository.

    Params:
        base_url (str): API base URL (e.g., http://localhost:8000).
        username (str): Owner username.
        agent_name (str): Repository name.
        bearer_token (Optional[str]): Bearer token for Authorization header.
        stytch_session (Optional[str]): Session token for 'stytch_session' cookie.

    Returns:
        requests.Response: HTTP response object.
    """
    url = f"{base_url}/api/v1/agents/delete/owner/{username}/agent/{agent_name}"
    headers = {"Authorization": f"token {bearer_token}"} if bearer_token else {}
    cookies = {"stytch_session": stytch_session} if stytch_session else {}
    resp = requests.delete(url, headers=headers, cookies=cookies, timeout=30)
    resp.raise_for_status()
    return resp

token = os.getenv("MODAIC_TOKEN")
base_url = os.getenv("MODAIC_API_URL")
print("token", token)
delete_agent_repo(base_url, "swagginty", "delete-this", bearer_token=token)