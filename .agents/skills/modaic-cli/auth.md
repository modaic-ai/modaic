# Authentication

## login

```bash
modaic login --workspace acme
```

Match the terminal code and device, then have the user approve the request they
started. Leave the CLI polling. On a headless host, use `modaic login --headless`
and open the printed URL on another device. Non-interactive terminals use
headless mode automatically. Browser sessions authenticate CLI commands;
programmatic SDK clients and CI use API keys.

| Flag | What it does |
| --- | --- |
| --profile | Store a named session; default: default. |
| --workspace | Bind the browser session to a personal or organization workspace. |
| --headless, --no-browser | Print a URL without opening the local browser. |
| --storage | keyring (default), or explicit plaintext file storage on POSIX. |
| --timeout | Login wait in seconds; default: 600. |
| --token-stdin | Verify and save an API key from the first stdin line instead. |
| -h, --help | Show command help. |

OS keyring holds session secrets by default. File fallback uses private
permissions and is unsupported on Windows. Access tokens expire; normal CLI
commands refresh them automatically near expiry. If refresh fails or the session
is revoked, log in again. Do not interpret access-token expiry as the lifetime
of the entire refreshable login.

Use --workspace when commands should be authorized and billed to one selected
workspace. Login accepts the account's own profile or verifies current
organization membership, and the API rechecks it for later requests. Omit the
flag for an unbound personal session. --workspace is not
accepted with --token-stdin because an API key already carries its workspace
scope.

## Stored API-key login

1. Have the user create an API key in Modaic settings under API keys.
2. Run the following in the user's interactive terminal:

```bash
modaic login --token-stdin
```

3. Have the user paste the key and press Enter. Stdin is not masked; do not
   request or echo the key through chat or tool output. The CLI verifies the
   key, then saves it locally. For an existing secret-manager workflow, pipe
   the secret directly to stdin instead of putting it in a shell command.
4. Run commands normally:

```bash
modaic models list
```

The stored key is plaintext in ~/.config/modaic/credentials.json by default on
macOS/Linux, or %APPDATA%/modaic/credentials.json on Windows. XDG_CONFIG_HOME
or MODAIC_CONFIG_DIR can change the location. POSIX permissions are 0600 for
the key file and 0700 for its directory. Browser keyring storage does not apply
to this API-key file. Saving a key does not make it refreshable.

## status

```bash
modaic auth status
modaic --json auth status
```

| Flag | What it does |
| --- | --- |
| --profile | Session to inspect; default: default. |
| -h, --help | Show command help. |

Credentials are selected in order: MODAIC_API_KEY, a key saved by login
--token-stdin, then the selected browser profile. This is selection, not fallback
on failure: a rejected key does not cause a retry using the browser session.
A stored key can therefore override a newly created browser login.

Status reports the selected credential source and stored session metadata;
it does not check remote validity. For browser sessions, expired reports the
stored access-token expiry, which normal commands may refresh. A bound browser
profile also reports its workspace.

## Profiles

```bash
modaic login --profile work --workspace acme
MODAIC_PROFILE=work modaic models list
modaic auth status --profile work
```

MODAIC_PROFILE selects browser sessions for normal commands. Login, logout,
and auth status default to default, so pass --profile explicitly for another
profile. Environment and stored API keys still take precedence over profiles.

## logout

```bash
modaic logout --profile work
```

| Flag | What it does |
| --- | --- |
| --profile | Session to remove; default: default. |
| --local-only | Remove local session data without attempting revocation. |
| --api-key | Remove the stored API-key file instead of a browser session. |
| -h, --help | Show command help. |

Browser logout attempts remote revocation and removes local session data.
If revocation cannot be confirmed, local data is still removed and the command
exits 3. --local-only leaves the remote session active. Removing a stored API
key only deletes the local copy; it does not revoke the key in Modaic.
Environment API keys remain set.

To switch from stored keys to browser login, remove the key sources first in
the relevant shell:

```bash
unset MODAIC_API_KEY
modaic logout --api-key
modaic login
```

## Compatibility aliases

modaic auth login is an alias for browser login and accepts --profile,
--workspace, --headless/--no-browser, --storage, --timeout, and --help; it does not accept
--token-stdin. modaic auth logout accepts --profile, --local-only, and --help;
it does not accept --api-key. Prefer the top-level login/logout commands.

## Troubleshooting

- Wrong account: inspect credential source before changing browser profiles.
- No stored login: run login for the intended profile.
- Keyring unavailable: configure an OS keyring or explicitly choose file storage
  on POSIX; do not silently switch storage.
- Login denied/expired/timed out: start a fresh request and match its new code.
- Remote revocation failed: report local removal and the unconfirmed revocation;
  do not claim the session was revoked.
