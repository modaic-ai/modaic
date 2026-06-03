# Modaic Arbiters in LangSmith

You can use a Modaic Arbiter as an **LM judge** in
[LangSmith](https://smith.langchain.com/) by routing requests through
Modaic's OpenAI-compatible chat-completions endpoint.

## 1. Create the Arbiter

See `arbiter.py` (or the SKILL.md "Defining an Arbiter" section) for the
full Signature + `modaic.Predict` + `as_arbiter()` flow. Push it to Modaic
Hub at, e.g., `your-org/correctness`.

## 2. Configure the LangSmith provider

In LangSmith:

1. Go to **Prompts** → **+ Prompt**.
2. Click the model name to configure provider settings.
3. Set the provider to **OpenAI Compatible Endpoint**.
4. Set the model to your Arbiter's repo path: `your-org/correctness`.
5. Set **Provider API** to **Chat Completions**.
6. Set **Base URL** to:

   ```
   https://api.modaic.dev/api/v1/arbiters
   ```

7. Save the provider configuration.

## 3. Set the API key

Set `OPENAI_API_KEY` to your Modaic token. LangSmith supports two
locations:

- **Browser secrets** — local to your browser.
- **Workspace secrets** — shared across the workspace.

Either works.

## 4. Template the prompt

Modaic's chat-completions endpoint does **not** run a vanilla LLM
completion. It parses the message body, extracts XML-tagged input fields
matching the Arbiter's signature, and invokes the Arbiter on those
fields. Templates must wrap each input field in an XML tag whose name
matches the signature's input field name.

For an Arbiter with input fields `title` and `content`:

```xml
<run>
<title>
{inputs.title}
</title>
<content>
{inputs.content}
</content>
</run>
```

`{inputs.title}` is LangSmith's templating — it pulls the `title` field
off a dataset row or trace input. The XML tag `<title>...</title>` is
what tells Modaic which Arbiter input it maps to.

> **The XML field names must exactly match the input field names on the
> `dspy.Signature`.** A typo silently fails: Modaic gets a missing input
> field and the run errors out.

## Notes

- Confidence scores from the Arbiter are not directly surfaced through
  the LangSmith chat-completions surface; for direct access to
  `confidence`, call the Arbiter via the Modaic SDK / API instead (see
  `run_arbiter.py`).
- For setup help, ping Modaic on
  [Discord](https://discord.com/invite/5NZ3GZNq5k) or email
  team@modaic.dev.
