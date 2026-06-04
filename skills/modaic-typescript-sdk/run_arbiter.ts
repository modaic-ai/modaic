/**
 * Runnable example: run an existing Arbiter that's already on Modaic Hub.
 *
 * This pushes nothing. It constructs a handle to a repo that already exists with
 * `new Arbiter("<owner>/<name>")` and calls predict(). The Modaic server runs the
 * LLM — a pure HTTP call authenticated with MODAIC_TOKEN.
 *
 * Run it (after create_arbiter.ts, or against any judge you own):
 *   npm add modaic
 *   export MODAIC_TOKEN="your-access-token"   # from https://modaic.dev
 *   npx tsx run_arbiter.ts                     # or: bun run run_arbiter.ts
 *
 * Optionally override which judge to run:
 *   MODAIC_REPO="your-username/support-triage" npx tsx run_arbiter.ts
 */

import { Arbiter } from "modaic";

const repo = process.env.MODAIC_REPO ?? "your-org/support-triage";

async function main() {
  if (!process.env.MODAIC_TOKEN) {
    throw new Error("MODAIC_TOKEN is not set. Get one at https://modaic.dev.");
  }

  // No network call here — predict() runs the judge server-side. Pass `rev` to
  // pin a branch/tag/commit: new Arbiter(repo, { rev: "v1" }).
  const arbiter = new Arbiter(repo);

  // Keys must match the signature's input fields.
  const result = await arbiter.predict({
    ticket: "My payment failed twice in a row.",
  });

  console.log("queue:    ", result.output?.queue);
  console.log("reasoning:", result.reasoning);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
