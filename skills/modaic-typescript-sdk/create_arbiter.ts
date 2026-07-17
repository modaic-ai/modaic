/**
 * Runnable example: define an Arbiter (an LLM judge) and push it to Modaic Hub.
 *
 * `Arbiter.create` writes the judge's config.json (signature schema) and
 * program.json (stored prompt) and pushes them to your Modaic profile via git.
 * The Modaic server runs the LLM — this client never does.
 *
 * Run it:
 *   npm add modaic zod
 *   export MODAIC_TOKEN="your-access-token"   # from https://modaic.dev
 *   npx tsx create_arbiter.ts                 # or: bun run create_arbiter.ts
 *
 * Optionally override where it pushes:
 *   MODAIC_REPO="your-username/support-triage" npx tsx create_arbiter.ts
 */

import { Arbiter, Signature, Enum } from "modaic";
import { z } from "zod";

const repo = process.env.MODAIC_REPO ?? "your-org/support-triage";

async function main() {
  if (!process.env.MODAIC_TOKEN) {
    throw new Error("MODAIC_TOKEN is not set. Get one at https://modaic.dev.");
  }

  // Define what the judge sees (input) and decides (output). Outputs use Enum so
  // the judge has a finite space to calibrate confidence against. Do NOT add a
  // `reasoning` output — create() injects one automatically.
  const signature = new Signature({
    instructions:
      "Classify the support ticket into the right queue. Use `billing` for " +
      "payment, refund, or invoice issues; `technical` for product bugs and " +
      "integration errors; `account` for login, profile, or permission questions.",
    input: z.object({
      ticket: z.string().describe("The user-submitted support ticket"),
    }),
    output: z.object({
      queue: Enum("billing", "technical", "account").describe(
        "Which queue should own this ticket",
      ),
    }),
  });

  // Create the repo on Modaic Hub and push the judge. Private by default.
  const arbiter = await Arbiter.create({
    repo,
    signature,
    // LiteLLM model string the server runs the judge with ("<provider>/<model>").
    // The provider's API key must be set on Modaic Hub (here: TOGETHER_API_KEY).
    model: "together_ai/openai/gpt-oss-120b",
    commit_message: "initial judge",
    private: true,
  });
  console.log(`Pushed ${arbiter.repo} (branch: ${arbiter.branch}, rev: ${arbiter.rev})`);

  // Run it — the Modaic server runs the LLM, not this client.
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
