Set up a CBORG-backed Codex CLI profile on this Perlmutter account. The normal OpenAI Codex configuration must continue to work unchanged when I run:

codex

I want an optional CBORG session when I run:

codex-cborg-think

The CBORG endpoint is:

https://api.cborg.lbl.gov/v1

The credential is already available, or will later be made available, through:

CBORG_API_KEY

Do not print, expose, copy, or store the secret value. Do not log the value. Do not place the value directly inside any TOML file or shell startup file.

The main CBORG model I want to use is:

lbl/cborg-deepthought

Also create a convenient alias for:

lbl/cborg-coder

The setup must address all of the following:

1. Safely modify the main Codex config.

2. Create a separate CBORG profile file.

3. Create shell aliases suitable for Perlmutter’s Bash environment.

4. Create a CBORG-specific model metadata catalog so Codex does not emit:

   Model metadata for `lbl/cborg-deepthought` not found. Defaulting to fallback metadata; this can degrade performance and cause issues.

5. Validate the complete setup without altering my default OpenAI Codex behavior.

Important constraints:

* Make timestamped backups before modifying any existing file.
* Make the smallest possible changes.
* Do not remove existing settings from ~/.codex/config.toml.
* Do not overwrite unrelated shell aliases or environment settings.
* Do not place CBORG-specific model_catalog_json in the main config; scope it to the CBORG profile only.
* Do not guess the custom catalog JSON schema. Inspect the catalog schema used by the installed Codex binary.
* Do not invent model capabilities. Use values reported by the live CBORG endpoint where available. If information is absent, use conservative fallback values only where Codex requires them, and document the uncertainty.
* If a required Codex command differs in the installed version, inspect `codex --help` and adapt.
* If Codex is not installed or is not on PATH, stop and report that clearly rather than attempting an unrelated installation.
* Do not modify my default OpenAI login or run `codex logout`.
* Do not make any unrelated changes.

Proceed as follows.

## Step 1: Inspect the local environment

Run:

set -euo pipefail
echo "SHELL=$SHELL"
echo "HOME=$HOME"
echo "CODEX_HOME=${CODEX_HOME:-$HOME/.codex}"
command -v codex
codex --version
codex --help | sed -n '1,220p'

Define:

CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"

Confirm whether Bash reads ~/.bashrc for interactive shells on this account. Use ~/.bashrc for aliases unless inspection shows another already-established alias file is more appropriate.

Inspect existing Codex files without exposing credentials:

mkdir -p "$CODEX_DIR"
ls -la "$CODEX_DIR"
test -f "$CODEX_DIR/config.toml" && sed -n '1,260p' "$CODEX_DIR/config.toml" || true

Search for any existing CBORG or legacy profile entries:

grep -RniE 'cborg|model_providers|profiles.|profile[[:space:]]*=' "$CODEX_DIR" 2>/dev/null || true

Do not modify anything yet.

## Step 2: Back up existing files

Create a timestamp:

TS="$(date +%Y%m%d-%H%M%S)"

Before modifying an existing file, make a backup such as:

cp -p "$CODEX_DIR/config.toml" "$CODEX_DIR/config.toml.backup-$TS"

Do the same for:

* ~/.bashrc
* any existing CBORG profile file
* any existing CBORG model-catalog file

Create directories as needed:

mkdir -p "$CODEX_DIR/model-catalogs"

## Step 3: Update the main Codex config safely

The provider definition belongs in the user-level main config:

"$CODEX_DIR/config.toml"

Append the following block only if `[model_providers.cborg]` is not already present:

[model_providers.cborg]
name = "CBorg API"
base_url = "https://api.cborg.lbl.gov/v1"
env_key = "CBORG_API_KEY"
supports_websockets = false
wire_api = "responses"

If `[model_providers.cborg]` already exists:

* inspect it;
* preserve it if equivalent;
* make the smallest correction if needed;
* do not create a duplicate block.

Do not add:

* `profile = "..."`
* `[profiles.*]`
* `model_provider = "cborg"` at the top level of the main config
* `model_catalog_json` at the top level of the main config

The default `codex` invocation must continue to use the existing normal OpenAI configuration.

## Step 4: Query the live CBORG endpoint

First verify that the credential variable exists without printing it:

if [[ -z "${CBORG_API_KEY:-}" ]]; then
echo "CBORG_API_KEY is not currently set. Configuration files can still be created, but live endpoint validation must be deferred."
else
echo "CBORG_API_KEY is set."
fi

If it is set, query the live endpoint safely:

curl -fsS \
-H "Authorization: Bearer ${CBORG_API_KEY}" \
"https://api.cborg.lbl.gov/v1/models" \
> /tmp/cborg-models-live.json

Validate and inspect the response without printing credentials:

python -m json.tool /tmp/cborg-models-live.json > /dev/null
python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("/tmp/cborg-models-live.json").read_text())
items = data.get("data", data if isinstance(data, list) else [])
ids = sorted(
item.get("id")
for item in items
if isinstance(item, dict) and item.get("id")
)
for model_id in ids:
if model_id.startswith("lbl/"):
print(model_id)
PY

Confirm whether these are available:

lbl/cborg-deepthought
lbl/cborg-coder
lbl/gemma-4-thinking
lbl/gpt-oss-120b-high
lbl/cborg-mini

If any are absent, report that clearly. At minimum, continue only if `lbl/cborg-deepthought` exists.

## Step 5: Inspect the installed Codex metadata schema

Do not guess the schema.

Run:

codex debug models --bundled > /tmp/codex-bundled-models.json
python -m json.tool /tmp/codex-bundled-models.json > /dev/null

Inspect representative complete entries:

python - <<'PY'
import json
from pathlib import Path
from pprint import pprint

data = json.loads(Path("/tmp/codex-bundled-models.json").read_text())
models = data.get("models", data.get("data", data if isinstance(data, list) else []))
print("top-level type:", type(data).**name**)
if isinstance(data, dict):
print("top-level keys:", sorted(data.keys()))
print("number of models:", len(models) if isinstance(models, list) else "unknown")
if isinstance(models, list):
for item in models[:3]:
pprint(item)
print("---")
PY

Identify the exact JSON shape expected by this installed Codex version, including:

* top-level structure;
* required fields;
* model identifier field;
* context-window field;
* output-token field;
* reasoning-related fields;
* tool-use fields;
* any visibility or capability fields.

If the schema remains unclear, stop and explain what is unclear instead of writing an invalid catalog.

## Step 6: Create a conservative CBORG metadata catalog

Create:

"$CODEX_DIR/model-catalogs/cborg-models.json"

Include entries for available CBORG models that are relevant to Codex:

* lbl/cborg-deepthought
* lbl/cborg-coder
* lbl/gemma-4-thinking
* lbl/gpt-oss-120b-high
* lbl/cborg-mini

Use the exact schema discovered from:

codex debug models --bundled

Use live CBORG endpoint metadata when the endpoint provides limits or capabilities.

If the live endpoint does not expose a required value:

* use a conservative value sufficient for correct Codex behavior;
* prefer underestimating rather than overestimating context or output limits;
* document each assumption in a short Markdown report:

  "$CODEX_DIR/model-catalogs/cborg-models.NOTES.md"

Do not claim unsupported capabilities merely because a similarly named OpenAI model has them.

Validate the completed file:

python -m json.tool "$CODEX_DIR/model-catalogs/cborg-models.json" > /dev/null

## Step 7: Create a separate CBORG profile

Create or minimally update:

"$CODEX_DIR/cborg-think.config.toml"

It should contain top-level profile keys, not a `[profiles.*]` table:

model = "lbl/cborg-deepthought"
model_provider = "cborg"
model_reasoning_effort = "high"
personality = "pragmatic"
model_catalog_json = "<EXPANDED_ABSOLUTE_PATH_TO_CODEX_DIR>/model-catalogs/cborg-models.json"

Replace `<EXPANDED_ABSOLUTE_PATH_TO_CODEX_DIR>` with the actual absolute path. Do not use `~` inside the TOML path.

Do not nest these values under:

[profiles.cborg-think]

Do not add this profile selector to the main config:

profile = "cborg-think"

The CBORG profile must only take effect when explicitly selected.

## Step 8: Add Bash aliases

Append the following aliases to ~/.bashrc only if equivalent aliases are not already present:

alias codex-cborg-think='codex --profile cborg-think -m lbl/cborg-deepthought'
alias codex-cborg-coder='codex --profile cborg-think -m lbl/cborg-coder'

Use a clearly marked block:

# >>> CBORG Codex aliases >>>

alias codex-cborg-think='codex --profile cborg-think -m lbl/cborg-deepthought'
alias codex-cborg-coder='codex --profile cborg-think -m lbl/cborg-coder'

# <<< CBORG Codex aliases <<<

Do not store `CBORG_API_KEY` in ~/.bashrc.

Reload aliases for the current shell:

source ~/.bashrc

Verify:

type codex-cborg-think
type codex-cborg-coder

## Step 9: Validate the effective configuration

First verify that normal OpenAI Codex remains available:

codex --version

Do not run `codex logout`.

Then inspect the CBORG profile catalog:

codex --profile cborg-think debug models > /tmp/codex-cborg-visible-models.json

Validate:

python -m json.tool /tmp/codex-cborg-visible-models.json > /dev/null

Search for entries:

grep -n 'lbl/cborg-deepthought' /tmp/codex-cborg-visible-models.json
grep -n 'lbl/cborg-coder' /tmp/codex-cborg-visible-models.json

If `CBORG_API_KEY` is set, run a non-destructive test:

codex exec \
--profile cborg-think \
-m lbl/cborg-deepthought \
"Respond with exactly: CBORG_CODEX_OK"

Capture stdout and stderr:

codex exec \
--profile cborg-think \
-m lbl/cborg-deepthought \
"Respond with exactly: CBORG_CODEX_OK" \
> /tmp/cborg-codex-test.stdout \
2> /tmp/cborg-codex-test.stderr

Inspect:

cat /tmp/cborg-codex-test.stdout
cat /tmp/cborg-codex-test.stderr

Verify all of the following:

* response contains `CBORG_CODEX_OK`;
* there is no 401 error;
* there is no request to `https://api.openai.com/v1/responses`;
* there is no warning that metadata for `lbl/cborg-deepthought` is missing;
* the CBORG endpoint remains `https://api.cborg.lbl.gov/v1`;
* running plain `codex` still uses the default normal OpenAI setup.

If `CBORG_API_KEY` is not currently set, complete the file changes but clearly mark live validation as pending.

## Step 10: Provide a concise report

At the end, report:

1. Codex CLI version found.
2. Shell and CODEX_HOME used.
3. Exact files created.
4. Exact files modified.
5. Backup files created.
6. Final `[model_providers.cborg]` block.
7. Final `cborg-think.config.toml` content.
8. Alias block added to ~/.bashrc.
9. CBORG model-catalog entries created.
10. Any conservative metadata assumptions made.
11. Validation commands run.
12. Whether the missing-metadata warning disappeared.
13. Whether the CBORG endpoint was used successfully.
14. Whether plain `codex` remains unchanged.
15. Any remaining manual action, such as exporting `CBORG_API_KEY`.

Do not make unrelated edits.
