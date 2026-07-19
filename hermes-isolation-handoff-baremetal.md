# Handoff Brief (Bare-Metal / Offline Edition): Per-User Isolation for OpenWebUI + Hermes Agent

**Audience:** Hermes Agent (you will help implement this on the host you run on)
**Constraints:** No Docker. Fully offline box. OpenWebUI installed via conda; Hermes Agent installed offline, gateways run bare-metal.
**Goal:** Each OpenWebUI user gets their own Hermes agent with full capability (terminal, files, memory, skills, cron), but kernel-enforced separation: a user's agent cannot see any other user's files, no matter what command it runs.

**Isolation mechanism:** systemd's built-in sandboxing (mount namespaces via `ProtectSystem`/`ProtectHome`/`BindPaths`) plus one dedicated OS user per webui user. Nothing needs to be downloaded — systemd is already PID 1 on the box. bubblewrap is listed as an alternative in Appendix A if you prefer it or systemd is too old.

---

## Part 0 — VERIFY THESE ASSUMPTIONS FIRST

Hermes: confirm each item against the actual system before changing anything. If any assumption is wrong, STOP and report what you found instead.

| # | Assumption | How to verify |
|---|-----------|---------------|
| A1 | OpenWebUI is connected to Hermes Agent API server(s) (OpenAI-compatible), per-profile on ports 8650+ or single on 8642. | OpenWebUI Admin → Connections; `curl -H "Authorization: Bearer <key>" http://127.0.0.1:8642/v1/models` (and 8650+). |
| A2 | All gateways currently run as one OS user; tool calls execute as that user with host-wide filesystem visibility. | `ps aux \| grep -i hermes` → user column; from a chat run `id`, `pwd`. |
| A3 | Profile separation is config/memory/skills only, not filesystem. | From profile A's chat: `ls ~/.hermes/profiles/` — if other profiles are visible, confirmed. |
| A4 | systemd is the init system and version ≥ 240 (Ubuntu 20.04+, Debian 11+ all qualify). | `systemctl --version` — record the number. ≥ 247 preferred (`TemporaryFileSystem` + full `BindPaths` behavior is smoother). |
| A5 | The model server (llama.cpp / vLLM / Ollama serving Hermes weights) listens on a known local port. | Find the endpoint in Hermes config; `ss -tlnp` to confirm the port. |
| A6 | OpenWebUI runs bare-metal (conda) on the same host, so `127.0.0.1:<port>` connections work with no Docker networking translation. | `ss -tlnp \| grep 8080` (or its port); confirm how it was launched. |
| A7 | User data can be consolidated under `/srv/users/<name>/`. | Check where user folders live now; plan migration. |
| A8 | Reproduce the escape once, from a non-admin user's chat: `cat /etc/passwd`, `ls ~`, `ls ~/.hermes/profiles`. Record results — this is the "before" state. | (as stated) |
| A9 | `sudo`/root access is available to create users and install systemd units. | `sudo -v`. |

Also report: hermes-agent version, how gateways are currently started (manual shell? script? tmux?), ports in use, RAM/CPU of the box, and `bwrap --version` if present (optional).

---

## Part 1 — What the problem actually is

OpenWebUI is only a chat frontend here. Every tool call executes inside the Hermes gateway process, as the OS user who launched it, on the host filesystem. Three things look like boundaries and are not:

1. **OpenWebUI accounts** — control which connection/model a user may chat with; invisible to the filesystem.
2. **Hermes profiles** — genuinely separate config, memory, skills, keys; but all run as the same OS user, so profile `alice` can `cat ~/.hermes/profiles/bob/.env` and steal Bob's `API_SERVER_KEY`.
3. **Workspace setting** — a default `cwd`, not a jail. `cd /` works.

Path-validation or command-filtering inside Hermes/OpenWebUI is not a fix: `../`, symlinks, `$(...)`, and plain persuasion of the model all defeat string-level checks. The boundary must live in the kernel, below anything the agent can influence.

Secondary hole: the per-profile API ports are bound on the host, so any local process that learns a key can curl any user's agent, bypassing OpenWebUI login. The `API_SERVER_KEY` is the real auth boundary and currently sits on a shared filesystem.

---

## Part 2 — The fix: one sandboxed gateway per user (systemd, no Docker)

Concept: keep the exact architecture you have (OpenWebUI → per-profile API servers on 8650+). Change only *how each gateway process is launched*: as its own OS user, inside a systemd sandbox whose mount namespace exposes only that user's data. Everything else on the filesystem is either invisible or read-only.

```
OpenWebUI (conda, bare-metal, :8080)
   │  http://127.0.0.1:<port>/v1 per connection
   ├── hermes-agent@alice.service  :8650  user=agent-alice  sees /srv/users/alice only
   ├── hermes-agent@bob.service    :8651  user=agent-bob    sees /srv/users/bob   only
   └── admin gateway               :8642  (unsandboxed or lightly sandboxed, admin only)
                     │
             http://127.0.0.1:<model-port>  shared model server (GPU shared, one copy of weights)
```

### Step 1 — Per-user OS accounts and data layout

```bash
for u in alice bob; do
  sudo useradd --system --create-home --home-dir /srv/users/$u --shell /usr/sbin/nologin agent-$u
  sudo mkdir -p /srv/users/$u/{workspace,hermes-home}
  sudo chown -R agent-$u:agent-$u /srv/users/$u
  sudo chmod 750 /srv/users/$u          # other users' agents can't even traverse
done
```

Migrate existing state so memory/skills survive: copy `~/.hermes/profiles/alice/*` into `/srv/users/alice/hermes-home/` and chown it. (Hermes: verify the exact profile-dir → `HERMES_HOME` layout mapping for the installed version before copying — the sandboxed agent will run with `HERMES_HOME=/srv/users/<u>/hermes-home` and its **default** profile; profiles are no longer used for separation.)

Hermes-agent code itself stays where it is (e.g. `/opt/hermes-agent` or the current install path), readable by all agent users. Its venv likewise. Nothing is duplicated per user.

### Step 2 — Per-user environment file (holds the API key)

```bash
# /etc/hermes/alice.env   (root-owned, mode 600)
API_SERVER_ENABLED=true
API_SERVER_HOST=127.0.0.1
API_SERVER_PORT=8650
API_SERVER_KEY=<new-random-key-alice>
HERMES_HOME=/srv/users/alice/hermes-home
# model endpoint — Hermes: set the real config key your version uses:
# e.g. OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Repeat per user with unique port + key. Keys live in `/etc/hermes/*.env` (root-only) and are injected by systemd — they are NOT readable from inside any sandbox, including the user's own shell tool.

### Step 3 — The templated systemd unit

```ini
# /etc/systemd/system/hermes-agent@.service
[Unit]
Description=Hermes Agent gateway for user %i
After=network.target

[Service]
User=agent-%i
Group=agent-%i
EnvironmentFile=/etc/hermes/%i.env
# Hermes: replace with the real venv python + gateway entrypoint / foreground flag:
ExecStart=/opt/hermes-agent/venv/bin/hermes gateway --foreground
WorkingDirectory=/srv/users/%i/workspace
Restart=on-failure

# ----- Sandbox: this is the actual fix -----
ProtectSystem=strict            # whole OS read-only
ProtectHome=tmpfs               # /home and /root replaced by empty tmpfs
TemporaryFileSystem=/srv/users  # hide ALL user dirs...
BindPaths=/srv/users/%i         # ...then bind back only this user's dir, read-write
PrivateTmp=yes
NoNewPrivileges=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
LockPersonality=yes
CapabilityBoundingSet=
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
SystemCallFilter=@system-service
# Egress lockdown: agent may talk ONLY to localhost (model server, own API bind).
# Box is offline anyway, but this makes it structural:
IPAddressDeny=any
IPAddressAllow=localhost

# Resource caps per user
MemoryMax=2G
CPUQuota=200%
TasksMax=256

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-agent@alice hermes-agent@bob
```

What the agent now sees from its own terminal tool: `/` is the real OS but read-only; `/srv/users` contains **only** its own folder; `/home` is empty; `/etc/hermes` keys unreadable; writes succeed only under `/srv/users/<self>` and `/tmp` (private). `cd /` is allowed and harmless.

You also get for free: auto-restart on crash, boot-time start, `journalctl -u hermes-agent@alice` per-user logs — replacing whatever tmux/script launch is in place now.

### Step 4 — Rewire OpenWebUI connections

No Docker networking involved — plain localhost:

| Connection | URL | API Key |
|---|---|---|
| Alice | `http://127.0.0.1:8650/v1` | contents of alice.env key |
| Bob | `http://127.0.0.1:8651/v1` | contents of bob.env key |

Admin → Users: assign each model to only its own user; remove default-visible models for the "user" role. (OpenWebUI stores connections in its DB — edit in the Admin UI, not just env vars.)

Rotate every key as part of this step: the old ones lived on a shared filesystem and must be treated as leaked.

### Step 5 — Close the port-sharing hole

All API servers bind 127.0.0.1, so nothing is exposed off-box. On-box, any local process could still curl another user's port — but now the keys are root-only and never inside any sandbox, so a stolen-key attack requires root, at which point all bets are off anyway. If you want belt-and-suspenders, add an nftables rule restricting each 865x port to the OpenWebUI process's UID (owner match) — optional, note in report if you do it.

### Step 6 — Admin access

Run the admin's gateway as your normal login user with no sandbox (current behavior), bound to a distinct port (8642), connected in OpenWebUI as a model visible only to the admin account. Optionally give it `BindPaths=/srv/users` read-only via its own unit for oversight of user folders.

### Step 7 — Acceptance tests (repeat A8, expect opposite results)

From each non-admin user's chat, via the agent's terminal:

1. `ls /srv/users` → only their own name.
2. `cat /etc/hermes/alice.env` (their OWN env file) → permission denied / not found.
3. Any path to another user's data → does not exist in this namespace.
4. `touch /opt/test` or `touch /etc/x` → read-only filesystem.
5. `ls /home` → empty.
6. Capability preserved: memory persists across sessions; skills load and newly-saved skills appear in `/srv/users/<u>/hermes-home`; cron jobs fire with the browser closed (`systemctl status hermes-agent@<u>` stays active).
7. Bomb test: memory-hungry command in one chat → that unit hits MemoryMax (check `journalctl`), other users unaffected.
8. `systemd-analyze security hermes-agent@alice.service` → record the exposure score (aim well under 5).

---

## Part 3 — Rollout order

1. Verify Part 0; report discrepancies (especially A4 systemd version and the exact gateway ExecStart command).
2. Create `agent-alice` + sandbox unit only; leave the existing setup running in parallel on its old port.
3. Point a NEW OpenWebUI connection ("alice-sandboxed") at 8650; run acceptance tests 1–8.
4. Migrate remaining users one at a time (copy profile state → chown → enable unit → repoint connection → stop old gateway).
5. Kill all unsandboxed per-profile gateways; keep only the admin path.
6. Rotate all keys; store only in `/etc/hermes/*.env` (600, root).
7. (Optional, only after 1–6 pass) Shared skills pool per Part 4 — verify A10 first, then add the read-only mount and promotion workflow.

---

## Part 4 — Shared skills pool (optional, add later)

Motivation: users' agents each write and accumulate their own skills privately. At some point we may want a curated pool of skills that ALL agents can use, without weakening per-user isolation.

**Design principle:** sharing and separation are not opposites — sandbox mounts are per-directory. The pool is a read-only mount added to every sandbox; each user's private skills stay read-write and private.

```
/srv/shared/skills/                       ← curated pool, owned root:root (or admin), 755
        │ read-only bind into every sandbox
        ├→ visible inside alice's agent (cannot modify)
        └→ visible inside bob's agent   (cannot modify)
/srv/users/<u>/hermes-home/skills/        ← private, read-write, as today
```

### 4.1 — Mount (one line in the unit template)

```ini
# add to [Service] in /etc/systemd/system/hermes-agent@.service
BindReadOnlyPaths=/srv/shared/skills
```

(`TemporaryFileSystem=/srv/users` only masks `/srv/users`, so `/srv/shared` is visible read-only already via `ProtectSystem=strict`; the explicit `BindReadOnlyPaths` line documents intent and survives future layout changes. bwrap alternative: `--ro-bind /srv/shared/skills /srv/shared/skills`.)

### 4.2 — VERIFY ITEM for Hermes (blocking for this part)

| # | Assumption | How to verify |
|---|-----------|---------------|
| A10 | The installed hermes-agent skill loader can pick up skills from a second directory (search path, config option, or by following a symlink inside the skills dir). | Inspect the skill-loading code/config for the installed version. Test: symlink `hermes-home/skills/_shared → /srv/shared/skills` in one sandbox, place a dummy skill in the pool, confirm it loads and executes; confirm the agent CANNOT save/modify skills there (write must fail read-only, and the agent should fall back to its private dir without erroring). |

If A10 fails (loader refuses symlinks/secondary dirs), fallback: a root cron/systemd timer that rsyncs `/srv/shared/skills/` into each user's private skills dir (`--delete` scoped to a `_shared/` subfolder). Less elegant, same result; note which path was taken.

### 4.3 — Promotion workflow (how skills enter the pool)

Agents must never write to the pool directly. A shared *writable* skills dir is a cross-user code-injection channel: a prompt-injected agent for user A plants a malicious skill, user B's agent later executes it inside B's sandbox with B's data. (Same failure mode as the documented marketplace/skill-supply-chain incidents.) Therefore:

1. A user's agent saves a self-written skill privately, as normal.
2. Admin reviews it (directly, or via the admin gateway which mounts `/srv/users` read-only) — read the skill's code fully before promoting; treat it as untrusted input.
3. Admin copies it to `/srv/shared/skills/`; it becomes available to everyone on next skill load, immutable to all agents.
4. Removal/rollback = admin deletes/replaces the file; keep the pool in a git repo (`git init /srv/shared/skills`) for history and diff-based review.

### 4.4 — Same pattern for other shared resources

Any additional shared, read-only material is just more `BindReadOnlyPaths` lines — no new mechanism:

```ini
BindReadOnlyPaths=/srv/shared/skills /srv/shared/datasets /srv/shared/docs
```

Only if genuinely needed, a shared **read-write** exchange dir (`/srv/shared/exchange`, `BindPaths=`) can be added — but understand this downgrades the trust model from "mutually untrusted users" to "trusted collaborators" for anything that touches that dir, and files found there must be treated as untrusted by every agent. Default: don't.

---

## Appendix A — bubblewrap alternative (if systemd sandboxing is unavailable or too old)

Same mount-namespace guarantee, launched manually or from a plain unit:

```bash
sudo -u agent-alice bwrap \
  --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --ro-bind /bin /bin --ro-bind /opt/hermes-agent /opt/hermes-agent \
  --ro-bind /etc/resolv.conf /etc/resolv.conf \
  --proc /proc --dev /dev --unshare-pid \
  --tmpfs /home \
  --bind /srv/users/alice/hermes-home /srv/users/alice/hermes-home \
  --bind /srv/users/alice/workspace /srv/users/alice/workspace \
  --chdir /srv/users/alice/workspace \
  --setenv HERMES_HOME /srv/users/alice/hermes-home \
  /opt/hermes-agent/venv/bin/hermes gateway --foreground
```

`bwrap` is in the standard Ubuntu/Debian repos (`bubblewrap` package) — if not already installed, it can be pulled from offline apt media. systemd remains the recommended path since it needs zero packages and adds supervision, logging, and resource caps in the same file.

## Appendix B — Why not Docker here

Docker/Podman would work (previous draft), but on an offline bare-metal box it adds: transferring images offline, a daemon (or rootless setup), storage-driver care, and networking translation between the conda-installed OpenWebUI and containers. systemd sandboxing achieves the identical kernel guarantee (mount namespaces) with tooling already running as PID 1, keeps everything on localhost, and is easier to audit with `systemd-analyze security`. If the deployment later moves to a connected machine or needs per-user package installs diverging from the host, revisit containers.
