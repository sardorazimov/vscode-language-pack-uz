#!/usr/bin/env node
import Anthropic from "@anthropic-ai/sdk";
import pLimit from "p-limit";
import fs from "node:fs/promises";
import path from "node:path";

const REPO_ROOT = path.resolve(import.meta.dirname, "..");
const BATCH_DIR = path.join(REPO_ROOT, "_translation-batches");
const CACHE_DIR = path.join(REPO_ROOT, "_translation-cache");
const TRANSLATIONS_DIR = path.join(REPO_ROOT, "translations");

const MODEL = "claude-sonnet-4-6";
const MAX_TOKENS = 16000;
const CONCURRENCY = 8;
const MAX_RETRIES = 3;

const argv = process.argv.slice(2);
function readArg(name) {
  const i = argv.indexOf(name);
  if (i < 0) return null;
  return argv[i + 1];
}
const LIMIT = readArg("--limit") ? parseInt(readArg("--limit"), 10) : Infinity;
const APPLY_ONLY = argv.includes("--apply-only");
const SKIP_APPLY = argv.includes("--skip-apply");

const SYSTEM_PROMPT = `You translate VS Code UI strings from English into modern Uzbek (Oʻzbek tili, Latin script).

INPUT
Each user message is a JSONL chunk. Each line is a JSON object:
  { "path": [module, key], "en_ploc": "<pseudo-English with Latin diacritics>", "tr": "<Turkish translation>" }
- "en_ploc" is Microsoft's pseudo-localized English. Decode by reversing this letter map (case preserved): Æ→A, æ→a, Ç→C, ç→c, Ð→D, ð→d, Ë→E, ë→e, Ï→I, ï→i, Ñ→N, ñ→n, Ø→O, ø→o, Ü→U, µ→u, Ý→Y, ÿ→y. Read the result as plain English.
- "tr" is the official Turkish translation, useful as a semantic cross-reference (do NOT translate FROM Turkish — translate from the decoded English).

OUTPUT
A STRICT JSON array, one object per input row, in the same order:
  [{"path": [...], "uz": "<Uzbek translation>"}, ...]
No prose, no markdown, no code fences, no explanations. Just the JSON array.

UZBEK STYLE RULES
- Use the Latin alphabet. ALWAYS use the modifier letter ʻ (U+02BB) — never the ASCII apostrophe ' — for the oʻ / gʻ digraphs (Oʻzbek, ishga tushirish, oʻchirish, oʻzgartirish, etc.). Examples: oʻzgartirish, oʻchirish, gʻalati, soʻrov.
- Preserve case patterns. "Save File" → "Faylni Saqlash" (Title Case). "save file" → "faylni saqlash". "SAVE" → "SAQLASH". Sentence case stays sentence case.
- For technical terms with no clean Uzbek equivalent, keep the English term: Snippet, Repository, Stash, Webview, Notebook, IntelliSense, JSON, YAML, URL, API, HTTP, Git, npm, Node.js, Webpack, Docker, Kubernetes, SSH, SQL, Markdown.
- Keep ASCII technical identifiers (class names, file names, function names) exactly as-is.

PLACEHOLDER PRESERVATION (CRITICAL)
Copy these literally, in the same positions, with no translation or whitespace change:
- {0}, {1}, {2}, ... (positional)
- {name}, {var} (named braces)
- %s, %d, %i, %f (printf)
- \${name} (template literal)
- && (mnemonic / accelerator marker — keep glued to its letter, e.g. &&File → &&Fayl, "&&Run" → "&&Ishga tushirish")
- \`code\`, **bold**, [link](url), \\n, \\t (Markdown / escapes — copy verbatim)
- HTML/XML tags and entities: <b>, </b>, &amp; etc.
- Trailing colons, ellipses (…), and punctuation.

TERMINOLOGY (use consistently)
- File → Fayl, Folder → Papka, Save → Saqlash, Open → Ochish, Close → Yopish, New → Yangi
- Edit → Tahrirlash, View → Koʻrinish, Search → Qidirish, Replace → Almashtirish, Find → Topish
- Settings → Sozlamalar, Preferences → Afzalliklar, Extensions → Kengaytmalar
- Workspace → Ish joyi, Editor → Muharrir, Window → Oyna, Panel → Panel
- Run → Ishga tushirish, Stop → Toʻxtatish, Pause → Pauza, Restart → Qayta ishga tushirish
- Debug → Debug (keep English in menu labels for compactness; use "Nosozliklarni tuzatish" only in long descriptions)
- Terminal → Terminal, Output → Chiqish, Console → Konsol, Problems → Muammolar
- Source Control → Manba nazorati, Commit → Tasdiqlash, Branch → Shoxa, Merge → Birlashtirish, Pull → Tortish, Push → Yuborish
- Show → Koʻrsatish, Hide → Yashirish, Toggle → Almashtirish, Enable → Yoqish, Disable → Oʻchirish
- Yes → Ha, No → Yoʻq, OK → OK, Cancel → Bekor qilish, Apply → Qoʻllash, Reset → Tiklash
- Error → Xato, Warning → Ogohlantirish, Info → Maʼlumot, Success → Muvaffaqiyat
- Loading → Yuklanmoqda, Saving → Saqlanmoqda, Done → Bajarildi
- Selection → Tanlov, Multi-Cursor → Koʻp kursor, Format → Formatlash
- Comment → Izoh, Uncomment → Izohni olib tashlash, Line → Satr, Column → Ustun
- Quick Open → Tezkor ochish, Command Palette → Buyruqlar paneli
- Keyboard Shortcut → Klaviatura yorligʻi, Snippet → Snippet (keep)
- Notification → Bildirishnoma, Tooltip → Eslatma

Be concise. Translate the meaning, not word-for-word. Aim for natural Uzbek that a native developer would write in a UI.`;

const client = new Anthropic();
const concurrency = pLimit(CONCURRENCY);

let totalInputTokens = 0;
let totalOutputTokens = 0;
let totalCacheReadTokens = 0;
let totalCacheCreationTokens = 0;
let chunksSucceeded = 0;
let chunksFailed = 0;
const failedChunks = [];

async function listChunks() {
  const all = await fs.readdir(BATCH_DIR);
  return all
    .filter((f) => f.endsWith(".jsonl") && f !== "MVP-priority.jsonl")
    .sort();
}

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true });
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function parseJsonArray(text) {
  // Defensive: strip surrounding code fences or prose, find the JSON array
  let t = text.trim();
  if (t.startsWith("```")) {
    t = t.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  }
  const firstBracket = t.indexOf("[");
  const lastBracket = t.lastIndexOf("]");
  if (firstBracket >= 0 && lastBracket > firstBracket) {
    t = t.slice(firstBracket, lastBracket + 1);
  }
  return JSON.parse(t);
}

async function callClaude(userMessage, attempt = 1) {
  try {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      system: [
        {
          type: "text",
          text: SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" },
        },
      ],
      messages: [{ role: "user", content: userMessage }],
    });
    totalInputTokens += response.usage.input_tokens ?? 0;
    totalOutputTokens += response.usage.output_tokens ?? 0;
    totalCacheReadTokens += response.usage.cache_read_input_tokens ?? 0;
    totalCacheCreationTokens += response.usage.cache_creation_input_tokens ?? 0;
    const textBlock = response.content.find((b) => b.type === "text");
    if (!textBlock) throw new Error("no text block in response");
    return { text: textBlock.text, usage: response.usage };
  } catch (err) {
    if (err instanceof Anthropic.RateLimitError && attempt <= MAX_RETRIES) {
      const wait = Math.pow(2, attempt - 1) * 1000;
      console.error(`  [rate-limit] backoff ${wait}ms (attempt ${attempt}/${MAX_RETRIES})`);
      await sleep(wait);
      return callClaude(userMessage, attempt + 1);
    }
    if (err instanceof Anthropic.APIError && err.status >= 500 && attempt <= MAX_RETRIES) {
      const wait = Math.pow(2, attempt - 1) * 1000;
      console.error(`  [api ${err.status}] backoff ${wait}ms (attempt ${attempt}/${MAX_RETRIES})`);
      await sleep(wait);
      return callClaude(userMessage, attempt + 1);
    }
    throw err;
  }
}

async function translateChunk(chunkFile, idx, total) {
  const chunkPath = path.join(BATCH_DIR, chunkFile);
  const cachePath = path.join(CACHE_DIR, chunkFile + ".json");

  try {
    const cached = await fs.readFile(cachePath, "utf8");
    JSON.parse(cached); // validate
    console.log(`[${idx}/${total}] ${chunkFile} → cached, skipped`);
    chunksSucceeded++;
    return;
  } catch {
    // not cached, proceed
  }

  const t0 = Date.now();
  const jsonl = await fs.readFile(chunkPath, "utf8");
  const rowCount = jsonl.trim().split("\n").length;
  const userMessage =
    `Translate each row's en_ploc to Uzbek. Output a JSON array, one object per row, in the same order.\n\n` +
    jsonl;

  let parsed;
  try {
    const { text } = await callClaude(userMessage);
    try {
      parsed = parseJsonArray(text);
    } catch (parseErr) {
      // retry once with explicit reminder
      console.error(`  [${chunkFile}] JSON parse fail, retrying with reminder`);
      const reminder =
        userMessage +
        `\n\nIMPORTANT: Output ONLY a valid JSON array. No prose, no markdown, no code fences.`;
      const { text: text2 } = await callClaude(reminder);
      parsed = parseJsonArray(text2);
    }
  } catch (err) {
    chunksFailed++;
    failedChunks.push(chunkFile);
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[${idx}/${total}] ${chunkFile} → FAILED: ${msg}`);
    return;
  }

  if (!Array.isArray(parsed)) {
    chunksFailed++;
    failedChunks.push(chunkFile);
    console.error(`[${idx}/${total}] ${chunkFile} → FAILED: response is not an array`);
    return;
  }

  await fs.writeFile(cachePath, JSON.stringify(parsed, null, 2), "utf8");
  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[${idx}/${total}] ${chunkFile} → ${parsed.length}/${rowCount} keys in ${dt}s`);
  chunksSucceeded++;
}

// Source chunk basename → translation file path. e.g.
//   uz-translate-input-main.chunk.000.jsonl   → translations/main.i18n.json
//   uz-translate-input-vscode.git.chunk.001.jsonl → translations/extensions/vscode.git.i18n.json
function chunkToTargetFile(chunkFile) {
  const m = chunkFile.match(/^uz-translate-input-(.+?)\.chunk\.\d+\.jsonl$/);
  if (!m) return null;
  const src = m[1];
  if (src === "main") return path.join(TRANSLATIONS_DIR, "main.i18n.json");
  return path.join(TRANSLATIONS_DIR, "extensions", `${src}.i18n.json`);
}

function setPathOnContents(obj, pathArray, value) {
  let cur = obj.contents;
  if (!cur) return false;
  for (let i = 0; i < pathArray.length - 1; i++) {
    const seg = pathArray[i];
    if (cur[seg] === undefined || cur[seg] === null) return false;
    cur = cur[seg];
  }
  const last = pathArray[pathArray.length - 1];
  if (!(last in cur)) return false;
  cur[last] = value;
  return true;
}

async function applyTranslations() {
  // Group cache entries by target file
  const cacheFiles = (await fs.readdir(CACHE_DIR)).filter((f) => f.endsWith(".jsonl.json"));
  const byTarget = new Map();
  for (const cf of cacheFiles) {
    const chunkFile = cf.replace(/\.json$/, "");
    const target = chunkToTargetFile(chunkFile);
    if (!target) {
      console.error(`  cannot map cache file ${cf} to a target — skipping`);
      continue;
    }
    if (!byTarget.has(target)) byTarget.set(target, []);
    byTarget.get(target).push(cf);
  }

  let totalApplied = 0;
  let totalSkippedEmpty = 0;
  let totalMissingPath = 0;
  for (const [target, cfs] of byTarget) {
    const raw = await fs.readFile(target, "utf8");
    const obj = JSON.parse(raw);
    let applied = 0;
    let skippedEmpty = 0;
    let missingPath = 0;
    for (const cf of cfs) {
      const data = JSON.parse(await fs.readFile(path.join(CACHE_DIR, cf), "utf8"));
      for (const row of data) {
        if (!row || !Array.isArray(row.path) || typeof row.uz !== "string") {
          skippedEmpty++;
          continue;
        }
        if (row.uz.trim() === "") {
          skippedEmpty++;
          continue;
        }
        const ok = setPathOnContents(obj, row.path, row.uz);
        if (ok) applied++;
        else missingPath++;
      }
    }
    await fs.writeFile(target, JSON.stringify(obj, null, 2), "utf8");
    const rel = path.relative(REPO_ROOT, target);
    console.log(`  merged → ${rel}: ${applied} applied, ${skippedEmpty} empty, ${missingPath} missing-path`);
    totalApplied += applied;
    totalSkippedEmpty += skippedEmpty;
    totalMissingPath += missingPath;
  }
  console.log(`\nMerge total: ${totalApplied} applied, ${totalSkippedEmpty} empty/skipped, ${totalMissingPath} missing-path`);
  return { totalApplied, totalSkippedEmpty, totalMissingPath };
}

async function main() {
  await ensureDir(CACHE_DIR);

  if (APPLY_ONLY) {
    console.log("Apply-only mode: merging cached translations into translation files\n");
    await applyTranslations();
    return;
  }

  let chunks = await listChunks();
  const total = chunks.length;
  if (Number.isFinite(LIMIT)) chunks = chunks.slice(0, LIMIT);
  console.log(
    `Processing ${chunks.length}/${total} chunks; model=${MODEL}; concurrency=${CONCURRENCY}; max_tokens=${MAX_TOKENS}\n`,
  );

  const t0 = Date.now();
  await Promise.all(
    chunks.map((chunk, i) =>
      concurrency(() => translateChunk(chunk, i + 1, chunks.length)),
    ),
  );
  const wallSec = ((Date.now() - t0) / 1000).toFixed(1);

  console.log(`\n=== Translation pass complete ===`);
  console.log(`Chunks succeeded: ${chunksSucceeded}`);
  console.log(`Chunks failed:    ${chunksFailed}`);
  if (failedChunks.length) {
    console.log(`Failed chunk names:`);
    for (const fc of failedChunks) console.log(`  - ${fc}`);
  }
  console.log(`Wall time: ${wallSec}s`);
  console.log(`API tokens (this run):`);
  console.log(`  input (uncached):    ${totalInputTokens}`);
  console.log(`  cache write:         ${totalCacheCreationTokens}`);
  console.log(`  cache read:          ${totalCacheReadTokens}`);
  console.log(`  output:              ${totalOutputTokens}`);

  if (!SKIP_APPLY && chunksSucceeded > 0) {
    console.log(`\nMerging cached translations into translations/ ...`);
    await applyTranslations();
  }
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
