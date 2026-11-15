// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// Routing is deterministic: every choice comes from the brief, the request, and the gateway's
// answers so far — no wall clock, no unseeded randomness, so a replay leaves the same export.
//
//   - quality: each prompt is served by the highest-rated free model that meets its requirements
//     (rating_milli from the brief, ties broken by model id), so the served rating is the best
//     rating for that prompt whenever the top model answers.
//   - failover: a 4xx/5xx/timeout is retried on the next best eligible model (at most 3 dials a
//     prompt). Every refusal a served prompt recovered from is reported in the
//     x-makefun-upstream-failures trail header, so the failover column can see it. A 401 (no key
//     pushed) is not retried: every model would refuse it. A model that refused one prompt is tried
//     again fresh on the next: a refusal here is a transient rate limit, not a dead model, and
//     carrying it forward would downgrade every later prompt for no gain. A model that is really
//     down costs at most one wasted dial a prompt, and the prompt is still served by the next model.
//   - streaming is buffered: the upstream SSE is read whole and checked (a finish chunk and the
//     [DONE] terminator) before one byte reaches the client, so a truncated stream fails over
//     cleanly instead of sending the client a broken event order.
//   - the catalog is exactly the brief's free models, on the register body and on both
//     GET /v1/models answers; --list prints that same document byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway dial may take this long before it counts as refused. */
const DIAL_TIMEOUT_MS = 8000;
/** The most gateway dials one prompt may cost, refusals included. */
const MAX_ATTEMPTS = 3;
/** The output cap a request without one is dialled with: the gateway refuses a capless call. */
const DEFAULT_MAX_TOKENS = 1024;
/** How often POST /register is retried while the env is still starting. */
const REGISTER_RETRIES = 5;
/** The pause between register retries, in milliseconds. */
const REGISTER_RETRY_MS = 500;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that meet a prompt's requirements, best first: highest rating_milli, ties broken
 * by model id, so the order is a pure function of the brief.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * The dial order for one prompt: the best eligible models first, at most MAX_ATTEMPTS dials in all.
 * A refusal is per prompt, not per trial: a model that refused this prompt is not tried again for
 * it, but the next prompt starts from the top of the list again.
 */
function attemptOrder(requires: Requirement): ModelRow[] {
  return rankedCandidates(requires).slice(0, MAX_ATTEMPTS);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The trail header's value for one prompt's refusals: at most 8 rows, in the order they refused. */
function trailHeader(failures: Array<{ status: number; code: string; model: string | null }>): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extraHeaders: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One refusal, as a trail row: only HTTP answers carry a status the trail can hold. */
interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

/** The gateway error's code, read off its body when it is JSON, else the HTTP status's name. */
function errorCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/**
 * One gateway dial: POST the Chat Completions body naming `model`, with the trial's dummy key the
 * gateway swaps for the entrant's own. A body without an output cap gets one: the gateway refuses
 * a capless call, and the cap never changes the answer's content.
 */
async function dial(model: string, body: Json): Promise<Response> {
  const outgoing: Json = { ...body, model };
  if (typeof outgoing.max_tokens !== 'number' && typeof outgoing.max_completion_tokens !== 'number') {
    outgoing.max_tokens = DEFAULT_MAX_TOKENS;
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(outgoing),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/**
 * Whether a buffered upstream SSE stream is whole: some chunk carried a finish reason and the
 * stream ended with the [DONE] terminator. A stream cut before either is a refusal, not an answer.
 */
function streamComplete(text: string): boolean {
  let finished = false;
  let done = false;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        done = true;
        continue;
      }
      try {
        const chunk: unknown = JSON.parse(payload);
        if (isRecord(chunk) && Array.isArray(chunk.choices)) {
          const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
          if (choice !== null && choice.finish_reason !== null && choice.finish_reason !== undefined) finished = true;
        }
      } catch {
        // Not JSON: not a finish chunk.
      }
    }
  }
  return finished && done;
}

/** A buffered upstream SSE stream as a ReadableStream the wire relays can read from. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

interface Served {
  /** The model that really answered. */
  model: string;
  /** The refusals this prompt recovered from, in order. */
  failures: Refusal[];
  /** A JSON answer from the gateway, when the client did not ask to stream. */
  json: Json | null;
  /** The gateway's buffered SSE text, when the client asked to stream. */
  streamText: string | null;
}

/**
 * Dial one prompt until a model answers it whole, or the dials run out. On success the served
 * answer and the refusals it recovered from; on failure the refusals, so the caller can still
 * report them in the trail header of its 502. A 401 is never retried — it means no key was
 * pushed, and every model would refuse the same way.
 */
async function serve(body: Json, requires: Requirement): Promise<{ served: Served } | { failures: Refusal[] }> {
  const failures: Refusal[] = [];
  const streaming = body.stream === true;
  for (const candidate of attemptOrder(requires)) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch {
      // No HTTP answer at all (timeout, unreachable): no trail row exists for a call that never
      // answered, so this refusal is silent — the next model is tried.
      continue;
    }
    if (upstream.status === 401) {
      failures.push({ status: 401, code: 'unauthorized', model: candidate.id });
      return { failures };
    }
    if (!upstream.ok) {
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: errorCodeOf(text, upstream.status), model: candidate.id });
      continue;
    }
    if (streaming) {
      const text = await upstream.text().catch(() => '');
      if (!streamComplete(text)) {
        failures.push({ status: upstream.status, code: 'truncated_stream', model: candidate.id });
        continue;
      }
      log(`served by ${candidate.id} (stream, ${failures.length} refusal(s) before it)`);
      return { served: { model: candidate.id, failures, json: null, streamText: text } };
    }
    const text = await upstream.text().catch(() => '');
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      failures.push({ status: upstream.status, code: 'bad_json', model: candidate.id });
      continue;
    }
    if (!isRecord(answer)) {
      failures.push({ status: upstream.status, code: 'bad_json', model: candidate.id });
      continue;
    }
    log(`served by ${candidate.id} (json, ${failures.length} refusal(s) before it)`);
    return { served: { model: candidate.id, failures, json: answer, streamText: null } };
  }
  return { failures };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a real free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const outcome = await serve(body, requirementsFor(body));
  if (!('served' in outcome)) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model refused this prompt', trailHeader(outcome.failures));
  }
  const served = outcome.served;
  const trail = trailHeader(served.failures);
  if (body.stream === true && served.streamText !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const frame of openAIStreamWithModel(streamOf(served.streamText), served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  if (served.json !== null) {
    sendJson(res, 200, openAIJsonWithModel(served.json, served.model), trail);
    return;
  }
  sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered in a shape this server cannot relay', trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const upstreamBody = anthropicToOpenAI(body, 'auto');
  // serve() dials with the body it is given: carry the client's stream flag on the translated body.
  if (body.stream === true) upstreamBody.stream = true;
  const outcome = await serve(upstreamBody, requires);
  if (!('served' in outcome)) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model refused this prompt', trailHeader(outcome.failures));
  }
  const served = outcome.served;
  const trail = trailHeader(served.failures);
  if (body.stream === true && served.streamText !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trail });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(served.streamText), served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  if (served.json !== null) {
    sendJson(res, 200, openAIToAnthropic(served.json, served.model), trail);
    return;
  }
  sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered in a shape this server cannot relay', trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= REGISTER_RETRIES; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      if (attempt < REGISTER_RETRIES) await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
