// Per-model health: a circuit breaker and a slow-model demotion, both deterministic.
//
// Routing in this entry is a pure function of the brief, the prompt sequence, and the gateway's
// answers — no clock, no randomness — so a replay leaves the same export. This module keeps that
// promise: it never reads the clock and never rolls a die. Its only inputs are the prompt index
// (the arrival order) and the outcome of each gateway dial, both of which the trial's cassette
// records, so a replay reaches the same routing decisions as the live pass.
//
//   - circuit breaker: a model that refuses twice in a row (a 4xx/5xx, a truncated stream, a bad
//     JSON body, or a dial that never answered) is rested for the next two prompts. A rest is
//     counted in prompts, not seconds: a wall-clock cooldown would make the routing depend on
//     when the trial ran. A model that answers again is forgiven at once: a refusal here is a
//     transient rate limit, not a dead model, and the rest is what keeps a really-down model
//     from eating one dial a prompt.
//   - slow demotion: a dial that never answers (the 8 s timeout, or the gateway unreachable) is
//     the only latency signal this module trusts, because it is a binary gateway outcome, not a
//     measured duration. A model with an unrecovered no-answer is demoted below every model
//     without one in the dial order, so a slow model stops being tried first. A success clears
//     the demotion: one slow dial is a hiccup, not a verdict. Measured milliseconds are
//     deliberately never used for routing: they would differ between a live pass and its replay.
//   - a 401 is never recorded: it means no key was pushed, and every model would refuse the
//     same way, so it says nothing about any one model's health.

import type { ModelRow } from './providers.ts';

/** How one gateway dial refused, as the caller observed it. */
export type RefusalKind = 'http' | 'timeout' | 'truncated' | 'bad_json';

/** Consecutive refusals after which a model is rested. */
export const REST_AFTER_CONSECUTIVE = 2;
/** Prompts a rested model is skipped for, counted from the prompt after the one that rested it. */
export const REST_PROMPTS = 2;

/** One model's health record: every counter the tracker keeps for it. */
export interface ModelHealth {
  id: string;
  /** Refusals since the last answer, or since the last rest. */
  consecutive: number;
  /** The first prompt index at which the model may be dialled again (-1: never rested). */
  restUntilPrompt: number;
  /** No-answer dials since the last answer: above zero the model is demoted. */
  slowStrikes: number;
  /** Dials in all, refusals and answers both. */
  dials: number;
  /** Refusals in all, every kind. */
  refusals: number;
  /** Dials that never answered, timeouts included. */
  timeouts: number;
  /** Answers in all. */
  successes: number;
  /** Times the breaker rested this model. */
  rests: number;
}

/** One model's health as a snapshot reports it, safe to log or to compare. */
export interface HealthSnapshot {
  model: string;
  consecutive: number;
  restedUntilPrompt: number;
  demoted: boolean;
  dials: number;
  refusals: number;
  timeouts: number;
  successes: number;
  rests: number;
}

/** What recording one refusal did: nothing new, a demotion, or a rest. */
export type RecordOutcome = 'noted' | 'demoted' | 'rested';

/** The trial's totals: every dial, refusal, timeout, answer and rest so far. */
export interface HealthTotals {
  dials: number;
  refusals: number;
  timeouts: number;
  successes: number;
  rests: number;
  demoted: number;
  rested: number;
}

/** A fresh, empty health record for one model id. */
function emptyRecord(id: string): ModelHealth {
  return {
    id,
    consecutive: 0,
    restUntilPrompt: -1,
    slowStrikes: 0,
    dials: 0,
    refusals: 0,
    timeouts: 0,
    successes: 0,
    rests: 0,
  };
}

/** A fresh snapshot of one record, with the demotion spelled out. */
function snapshotOf(record: ModelHealth): HealthSnapshot {
  return {
    model: record.id,
    consecutive: record.consecutive,
    restedUntilPrompt: record.restUntilPrompt,
    demoted: record.slowStrikes > 0,
    dials: record.dials,
    refusals: record.refusals,
    timeouts: record.timeouts,
    successes: record.successes,
    rests: record.rests,
  };
}

/**
 * The per-model health tracker for one trial. One instance lives for the whole trial: the
 * records are the trial's memory of which models refused, which answered, and which are rested.
 */
export class HealthTracker {
  records = new Map<string, ModelHealth>();

  /** The record for one model, created on first sight. */
  recordFor(id: string): ModelHealth {
    let record = this.records.get(id);
    if (record === undefined) {
      record = emptyRecord(id);
      this.records.set(id, record);
    }
    return record;
  }

  /**
   * One refusal for one model on one prompt. A timeout is also a slow strike. Two refusals in a
   * row rest the model for the next two prompts and clear its consecutive count, so the rest is
   * one event per streak, not one per refusal. Returns what the refusal caused.
   */
  record(id: string, kind: RefusalKind, promptIndex: number): RecordOutcome {
    const record = this.recordFor(id);
    record.dials += 1;
    record.refusals += 1;
    record.consecutive += 1;
    if (kind === 'timeout') {
      record.timeouts += 1;
      record.slowStrikes += 1;
    }
    if (record.consecutive >= REST_AFTER_CONSECUTIVE) {
      record.rests += 1;
      record.consecutive = 0;
      record.restUntilPrompt = promptIndex + 1 + REST_PROMPTS;
      return 'rested';
    }
    if (kind === 'timeout') return 'demoted';
    return 'noted';
  }

  /** One answer for one model: the streak and the demotion are both forgiven at once. */
  success(id: string): void {
    const record = this.recordFor(id);
    record.dials += 1;
    record.successes += 1;
    record.consecutive = 0;
    record.slowStrikes = 0;
  }

  /** Whether one model is rested on one prompt: rested until its rest-until prompt arrives. */
  isRested(id: string, promptIndex: number): boolean {
    const record = this.records.get(id);
    if (record === undefined) return false;
    return promptIndex < record.restUntilPrompt;
  }

  /** Whether one model is demoted: it has a no-answer dial it has not answered since. */
  isDemoted(id: string): boolean {
    const record = this.records.get(id);
    if (record === undefined) return false;
    return record.slowStrikes > 0;
  }

  /**
   * The dial order for one prompt: the ranked candidates with rested models removed, demoted
   * models last, and the rest in rating order (ties by model id), exactly as the ranking does.
   * A demoted model is still dialled when it is all that is left, or when it still lands in the
   * top three: demotion reorders, it never excludes.
   */
  orderCandidates(models: readonly ModelRow[], promptIndex: number): ModelRow[] {
    const eligible: ModelRow[] = [];
    for (const model of models) {
      if (!this.isRested(model.id, promptIndex)) eligible.push(model);
    }
    eligible.sort((a, b) => {
      const demotedA = this.isDemoted(a.id) ? 1 : 0;
      const demotedB = this.isDemoted(b.id) ? 1 : 0;
      if (demotedA !== demotedB) return demotedA - demotedB;
      if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
      if (a.id < b.id) return -1;
      if (a.id > b.id) return 1;
      return 0;
    });
    return eligible;
  }

  /** Every model's health, in model-id order, for logging or for a test to read. */
  snapshot(): HealthSnapshot[] {
    const out: HealthSnapshot[] = [];
    for (const record of this.records.values()) out.push(snapshotOf(record));
    out.sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
    return out;
  }

  /** The trial's totals on one prompt: dials, refusals, answers, rests, and how many models are demoted or rested right now. */
  totals(promptIndex: number): HealthTotals {
    let dials = 0;
    let refusals = 0;
    let timeouts = 0;
    let successes = 0;
    let rests = 0;
    let demoted = 0;
    let rested = 0;
    for (const record of this.records.values()) {
      dials += record.dials;
      refusals += record.refusals;
      timeouts += record.timeouts;
      successes += record.successes;
      rests += record.rests;
      if (record.slowStrikes > 0) demoted += 1;
      if (promptIndex < record.restUntilPrompt) rested += 1;
    }
    return { dials, refusals, timeouts, successes, rests, demoted, rested };
  }

  /** One log line for the trial's totals on one prompt: the shape of the trial so far, in numbers. */
  describe(promptIndex: number): string {
    const total = this.totals(promptIndex);
    const parts = [
      `${total.dials} dial(s)`,
      `${total.successes} answer(s)`,
      `${total.refusals} refusal(s)`,
      `${total.timeouts} no-answer(s)`,
      `${total.rests} rest(s)`,
      `${total.demoted} demoted`,
      `${total.rested} rested`,
    ];
    return `health: ${parts.join(', ')}`;
  }

  /** One log line naming every model that is demoted or rested on one prompt, or '' when none is. */
  describeUnhealthy(promptIndex: number): string {
    const names: string[] = [];
    for (const record of this.records.values()) {
      if (promptIndex < record.restUntilPrompt) names.push(`${record.id} (rested until prompt ${record.restUntilPrompt})`);
      else if (record.slowStrikes > 0) names.push(`${record.id} (demoted, ${record.slowStrikes} no-answer(s))`);
    }
    if (names.length === 0) return '';
    return `health: ${names.join('; ')}`;
  }
}
