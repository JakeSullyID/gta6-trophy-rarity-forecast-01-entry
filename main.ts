// Dilution and progression forecaster incorporating v2 hazard decay
import * as fs from 'node:fs';
import * as path from 'node:path';

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface ReadHistory {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

interface CaseInput {
  schema: string;
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: {
    name: string;
    platform: string;
    release_date: string;
    source: string;
  };
  trophies: Trophy[];
  history: ReadHistory[];
  notes: string;
}

interface TrophyOutput {
  id: number;
  q: number[];
}

interface OutputJson {
  trophies: TrophyOutput[];
}

function logit(p: number): number {
  const prob = Math.max(0.0001, Math.min(0.9999, p / 100));
  return Math.log(prob / (1 - prob));
}

function invLogit(val: number): number {
  const p = 1 / (1 + Math.exp(-val));
  return Math.max(0, Math.min(100, p * 100));
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function linearRegression(x: number[], y: number[]): { slope: number; intercept: number; r2: number } {
  const n = x.length;
  if (n === 0) return { slope: 0, intercept: 0, r2: 0 };
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - mx) * (y[i] - my);
    den += (x[i] - mx) * (x[i] - mx);
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = my - slope * mx;

  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const pred = slope * x[i] + intercept;
    ssRes += (y[i] - pred) * (y[i] - pred);
    ssTot += (y[i] - my) * (y[i] - my);
  }
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  return { slope, intercept, r2 };
}

// Estimate v2 hazard decay per trophy from flat-segment history halves
function processCase(inputData: CaseInput): OutputJson {
  const baseTime = new Date('2026-11-19T00:00:00Z').getTime();
  const history = inputData.history;
  const targetTime = (new Date(inputData.target).getTime() - baseTime) / 3600000;
  const horizonHours = inputData.horizon_minutes / 60;
  const H_days = horizonHours / 24;

  // Build reads with hours
  const reads = history.map((h) => {
    const hours = (new Date(h.slot).getTime() - baseTime) / 3600000;
    return {
      slot: h.slot,
      hours,
      P: h.players_tracked,
      rates: h.rates,
    };
  });

  const numTrophies = inputData.trophies.length;
  // Initialize earner counts E[read_idx][trophy_idx]
  const E_matrix: number[][] = [];
  let prevRates: (number | null)[] = new Array(numTrophies).fill(null);

  for (let r = 0; r < reads.length; r++) {
    const P = reads[r].P;
    const currentRates = reads[r].rates;
    const rowE: number[] = [];
    for (let t = 0; t < numTrophies; t++) {
      let rate = currentRates[t];
      if (rate === null || rate === undefined) {
        rate = prevRates[t] !== null ? prevRates[t]! : 0;
      }
      prevRates[t] = rate;
      rowE.push((rate * P) / 100);
    }
    E_matrix.push(rowE);
  }

  // 1. Segment history: refresh boundary = consecutive read pair with P_next > P_prev
  const segments: { startIdx: number; endIdx: number; P: number }[] = [];
  let segStart = 0;
  for (let r = 1; r < reads.length; r++) {
    if (reads[r].P > reads[r - 1].P) {
      segments.push({ startIdx: segStart, endIdx: r - 1, P: reads[segStart].P });
      segStart = r;
    }
  }
  segments.push({ startIdx: segStart, endIdx: reads.length - 1, P: reads[segStart].P });

  // 2. Player growth P(t)
  // Collect post-refresh reads or unique P values
  const postRefreshReads = [0];
  for (let r = 1; r < reads.length; r++) {
    if (reads[r].P > reads[r - 1].P) {
      postRefreshReads.push(r);
    }
  }

  const distinctPValues = new Set(reads.map((r) => r.P));
  let P_target = reads[reads.length - 1].P;
  let s_res = 0;
  let useFit = false;

  if (distinctPValues.size >= 3) {
    useFit = true;
    const lnX: number[] = [];
    const lnP: number[] = [];
    for (const idx of postRefreshReads) {
      lnX.push(Math.log(reads[idx].hours + 1));
      lnP.push(Math.log(reads[idx].P));
    }
    const fit = linearRegression(lnX, lnP);
    const rawTargetP = Math.exp(fit.intercept + fit.slope * Math.log(targetTime + 1));
    const lastP = reads[reads.length - 1].P;
    P_target = Math.max(lastP, Math.min(lastP * 20, rawTargetP));

    let ssRes = 0;
    for (let i = 0; i < lnX.length; i++) {
      const pred = fit.intercept + fit.slope * lnX[i];
      ssRes += (lnP[i] - pred) * (lnP[i] - pred);
    }
    s_res = lnX.length > 2 ? Math.sqrt(ssRes / (lnX.length - 2)) : 0;
  } else {
    // Clock prior: 282-minute clock (~4.7 hours), anchor first read
    const anchorHours = reads.length > 0 ? reads[0].hours : 0;
    const lastReadHours = reads[reads.length - 1].hours;
    let mult = 6.06;
    let currentP = reads[reads.length - 1].P;
    const intervalHours = 282 / 60;
    let k = Math.floor((lastReadHours - anchorHours) / intervalHours) + 1;
    let nextRefreshTime = anchorHours + k * intervalHours;

    while (nextRefreshTime <= targetTime + 0.05) {
      currentP = currentP * mult;
      mult = Math.max(1.2, mult / 1.5);
      k++;
      nextRefreshTime = anchorHours + k * intervalHours;
    }
    const lastP = reads[reads.length - 1].P;
    P_target = Math.max(lastP, Math.min(lastP * 20, currentP));
  }

  // 3. v2 Hazard decay estimation per trophy
  const t_first = reads.length > 0 ? reads[0].hours : 0;
  const t_last = reads[reads.length - 1].hours;

  const trophyParams: { lambda_late: number; delta: number }[] = [];

  for (let t = 0; t < numTrophies; t++) {
    // Collect flat segments with length >= 2 reads
    const usableSegments: { times: number[]; earners: number[]; P: number }[] = [];
    for (const seg of segments) {
      if (seg.endIdx - seg.startIdx + 1 >= 2) {
        const times: number[] = [];
        const earners: number[] = [];
        for (let r = seg.startIdx; r <= seg.endIdx; r++) {
          times.push(reads[r].hours);
          earners.push(E_matrix[r][t]);
        }
        usableSegments.push({ times, earners, P: seg.P });
      }
    }

    // Compute hazard per usable segment
    const segmentHazards: { lambda: number; midpoint: number }[] = [];
    for (const us of usableSegments) {
      const reg = linearRegression(us.times, us.earners);
      const dEdt = reg.slope;
      const P = us.P;
      const avgE = mean(us.earners);
      const hazard = dEdt / Math.max(P - avgE, 1);
      const midpoint = mean(us.times);
      segmentHazards.push({ lambda: Math.max(0, Math.min(0.5, hazard)), midpoint });
    }

    let lambda_late = 0;
    let delta = 0;

    if (segmentHazards.length >= 2) {
      // Split into early and late halves based on midpoint
      segmentHazards.sort((a, b) => a.midpoint - b.midpoint);
      const midIdx = Math.floor(segmentHazards.length / 2);
      const earlySegs = segmentHazards.slice(0, midIdx);
      const lateSegs = segmentHazards.slice(midIdx);

      const lam_early = mean(earlySegs.map((s) => s.lambda));
      lambda_late = mean(lateSegs.map((s) => s.lambda));

      const t_early_mid = mean(earlySegs.map((s) => s.midpoint));
      const t_late_mid = mean(lateSegs.map((s) => s.midpoint));
      const dt_mid = Math.max(0.1, t_late_mid - t_early_mid);

      if (lam_early > 0 && lambda_late > 0) {
        delta = Math.log(lam_early / lambda_late) / dt_mid;
        delta = Math.max(0, Math.min(0.02, delta));
      } else {
        delta = 0;
      }
    } else if (segmentHazards.length === 1) {
      lambda_late = segmentHazards[0].lambda;
      delta = 0.01; // prior delta when only one window
    } else {
      // Fallback tier prior
      const tier = inputData.trophies[t].tier;
      if (tier === 'platinum') lambda_late = 0.001;
      else if (tier === 'gold') lambda_late = 0.005;
      else if (tier === 'silver') lambda_late = 0.01;
      else lambda_late = 0.02;
      delta = 0.01;
    }

    lambda_late = Math.max(0, Math.min(0.5, lambda_late));
    trophyParams.push({ lambda_late, delta });
  }

  // 4. Refresh joiner fraction f_t
  const trophyJoiners: number[] = [];
  for (let t = 0; t < numTrophies; t++) {
    const fValues: number[] = [];
    for (let r = 1; r < reads.length; r++) {
      if (reads[r].P > reads[r - 1].P) {
        const P_before = reads[r - 1].P;
        const P_after = reads[r].P;
        const E_before = E_matrix[r - 1][t];
        const E_after = E_matrix[r][t];
        const dt_hours = reads[r].hours - reads[r - 1].hours;
        const curTime = reads[r - 1].hours;
        const lam = trophyParams[t].lambda_late * Math.exp(-trophyParams[t].delta * Math.max(0, curTime - t_last));
        const num = E_after - E_before - lam * (P_before - E_before) * dt_hours;
        const den = Math.max(1, P_after - P_before);
        const f = Math.max(0, Math.min(1, num / den));
        fValues.push(f);
      }
    }
    let f_t = 0;
    if (fValues.length >= 2) {
      fValues.sort((a, b) => a - b);
      f_t = fValues[Math.floor(fValues.length / 2)];
    } else if (fValues.length === 1) {
      f_t = fValues[0];
    } else {
      const lastRate = reads[reads.length - 1].rates[t] || 0;
      f_t = lastRate > 90 ? 0.95 : 0;
    }
    trophyJoiners.push(f_t);
  }

  // 5. Forecast to target (step forward in 30-minute steps = 0.5 hours)
  const lastReadHours = reads[reads.length - 1].hours;
  const dt_step = 0.5;
  const steps = Math.max(1, Math.round((targetTime - lastReadHours) / dt_step));

  // Current state at last read
  let currentP = reads[reads.length - 1].P;
  const currentE = [...E_matrix[reads.length - 1]];

  // If using prior path for P(t)
  const anchorHours = reads.length > 0 ? reads[0].hours : 0;
  const intervalHours = 282 / 60;
  let nextRefreshIdx = Math.floor((lastReadHours - anchorHours) / intervalHours) + 1;
  let nextRefreshTime = anchorHours + nextRefreshIdx * intervalHours;
  let currentMult = 6.06;
  // Adjust currentMult based on how many refreshes already passed
  for (let k = 1; k < nextRefreshIdx; k++) {
    currentMult = Math.max(1.2, currentMult / 1.5);
  }

  for (let s = 0; s < steps; s++) {
    const t_start = lastReadHours + s * dt_step;
    const t_end = t_start + dt_step;

    let nextP = currentP;
    let dP = 0;

    if (useFit) {
      // Interpolate fitted ln P curve
      const lnX_start = Math.log(t_start + 1);
      const lnX_end = Math.log(t_end + 1);
      // Re-fit to get slope/intercept
      const lnX: number[] = [];
      const lnP: number[] = [];
      for (const idx of postRefreshReads) {
        lnX.push(Math.log(reads[idx].hours + 1));
        lnP.push(Math.log(reads[idx].P));
      }
      const fit = linearRegression(lnX, lnP);
      const pStart = Math.exp(fit.intercept + fit.slope * lnX_start);
      const pEnd = Math.exp(fit.intercept + fit.slope * lnX_end);
      nextP = Math.max(currentP, Math.min(reads[reads.length - 1].P * 20, pEnd));
      dP = nextP - currentP;
    } else {
      // Prior clock path
      if ((t_end >= nextRefreshTime && t_start < nextRefreshTime) || (s === steps - 1 && Math.abs(nextRefreshTime - t_end) <= 0.08)) {
        dP = currentP * (currentMult - 1);
        nextP = currentP + dP;
        currentMult = Math.max(1.2, currentMult / 1.5);
        nextRefreshData: {
          nextRefreshIdx++;
          nextRefreshTime = anchorHours + nextRefreshIdx * intervalHours;
        }
      } else {
        dP = 0;
        nextP = currentP;
      }
    }

    // Update E for each trophy
    for (let t = 0; t < numTrophies; t++) {
      const lam = trophyParams[t].lambda_late * Math.exp(-trophyParams[t].delta * Math.max(0, t_start - t_last));
      const f = trophyJoiners[t];
      const dEdt = lam * (currentP - currentE[t]) + (dP > 0 ? (dP / dt_step) * f : 0);
      currentE[t] = Math.max(0, Math.min(nextP, currentE[t] + dEdt * dt_step));
      if (dP > 0 && t_end >= nextRefreshTime) {
        // discrete refresh jump
        currentE[t] = Math.max(0, Math.min(nextP, currentE[t] + dP * f));
      }
    }
    currentP = nextP;
  }

  const finalP = currentP > 0 ? currentP : 1;

  // 6. Spread and quantiles
  const trophiesOutput: TrophyOutput[] = [];
  const zValues = [-1.645, -0.674, 0, 0.674, 1.645];

  for (let t = 0; t < numTrophies; t++) {
    const E_final = currentE[t];
    const m_t = Math.max(0.01, Math.min(99.99, (100 * E_final) / finalP));

    let sP = 0.45;
    if (useFit) {
      const sP_raw = 0.35 * Math.log(P_target / finalP); // or P_high
      sP = Math.max(0.15, Math.min(0.8, Math.max(s_res, sP_raw)));
    }
    const sE = 2 / Math.sqrt(E_final + 1);
    const sM = 0.25 + 0.15 * (H_days / 4.5);

    let s_t = Math.sqrt(sP * sP + sE * sE + sM * sM);
    s_t = Math.max(0.30, Math.min(2.00, s_t));

    const baseLogit = logit(m_t);
    const qRaw = zValues.map((z) => invLogit(baseLogit + z * s_t));

    // Ensure non-decreasing and rounded/clamped
    qRaw.sort((a, b) => a - b);
    const q = qRaw.map((val) => Number(Math.max(0, Math.min(100, val)).toFixed(4)));

    trophiesOutput.push({
      id: inputData.trophies[t].id,
      q,
    });
  }

  return { trophies: trophiesOutput };
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || 'input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || 'output';

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const files = fs.readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();

  for (const file of files) {
    const inputPath = path.join(inputDir, file);
    const rawContent = fs.readFileSync(inputPath, 'utf8');
    const caseData: CaseInput = JSON.parse(rawContent);

    const result = processCase(caseData);

    const baseName = path.basename(file, '.in');
    const outputPath = path.join(outputDir, `${baseName}.out`);
    fs.writeFileSync(outputPath, JSON.stringify(result) + '\n', 'utf8');
  }
}

main();
