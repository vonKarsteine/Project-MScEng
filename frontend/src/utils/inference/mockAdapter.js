const HEATMAP_SIZE = 384;

function stableOffset(...parts) {
  const text = parts.join('|');
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (Math.abs(hash) % 1000) / 1000;
}

function makeRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return `mock-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

/**
 * A transparent-background RGBA blob so the overlay composites over the X-ray the
 * same way a real Grad-CAM PNG does. Its position is hashed from the filename, so a
 * given case always renders the same map -- the mock stays reproducible for a demo.
 */
function synthesizeHeatmap(seed) {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = HEATMAP_SIZE;
  canvas.height = HEATMAP_SIZE;
  const context = canvas.getContext('2d');
  if (!context) return null;

  const cx = HEATMAP_SIZE * (0.36 + stableOffset(seed, 'cx') * 0.28);
  const cy = HEATMAP_SIZE * (0.42 + stableOffset(seed, 'cy') * 0.22);
  const radius = HEATMAP_SIZE * (0.24 + stableOffset(seed, 'r') * 0.12);

  const gradient = context.createRadialGradient(cx, cy, radius * 0.08, cx, cy, radius);
  gradient.addColorStop(0, 'rgba(239, 68, 68, 0.92)');
  gradient.addColorStop(0.35, 'rgba(245, 158, 11, 0.66)');
  gradient.addColorStop(0.65, 'rgba(16, 185, 129, 0.38)');
  gradient.addColorStop(1, 'rgba(37, 99, 235, 0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, HEATMAP_SIZE, HEATMAP_SIZE);

  return canvas.toDataURL('image/png');
}

export async function mockInference({ profileId, xray, mriStatus }) {
  const started = performance.now();
  const seed = xray?.name || 'xray';
  const pairedMri = mriStatus?.state === 'paired';
  const profileNeedsMri = profileId !== 'xray_cmodes_oof';

  const center = 1 + stableOffset(profileId, seed) * 2.5 + (pairedMri ? 0.15 : 0);
  const raw = Array.from({ length: 5 }, (_, index) => Math.exp(-0.85 * Math.abs(index - center)));
  const total = raw.reduce((sum, value) => sum + value, 0);
  const classProbs = raw.map((value) => value / total);
  const klGrade = classProbs.reduce((best, value, index) => (value > classProbs[best] ? index : best), 0);
  const confidence = Math.max(...classProbs);
  const uncertainty =
    -classProbs.reduce((sum, value) => sum + value * Math.log(Math.max(value, 1e-8)), 0) / Math.log(5);

  const mriUsed = pairedMri && profileNeedsMri;
  const missingModalityFallback = !pairedMri && profileNeedsMri;
  const executedProfile = mriUsed || !profileNeedsMri ? profileId : 'xray_cmodes_oof';
  const switchFlag = stableOffset(profileId, seed, 'flag') > 0.5;

  return {
    requestId: makeRequestId(),
    prediction: { klGrade, confidence, classProbs, uncertainty },
    classProbs,
    uncertainty,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    metadata: {
      modelProfile: profileId,
      executedProfile,
      quantization: 'mock',
      runtime: 'mock',
      mriUsed,
      missingModalityFallback,
      xrayFilename: xray?.name || null,
    },
    route: {
      routeId: executedProfile,
      runtime: 'mock',
      mriUsed,
      missingModalityFallback,
      switchScore: Number((stableOffset(profileId, seed, 'switch') * 0.18).toFixed(3)),
      switchFlag,
    },
    heatmap: {
      available: true,
      dataUrl: synthesizeHeatmap(seed),
      method: 'synthetic-demo',
      layers: ['saliency'],
    },
  };
}
