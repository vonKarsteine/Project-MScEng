import { API_BASE, apiInference, fetchProfilesFromApi } from './apiAdapter.js';
import { mockInference } from './mockAdapter.js';
import { onnxInference } from './onnxAdapter.js';

export { API_BASE };

export const MODES = ['api', 'mock', 'onnx'];

export const MODE_LABELS = {
  api: 'API runtime',
  mock: 'Deterministic mock',
  onnx: 'ONNX Runtime Web',
};

/**
 * Used only when GET /api/profiles is unreachable, so the workbench still renders
 * a usable dropdown offline. The API response is the source of truth whenever it
 * answers -- this list is never merged with it.
 */
export const FALLBACK_PROFILES = [
  { id: 'fusion_cmodes_oof', name: 'Fusion C-MODES', modalities: ['xray', 'mri'] },
  { id: 'fusion_traditional_ensemble', name: 'Fusion OOF Stack', modalities: ['xray', 'mri'] },
  { id: 'rckf_cv2_simsiam', name: 'RCKF ConvNeXt V2 + SimSiam SSL', modalities: ['xray', 'mri'] },
  { id: 'xray_cmodes_oof', name: 'X-ray C-MODES', modalities: ['xray'] },
];

/**
 * The single MRI-completeness predicate for the whole frontend. The fusion route
 * consumes a two-channel (T2, R2) volume, so "MRI is available" means both channels
 * are present -- never just one, and never "some MRI field was set".
 */
export function pairedMriAvailable({ mri, mriR2 } = {}) {
  return Boolean(mri && mriR2);
}

export function mriChannelStatus({ mri, mriR2 } = {}) {
  if (pairedMriAvailable({ mri, mriR2 })) {
    return {
      state: 'paired',
      fields: ['mri_t2', 'mri_r2'],
      missing: null,
      message: 'T2 and R2 uploaded. The fusion route can run on the paired volume.',
    };
  }
  if (mri || mriR2) {
    const missing = mri ? 'R2' : 'T2';
    return {
      state: 'partial',
      fields: ['mri'],
      missing,
      message: `Only the ${mri ? 'T2' : 'R2'} channel was uploaded. It is sent as a single MRI field, but the fusion route needs a complete T2 + R2 pair, so this case falls back to the X-ray route. Add ${missing} to enable fusion.`,
    };
  }
  return {
    state: 'none',
    fields: [],
    missing: 'T2 and R2',
    message: 'No MRI uploaded. X-ray-only route.',
  };
}

const HEATMAP_REASONS = {
  not_configured: 'This runtime does not expose Grad-CAM. Only the X-ray validation substitute returns a saliency map.',
  'grad-cam': 'Grad-CAM ran but produced no usable activation for the target layer.',
  none: 'This route returned no saliency map.',
};

/**
 * Collapses the three shapes the backend can return -- a populated Grad-CAM payload,
 * `{available: false, method: "not_configured"}` with no `dataUrl` key at all, and the
 * mock's synthetic map -- into one always-safe object. Never index into `dataUrl`
 * before checking `available`.
 */
export function normalizeHeatmap(heatmap) {
  const source = heatmap && typeof heatmap === 'object' ? heatmap : {};
  const dataUrl = typeof source.dataUrl === 'string' && source.dataUrl.length > 0 ? source.dataUrl : null;
  const declared = source.available === undefined ? Boolean(dataUrl) : Boolean(source.available);
  const method = source.method || (dataUrl ? 'unknown' : 'none');
  const available = declared && Boolean(dataUrl);
  return {
    available,
    dataUrl: available ? dataUrl : null,
    method,
    reason: available ? null : HEATMAP_REASONS[method] || `No saliency map available (${method}).`,
  };
}

function normalizeResult(payload, request) {
  const source = payload && typeof payload === 'object' ? payload : {};
  const prediction = source.prediction || {};
  const classProbs = Array.isArray(prediction.classProbs)
    ? prediction.classProbs
    : Array.isArray(source.classProbs)
      ? source.classProbs
      : [];
  return {
    ...source,
    prediction: { ...prediction, classProbs },
    classProbs,
    heatmap: normalizeHeatmap(source.heatmap),
    route: source.route || null,
    metadata: source.metadata || null,
    mriStatus: request.mriStatus,
  };
}

export async function loadProfiles(signal) {
  try {
    const profiles = await fetchProfilesFromApi(signal);
    return { profiles, source: 'api', error: null };
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    return {
      profiles: FALLBACK_PROFILES,
      source: 'fallback',
      error: error instanceof Error ? error.message : 'profile fetch failed',
    };
  }
}

export async function runInference(request) {
  const enriched = { ...request, mriStatus: mriChannelStatus(request) };
  let payload;
  if (enriched.mode === 'mock') payload = await mockInference(enriched);
  else if (enriched.mode === 'onnx') payload = await onnxInference(enriched);
  else payload = await apiInference(enriched);
  return normalizeResult(payload, enriched);
}
