export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * MRI field naming mirrors the backend multipart parser exactly: a complete pair
 * goes out as `mri_t2` + `mri_r2`, a lone channel as `mri`.
 *
 * The runtime can only assemble the (2, 32, 384, 384) T2/R2 tensor from a complete
 * pair, so a lone channel reaches the server but cannot drive the fusion route.
 * It is still sent rather than discarded, and `request.mriStatus` (computed once by
 * `runInference`) is what raises the shortfall as a visible notice.
 */
function appendMri(body, request) {
  const state = request.mriStatus?.state;
  if (state === 'paired') {
    body.append('mri_t2', request.mri, request.mri.name);
    body.append('mri_r2', request.mriR2, request.mriR2.name);
  } else if (state === 'partial') {
    const lone = request.mri || request.mriR2;
    body.append('mri', lone, lone.name);
  }
}

export async function apiInference(request) {
  if (!request.xray) throw new Error('X-ray file is required');

  const body = new FormData();
  body.append('profileId', request.profileId);
  body.append('xray', request.xray, request.xray.name);
  appendMri(body, request);

  let response;
  try {
    response = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body });
  } catch {
    throw new Error(`Cannot reach the inference API at ${API_BASE}. Is it running?`);
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `API responded ${response.status}`);
  }
  return payload;
}

export async function fetchProfilesFromApi(signal) {
  const response = await fetch(`${API_BASE}/api/profiles`, { signal });
  if (!response.ok) throw new Error(`API responded ${response.status}`);
  const payload = await response.json();
  const profiles = Array.isArray(payload) ? payload : payload?.profiles;
  if (!Array.isArray(profiles) || profiles.length === 0) {
    throw new Error('API returned no profiles');
  }
  return profiles.filter((profile) => profile && profile.id);
}
