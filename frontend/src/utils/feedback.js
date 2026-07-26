const STORAGE_KEY = 'koa_multimodal_v7_feedback';

/**
 * Browsers cap localStorage at roughly 5 MB per origin, and every entry embeds two
 * base64 384x384 PNGs (processed frame + annotated frame), so the budget is spent in
 * tens of cases rather than thousands. Warn well before the write starts failing.
 */
export const STORAGE_LIMIT_BYTES = 5 * 1024 * 1024;
export const STORAGE_WARN_BYTES = Math.round(STORAGE_LIMIT_BYTES * 0.75);

function serialize(items) {
  return JSON.stringify(items);
}

export function measureBytes(items) {
  try {
    return new Blob([serialize(items)]).size;
  } catch {
    return serialize(items).length;
  }
}

export function storageStatus(items) {
  const bytes = measureBytes(items);
  const ratio = bytes / STORAGE_LIMIT_BYTES;
  if (bytes >= STORAGE_LIMIT_BYTES) return { bytes, ratio, level: 'full' };
  if (bytes >= STORAGE_WARN_BYTES) return { bytes, ratio, level: 'warn' };
  return { bytes, ratio, level: 'ok' };
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function loadFeedback() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveFeedback(items) {
  try {
    localStorage.setItem(STORAGE_KEY, serialize(items));
    return { ok: true, error: null };
  } catch (error) {
    return {
      ok: false,
      error:
        'Local feedback store is full. Export the queue to JSON and clear it before adding more annotated cases.',
      cause: error instanceof Error ? error.message : String(error),
    };
  }
}

export function exportFeedback(items) {
  const payload = JSON.stringify(
    { schema: 'koa-multimodal-v7-feedback', exportedAt: new Date().toISOString(), count: items.length, items },
    null,
    2
  );
  const blob = new Blob([payload], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `koa_feedback_export_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
