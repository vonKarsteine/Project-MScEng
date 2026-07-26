const ONNX_MODEL_URL = import.meta.env.VITE_ONNX_MODEL_URL || '';

/**
 * The in-browser runtime tier described in the design, deliberately left inert.
 * `onnxruntime-web` is a declared dependency and the mode stays selectable, but no
 * quantized graph has been exported from this package, so the adapter reports that
 * plainly instead of silently degrading to another backend.
 */
export async function onnxInference() {
  if (!ONNX_MODEL_URL) {
    throw new Error(
      'ONNX mode is not enabled: set VITE_ONNX_MODEL_URL to an exported qat-v2 graph and rebuild. ' +
        'Use api or mock mode in the meantime.'
    );
  }
  throw new Error(
    `ONNX mode found VITE_ONNX_MODEL_URL=${ONNX_MODEL_URL} but in-browser session loading is not implemented in this build. ` +
      'Use api or mock mode.'
  );
}
