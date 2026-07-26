# KOA Multimodal v7 — Clinical Inference Workbench

A browser-first diagnostic prototype for 5-class Kellgren–Lawrence knee osteoarthritis grading from
paired X-ray + MRI. It is the presentation layer over the v7 inference API and is organised in four
tiers:

1. **Presentation** — upload, Konva alignment canvas, saliency overlay, ROI annotation.
2. **Inference abstraction** — three swappable adapters behind one normalised response shape.
3. **Model runtime** — the API-backed PyTorch runtime, with an in-browser ONNX slot reserved.
4. **Feedback** — reviewer decisions kept in `localStorage` and exported as JSON. Nothing is POSTed back.

## Run it

```bash
npm ci
npm run dev     # http://127.0.0.1:5173
npm run build   # production bundle into dist/
```

Copy `.env.example` to `.env` and adjust as needed.

## Inference modes

| Mode   | Backend                                                                            |
| ------ | ---------------------------------------------------------------------------------- |
| `api`  | `POST {VITE_API_BASE_URL}/api/predict` (multipart). The real runtime.               |
| `mock` | Deterministic in-browser stub, seeded by filename. No backend needed for a demo.    |
| `onnx` | Reserved for `onnxruntime-web`. Not enabled — it reports that rather than faking a result. |

The mode is a dropdown in the **Model** panel and is switchable at any time.

## Environment variables

| Variable              | Default                 | Purpose                                                    |
| --------------------- | ----------------------- | ---------------------------------------------------------- |
| `VITE_API_BASE_URL`   | `http://127.0.0.1:8000` | Base URL for `/api/predict` and `/api/profiles`.            |
| `VITE_INFERENCE_MODE` | `api`                   | **Initial** mode only — the selector overrides it at runtime. |
| `VITE_ONNX_MODEL_URL` | *(empty)*               | URL of an exported ONNX graph for the ONNX adapter.         |

Vite inlines `VITE_*` values at build time, so changing them requires a restart (dev) or rebuild.

## Behaviour worth knowing

**The model never sees the raw upload.** The X-ray is re-rendered through the alignment canvas onto an
offscreen 384×384 PNG, and that file is what is POSTed. `OUTPUT_SIZE` in
`src/components/AlignmentCanvas.jsx` governs both that export and the `*Px` ROI coordinates written
into feedback entries, so changing it moves both together. The saliency overlay is independent — it is
stretched over the on-screen crop region at whatever resolution the backend returns.

**MRI is paired or nothing.** The fusion route consumes a two-channel (T2, R2) volume. A complete pair
is sent as `mri_t2` + `mri_r2`; a lone channel is sent as a single `mri` field and the UI states
plainly that fusion needs the pair. `pairedMriAvailable()` in `src/utils/inference/index.js` is the one
predicate the whole frontend uses for this.

**Model profiles come from the API.** `GET /api/profiles` is fetched on mount; if it is unreachable the
workbench falls back to a built-in list and says which one is in use.

**Saliency maps are optional.** Only some routes return one. The overlay renders only when the response
is both `available` and carries a `dataUrl`; otherwise the panel shows the reason from `method`.

**Feedback is local only.** Entries live under the `koa_multimodal_v7_feedback` localStorage key with
ISO-8601 timestamps and embed the processed and annotated PNGs. Browsers cap that store at roughly 5 MB,
so the panel warns as the queue approaches the limit — export to JSON and clear it.
