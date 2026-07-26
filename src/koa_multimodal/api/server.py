"""The demo inference API. Standard library only, on purpose.

``ThreadingHTTPServer`` + ``BaseHTTPRequestHandler``, with multipart bodies parsed
by hand through :mod:`email.parser`. There is no FastAPI, no Flask, no uvicorn and
no pydantic here, and adding one would be a regression rather than a cleanup.

§4.6.3 presents this as a **serverless-style prototype**: the
claim being demonstrated is that the deployed grader needs a Python interpreter
and a checkpoint and nothing else. A web framework would make that claim false
while changing nothing a reviewer can see -- the endpoints, the response shape and
the latency would be identical, and the one property the section exists to
demonstrate would be gone. The cost is roughly sixty lines of routing and one
multipart parser, which is cheaper than the dependency it replaces.

**Two predictors, and only two.** The deterministic mock
(:mod:`koa_multimodal.api.predictors.mock`) and the real torch runtime
(:mod:`koa_multimodal.api.predictors.torch_runtime`). No third "substitute"
predictor is admitted -- in particular not a single-modality stand-in from
another project tree, reached through a hardcoded absolute path on one
developer's machine. Served under this API's response shape, a different
system's model does not demonstrate this one.

**Predictor selection is two-layered, and each layer answers one question.**

*Content type decides first.* A JSON body always gets the mock, even with a real
checkpoint configured: a JSON body carries filenames, not pixels, so there is
nothing for a model to look at. That makes ``POST /api/predict`` with a JSON body
a contract probe that works identically on every machine.

*Then availability decides.* A multipart body goes to the torch runtime when a
checkpoint is **configured and exists** -- both, checked separately -- and
otherwise to the mock, which says so in ``metadata.runtime`` and ``route.runtime``.
Answering 503 here would turn the entire workbench into an error page for a
checkout that deliberately ships no weights, so the answer is a labelled mock
instead: an honest synthetic answer is more useful than an opaque failure,
provided it is impossible to mistake for a real one. A checkpoint
that is configured and present but *broken* is a different case and is still an
error -- falling back there would hide a real fault.

**The predictor contract.** Both predictors return the same intermediate mapping,
and this module assembles the HTTP envelope from it exactly once:

``prediction``, ``route``, ``heatmap``, ``runtime``, ``quantization``,
``diagnostics``.

So the response shape is defined here and nowhere else, and a new predictor
cannot introduce a fourth variant of it.
"""

from __future__ import annotations

import json
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from koa_multimodal import __version__
from koa_multimodal.api.predictors import torch_runtime
from koa_multimodal.api.predictors.mock import mock_predict
from koa_multimodal.api.profiles import PROFILES, get_profile, resolve_profile_id
from koa_multimodal.config.paths import project_root
from koa_multimodal.core.errors import KoaError
from koa_multimodal.core.serialization import read_json

SERVICE_NAME = "koa_multimodal_v7"

#: ``results/published/`` holds figures transcribed from the dissertation, not the
#: output of a run. It is not a run record and is never served as one.
PUBLISHED_DIR = "published"

#: A demo server should not be knocked over by a single request. 64 MB comfortably
#: holds a 384x384 PNG plus two OAI NIfTI volumes.
MAX_BODY_BYTES = 64 * 1024 * 1024

_CORS_HEADERS = (
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Headers", "Content-Type"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ("Access-Control-Max-Age", "600"),
)


# ------------------------------------------------------------------- transport


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    """Every response leaves through here, so CORS cannot be attached selectively.

    Errors carry the same headers as successes. A 400 without them reaches the
    browser as an opaque network failure, and the workbench then reports "cannot
    reach the API" for a request the API answered precisely.
    """

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    for name, value in _CORS_HEADERS:
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(body)


def parse_multipart(raw: bytes, content_type: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Split a multipart body into ``(fields, files)``.

    ``email.parser`` wants a complete MIME document, so the ``Content-Type``
    header -- which carries the boundary -- is prepended before parsing. Parts
    with a filename become files; the rest become plain string fields.
    """

    preamble = ("Content-Type: %s\r\nMIME-Version: 1.0\r\n\r\n" % content_type).encode("utf-8")
    message = BytesParser(policy=email_default).parsebytes(preamble + raw)
    fields: Dict[str, str] = {}
    files: Dict[str, Dict[str, Any]] = {}
    if not message.is_multipart():
        return fields, files
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            files[name] = {
                "filename": filename,
                "content": payload,
                "contentType": part.get_content_type(),
                "bytes": len(payload),
            }
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


# ----------------------------------------------------------------- run records


def available_run_ids() -> List[str]:
    """Every record directory under ``results/``, minus ``published``.

    Listed from disk rather than hardcoded. A whitelist of ids in the handler
    would make adding a run an edit to the web layer, and would leave a freshly
    scaffolded record invisible until someone remembered to make that edit.
    """

    root = project_root() / "results"
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and entry.name != PUBLISHED_DIR
    )


def load_run_record(run_id: str) -> Optional[Dict[str, Any]]:
    """The curated record tree for one run, or ``None`` when there is no such run.

    ``run_id`` is checked against the real directory listing rather than
    sanitised, which is both simpler and stricter than filtering separators: a
    traversal attempt is not on the list, so it is a 404 like any other unknown
    id.
    """

    if run_id not in available_run_ids():
        return None
    directory = project_root() / "results" / run_id
    records: Dict[str, Any] = {}
    unreadable: List[str] = []
    for source in sorted(directory.glob("*.json")):
        try:
            records[source.stem] = read_json(source)
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append("%s: %s" % (source.name, exc))
    manifest = records.get("manifest") or {}
    return {
        "runId": run_id,
        "status": manifest.get("status"),
        "files": sorted(records),
        "unreadable": unreadable,
        "records": records,
    }


# ------------------------------------------------------------------ prediction


def _mri_state(paired: bool, partial: bool) -> str:
    if paired:
        return "paired"
    return "partial" if partial else "none"


def _assemble(
    inference: Dict[str, Any],
    *,
    requested_profile: str,
    latency_ms: int,
    filenames: Dict[str, Optional[str]],
    mri_state: str,
) -> Dict[str, Any]:
    """Build the HTTP envelope the workbench consumes, from a predictor result.

    ``classProbs`` and ``uncertainty`` are duplicated at the top level as well as
    inside ``prediction``. That is not redundancy for its own sake:
    ``frontend/src/utils/inference/index.js`` reads ``prediction.classProbs``
    first and falls back to the top-level array, so emitting both means the
    workbench renders the bars whichever branch it takes.
    """

    prediction = inference["prediction"]
    route = inference["route"]
    return {
        "requestId": str(uuid.uuid4()),
        "prediction": prediction,
        "classProbs": prediction["classProbs"],
        "uncertainty": prediction["uncertainty"],
        "latencyMs": latency_ms,
        "metadata": {
            "modelProfile": requested_profile,
            # What actually ran. The workbench flags a divergence between the two
            # in the result panel, which is how a missing-modality fallback
            # becomes visible rather than silent.
            "executedProfile": route["routeId"],
            "quantization": inference["quantization"],
            "runtime": inference["runtime"],
            "mriUsed": route["mriUsed"],
            "missingModalityFallback": route["missingModalityFallback"],
            "xrayFilename": filenames.get("xray"),
            "mriT2Filename": filenames.get("mri_t2"),
            "mriR2Filename": filenames.get("mri_r2"),
            "mriFilename": filenames.get("mri"),
            "mriState": mri_state,
            "diagnostics": inference.get("diagnostics", {}),
        },
        "route": route,
        "heatmap": inference["heatmap"],
    }


def predict_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON path: always the mock, because a JSON body carries no pixels.

    Accepted keys: ``profileId``; ``xray`` (a filename string or an object with a
    ``filename``); ``mriT2`` and ``mriR2``, same shape. MRI counts as available
    only when **both** channels are declared -- the same completeness rule the
    multipart path and the frontend apply, since a lone channel cannot form the
    two-channel volume the fusion routes consume.
    """

    started = time.perf_counter()
    profile_id = resolve_profile_id(payload.get("profileId") or payload.get("modelProfile"))
    xray_filename = _declared_filename(payload.get("xray")) or payload.get("xrayFilename")
    t2_filename = _declared_filename(payload.get("mriT2") or payload.get("mri"))
    r2_filename = _declared_filename(payload.get("mriR2"))
    paired = bool(t2_filename and r2_filename)
    partial = bool(t2_filename or r2_filename) and not paired

    inference = mock_predict(
        profile_id=profile_id,
        xray_filename=xray_filename,
        mri_paired=paired,
    )
    return _assemble(
        inference,
        requested_profile=profile_id,
        latency_ms=int(round((time.perf_counter() - started) * 1000.0)),
        filenames={
            "xray": xray_filename,
            "mri_t2": t2_filename if paired else None,
            "mri_r2": r2_filename if paired else None,
            "mri": t2_filename if partial else None,
        },
        mri_state=_mri_state(paired, partial),
    )


def _declared_filename(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        name = value.get("filename") or value.get("name")
        return str(name) if name else None
    return None


def predict_from_multipart(
    fields: Dict[str, str], files: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """The upload path. Torch when a checkpoint is configured *and* present.

    MRI field naming mirrors the frontend exactly: a complete pair arrives as
    ``mri_t2`` + ``mri_r2``, a lone channel as ``mri``. The lone channel is
    accepted, recorded and then not used -- it cannot form the ``(2, D, H, W)``
    tensor -- so the case runs the X-ray route and reports
    ``missingModalityFallback``.
    """

    started = time.perf_counter()
    xray = files.get("xray")
    if xray is None:
        raise _BadRequest("missing_xray", "multipart field 'xray' is required")

    profile_id = resolve_profile_id(fields.get("profileId") or fields.get("modelProfile"))
    t2 = files.get("mri_t2")
    r2 = files.get("mri_r2")
    lone = files.get("mri")
    paired = t2 is not None and r2 is not None
    partial = (lone is not None) or (t2 is None) != (r2 is None)

    if torch_runtime.runtime_available():
        runtime = torch_runtime.get_runtime()
        inference = runtime.predict(
            xray["content"],
            t2["content"] if paired else None,
            r2["content"] if paired else None,
        )
    else:
        inference = mock_predict(
            profile_id=profile_id,
            xray_filename=xray["filename"],
            mri_paired=paired,
        )

    return _assemble(
        inference,
        requested_profile=profile_id,
        latency_ms=int(round((time.perf_counter() - started) * 1000.0)),
        filenames={
            "xray": xray["filename"],
            "mri_t2": t2["filename"] if t2 else None,
            "mri_r2": r2["filename"] if r2 else None,
            "mri": lone["filename"] if lone else None,
        },
        mri_state=_mri_state(paired, partial),
    )


class _BadRequest(Exception):
    """A client-side problem: reported as 400 with a machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------- handler


class Handler(BaseHTTPRequestHandler):
    server_version = "KOAMultimodal/%s" % __version__
    protocol_version = "HTTP/1.1"

    # -- routing -----------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        _send_json(self, 200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/health":
                _send_json(self, 200, health_payload())
            elif path == "/api/profiles":
                _send_json(self, 200, {"profiles": PROFILES})
            elif path == "/api/results":
                _send_json(self, 200, {"runIds": available_run_ids()})
            elif path.startswith("/api/results/"):
                self._get_result(path.rsplit("/", 1)[-1])
            else:
                _send_json(self, 404, {"error": "not_found", "path": path})
        except Exception as exc:  # noqa: BLE001 - a demo server must not die on one request
            _send_json(self, 500, {"error": "internal_error", "detail": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        content_type = self.headers.get("Content-Type", "")
        try:
            raw = self._read_body()
        except _BadRequest as exc:
            _send_json(self, 413, {"error": exc.code, "detail": exc.detail})
            return

        try:
            if path == "/api/predict":
                self._post_predict(raw, content_type)
            elif path == "/api/feedback/export":
                self._post_feedback(raw)
            else:
                _send_json(self, 404, {"error": "not_found", "path": path})
        except _BadRequest as exc:
            _send_json(self, 400, {"error": exc.code, "detail": exc.detail})
        except FileNotFoundError as exc:
            # A configured checkpoint that vanished between the manifest check and
            # the load. Reported, never silently downgraded to the mock.
            _send_json(self, 503, {"error": "missing_checkpoint", "detail": str(exc)})
        except ValueError as exc:
            _send_json(self, 400, {"error": "invalid_upload", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            _send_json(
                self,
                500,
                {"error": "inference_failed", "detail": "%s: %s" % (type(exc).__name__, exc)},
            )

    # -- handlers ----------------------------------------------------------

    def _get_result(self, run_id: str) -> None:
        record = load_run_record(run_id)
        if record is None:
            _send_json(
                self,
                404,
                {
                    "error": "unknown_run_id",
                    "runId": run_id,
                    "available": available_run_ids(),
                },
            )
            return
        _send_json(self, 200, record)

    def _post_predict(self, raw: bytes, content_type: str) -> None:
        if content_type.startswith("multipart/form-data"):
            fields, files = parse_multipart(raw, content_type)
            _send_json(self, 200, predict_from_multipart(fields, files))
            return
        _send_json(self, 200, predict_from_json(self._json_body(raw)))

    def _post_feedback(self, raw: bytes) -> None:
        """Echo the submitted queue back. **Nothing is persisted.**

        Clinician feedback is patient-adjacent data and this is a demo with no
        access control, no audit log and no retention policy, so it stores none.
        The workbench keeps its queue in ``localStorage`` and exports JSON on the
        client; this endpoint exists so an integration can be demonstrated
        end to end, and it returns what it was given plus a receipt.
        """

        payload = self._json_body(raw)
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            items = [payload]
        _send_json(
            self,
            200,
            {
                "exportedAt": time.time(),
                "count": len(items),
                "items": items,
                "persisted": False,
                "note": (
                    "This endpoint stores nothing. The response is an echo; the "
                    "authoritative feedback queue lives in the browser's "
                    "localStorage and is exported client-side."
                ),
            },
        )

    # -- plumbing ----------------------------------------------------------

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > MAX_BODY_BYTES:
            raise _BadRequest(
                "payload_too_large",
                "Body is %d bytes; this demo server accepts at most %d."
                % (length, MAX_BODY_BYTES),
            )
        return self.rfile.read(length) if length else b""

    def _json_body(self, raw: bytes) -> Dict[str, Any]:
        if not raw.strip():
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _BadRequest("invalid_json", str(exc)) from None
        if not isinstance(payload, dict):
            raise _BadRequest("invalid_json", "Expected a JSON object at the top level")
        return payload

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Silenced: the CLI's own output is the interface, not an access log."""

        return


# ---------------------------------------------------------------- entry points


def health_payload() -> Dict[str, Any]:
    """``ok`` reflects the *service*, not the model.

    The API is healthy when it can answer, and it can always answer -- with the
    mock when no checkpoint is configured. Whether a real model is loaded is a
    separate fact, reported under ``runtime``, where ``configured`` and ``exists``
    are two independent booleans rather than one conflated one.
    """

    manifest = torch_runtime.runtime_manifest()
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": __version__,
        "time": time.time(),
        "runtime": {
            "configured": manifest["configured"],
            "exists": manifest["exists"],
            "modelId": manifest["modelId"],
            "checkpoint": manifest["checkpoint"],
            "envVar": manifest["envVar"],
            "loaded": manifest["loaded"],
            "activePredictor": "torch" if (manifest["configured"] and manifest["exists"]) else "mock",
        },
    }


def startup_check() -> Dict[str, Any]:
    """Build the app, exercise the mock end to end, and read ``results/``.

    Used by ``koa serve --check``. It binds no socket -- a start-up check that
    needs a free port is a check of the port. What it does verify is everything
    that can fail before a request arrives: the profile list is structurally
    sound, the mock produces a response matching the contract this module
    assembles, and the record tree is on disk and readable.

    Raises :class:`~koa_multimodal.core.errors.KoaError` on a contract failure so
    the CLI exits non-zero; a missing checkpoint is *not* a failure, because
    serving the mock is the documented behaviour of this checkout.
    """

    findings: List[str] = []

    probe = predict_from_json(
        {"profileId": "fusion_cmodes_oof", "xray": {"filename": "startup-check.png"}}
    )
    _assert_response_contract(probe)

    paired = predict_from_json(
        {
            "profileId": "fusion_cmodes_oof",
            "xray": {"filename": "startup-check.png"},
            "mriT2": "t2.nii.gz",
            "mriR2": "r2.nii.gz",
        }
    )
    _assert_response_contract(paired)
    if not paired["route"]["mriUsed"]:
        raise KoaError(
            "A complete T2 + R2 pair was reported as unusable by the mock; the "
            "MRI completeness rule and the response envelope disagree."
        )
    if probe["route"]["missingModalityFallback"] is not True:
        raise KoaError(
            "A multimodal profile with no MRI did not report a missing-modality "
            "fallback; the workbench would show a fusion result for an X-ray-only case."
        )

    results_dir = project_root() / "results"
    if not results_dir.is_dir():
        raise KoaError(
            "results/ does not exist. The record tree is scaffolded before a run "
            "and served read-only by GET /api/results/<run_id>; without it the "
            "API cannot answer that route."
        )
    run_ids = available_run_ids()
    if not run_ids:
        findings.append(
            "results/ contains no run record directories; GET /api/results/<id> "
            "will 404 for every id."
        )

    manifest = torch_runtime.runtime_manifest()
    if manifest["configured"] and not manifest["exists"]:
        findings.append(
            "%s is set to %r, which does not exist. Multipart predictions will be "
            "served by the mock." % (manifest["envVar"], manifest["checkpoint"])
        )
    elif not manifest["configured"]:
        findings.append(
            "No deployment checkpoint configured (%s unset); every prediction is "
            "served by the labelled mock. This checkout ships no weights."
            % manifest["envVar"]
        )

    return {
        "ok": True,
        "service": SERVICE_NAME,
        "version": __version__,
        "routes": ROUTES,
        "profiles": [profile["id"] for profile in PROFILES],
        "mockProbe": {
            "runtime": probe["metadata"]["runtime"],
            "klGrade": probe["prediction"]["klGrade"],
            "classProbCount": len(probe["classProbs"]),
            "heatmapMethod": probe["heatmap"]["method"],
            "heatmapAvailable": probe["heatmap"]["available"],
        },
        "records": {"root": str(results_dir), "runIds": run_ids},
        "runtime": manifest,
        "findings": findings,
    }


#: Advertised by the start-up check so a reviewer can see the surface at a glance.
ROUTES: Tuple[str, ...] = (
    "GET /health",
    "GET /api/profiles",
    "GET /api/results",
    "GET /api/results/<run_id>",
    "POST /api/predict",
    "POST /api/feedback/export",
    "OPTIONS *",
)

_REQUIRED_TOP_LEVEL = ("requestId", "prediction", "classProbs", "uncertainty", "latencyMs", "metadata", "route", "heatmap")
_REQUIRED_PREDICTION = ("klGrade", "classProbs", "confidence", "uncertainty")
_REQUIRED_ROUTE = ("routeId", "runtime", "mriUsed", "missingModalityFallback", "switchScore", "switchFlag")
_REQUIRED_METADATA = (
    "modelProfile",
    "executedProfile",
    "quantization",
    "runtime",
    "mriUsed",
    "missingModalityFallback",
    "xrayFilename",
)


def _assert_response_contract(response: Dict[str, Any]) -> None:
    """Check the envelope against what the workbench actually reads.

    Every key here is dereferenced by ``frontend/src/utils/inference/index.js``,
    ``components/ResultPanel.jsx`` or ``App.jsx``. Checking the shape in a start-up
    probe means a field lost in a refactor surfaces as a failed
    ``koa serve --check`` rather than as a dash in a clinician's result panel.
    """

    for group, keys, node in (
        ("response", _REQUIRED_TOP_LEVEL, response),
        ("prediction", _REQUIRED_PREDICTION, response.get("prediction", {})),
        ("route", _REQUIRED_ROUTE, response.get("route", {})),
        ("metadata", _REQUIRED_METADATA, response.get("metadata", {})),
    ):
        missing = [key for key in keys if key not in node]
        if missing:
            raise KoaError("Response %s is missing %s" % (group, missing))

    probabilities = response["prediction"]["classProbs"]
    if len(probabilities) != 5:
        raise KoaError(
            "Expected 5 Kellgren-Lawrence posteriors, got %d" % len(probabilities)
        )
    total = sum(float(value) for value in probabilities)
    if abs(total - 1.0) > 1e-4:
        raise KoaError("Class posteriors sum to %.6f, not 1" % total)
    grade = response["prediction"]["klGrade"]
    if grade != max(range(5), key=lambda index: probabilities[index]):
        raise KoaError("klGrade %r is not the argmax of the reported posteriors" % grade)
    if get_profile(response["metadata"]["modelProfile"]) is None:
        raise KoaError(
            "Response names profile %r, which is not in the served profile list"
            % response["metadata"]["modelProfile"]
        )


class DemoServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` with two demo-appropriate manners.

    ``daemon_threads`` means an in-flight request cannot keep the process alive
    after Ctrl-C -- a server that has to be killed twice reads as a hang.

    :meth:`handle_error` swallows the three connection-teardown errors. Under
    HTTP/1.1 keep-alive a client that closes its socket after the response --
    which every ``curl``, ``fetch`` and ``Invoke-WebRequest`` does -- leaves the
    handler blocked in ``readline`` waiting for the next request on that
    connection, and ``socketserver``'s default handler prints a full traceback
    for it. The exchange succeeded; the trace is noise, and noise in a demo
    console is indistinguishable from a fault to the person watching it. Every
    other error still goes through the default path.
    """

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        exception = sys.exc_info()[1]
        if isinstance(exception, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the API until interrupted."""

    httpd = DemoServer((host, port), Handler)
    manifest = torch_runtime.runtime_manifest()
    predictor = "torch runtime" if (manifest["configured"] and manifest["exists"]) else "mock (no checkpoint configured)"
    print("KOA Multimodal v%s API on http://%s:%d" % (__version__, host, port))
    print("Predictor for multipart uploads: %s" % predictor)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
