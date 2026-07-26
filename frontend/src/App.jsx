import React, { useEffect, useMemo, useRef, useState } from 'react';
import FileSlot from './components/FileSlot.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import FeedbackPanel from './components/FeedbackPanel.jsx';
import AlignmentCanvas from './components/AlignmentCanvas.jsx';
import {
  FALLBACK_PROFILES,
  MODES,
  MODE_LABELS,
  loadProfiles,
  mriChannelStatus,
  runInference,
} from './utils/inference/index.js';
import { exportFeedback, loadFeedback, saveFeedback, storageStatus } from './utils/feedback.js';

const initialMode = import.meta.env.VITE_INFERENCE_MODE || 'api';

function makeFeedbackId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return `feedback-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const canvasRef = useRef(null);
  const [xray, setXray] = useState(null);
  const [mri, setMri] = useState(null);
  const [mriR2, setMriR2] = useState(null);
  const [profiles, setProfiles] = useState(FALLBACK_PROFILES);
  const [profileSource, setProfileSource] = useState('pending');
  const [profileId, setProfileId] = useState(FALLBACK_PROFILES[0].id);
  const [mode, setMode] = useState(MODES.includes(initialMode) ? initialMode : 'api');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showRoi, setShowRoi] = useState(true);
  const [zoom, setZoom] = useState(100);
  const [enableCrop, setEnableCrop] = useState(true);
  const [showGuides, setShowGuides] = useState(true);
  const [alignment, setAlignment] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [roiMode, setRoiMode] = useState(null);
  const [roiSelections, setRoiSelections] = useState([]);
  const [feedbackItems, setFeedbackItems] = useState(() => loadFeedback());
  const [storageError, setStorageError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    loadProfiles(controller.signal)
      .then((outcome) => {
        setProfiles(outcome.profiles);
        setProfileSource(outcome.source);
        setProfileId((current) =>
          outcome.profiles.some((profile) => profile.id === current) ? current : outcome.profiles[0].id
        );
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const outcome = saveFeedback(feedbackItems);
    setStorageError(outcome.ok ? '' : outcome.error);
  }, [feedbackItems]);

  useEffect(() => {
    setResult(null);
    setProcessedImage(null);
    setRoiSelections([]);
    setRoiMode(null);
  }, [xray]);

  const storage = useMemo(() => storageStatus(feedbackItems), [feedbackItems]);
  const mriStatus = useMemo(() => mriChannelStatus({ mri, mriR2 }), [mri, mriR2]);
  const activeProfile = useMemo(
    () => profiles.find((profile) => profile.id === profileId) || null,
    [profileId, profiles]
  );
  const profileWantsMri = Boolean(activeProfile?.modalities?.includes('mri'));
  const heatmapAvailable = Boolean(result?.heatmap?.available);

  function resetReviewState() {
    setResult(null);
    setProcessedImage(null);
    setRoiSelections([]);
    setRoiMode(null);
    setError('');
  }

  function handleNewCase() {
    resetReviewState();
    setXray(null);
    setMri(null);
    setMriR2(null);
    setAlignment(null);
    setZoom(100);
    setEnableCrop(true);
    setShowGuides(true);
    setShowHeatmap(true);
    setShowRoi(true);
  }

  async function handleRun() {
    if (!xray || loading) return;
    setLoading(true);
    setError('');
    try {
      // The raw upload is never sent: the canvas re-renders the aligned frame onto an
      // offscreen 384x384 PNG and that file is what goes to the model.
      const processed = await canvasRef.current?.exportProcessedImage();
      if (!processed) throw new Error('Processed image export failed');
      const inferenceImage = new File([processed.blob], processed.filename, { type: 'image/png' });
      const next = await runInference({ mode, profileId, xray: inferenceImage, mri, mriR2 });
      setProcessedImage(processed);
      setResult(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
    } finally {
      setLoading(false);
    }
  }

  function handleFeedback(item) {
    const feedbackLayer = canvasRef.current?.exportFeedbackImage?.() || null;
    const entry = {
      feedbackId: makeFeedbackId(),
      requestId: result?.requestId || null,
      doctorId: 'local-reviewer',
      doctorDecision: item.doctorDecision,
      correctedGrade: item.correctedGrade,
      roiSelections: feedbackLayer?.selections || canvasRef.current?.serializeSelections() || [],
      comments: item.comments || '',
      timestamp: new Date().toISOString(),
      modelProfile: profileId,
      inferenceMode: mode,
      xrayFilename: xray?.name || null,
      mriFilename: mri?.name || null,
      mriR2Filename: mriR2?.name || null,
      processedImageFilename: processedImage?.filename || null,
      processedImageDataUrl: processedImage?.dataUrl || null,
      feedbackImageFilename: feedbackLayer?.filename || null,
      feedbackImageDataUrl: feedbackLayer?.dataUrl || null,
      alignment: processedImage?.alignment || alignment,
      predictedGrade: result?.prediction?.klGrade ?? null,
      classProbs: result?.prediction?.classProbs || result?.classProbs || [],
      route: result?.route || null,
      metadata: result?.metadata || null,
    };
    setFeedbackItems((items) => [...items, entry]);
  }

  function handleClearQueue() {
    if (feedbackItems.length === 0) return;
    const confirmed = window.confirm(
      `Delete all ${feedbackItems.length} locally stored feedback entries? Export first if you need them.`
    );
    if (confirmed) setFeedbackItems([]);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <p className="eyebrow">KOA Multimodal v7</p>
          <h1>Clinical Inference Workbench</h1>
        </div>
        <div className="topbar-actions">
          <span className="pill pill-mode">{MODE_LABELS[mode] || mode}</span>
          <button
            type="button"
            className="ghost-button"
            disabled={feedbackItems.length === 0}
            onClick={() => exportFeedback(feedbackItems)}
          >
            Export feedback
          </button>
          <button
            type="button"
            className="ghost-button"
            disabled={feedbackItems.length === 0}
            onClick={handleClearQueue}
          >
            Clear queue
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <aside className="control-panel">
          <section className="panel-block">
            <h2>Study inputs</h2>
            <FileSlot
              label="X-ray"
              required
              file={xray}
              onChange={setXray}
              accept="image/png,image/jpeg"
              hint="PNG or JPEG. Re-rendered to a 384x384 frame before inference."
            />
            <FileSlot
              label="MRI T2"
              file={mri}
              onChange={setMri}
              accept=".nii,.nii.gz,image/png,image/jpeg"
            />
            <FileSlot
              label="MRI R2"
              file={mriR2}
              onChange={setMriR2}
              accept=".nii,.nii.gz,image/png,image/jpeg"
            />
            {mriStatus.state === 'partial' ? (
              <p className="status-note">{mriStatus.message}</p>
            ) : null}
            {mriStatus.state === 'none' && profileWantsMri ? (
              <p className="status-line status-muted">
                This profile uses MRI. Without a T2 + R2 pair it falls back to the X-ray route.
              </p>
            ) : null}
          </section>

          <section className="panel-block">
            <h2>Model</h2>
            <label className="field">
              <span className="field-label">Inference mode</span>
              <select value={mode} onChange={(event) => setMode(event.target.value)}>
                {MODES.map((item) => (
                  <option key={item} value={item}>
                    {MODE_LABELS[item] || item}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span className="field-label">Profile</span>
              <select value={profileId} onChange={(event) => setProfileId(event.target.value)}>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name || profile.id}
                  </option>
                ))}
              </select>
            </label>
            <p className="roi-status">
              {profileSource === 'api'
                ? 'Profiles loaded from the API.'
                : profileSource === 'fallback'
                  ? 'API unreachable. Using the built-in fallback profile list.'
                  : 'Loading profiles...'}
            </p>
          </section>

          <section className="panel-block">
            <h2>Alignment</h2>
            <label className="range-control">
              <span>Zoom</span>
              <input
                type="range"
                min="60"
                max="180"
                step="5"
                value={zoom}
                onChange={(event) => setZoom(Number(event.target.value))}
                disabled={!xray || !!result}
              />
              <strong>{zoom}%</strong>
            </label>
            <div className="check-row">
              <label>
                <input
                  type="checkbox"
                  checked={enableCrop}
                  onChange={(event) => setEnableCrop(event.target.checked)}
                  disabled={!!result}
                />
                Crop frame
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showGuides}
                  onChange={(event) => setShowGuides(event.target.checked)}
                />
                Guides
              </label>
            </div>
          </section>

          <section className="panel-block">
            <h2>Overlays</h2>
            <div className="check-row">
              <label className={!heatmapAvailable ? 'is-disabled' : undefined}>
                <input
                  type="checkbox"
                  checked={showHeatmap && heatmapAvailable}
                  onChange={(event) => setShowHeatmap(event.target.checked)}
                  disabled={!heatmapAvailable}
                />
                Heatmap
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showRoi}
                  onChange={(event) => setShowRoi(event.target.checked)}
                />
                ROI
              </label>
            </div>
            {result && !heatmapAvailable ? (
              <p className="status-line status-muted">{result.heatmap?.reason}</p>
            ) : null}
          </section>

          <div className="run-actions">
            <button
              type="button"
              className="primary-button"
              disabled={!xray || loading}
              onClick={handleRun}
            >
              {loading ? 'Running inference...' : 'Run inference'}
            </button>
            <div className="case-actions">
              <button
                type="button"
                className="ghost-button"
                disabled={!result || loading}
                onClick={resetReviewState}
              >
                Revise alignment
              </button>
              <button
                type="button"
                className="ghost-button"
                disabled={loading || (!xray && !mri && !mriR2 && !result)}
                onClick={handleNewCase}
              >
                New case
              </button>
            </div>
            {error ? <p className="error-text">{error}</p> : null}
          </div>
        </aside>

        <section className="viewer-panel">
          <AlignmentCanvas
            ref={canvasRef}
            xray={xray}
            heatmap={heatmapAvailable ? result.heatmap.dataUrl : null}
            showHeatmap={showHeatmap && heatmapAvailable}
            zoom={zoom}
            enableCrop={enableCrop}
            showGuides={showGuides}
            roiMode={showRoi ? roiMode : null}
            selections={showRoi ? roiSelections : []}
            onSelectionsChange={setRoiSelections}
            onAlignmentChange={setAlignment}
            hasInference={!!result}
          />
        </section>

        <aside className="result-panel">
          <ResultPanel result={result} loading={loading} />
          <FeedbackPanel
            disabled={!result}
            onSubmit={handleFeedback}
            count={feedbackItems.length}
            roiMode={roiMode}
            onRoiModeChange={setRoiMode}
            onClearSelections={() => setRoiSelections([])}
            selectionsCount={roiSelections.length}
            storage={storage}
            storageError={storageError}
          />
        </aside>
      </main>
    </div>
  );
}
