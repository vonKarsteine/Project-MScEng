import React from 'react';
import ClassProbabilityBars from './ClassProbabilityBars.jsx';

function formatNumber(value, digits = 3) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}

function formatText(value) {
  return value === null || value === undefined || value === '' ? '-' : String(value);
}

export default function ResultPanel({ result, loading }) {
  const prediction = result?.prediction;
  const route = result?.route;
  const metadata = result?.metadata;
  const heatmap = result?.heatmap;
  const mriStatus = result?.mriStatus;

  const executedProfile = metadata?.executedProfile ?? route?.routeId;
  const requestedProfile = metadata?.modelProfile;
  const reroutedByFallback =
    requestedProfile && executedProfile && requestedProfile !== executedProfile;
  const switched = Boolean(route?.switchFlag);

  return (
    <section className="panel-block result-block">
      <div className="panel-heading">
        <h2>Clinical result</h2>
        {loading ? <span className="pill pill-live">running</span> : null}
      </div>

      <div className="grade-display">
        <span>Kellgren-Lawrence</span>
        <strong>{loading ? '--' : (prediction?.klGrade ?? '--')}</strong>
      </div>

      <ClassProbabilityBars classProbs={result?.classProbs} predictedGrade={prediction?.klGrade} />

      <dl className="metric-list">
        <div>
          <dt>Confidence</dt>
          <dd>{formatNumber(prediction?.confidence)}</dd>
        </div>
        <div>
          <dt>Uncertainty</dt>
          <dd>{formatNumber(prediction?.uncertainty)}</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd>{typeof result?.latencyMs === 'number' ? `${result.latencyMs} ms` : '-'}</dd>
        </div>
        <div>
          <dt>MRI used</dt>
          <dd>{prediction ? (route?.mriUsed ? 'yes' : 'no') : '-'}</dd>
        </div>
      </dl>

      <div className="detail-section">
        <h3>Execution</h3>
        <dl className="detail-list">
          <div>
            <dt>Runtime</dt>
            <dd>{formatText(metadata?.runtime ?? route?.runtime)}</dd>
          </div>
          <div>
            <dt>Quantization</dt>
            <dd>{formatText(metadata?.quantization)}</dd>
          </div>
          <div>
            <dt>Requested profile</dt>
            <dd>{formatText(requestedProfile)}</dd>
          </div>
          <div>
            <dt>Executed profile</dt>
            <dd className={reroutedByFallback ? 'is-diverged' : undefined}>
              {formatText(executedProfile)}
            </dd>
          </div>
          {switched ? (
            <>
              <div>
                <dt>Route switch</dt>
                <dd>
                  <span className="pill pill-switch">C-MODES switched</span>
                </dd>
              </div>
              <div>
                <dt>Switch score</dt>
                <dd>{formatNumber(route?.switchScore)}</dd>
              </div>
            </>
          ) : null}
        </dl>
      </div>

      {result ? (
        <div className="detail-section">
          <h3>Saliency</h3>
          {heatmap?.available ? (
            <p className="status-line status-ok">
              Overlay active &middot; <code>{heatmap.method}</code>
            </p>
          ) : (
            <p className="status-line status-muted">{heatmap?.reason || 'No saliency map available.'}</p>
          )}
        </div>
      ) : null}

      {route?.missingModalityFallback ? (
        <p className="status-note">
          Missing-modality fallback active. The paired MRI route was unavailable for this case, so the
          X-ray route produced this grade.
        </p>
      ) : null}

      {mriStatus?.state === 'partial' ? (
        <p className="status-note">{mriStatus.message}</p>
      ) : null}

      {!result && !loading ? (
        <p className="status-line status-muted">
          No inference yet. Align the X-ray and run the selected profile.
        </p>
      ) : null}
    </section>
  );
}
