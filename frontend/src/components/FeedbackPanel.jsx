import React, { useState } from 'react';
import { formatBytes } from '../utils/feedback.js';

const ROI_TOOLS = [
  ['rect', 'Rect'],
  ['polygon', 'Polygon'],
  ['eraser', 'Erase'],
];

export default function FeedbackPanel({
  disabled,
  onSubmit,
  count,
  roiMode,
  onRoiModeChange,
  onClearSelections,
  selectionsCount,
  storage,
  storageError,
}) {
  const [doctorDecision, setDoctorDecision] = useState('approve');
  const [correctedGrade, setCorrectedGrade] = useState('');
  const [comments, setComments] = useState('');

  function submit() {
    onSubmit({
      doctorDecision,
      correctedGrade:
        doctorDecision === 'approve' || correctedGrade === '' ? undefined : Number(correctedGrade),
      comments,
    });
    setComments('');
    setCorrectedGrade('');
  }

  return (
    <section className="panel-block">
      <div className="panel-heading">
        <h2>Reviewer feedback</h2>
        <span className="pill">{count} saved</span>
      </div>

      <div className="field">
        <span className="field-label">Decision</span>
        <div className="segmented-control" role="group" aria-label="Reviewer decision">
          {['approve', 'reject'].map((item) => (
            <button
              type="button"
              key={item}
              className={item === doctorDecision ? 'active' : ''}
              aria-pressed={item === doctorDecision}
              onClick={() => setDoctorDecision(item)}
              disabled={disabled}
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      <label className="field">
        <span className="field-label">Corrected grade</span>
        <select
          value={correctedGrade}
          onChange={(event) => setCorrectedGrade(event.target.value)}
          disabled={disabled || doctorDecision === 'approve'}
        >
          <option value="">No correction</option>
          {[0, 1, 2, 3, 4].map((grade) => (
            <option key={grade} value={grade}>
              KL{grade}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        <span className="field-label">Clinical note</span>
        <textarea
          value={comments}
          onChange={(event) => setComments(event.target.value)}
          placeholder="Observations, disagreement rationale, image quality"
          disabled={disabled}
        />
      </label>

      <div className="roi-tools">
        <h3>ROI annotation</h3>
        <div className="roi-toolbar" role="group" aria-label="ROI annotation tool">
          {ROI_TOOLS.map(([value, label]) => (
            <button
              type="button"
              key={value}
              className={`roi-btn ${roiMode === value ? 'active' : ''}`}
              aria-pressed={roiMode === value}
              disabled={disabled}
              onClick={() => onRoiModeChange(roiMode === value ? null : value)}
            >
              {label}
            </button>
          ))}
          <button
            type="button"
            className="roi-btn"
            disabled={disabled || selectionsCount === 0}
            onClick={onClearSelections}
          >
            Clear
          </button>
        </div>
        <p className="roi-status">
          {selectionsCount} region{selectionsCount === 1 ? '' : 's'} marked on the processed frame
        </p>
      </div>

      <button type="button" className="primary-button" disabled={disabled} onClick={submit}>
        Add to feedback queue
      </button>

      {storageError ? <p className="error-text">{storageError}</p> : null}
      {!storageError && storage?.level === 'warn' ? (
        <p className="status-note">
          Local feedback store at {formatBytes(storage.bytes)} of the ~5 MB browser limit. Entries embed
          full-resolution PNGs; export and clear the queue soon.
        </p>
      ) : null}
      {!storageError && storage?.level === 'ok' && count > 0 ? (
        <p className="roi-status">{formatBytes(storage.bytes)} stored locally, nothing uploaded.</p>
      ) : null}
    </section>
  );
}
