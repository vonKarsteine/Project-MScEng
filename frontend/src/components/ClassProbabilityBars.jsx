import React from 'react';

const KL_LABELS = ['KL0', 'KL1', 'KL2', 'KL3', 'KL4'];
const KL_DESCRIPTIONS = [
  'none',
  'doubtful',
  'minimal',
  'moderate',
  'severe',
];

function formatPercent(value) {
  const percent = value * 100;
  if (percent > 0 && percent < 0.1) return '<0.1%';
  return `${percent.toFixed(1)}%`;
}

export default function ClassProbabilityBars({ classProbs, predictedGrade }) {
  const probs = Array.isArray(classProbs) && classProbs.length === 5 ? classProbs : null;

  if (!probs) {
    return (
      <div className="prob-bars prob-bars-empty">
        <p>Run inference to see the ordinal grade distribution.</p>
      </div>
    );
  }

  const argmax = probs.reduce((best, value, index) => (value > probs[best] ? index : best), 0);
  const highlighted = typeof predictedGrade === 'number' ? predictedGrade : argmax;
  const peak = Math.max(...probs, 1e-6);

  return (
    <div className="prob-bars" role="list" aria-label="Predicted probability per Kellgren-Lawrence grade">
      {probs.map((value, index) => {
        const isPredicted = index === highlighted;
        return (
          <div
            className={`prob-row${isPredicted ? ' is-predicted' : ''}`}
            role="listitem"
            key={KL_LABELS[index]}
          >
            <span className="prob-label">
              {KL_LABELS[index]}
              <span className="prob-label-note"> {KL_DESCRIPTIONS[index]}</span>
            </span>
            <span className="prob-track">
              {/* Bars are scaled against the peak so low-entropy distributions stay legible;
                  the printed percentage is always the absolute probability. */}
              <span className="prob-fill" style={{ width: `${Math.max(1.5, (value / peak) * 100)}%` }} />
            </span>
            <span className="prob-value">
              {formatPercent(value)}
              {isPredicted ? <span className="sr-only"> predicted grade</span> : null}
            </span>
          </div>
        );
      })}
    </div>
  );
}
