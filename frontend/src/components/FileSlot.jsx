import React, { useEffect, useId, useState } from 'react';

export default function FileSlot({ label, required = false, file, onChange, accept, hint }) {
  const [preview, setPreview] = useState('');
  const inputId = useId();

  useEffect(() => {
    if (!file || !file.type?.startsWith('image/')) {
      setPreview('');
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className={`file-slot${file ? ' is-filled' : ''}`}>
      <div className="file-slot-header">
        <label htmlFor={inputId}>{label}</label>
        {required ? <strong>required</strong> : <em>optional</em>}
      </div>
      <label className="file-drop" htmlFor={inputId}>
        <input
          id={inputId}
          type="file"
          accept={accept}
          onChange={(event) => {
            onChange(event.target.files?.[0] || null);
            event.target.value = '';
          }}
        />
        {preview ? (
          <img src={preview} alt={`${label} preview`} />
        ) : (
          <span>{file?.name || 'Select file'}</span>
        )}
      </label>
      {hint && !file ? <p className="file-hint">{hint}</p> : null}
      {file ? (
        <div className="file-slot-footer">
          <span className="file-name" title={file.name}>
            {file.name}
          </span>
          <button type="button" className="text-button" onClick={() => onChange(null)}>
            Clear
          </button>
        </div>
      ) : null}
    </div>
  );
}
