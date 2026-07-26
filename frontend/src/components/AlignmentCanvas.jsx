import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Line, Circle, Transformer, Text } from 'react-konva';

/**
 * The model never sees the raw upload. The X-ray is re-rendered through this canvas
 * onto an offscreen OUTPUT_SIZE x OUTPUT_SIZE PNG and *that* is what is POSTed, so the
 * server receives exactly the frame the reviewer aligned.
 *
 * OUTPUT_SIZE therefore has two jobs at once: it is the pixel size of that exported
 * PNG, and it is the coordinate space of the `*Px` ROI values written into feedback
 * entries. Changing it moves both together -- ROI pixel coordinates are only
 * meaningful against the resolution of the frame they were drawn on.
 *
 * The heatmap overlay is deliberately unaffected: it is stretched over the on-screen
 * crop region at whatever resolution the backend returns.
 */
const OUTPUT_SIZE = 384;

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function useContainerSize(ref) {
  const [size, setSize] = useState({ width: 720, height: 560 });

  useEffect(() => {
    function update() {
      if (!ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      setSize({
        width: Math.max(320, Math.floor(rect.width)),
        height: Math.max(360, Math.floor(rect.height)),
      });
    }

    update();
    window.addEventListener('resize', update);
    if (typeof ResizeObserver === 'undefined') {
      return () => window.removeEventListener('resize', update);
    }

    const observer = new ResizeObserver(update);
    if (ref.current) observer.observe(ref.current);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', update);
    };
  }, [ref]);

  return size;
}

function useObjectUrl(file) {
  const [url, setUrl] = useState('');

  useEffect(() => {
    if (!file || !file.type?.startsWith('image/')) {
      setUrl('');
      return undefined;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);

  return url;
}

function useHtmlImage(src) {
  const [image, setImage] = useState(null);

  useEffect(() => {
    if (!src) {
      setImage(null);
      return undefined;
    }
    let cancelled = false;
    const next = new window.Image();
    next.onload = () => {
      if (!cancelled) setImage(next);
    };
    next.onerror = () => {
      if (!cancelled) setImage(null);
    };
    next.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);

  return image;
}

function fitImageBox(image, size, zoom) {
  if (!image) return null;
  const scale =
    Math.min((size.width * 0.86) / image.width, (size.height * 0.86) / image.height) * (zoom / 100);
  const width = image.width * scale;
  const height = image.height * scale;
  return {
    x: (size.width - width) / 2,
    y: (size.height - height) / 2,
    width,
    height,
  };
}

function activeRegion(size, cropRegion, enableCrop) {
  if (enableCrop) return cropRegion;
  return { x: 0, y: 0, width: size.width, height: size.height };
}

function normalizeRect(rect, region) {
  const x1 = clamp((rect.x - region.x) / region.width);
  const y1 = clamp((rect.y - region.y) / region.height);
  const x2 = clamp((rect.x + rect.width - region.x) / region.width);
  const y2 = clamp((rect.y + rect.height - region.y) / region.height);
  return {
    type: 'rect',
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1),
  };
}

function normalizePolygon(selection, region) {
  const points = [];
  for (let index = 0; index < selection.points.length; index += 2) {
    points.push([
      clamp((selection.points[index] - region.x) / region.width),
      clamp((selection.points[index + 1] - region.y) / region.height),
    ]);
  }
  return { type: 'polygon', points };
}

function toOutputRect(normalized) {
  return {
    ...normalized,
    xPx: Math.round(normalized.x * OUTPUT_SIZE),
    yPx: Math.round(normalized.y * OUTPUT_SIZE),
    widthPx: Math.round(normalized.width * OUTPUT_SIZE),
    heightPx: Math.round(normalized.height * OUTPUT_SIZE),
  };
}

function toOutputPolygon(normalized) {
  return {
    ...normalized,
    pointsPx: normalized.points.map(([x, y]) => [
      Math.round(x * OUTPUT_SIZE),
      Math.round(y * OUTPUT_SIZE),
    ]),
  };
}

function selectionToOutput(selection, region) {
  if (selection.type === 'rect') return toOutputRect(normalizeRect(selection, region));
  return toOutputPolygon(normalizePolygon(selection, region));
}

function drawProcessedFrame(context, image, imageBox, region) {
  context.fillStyle = '#000000';
  context.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  const scaleX = OUTPUT_SIZE / region.width;
  const scaleY = OUTPUT_SIZE / region.height;
  context.drawImage(
    image,
    (imageBox.x - region.x) * scaleX,
    (imageBox.y - region.y) * scaleY,
    imageBox.width * scaleX,
    imageBox.height * scaleY
  );
}

function drawSelectionOverlay(context, outputSelection) {
  context.save();
  context.lineWidth = 3;
  context.strokeStyle = '#ef4444';
  context.fillStyle = 'rgba(239, 68, 68, 0.18)';
  context.setLineDash([8, 4]);

  if (outputSelection.type === 'rect') {
    context.fillRect(
      outputSelection.xPx,
      outputSelection.yPx,
      outputSelection.widthPx,
      outputSelection.heightPx
    );
    context.strokeRect(
      outputSelection.xPx,
      outputSelection.yPx,
      outputSelection.widthPx,
      outputSelection.heightPx
    );
  } else if (outputSelection.pointsPx.length >= 3) {
    context.beginPath();
    outputSelection.pointsPx.forEach(([x, y], index) => {
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fill();
    context.stroke();
  }

  context.restore();
}

function safeBaseName(name) {
  return (
    (name || 'xray')
      .replace(/\.[^.]+$/, '')
      .replace(/[^a-zA-Z0-9_-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 80) || 'xray'
  );
}

const AlignmentCanvas = forwardRef(function AlignmentCanvas(
  {
    xray,
    heatmap,
    showHeatmap,
    zoom,
    enableCrop,
    showGuides,
    roiMode,
    selections,
    onSelectionsChange,
    onAlignmentChange,
    hasInference,
  },
  ref
) {
  const containerRef = useRef(null);
  const stageRef = useRef(null);
  const imageRef = useRef(null);
  const transformerRef = useRef(null);
  const selectionRefs = useRef([]);
  const size = useContainerSize(containerRef);
  const imageUrl = useObjectUrl(xray);
  const xrayImage = useHtmlImage(imageUrl);
  const heatmapImage = useHtmlImage(heatmap);

  const [imageBox, setImageBox] = useState(null);
  const [scaleEnabled, setScaleEnabled] = useState(false);
  const [dragEnabled, setDragEnabled] = useState(true);
  const [isDrawing, setIsDrawing] = useState(false);
  const [selectionStart, setSelectionStart] = useState(null);
  const [currentSelection, setCurrentSelection] = useState(null);

  const cropRegion = useMemo(
    () => ({
      x: size.width * 0.15,
      y: size.height * 0.1,
      width: size.width * 0.7,
      height: size.height * 0.8,
    }),
    [size]
  );

  const outputRegion = useMemo(
    () => activeRegion(size, cropRegion, enableCrop),
    [cropRegion, enableCrop, size]
  );

  useEffect(() => {
    setImageBox(fitImageBox(xrayImage, size, zoom));
  }, [xrayImage, size, zoom]);

  useEffect(() => {
    if (onAlignmentChange) {
      onAlignmentChange({
        imageBox,
        cropRegion,
        activeRegion: outputRegion,
        canvasSize: size,
        outputSize: { width: OUTPUT_SIZE, height: OUTPUT_SIZE },
        cropEnabled: enableCrop,
      });
    }
  }, [cropRegion, enableCrop, imageBox, onAlignmentChange, outputRegion, size]);

  useEffect(() => {
    if (transformerRef.current && imageRef.current && scaleEnabled && xrayImage && !hasInference) {
      transformerRef.current.nodes([imageRef.current]);
      transformerRef.current.getLayer()?.batchDraw();
    } else if (transformerRef.current) {
      transformerRef.current.nodes([]);
    }
  }, [hasInference, scaleEnabled, xrayImage]);

  useImperativeHandle(ref, () => ({
    async exportProcessedImage() {
      if (!xrayImage || !imageBox) throw new Error('X-ray image is not ready');
      const canvas = document.createElement('canvas');
      canvas.width = OUTPUT_SIZE;
      canvas.height = OUTPUT_SIZE;
      const context = canvas.getContext('2d');
      drawProcessedFrame(context, xrayImage, imageBox, outputRegion);
      const dataUrl = canvas.toDataURL('image/png');
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('Processed image export failed');
      return {
        blob,
        dataUrl,
        filename: `processed-${safeBaseName(xray?.name)}.png`,
        alignment: {
          imageBox,
          cropRegion,
          activeRegion: outputRegion,
          canvasSize: size,
          outputSize: { width: OUTPUT_SIZE, height: OUTPUT_SIZE },
          cropEnabled: enableCrop,
        },
      };
    },
    serializeSelections() {
      return (selections || []).map((selection) => selectionToOutput(selection, outputRegion));
    },
    exportFeedbackImage() {
      if (!xrayImage || !imageBox) return null;
      const outputSelections = (selections || []).map((selection) =>
        selectionToOutput(selection, outputRegion)
      );
      const canvas = document.createElement('canvas');
      canvas.width = OUTPUT_SIZE;
      canvas.height = OUTPUT_SIZE;
      const context = canvas.getContext('2d');
      drawProcessedFrame(context, xrayImage, imageBox, outputRegion);
      outputSelections.forEach((selection) => drawSelectionOverlay(context, selection));
      return {
        dataUrl: canvas.toDataURL('image/png'),
        filename: `feedback-${safeBaseName(xray?.name)}.png`,
        selections: outputSelections,
      };
    },
  }));

  const updateImageBoxFromNode = useCallback(() => {
    const node = imageRef.current;
    if (!node) return;
    const nextBox = {
      x: node.x(),
      y: node.y(),
      width: Math.max(32, node.width() * node.scaleX()),
      height: Math.max(32, node.height() * node.scaleY()),
    };
    node.scaleX(1);
    node.scaleY(1);
    setImageBox(nextBox);
  }, []);

  const stagePoint = useCallback((event) => event.target.getStage()?.getPointerPosition(), []);

  const handleMouseDown = useCallback(
    (event) => {
      if (!hasInference || !roiMode) return;
      const point = stagePoint(event);
      if (!point) return;

      if (roiMode === 'eraser') {
        const name = event.target.name();
        if (name.startsWith('selection-')) {
          const index = Number(name.replace('selection-', ''));
          onSelectionsChange((selections || []).filter((_, itemIndex) => itemIndex !== index));
        }
        return;
      }

      if (roiMode === 'rect') {
        setIsDrawing(true);
        setSelectionStart(point);
        setCurrentSelection({ type: 'rect', x: point.x, y: point.y, width: 0, height: 0 });
      }

      if (roiMode === 'polygon') {
        setIsDrawing(true);
        setCurrentSelection((previous) => {
          const points = previous?.points || [];
          return { type: 'polygon', points: [...points, point.x, point.y] };
        });
      }
    },
    [hasInference, onSelectionsChange, roiMode, selections, stagePoint]
  );

  const handleMouseMove = useCallback(
    (event) => {
      if (!isDrawing || roiMode !== 'rect' || !selectionStart) return;
      const point = stagePoint(event);
      if (!point) return;
      setCurrentSelection({
        type: 'rect',
        x: Math.min(selectionStart.x, point.x),
        y: Math.min(selectionStart.y, point.y),
        width: Math.abs(point.x - selectionStart.x),
        height: Math.abs(point.y - selectionStart.y),
      });
    },
    [isDrawing, roiMode, selectionStart, stagePoint]
  );

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || roiMode !== 'rect') return;
    if (currentSelection && currentSelection.width > 8 && currentSelection.height > 8) {
      onSelectionsChange([...(selections || []), currentSelection]);
    }
    setIsDrawing(false);
    setSelectionStart(null);
    setCurrentSelection(null);
  }, [currentSelection, isDrawing, onSelectionsChange, roiMode, selections]);

  const handleContextMenu = useCallback(
    (event) => {
      event.evt.preventDefault();
      if (roiMode === 'polygon' && currentSelection?.points?.length >= 6) {
        onSelectionsChange([...(selections || []), currentSelection]);
        setCurrentSelection(null);
        setIsDrawing(false);
      }
    },
    [currentSelection, onSelectionsChange, roiMode, selections]
  );

  const handleSelectionDragEnd = useCallback(
    (index) => {
      const node = selectionRefs.current[index];
      if (!node) return;
      const updated = [...(selections || [])];
      if (updated[index]?.type === 'rect') {
        updated[index] = { ...updated[index], x: node.x(), y: node.y() };
      } else if (updated[index]?.type === 'polygon') {
        const dx = node.x();
        const dy = node.y();
        const points = updated[index].points.map(
          (value, pointIndex) => value + (pointIndex % 2 === 0 ? dx : dy)
        );
        updated[index] = { ...updated[index], points };
        node.position({ x: 0, y: 0 });
      }
      onSelectionsChange(updated);
    },
    [onSelectionsChange, selections]
  );

  if (!xrayImage || !imageBox) {
    return (
      <div className="canvas-container" ref={containerRef}>
        <div className="canvas-placeholder">
          <svg className="empty-icon" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M3 15l5-5 4 4 3-3 6 6" />
          </svg>
          <p>Upload a knee X-ray to begin alignment</p>
          <p className="canvas-placeholder-sub">
            The framed region is exported at {OUTPUT_SIZE}&times;{OUTPUT_SIZE} and sent to the model.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="canvas-container" ref={containerRef}>
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onContextMenu={handleContextMenu}
      >
        <Layer>
          <KonvaImage
            ref={imageRef}
            image={xrayImage}
            x={imageBox.x}
            y={imageBox.y}
            width={imageBox.width}
            height={imageBox.height}
            draggable={dragEnabled && !hasInference}
            onDragEnd={updateImageBoxFromNode}
            onTransformEnd={updateImageBoxFromNode}
          />
          {heatmapImage && showHeatmap ? (
            <KonvaImage
              image={heatmapImage}
              x={outputRegion.x}
              y={outputRegion.y}
              width={outputRegion.width}
              height={outputRegion.height}
              opacity={0.72}
              listening={false}
            />
          ) : null}
          {scaleEnabled && !hasInference ? (
            <Transformer
              ref={transformerRef}
              rotateEnabled={false}
              centeredScaling
              enabledAnchors={['top-left', 'top-right', 'bottom-left', 'bottom-right']}
              borderStroke="#ffffff"
              anchorStroke="#ffffff"
              anchorFill="#000000"
              anchorSize={10}
            />
          ) : null}
        </Layer>

        <Layer listening={false}>
          {enableCrop ? (
            <>
              <Rect x={0} y={0} width={size.width} height={cropRegion.y} fill="rgba(0,0,0,0.72)" />
              <Rect
                x={0}
                y={cropRegion.y + cropRegion.height}
                width={size.width}
                height={size.height - cropRegion.y - cropRegion.height}
                fill="rgba(0,0,0,0.72)"
              />
              <Rect
                x={0}
                y={cropRegion.y}
                width={cropRegion.x}
                height={cropRegion.height}
                fill="rgba(0,0,0,0.72)"
              />
              <Rect
                x={cropRegion.x + cropRegion.width}
                y={cropRegion.y}
                width={size.width - cropRegion.x - cropRegion.width}
                height={cropRegion.height}
                fill="rgba(0,0,0,0.72)"
              />
            </>
          ) : null}
          {showGuides ? (
            <>
              <Rect
                x={outputRegion.x}
                y={outputRegion.y}
                width={outputRegion.width}
                height={outputRegion.height}
                stroke="#10b981"
                strokeWidth={2}
                dash={[8, 4]}
                opacity={0.8}
              />
              <Line
                points={[
                  outputRegion.x + outputRegion.width / 2 - 18,
                  outputRegion.y + outputRegion.height / 2,
                  outputRegion.x + outputRegion.width / 2 + 18,
                  outputRegion.y + outputRegion.height / 2,
                ]}
                stroke="#10b981"
                strokeWidth={1}
                opacity={0.7}
              />
              <Line
                points={[
                  outputRegion.x + outputRegion.width / 2,
                  outputRegion.y + outputRegion.height / 2 - 18,
                  outputRegion.x + outputRegion.width / 2,
                  outputRegion.y + outputRegion.height / 2 + 18,
                ]}
                stroke="#10b981"
                strokeWidth={1}
                opacity={0.7}
              />
              <Text
                x={outputRegion.x}
                y={Math.max(8, outputRegion.y - 22)}
                text={`Processed inference frame (${OUTPUT_SIZE}px)`}
                fontSize={11}
                fill="#10b981"
                fontFamily="system-ui, sans-serif"
                opacity={0.9}
              />
            </>
          ) : null}
        </Layer>

        <Layer>
          {(selections || []).map((selection, index) =>
            selection.type === 'rect' ? (
              <Rect
                key={index}
                ref={(node) => {
                  selectionRefs.current[index] = node;
                }}
                name={`selection-${index}`}
                x={selection.x}
                y={selection.y}
                width={selection.width}
                height={selection.height}
                stroke="#ef4444"
                strokeWidth={2}
                dash={[4, 2]}
                fill="rgba(239, 68, 68, 0.14)"
                draggable={!roiMode}
                onDragEnd={() => handleSelectionDragEnd(index)}
              />
            ) : (
              <Line
                key={index}
                ref={(node) => {
                  selectionRefs.current[index] = node;
                }}
                name={`selection-${index}`}
                points={selection.points}
                stroke="#ef4444"
                strokeWidth={2}
                closed
                fill="rgba(239, 68, 68, 0.14)"
                draggable={!roiMode}
                onDragEnd={() => handleSelectionDragEnd(index)}
              />
            )
          )}
          {currentSelection?.type === 'rect' ? (
            <Rect
              x={currentSelection.x}
              y={currentSelection.y}
              width={currentSelection.width}
              height={currentSelection.height}
              stroke="#ef4444"
              strokeWidth={2}
              dash={[4, 2]}
              fill="rgba(239, 68, 68, 0.14)"
            />
          ) : null}
          {currentSelection?.type === 'polygon' ? (
            <>
              <Line
                points={currentSelection.points}
                stroke="#ef4444"
                strokeWidth={2}
                fill="rgba(239, 68, 68, 0.1)"
              />
              {Array.from({ length: currentSelection.points.length / 2 }).map((_, index) => (
                <Circle
                  key={index}
                  x={currentSelection.points[index * 2]}
                  y={currentSelection.points[index * 2 + 1]}
                  radius={4}
                  fill="#ef4444"
                  stroke="#ffffff"
                  strokeWidth={1}
                />
              ))}
            </>
          ) : null}
        </Layer>
      </Stage>

      <div className="canvas-toolbar">
        <button
          type="button"
          className={`toolbar-btn ${dragEnabled ? 'active' : ''}`}
          disabled={hasInference}
          aria-pressed={dragEnabled}
          onClick={() => setDragEnabled((value) => !value)}
          title="Drag the X-ray inside the frame"
        >
          Move
        </button>
        <button
          type="button"
          className={`toolbar-btn ${scaleEnabled ? 'active' : ''}`}
          disabled={hasInference}
          aria-pressed={scaleEnabled}
          onClick={() => setScaleEnabled((value) => !value)}
          title="Resize the X-ray with corner handles"
        >
          Scale
        </button>
      </div>
      <div className="canvas-hint">
        {hasInference
          ? roiMode
            ? 'Draw or erase ROI feedback on the processed frame. Right-click closes a polygon.'
            : 'Drag existing ROI selections to refine feedback'
          : 'Drag or scale the image, then run inference on the framed crop'}
      </div>
    </div>
  );
});

export default AlignmentCanvas;
