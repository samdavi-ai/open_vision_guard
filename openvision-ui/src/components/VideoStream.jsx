import React, { useEffect, useRef, useState, useCallback } from 'react';

/**
 * VideoStream renders the annotated JPEG frames and overlays invisible
 * click-targets on top of each detected person (for click-to-analyze).
 *
 * KEY FIX: The image uses objectFit:contain so there are black letterbox bars.
 * Click-targets must be offset to the actual rendered image position, not the
 * container top-left.
 */
export default function VideoStream({ wsUrl, setFps, onPersonClick }) {
  const [frameSrc, setFrameSrc]   = useState(null);
  const [detections, setDetections] = useState([]);
  const [dims, setDims]            = useState({ w: 1, h: 1 });
  const [imgBox, setImgBox]        = useState(null); // rendered image bounds relative to container

  const wsRef        = useRef(null);
  const containerRef = useRef(null);
  const imgRef       = useRef(null);
  const fpsCount     = useRef(0);

  /* ── FPS counter ── */
  useEffect(() => {
    const id = setInterval(() => { setFps(fpsCount.current); fpsCount.current = 0; }, 1000);
    return () => clearInterval(id);
  }, [setFps]);

  /* ── WebSocket ── */
  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      fpsCount.current++;
      setFrameSrc(`data:image/jpeg;base64,${d.frame}`);
      setDims({ w: d.width || 1, h: d.height || 1 });
      setDetections(d.detections || []);
    };
    return () => ws.close();
  }, [wsUrl]);

  /* ── Measure actual image bounds within container (handles letterbox) ── */
  const measureImage = useCallback(() => {
    const img = imgRef.current;
    const cnt = containerRef.current;
    if (!img || !cnt) return;

    const cRect = cnt.getBoundingClientRect();
    const iRect = img.getBoundingClientRect();

    setImgBox({
      left:   iRect.left - cRect.left,
      top:    iRect.top  - cRect.top,
      width:  iRect.width,
      height: iRect.height,
    });
  }, []);

  useEffect(() => {
    window.addEventListener('resize', measureImage);
    return () => window.removeEventListener('resize', measureImage);
  }, [measureImage]);

  /* ── Convert native bbox to CSS position inside container ── */
  const toCSS = (bbox) => {
    if (!imgBox) return null;
    const sx = imgBox.width  / dims.w;
    const sy = imgBox.height / dims.h;
    return {
      position: 'absolute',
      left:   imgBox.left + bbox[0] * sx,
      top:    imgBox.top  + bbox[1] * sy,
      width:  (bbox[2] - bbox[0]) * sx,
      height: (bbox[3] - bbox[1]) * sy,
    };
  };

  return (
    <div
      ref={containerRef}
      style={{ position: 'relative', width: '100%', height: '100%', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}
    >
      {frameSrc && (
        <img
          ref={imgRef}
          src={frameSrc}
          alt="AI Feed"
          onLoad={measureImage}
          style={{ display: 'block', maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
        />
      )}

      {/* Invisible click targets — positioned over each detected person */}
      {imgBox && detections.map((det) => {
        const css = toCSS(det.bbox);
        if (!css) return null;
        return (
          <div
            key={det.global_id}
            title={`Analyze: ${det.display_name || det.global_id}`}
            onClick={() => onPersonClick && onPersonClick(det.global_id)}
            style={{
              ...css,
              cursor: 'crosshair',
              zIndex: 20,
              border: '2px solid transparent',
              borderRadius: 4,
              transition: 'all 0.1s ease',
              boxSizing: 'border-box',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = 'rgba(59,130,246,0.18)';
              e.currentTarget.style.borderColor = 'rgba(59,130,246,0.85)';
              e.currentTarget.style.boxShadow = '0 0 12px rgba(59,130,246,0.4)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.borderColor = 'transparent';
              e.currentTarget.style.boxShadow = 'none';
            }}
          />
        );
      })}
    </div>
  );
}
