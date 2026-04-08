import React, { useEffect, useRef, useState, useCallback } from 'react';

/**
 * VideoStream renders the annotated JPEG frames and overlays interactive
 * click-targets on top of each detected person with enriched info on hover.
 */
export default function VideoStream({ wsUrl, setFps, onPersonClick }) {
  const [frameSrc, setFrameSrc]   = useState(null);
  const [detections, setDetections] = useState([]);
  const [dims, setDims]            = useState({ w: 1, h: 1 });
  const [imgBox, setImgBox]        = useState(null);
  const [hoveredId, setHoveredId]  = useState(null);
  const [streamMeta, setStreamMeta] = useState({ timestamp: null, lat: null, lon: null });

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
      setStreamMeta({ timestamp: d.timestamp, lat: d.latitude, lon: d.longitude });
    };
    return () => ws.close();
  }, [wsUrl]);

  /* ── Measure actual image bounds ── */
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

  /* ── Convert native bbox to CSS ── */
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

  const RISK_COLORS = { low: '#22c55e', medium: '#eab308', high: '#f97316', critical: '#ef4444' };
  const DIR_ARROWS = { left: '←', right: '→', towards: '↓', away: '↑', stationary: '•' };

  // Colors for non-person object categories
  const CATEGORY_COLORS = {
    vehicle: '#00c853', animal: '#ffab00', accessory: '#29b6f6',
    sports: '#ff9100', food: '#ff6e40', furniture: '#78909c',
    electronic: '#42a5f5', kitchen: '#ab47bc', other: '#90a4ae',
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

      {/* Stream Metadata HUD */}
      {streamMeta.timestamp && (
        <div style={{
          position: 'absolute', top: 12, left: 12, zIndex: 40,
          background: 'rgba(8,15,30,0.85)', backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6,
          padding: '6px 10px', color: 'var(--text-main)', fontSize: '0.68rem',
          display: 'flex', flexDirection: 'column', gap: 4, pointerEvents: 'none',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span style={{ color: 'var(--text-dim)' }}>TIME</span>
            <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{new Date(streamMeta.timestamp).toLocaleString()}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span style={{ color: 'var(--text-dim)' }}>GEO</span>
            <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#60a5fa' }}>{streamMeta.lat?.toFixed(4)}, {streamMeta.lon?.toFixed(4)}</span>
          </div>
        </div>
      )}

      {/* Detection overlays */}
      {imgBox && detections.map((det, idx) => {
        const css = toCSS(det.bbox);
        if (!css) return null;
        const isObject = det.is_object;

        if (isObject) {
          // ─── NON-PERSON OBJECT ───
          const catColor = CATEGORY_COLORS[det.object_category] || '#90a4ae';
          const isObjHovered = hoveredId === det.global_id;
          return (
            <div key={`obj-${det.global_id}-${idx}`}>
              <div
                onMouseEnter={() => setHoveredId(det.global_id)}
                onMouseLeave={() => setHoveredId(null)}
                style={{
                  ...css,
                  zIndex: 15,
                  border: `1px solid ${catColor}60`,
                  borderRadius: 3,
                  background: isObjHovered ? `${catColor}18` : 'transparent',
                  transition: 'all 0.15s ease',
                  boxSizing: 'border-box',
                  pointerEvents: 'auto',
                }}
              />
              {isObjHovered && (
                <div style={{
                  position: 'absolute',
                  left: css.left,
                  top: Math.max(0, css.top - 40),
                  zIndex: 30,
                  background: 'rgba(8,15,30,0.92)',
                  backdropFilter: 'blur(8px)',
                  border: `1px solid ${catColor}50`,
                  borderRadius: 5,
                  padding: '4px 8px',
                  pointerEvents: 'none',
                  boxShadow: `0 4px 12px rgba(0,0,0,0.5)`,
                  whiteSpace: 'nowrap',
                }}>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: catColor, textTransform: 'capitalize' }}>
                    {det.display_name}
                  </div>
                  <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>
                    {det.object_category} · {Math.round((det.confidence || 0) * 100)}% conf
                  </div>
                </div>
              )}
            </div>
          );
        }

        // ─── PERSON ───
        const isHovered = hoveredId === det.global_id;
        const riskColor = RISK_COLORS[det.risk_level] || RISK_COLORS.low;

        return (
          <div key={det.global_id}>
            {/* Clickable box */}
            <div
              title={`Analyze: ${det.display_name || det.global_id}`}
              onClick={() => onPersonClick && onPersonClick(det.global_id)}
              onMouseEnter={() => setHoveredId(det.global_id)}
              onMouseLeave={() => setHoveredId(null)}
              style={{
                ...css,
                cursor: 'crosshair',
                zIndex: 20,
                border: isHovered ? `2px solid ${riskColor}` : '2px solid transparent',
                borderRadius: 4,
                transition: 'all 0.15s ease',
                boxSizing: 'border-box',
                background: isHovered ? `${riskColor}18` : 'transparent',
                boxShadow: isHovered ? `0 0 12px ${riskColor}40` : 'none',
              }}
            />

            {/* Hover info tooltip */}
            {isHovered && (
              <div style={{
                position: 'absolute',
                left: css.left,
                top: Math.max(0, css.top - 76),
                zIndex: 30,
                background: 'rgba(8,15,30,0.92)',
                backdropFilter: 'blur(8px)',
                border: `1px solid ${riskColor}50`,
                borderRadius: 6,
                padding: '5px 8px',
                minWidth: 140,
                pointerEvents: 'none',
                boxShadow: `0 4px 16px rgba(0,0,0,0.5), 0 0 8px ${riskColor}20`,
              }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, marginBottom: 3, color: 'var(--text-main)' }}>
                  {det.display_name || det.global_id}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 8px', fontSize: '0.62rem' }}>
                  <span style={{ color: 'var(--text-dim)' }}>Risk</span>
                  <span style={{ color: riskColor, fontWeight: 600, textTransform: 'uppercase' }}>{Math.round(det.risk_score || 0)}/100 · {det.risk_level || 'low'}</span>
                  <span style={{ color: 'var(--text-dim)' }}>Act</span>
                  <span style={{ color: '#60a5fa', textTransform: 'capitalize' }}>{det.behaviour_label ? det.behaviour_label.replace(/_/g, ' ') : (det.pose_detail || det.activity || '—')}</span>
                  <span style={{ color: 'var(--text-dim)' }}>Move</span>
                  <span>{DIR_ARROWS[det.movement_direction] || '•'} {det.movement_direction || 'still'}</span>
                  {det.carried_objects?.length > 0 && <>
                    <span style={{ color: 'var(--text-dim)' }}>Luggage</span>
                    <span style={{ color: '#a78bfa', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{det.carried_objects.join(', ')}</span>
                  </>}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
