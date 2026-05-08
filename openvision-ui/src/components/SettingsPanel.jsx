import React, { useState, useEffect, useCallback } from 'react';
import { X, Settings, Zap, Shield, Eye, Package, SlidersHorizontal, RotateCcw } from 'lucide-react';

/* ── Small toggle switch ── */
function Toggle({ value, onChange, disabled }) {
  return (
    <div
      onClick={() => !disabled && onChange(!value)}
      style={{
        width: 40, height: 22, borderRadius: 11, flexShrink: 0,
        background: value ? 'var(--accent)' : 'rgba(255,255,255,0.12)',
        border: `1px solid ${value ? 'var(--accent)' : 'rgba(255,255,255,0.18)'}`,
        cursor: disabled ? 'not-allowed' : 'pointer',
        position: 'relative',
        transition: 'background 0.2s, border-color 0.2s',
        opacity: disabled ? 0.4 : 1,
        boxShadow: value ? '0 0 8px rgba(59,130,246,0.4)' : 'none',
      }}
    >
      <div style={{
        position: 'absolute',
        top: 3, left: value ? 20 : 3,
        width: 14, height: 14, borderRadius: '50%',
        background: '#fff',
        transition: 'left 0.18s cubic-bezier(0.4,0,0.2,1)',
        boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
      }} />
    </div>
  );
}

/* ── Slider row ── */
function SliderRow({ label, desc, value, min, max, step = 0.01, format, onChange }) {
  const display = format ? format(value) : value;
  return (
    <div style={{ marginBottom: '0.9rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <div>
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>{label}</div>
          {desc && <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 1 }}>{desc}</div>}
        </div>
        <span style={{
          fontSize: '0.78rem', fontFamily: 'monospace', fontWeight: 700,
          color: 'var(--accent)', background: 'rgba(59,130,246,0.12)',
          padding: '2px 8px', borderRadius: 5, alignSelf: 'flex-start',
        }}>{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
      />
    </div>
  );
}

/* ── Toggle row ── */
function ToggleRow({ label, desc, value, onChange, tag }) {
  const TAG_COLORS = { perf: '#8b5cf6', accuracy: '#3b82f6', alert: '#f97316', privacy: '#22c55e' };
  const tagColor = TAG_COLORS[tag] || '#94a3b8';
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
      padding: '0.65rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)',
      gap: 12,
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-main)' }}>{label}</span>
          {tag && (
            <span style={{
              fontSize: '0.58rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em',
              color: tagColor, background: `${tagColor}18`, padding: '1px 5px', borderRadius: 3,
            }}>{tag}</span>
          )}
        </div>
        {desc && <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 2, lineHeight: 1.4 }}>{desc}</div>}
      </div>
      <Toggle value={value} onChange={onChange} />
    </div>
  );
}

/* ── Section header ── */
function Section({ icon, title, children }) {
  return (
    <div style={{ marginBottom: '1.4rem' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 7,
        fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase',
        letterSpacing: '0.07em', color: 'var(--text-dim)',
        marginBottom: '0.6rem', paddingBottom: '0.4rem',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
      }}>
        {icon} {title}
      </div>
      {children}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   SETTINGS PANEL
══════════════════════════════════════════════════════════════ */
export default function SettingsPanel({ apiBase, onClose }) {
  const [cfg, setCfg] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [dirty, setDirty] = useState({});   // tracks which fields changed

  /* Load current config from backend */
  useEffect(() => {
    fetch(`${apiBase}/config`)
      .then(r => r.json())
      .then(data => setCfg(data))
      .catch(() => {});
  }, [apiBase]);

  const set = useCallback((key, val) => {
    setCfg(prev => ({ ...prev, [key]: val }));
    setDirty(prev => ({ ...prev, [key]: val }));
    setSaved(false);
  }, []);

  const handleSave = async () => {
    if (!Object.keys(dirty).length) return;
    setSaving(true);
    try {
      const r = await fetch(`${apiBase}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dirty),
      });
      if (r.ok) {
        setSaved(true);
        setDirty({});
        setTimeout(() => setSaved(false), 2500);
      }
    } catch (e) {}
    setSaving(false);
  };

  const hasDirty = Object.keys(dirty).length > 0;

  if (!cfg) return (
    <div style={OVERLAY}>
      <div style={{ ...PANEL, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)' }}>
        Loading settings…
      </div>
    </div>
  );

  return (
    <div style={OVERLAY} onClick={onClose}>
      <div style={PANEL} onClick={e => e.stopPropagation()}>

        {/* ── Header ── */}
        <div style={{
          padding: '1rem 1.2rem 0.85rem',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(59,130,246,0.3)',
            }}>
              <Settings size={16} color="#fff" />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>System Settings</div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>Changes apply instantly — no restart needed</div>
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
            color: 'var(--text-dim)', borderRadius: 8, width: 32, height: 32,
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <X size={16} />
          </button>
        </div>

        {/* ── Scrollable body ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.1rem 1.2rem' }}>

          {/* ── AI DETECTION ── */}
          <Section icon={<Zap size={13}/>} title="AI Detection">
            <ToggleRow
              label="Multi-Scale Detection (960px)"
              desc="Second pass at higher resolution — catches distant/small persons. Adds ~400ms latency per camera."
              tag="accuracy"
              value={cfg.multiscale_enabled}
              onChange={v => set('multiscale_enabled', v)}
            />
            <ToggleRow
              label="Small Object Re-inference"
              desc="Crops and re-infers small regions for missed detections. Adds ~200ms per camera."
              tag="accuracy"
              value={cfg.small_object_reinference}
              onChange={v => set('small_object_reinference', v)}
            />
            <ToggleRow
              label="Re-Detection on Track Loss"
              desc="Actively searches for a person after their track is lost. Adds ~150ms per camera."
              tag="accuracy"
              value={cfg.redetection_on_loss_enabled}
              onChange={v => set('redetection_on_loss_enabled', v)}
            />
            <ToggleRow
              label="Zero-Motion Gate"
              desc="Skip AI inference on completely static scenes to save compute. Can miss slow-walking persons."
              tag="perf"
              value={cfg.zero_motion_gate_enabled}
              onChange={v => set('zero_motion_gate_enabled', v)}
            />
          </Section>

          {/* ── ALERT FEATURES ── */}
          <Section icon={<Shield size={13}/>} title="Alert Features">
            <ToggleRow
              label="Weapon Detection"
              desc="Flags persons near detected weapons (knives, guns, etc.)"
              tag="alert"
              value={cfg.weapon_detection_enabled}
              onChange={v => set('weapon_detection_enabled', v)}
            />
            <ToggleRow
              label="Pose / Fall Detection"
              desc="Detects fall events using body pose analysis (requires MediaPipe)."
              tag="alert"
              value={cfg.pose_enabled}
              onChange={v => set('pose_enabled', v)}
            />
            <ToggleRow
              label="Face Recognition"
              desc="Match detected faces against known person database."
              tag="privacy"
              value={cfg.face_recognition_enabled}
              onChange={v => set('face_recognition_enabled', v)}
            />
          </Section>

          {/* ── DETECTION THRESHOLDS ── */}
          <Section icon={<SlidersHorizontal size={13}/>} title="Detection Thresholds">
            <SliderRow
              label="Person Confidence"
              desc="Minimum confidence to accept a person detection. Higher = fewer false positives."
              value={cfg.person_conf_threshold ?? 0.45}
              min={0.20} max={0.80} step={0.01}
              format={v => `${Math.round(v * 100)}%`}
              onChange={v => set('person_conf_threshold', v)}
            />
            <SliderRow
              label="Object (Bag) Confidence"
              desc="Minimum confidence for bag/object detection. Lower = catch more hidden bags."
              value={cfg.object_conf_threshold ?? 0.35}
              min={0.15} max={0.70} step={0.01}
              format={v => `${Math.round(v * 100)}%`}
              onChange={v => set('object_conf_threshold', v)}
            />
            <SliderRow
              label="Luggage Confidence"
              desc="Threshold specifically for bag classes (backpack, handbag, suitcase)."
              value={cfg.luggage_conf_threshold ?? 0.30}
              min={0.10} max={0.70} step={0.01}
              format={v => `${Math.round(v * 100)}%`}
              onChange={v => set('luggage_conf_threshold', v)}
            />
            <SliderRow
              label="Re-ID Similarity Threshold"
              desc="How similar two appearances must be to be the same person. Lower = more lenient matching."
              value={cfg.similarity_threshold ?? 0.70}
              min={0.40} max={0.95} step={0.01}
              format={v => `${Math.round(v * 100)}%`}
              onChange={v => set('similarity_threshold', v)}
            />
          </Section>

          {/* ── BAGGAGE & SESSION ── */}
          <Section icon={<Package size={13}/>} title="Baggage & Session Tracking">
            <SliderRow
              label="Exit Timeout"
              desc="Seconds a person must be absent before considered 'exited'. Raise for large spaces."
              value={cfg.session_exit_timeout_s ?? 20}
              min={5} max={60} step={1}
              format={v => `${v}s`}
              onChange={v => set('session_exit_timeout_s', v)}
            />
            <SliderRow
              label="Loitering Threshold"
              desc="Seconds before a stationary person triggers a loitering alert."
              value={cfg.loitering_threshold_seconds ?? 30}
              min={10} max={120} step={5}
              format={v => `${v}s`}
              onChange={v => set('loitering_threshold_seconds', v)}
            />
          </Section>

          {/* ── PERFORMANCE ── */}
          <Section icon={<Eye size={13}/>} title="Performance & Display">
            <SliderRow
              label="Detection Hold Frames"
              desc="How many frames a box stays visible after person disappears (prevents flicker)."
              value={cfg.temporal_hold_frames ?? 20}
              min={5} max={90} step={1}
              format={v => `${v} frames`}
              onChange={v => set('temporal_hold_frames', v)}
            />
            <SliderRow
              label="Alert Dedup Window"
              desc="Seconds before the same alert type fires again for the same person."
              value={cfg.alert_dedup_window_seconds ?? 30}
              min={5} max={120} step={5}
              format={v => `${v}s`}
              onChange={v => set('alert_dedup_window_seconds', v)}
            />
            <SliderRow
              label="JPEG Stream Quality"
              desc="Video quality sent to browser. Higher = sharper but more bandwidth."
              value={cfg.frame_jpeg_quality ?? 75}
              min={30} max={95} step={5}
              format={v => `${v}%`}
              onChange={v => set('frame_jpeg_quality', v)}
            />
          </Section>
        </div>

        {/* ── Footer: Save / Reset ── */}
        <div style={{
          padding: '0.85rem 1.2rem',
          borderTop: '1px solid rgba(255,255,255,0.08)',
          display: 'flex', gap: 8, alignItems: 'center',
          flexShrink: 0,
          background: 'rgba(0,0,0,0.2)',
        }}>
          {saved && (
            <span style={{ fontSize: '0.75rem', color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4, flex: 1 }}>
              ✓ Settings applied — taking effect immediately
            </span>
          )}
          {hasDirty && !saved && (
            <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', flex: 1 }}>
              {Object.keys(dirty).length} unsaved change{Object.keys(dirty).length > 1 ? 's' : ''}
            </span>
          )}
          {!hasDirty && !saved && <div style={{ flex: 1 }} />}

          <button
            onClick={onClose}
            style={{
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)',
              color: 'var(--text-dim)', padding: '0.45rem 1rem', borderRadius: 7,
              cursor: 'pointer', fontSize: '0.8rem', fontFamily: 'inherit',
            }}
          >
            Close
          </button>
          <button
            onClick={handleSave}
            disabled={!hasDirty || saving}
            style={{
              background: hasDirty ? 'var(--accent)' : 'rgba(59,130,246,0.3)',
              border: 'none', color: '#fff',
              padding: '0.45rem 1.2rem', borderRadius: 7, cursor: hasDirty ? 'pointer' : 'default',
              fontSize: '0.8rem', fontWeight: 700, fontFamily: 'inherit',
              opacity: saving ? 0.7 : 1,
              transition: 'background 0.2s',
              boxShadow: hasDirty ? '0 4px 14px rgba(59,130,246,0.35)' : 'none',
            }}
          >
            {saving ? 'Applying…' : 'Apply Settings'}
          </button>
        </div>
      </div>
    </div>
  );
}

const OVERLAY = {
  position: 'fixed', inset: 0, zIndex: 2000,
  background: 'rgba(0,0,0,0.75)',
  backdropFilter: 'blur(6px)',
  display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
  animation: 'fadeIn 0.15s ease',
};

const PANEL = {
  width: 480,
  maxWidth: '96vw',
  height: '100vh',
  background: 'linear-gradient(160deg, #0d1a2e 0%, #080f1e 100%)',
  borderLeft: '1px solid rgba(255,255,255,0.1)',
  boxShadow: '-24px 0 80px rgba(0,0,0,0.6)',
  display: 'flex',
  flexDirection: 'column',
  animation: 'slideInRight 0.25s cubic-bezier(0.4,0,0.2,1)',
};
