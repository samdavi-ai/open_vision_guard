import React, { useEffect, useState } from 'react';
import { X, Fingerprint, Activity, Shield, Clock } from 'lucide-react';

export default function ProfileHUD({ globalId, onClose }) {
  const [profile, setProfile] = useState(null);

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await fetch(`http://localhost:8000/identities/${globalId}`);
        if (res.ok) {
          const data = await res.json();
          setProfile(data);
        }
      } catch (err) {
        console.error("Error fetching profile", err);
      }
    };
    fetchProfile();
    
    // Poll for updates while HUD is open
    const id = setInterval(fetchProfile, 2000);
    return () => clearInterval(id);
  }, [globalId]);

  if (!profile) return null;

  const { metadata, history } = profile;
  const riskColor = metadata.risk_level === 'critical' ? 'var(--critical)' : 'var(--text-main)';

  return (
    <div style={{
      position: 'absolute', top: '1rem', left: '1rem', width: 320,
      background: 'rgba(15, 23, 42, 0.85)', backdropFilter: 'blur(16px)',
      border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: '1.25rem',
      zIndex: 50, boxShadow: '0 20px 40px rgba(0,0,0,0.5)', animation: 'slideIn 0.3s ease-out'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.5rem' }}>
        <h3 style={{ fontSize: '1.1rem', margin: 0 }}>{metadata.face_name || "Unknown Individual"}</h3>
        <X onClick={onClose} size={20} style={{ cursor: 'pointer', color: 'var(--text-dim)' }} />
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1rem' }}>
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.25rem', display:'flex', alignItems:'center', gap:4 }}><Fingerprint size={12}/> Identity ID</div>
          <div style={{ fontWeight: 600, fontSize: '0.85rem', wordBreak: 'break-all' }}>{globalId.substring(0, 12)}...</div>
        </div>
        
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.25rem', display:'flex', alignItems:'center', gap:4 }}><Shield size={12}/> Risk Level</div>
          <div style={{ fontWeight: 600, fontSize: '0.95rem', textTransform: 'capitalize', color: riskColor }}>{metadata.risk_level || 'low'}</div>
        </div>
        
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.25rem', display:'flex', alignItems:'center', gap:4 }}><Activity size={12}/> Activity</div>
          <div style={{ fontWeight: 600, fontSize: '0.95rem', textTransform: 'capitalize' }}>{metadata.activity || 'unknown'}</div>
        </div>
        
        <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.75rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '0.25rem', display:'flex', alignItems:'center', gap:4 }}><Clock size={12}/> First Seen</div>
          <div style={{ fontWeight: 600, fontSize: '0.8rem' }}>
            {history.length > 0 ? new Date(history[0].timestamp).toLocaleTimeString() : '-'}
          </div>
        </div>
      </div>
      
      <h4 style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>Event Timeline</h4>
      <div style={{ maxHeight: 150, overflowY: 'auto', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.75rem' }}>
        {history.length === 0 ? (
          <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>No history yet</div>
        ) : (
          history.map((ev, i) => (
            <div key={i} style={{ display: 'flex', gap: '0.75rem', fontSize: '0.8rem', marginBottom: '0.5rem' }}>
              <span style={{ color: 'var(--text-dim)' }}>{new Date(ev.timestamp).toLocaleTimeString()}</span>
              <span>{ev.activity || 'Detected'} at {ev.camera_id}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
