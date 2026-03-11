import React from 'react';
import { Bell, Flame, Activity } from 'lucide-react';

const AlertSidebar = ({ alerts }) => {
  return (
    <div className="w-80 bg-panel border-l border-border h-full flex flex-col z-20 shadow-2xl">
      <div className="p-6 border-b border-border/50 flex items-center justify-between bg-black/20">
        <h2 className="font-bold tracking-wider flex items-center gap-2 text-white">
          <Bell className="text-primary" size={20} />
          EVENT LOG
        </h2>
        <span className="bg-red-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full animate-pulse">
          {alerts.length} NEW
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {alerts.length === 0 ? (
          <div className="text-center text-gray-500 mt-10 text-sm">
            No events detected. System is actively monitoring.
          </div>
        ) : (
          alerts.map((alert, i) => {
            const isDanger = alert.text.toLowerCase().includes('weapon') || alert.text.toLowerCase().includes('fight') || alert.text.toLowerCase().includes('fall');
            const Icon = isDanger ? Flame : Activity;

            return (
              <div 
                key={i} 
                className={`p-3 rounded-lg border backdrop-blur-sm text-sm transition-all duration-300 transform hover:-translate-y-1 hover:shadow-lg
                  ${isDanger 
                    ? 'bg-red-500/10 border-red-500/30 text-red-100 shadow-[0_0_10px_rgba(239,68,68,0.1)]' 
                    : 'bg-primary/5 border-primary/20 text-gray-300 shadow-[0_0_10px_rgba(59,130,246,0.05)]'}`}
              >
                <div className="flex items-start gap-3">
                  <div className={`mt-0.5 ${isDanger ? 'text-red-500' : 'text-primary'}`}>
                    <Icon size={16} />
                  </div>
                  <div>
                    <p className={`font-semibold tracking-wide ${isDanger ? 'text-red-400' : 'text-gray-100'}`}>
                      {alert.text}
                    </p>
                    <div className="flex items-center gap-2 mt-1.5 text-[11px] font-mono opacity-60">
                      <span>{alert.camName}</span>
                      <span>•</span>
                      <span>{alert.time}</span>
                    </div>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  );
};

export default AlertSidebar;
