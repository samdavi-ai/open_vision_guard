import React from 'react';
import StreamPlayer from './StreamPlayer';

const CameraGrid = ({ cameras, onAlert }) => {
  // Determine grid columns based on number of cameras
  const cols = cameras.length === 1 ? 'grid-cols-1' :
               cameras.length <= 4 ? 'grid-cols-2' :
               'grid-cols-3';

  return (
    <div className={`grid ${cols} gap-4 h-full auto-rows-fr`}>
      {cameras.map((cam) => (
        <StreamPlayer key={cam.id} camera={cam} onAlert={onAlert} />
      ))}
    </div>
  );
};

export default CameraGrid;
