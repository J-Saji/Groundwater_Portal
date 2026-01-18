import React from 'react';

interface AlertToastProps {
  message: string;
  type: 'info' | 'warning' | 'error' | 'success';
  onClose: () => void;
}

export default function AlertToast({ message, type, onClose }: AlertToastProps) {
  React.useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const bgColor = {
    error: 'bg-red-50 border-red-300 text-red-800',
    warning: 'bg-yellow-50 border-yellow-300 text-yellow-800',
    success: 'bg-green-50 border-green-300 text-green-800',
    info: 'bg-blue-50 border-blue-300 text-blue-800'
  }[type];

  const icon = {
    error: '❌',
    warning: '⚠️',
    success: '✅',
    info: 'ℹ️'
  }[type];

  return (
    <div className={`fixed top-4 right-4 z-[10000] max-w-md p-4 rounded-lg shadow-2xl border-2 animate-fade-in ${bgColor}`}>
      <div className="flex items-center gap-2">
        <span className="text-2xl">{icon}</span>
        <span className="font-medium">{message}</span>
      </div>
    </div>
  );
}