import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { WebGPUCanvas } from './components/Scene';
import './style.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <main>
      <WebGPUCanvas />
    </main>
  </StrictMode>
);
