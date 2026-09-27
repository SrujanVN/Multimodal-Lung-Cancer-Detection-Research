import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './legacy.css';
import './index.css';
import './components/chat.css';

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);

