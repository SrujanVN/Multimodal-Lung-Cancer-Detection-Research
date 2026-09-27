import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], server: { proxy: { '/api': 'http://127.0.0.1:5000', '/diagnosis': 'http://127.0.0.1:5000', '/predict_metadata_form': 'http://127.0.0.1:5000', '/login': 'http://127.0.0.1:5000', '/register': 'http://127.0.0.1:5000', '/download_report': 'http://127.0.0.1:5000', '/chat': 'http://127.0.0.1:5000', '/documentation': 'http://127.0.0.1:5000', '/static': 'http://127.0.0.1:5000', '/uploads': 'http://127.0.0.1:5000', '/get_explanation': 'http://127.0.0.1:5000' } } });
