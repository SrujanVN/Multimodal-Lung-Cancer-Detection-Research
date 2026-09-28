import { routes } from '@vercel/config/v1';

const backendUrl = process.env.BACKEND_URL;

if (!backendUrl) {
  throw new Error('Set BACKEND_URL in Vercel to the public HTTPS origin of the Dockerized Flask backend.');
}

const parsedBackendUrl = new URL(backendUrl);
if (parsedBackendUrl.protocol !== 'https:' || parsedBackendUrl.pathname !== '/' || parsedBackendUrl.search || parsedBackendUrl.hash) {
  throw new Error('BACKEND_URL must be an HTTPS origin, for example https://api.example.com, without a path.');
}

const backend = parsedBackendUrl.origin;
const backendPaths = [
  'diagnosis',
  'predict_metadata_form',
  'download_report',
  'login',
  'register',
  'logout',
  'forgot_password',
  'moreinfo',
  'about',
  'home',
  'documentation',
  'chat',
];

export const config = {
  framework: 'vite',
  buildCommand: 'npm run build',
  outputDirectory: 'dist',
  rewrites: [
    routes.rewrite('/api/:path*', `${backend}/api/:path*`),
    routes.rewrite('/static/:path*', `${backend}/static/:path*`),
    routes.rewrite('/uploads/:path*', `${backend}/uploads/:path*`),
    ...backendPaths.flatMap((path) => [
      routes.rewrite(`/${path}`, `${backend}/${path}`),
      routes.rewrite(`/${path}/:path*`, `${backend}/${path}/:path*`),
    ]),
    routes.rewrite('/get_explanation/:path*', `${backend}/get_explanation/:path*`),
    routes.rewrite('/(.*)', '/index.html'),
  ],
  headers: [
    { source: '/api/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
    { source: '/uploads/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
    { source: '/static/uploads/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
    { source: '/static/reports/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
  ],
};
