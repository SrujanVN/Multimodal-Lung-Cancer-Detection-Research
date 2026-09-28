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
    { source: '/api/:path*', destination: `${backend}/api/:path*` },
    { source: '/static/:path*', destination: `${backend}/static/:path*` },
    { source: '/uploads/:path*', destination: `${backend}/uploads/:path*` },
    ...backendPaths.flatMap((path) => [
      { source: `/${path}`, destination: `${backend}/${path}` },
      { source: `/${path}/:path*`, destination: `${backend}/${path}/:path*` },
    ]),
    { source: '/get_explanation/:path*', destination: `${backend}/get_explanation/:path*` },
    { source: '/(.*)', destination: '/index.html' },
  ],
  headers: [
    { source: '/api/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
    { source: '/uploads/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
    { source: '/static/uploads/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
    { source: '/static/reports/:path*', headers: [{ key: 'Cache-Control', value: 'no-store' }] },
  ],
};
