const CACHE_NAME = 'peerhive-cache-v2';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './manifest.json',
  'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap',
  'https://unpkg.com/peerjs@1.5.4/dist/peerjs.min.js',
  'https://unpkg.com/lucide@latest'
];

// Instalação do Service Worker e caching de recursos essenciais
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Cacheando assets principais');
      return cache.addAll(ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// Ativação e limpeza de caches antigos
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Removendo cache antigo:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Interceptação de requisições (Cache First com fallback para Network)
self.addEventListener('fetch', (e) => {
  // Ignora requisições de APIs de terceiros (como PeerJS Cloud, QR Code generator, etc.) 
  // para que funcionem sempre via rede e não quebrem a sinalização
  if (e.request.url.includes('peerjs') && !e.request.url.includes('peerjs.min.js')) {
    return;
  }
  if (e.request.url.includes('qrserver.com') || e.request.url.includes('icons8.com')) {
    return;
  }

  e.respondWith(
    caches.match(e.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(e.request).then((networkResponse) => {
        // Salva cópia no cache se for um asset local e válido
        if (networkResponse.status === 200 && e.request.url.startsWith(self.location.origin)) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(e.request, responseToCache);
          });
        }
        return networkResponse;
      });
    }).catch(() => {
      // Fallback offline se o fetch falhar e o asset não estiver no cache
      if (e.request.mode === 'navigate') {
        return caches.match('./index.html');
      }
    })
  );
});
