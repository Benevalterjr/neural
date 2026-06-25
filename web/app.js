/* ==========================================================================
   PEERHIVE CLIENT ENGINE - P2P WebRTC Network Simulator
   ========================================================================== */

// Registro do Service Worker para PWA
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js')
      .then(reg => console.log('[Service Worker] Registrado com sucesso:', reg.scope))
      .catch(err => console.error('[Service Worker] Falha ao registrar:', err));
  });
}

// Configuração do PeerJS
let peer = null;
let myId = null;
let connections = {}; // Armazena canais de dados: peerId -> dataConnection

// Banco de Dados Local do Nó (HiveStore)
let localVideos = [];
// Tabela de Feromônios / Ponteiros: video_id -> { video: object, pointer: peerId, val: float }
let pheromones = {};

// Embedding Cache & Utilitários
const EMBEDDING_DIM = 128;

// Configuração de Simulação de Rede
const SIM_LATENCY_BASE = 50; // ms de RTT simulado base

/* ==========================================================================
   INICIALIZAÇÃO & INTERFACE DE CONEXÃO P2P
   ========================================================================== */

window.addEventListener('DOMContentLoaded', () => {
  initUI();
  setupPeer();
});

function setupPeer() {
  // Gera um ID amigável aleatório para reduzir fricção
  const randNum = Math.floor(1000 + Math.random() * 9000);
  const suggestedId = `peerhive-${randNum}`;

  // Conecta ao servidor de sinalização público do PeerJS
  // Usamos as opções padrão que se conectam ao PeerJS Cloud gratuito
  peer = new Peer(suggestedId, {
    debug: 1
  });

  peer.on('open', (id) => {
    myId = id;
    updateStatus('online', 'Online');
    
    const idBadge = document.getElementById('my-peer-id');
    idBadge.innerText = myId;
    document.getElementById('btn-copy-id').disabled = false;
    document.getElementById('btn-connect').disabled = false;
    
    logToConsole(`[SISTEMA] Conectado à sinalização! Seu ID de Nó é: ${myId}`, 'success');
    
    // Gerar Link de Auto-conexão e QR Code (Zero Fricção)
    generateFrictionlessConnectUI();
    
    // Verificar parâmetro na URL para auto-conexão
    checkUrlParams();
    
    // Adicionar alguns vídeos mock iniciais para este nó para não começar vazio
    addMockVideos();
  });

  peer.on('connection', (conn) => {
    setupConnectionEvents(conn);
  });

  peer.on('error', (err) => {
    console.error('PeerJS Error:', err);
    logToConsole(`[ERRO PEERJS] ${err.type}: ${err.message}`, 'error');
    if (err.type === 'unavailable-id') {
      logToConsole('[SISTEMA] Tentando gerar outro ID único...', 'info');
      // Tentar novamente com outro ID
      setTimeout(() => {
        const altId = `peerhive-${Math.floor(1000 + Math.random() * 9000)}`;
        peer = new Peer(altId);
      }, 1000);
    }
  });

  peer.on('disconnected', () => {
    updateStatus('offline', 'Desconectado');
    logToConsole('[SISTEMA] Desconectado do servidor de sinalização. Tentando reconectar...', 'error');
    peer.reconnect();
  });
}

// Configura eventos de troca de dados após a conexão estabelecida
function setupConnectionEvents(conn) {
  conn.on('open', () => {
    connections[conn.peer] = conn;
    logToConsole(`[CONEXÃO] Canal WebRTC DataChannel aberto com peer: ${conn.peer}`, 'success');
    
    // Medir RTT Inicial
    sendPing(conn.peer);
    
    // Sincronizar vizinhos
    updatePeersListUI();
    
    // Notificar conexão no swarm
    logToConsole(`[SWARM] Vizinho ${conn.peer} adicionado com sucesso.`, 'info');
  });

  conn.on('data', (data) => {
    handleIncomingMessage(conn.peer, data);
  });

  conn.on('close', () => {
    handlePeerDisconnection(conn.peer);
  });

  conn.on('error', (err) => {
    logToConsole(`[ERRO CONEXÃO] Com ${conn.peer}: ${err.message}`, 'error');
    handlePeerDisconnection(conn.peer);
  });
}

function connectToPeer(targetId) {
  if (!targetId || targetId.trim() === '') return;
  targetId = targetId.trim();

  if (targetId === myId) {
    logToConsole('[AVISO] Você não pode se conectar a si mesmo.', 'error');
    return;
  }

  if (connections[targetId]) {
    logToConsole(`[AVISO] Você já está conectado ao peer ${targetId}.`, 'info');
    return;
  }

  logToConsole(`[CONEXÃO] Iniciando aperto de mão WebRTC com ${targetId}...`, 'info');
  
  const conn = peer.connect(targetId, {
    serialization: 'json'
  });
  
  setupConnectionEvents(conn);
}

function handlePeerDisconnection(peerId) {
  if (connections[peerId]) {
    delete connections[peerId];
    logToConsole(`[SWARM] Peer ${peerId} saiu da rede (Churn detectado).`, 'healing');
    updatePeersListUI();
    
    // IMPORTANTE: Não apagamos os feromônios do peerId que saiu imediatamente.
    // Isso é feito para que possamos demonstrar a necessidade do Self-Healing
    // quando o usuário tentar usar uma pista/feromônio que aponta para um nó que morreu!
  }
}

/* ==========================================================================
   SINALIZAÇÃO & MENSAGENS P2P (PROTOCOLO PEERHIVE)
   ========================================================================== */

function handleIncomingMessage(senderId, msg) {
  switch (msg.type) {
    case 'PING':
      // Responde com PONG
      sendToPeer(senderId, { type: 'PONG', timestamp: msg.timestamp });
      break;

    case 'PONG':
      const rtt = Date.now() - msg.timestamp;
      if (connections[senderId]) {
        connections[senderId].rtt = rtt;
        updatePeersListUI();
        logToConsole(`[PING] RTT medido com ${senderId}: ${rtt}ms`, 'system');
      }
      break;

    case 'GOSSIP_INDEX':
      // Recebeu anúncio de vídeo indexado por Gossip
      handleGossipIndex(senderId, msg);
      break;

    case 'VECTOR_SEARCH_REQ':
      // Recebeu requisição de busca vetorial
      handleVectorSearchReq(senderId, msg);
      break;

    case 'VECTOR_SEARCH_RES':
      // Recebeu resposta de busca vetorial
      handleVectorSearchRes(senderId, msg);
      break;

    case 'DOWNLOAD_REQ':
      // Alguém quer baixar um vídeo local deste nó
      handleDownloadReq(senderId, msg);
      break;

    case 'DOWNLOAD_RES':
      // Resposta com os dados do download
      handleDownloadRes(senderId, msg);
      break;

    default:
      console.warn('Mensagem desconhecida:', msg);
  }
}

function sendToPeer(peerId, msg) {
  const conn = connections[peerId];
  if (conn && conn.open) {
    conn.send(msg);
  } else {
    logToConsole(`[ERRO] Não foi possível enviar para ${peerId}. Conexão fechada.`, 'error');
  }
}

function sendPing(peerId) {
  sendToPeer(peerId, { type: 'PING', timestamp: Date.now() });
}

/* ==========================================================================
   LÓGICA HIVESTORE & EMBEDDINGS (MOCK VETORIAL)
   ========================================================================== */

// Gera um embedding de 128-dim consistente para um texto
function getEmbeddingForText(text) {
  // LCG Pseudo-Random Number Generator baseado em hash simples da string
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = text.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  const vec = [];
  let seed = hash;
  
  // Função pseudo-aleatória determinística
  const nextRand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 2 - 1; // Entre -1.0 e 1.0
  };

  for (let i = 0; i < EMBEDDING_DIM; i++) {
    vec.push(nextRand());
  }

  // Normalizar vetor (Norma Euclidiana = 1)
  let norm = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  for (let i = 0; i < EMBEDDING_DIM; i++) vec[i] /= norm;

  return vec;
}

// Similaridade de cosseno
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    dotProduct += vecA[i] * vecB[i];
  }
  return dotProduct; // Como ambos estão normalizados, a norma é 1, então similaridade = dotProduct
}

// Indexa um vídeo localmente e propaga via Gossip
function indexLocalVideo(title, sizeType) {
  const videoId = 'vid-' + Math.floor(100000 + Math.random() * 900000);
  const sizeBytes = sizeType === 'hnerv' ? 10240 : 3145728; // 10KB vs 3MB
  const embedding = getEmbeddingForText(title);

  const video = {
    id: videoId,
    title: title,
    sizeType: sizeType,
    sizeBytes: sizeBytes,
    embedding: embedding,
    owner: myId
  };

  localVideos.push(video);
  updateLocalVideosUI();
  logToConsole(`[HIVESTORE] Vídeo '${title}' indexado localmente.`, 'success');

  // Propagar anúncio via Gossip para todos os vizinhos conectados
  propagateGossipIndex(video, 1);
}

/* ==========================================================================
   LÓGICA GOSSIP DE INDEXAÇÃO (PONTEIROS / FEROMÔNIOS)
   ========================================================================== */

function propagateGossipIndex(video, hopCount) {
  const gossipMsg = {
    type: 'GOSSIP_INDEX',
    video: video,
    hopCount: hopCount,
    senderId: myId
  };

  const peersList = Object.keys(connections);
  if (peersList.length === 0) {
    logToConsole('[GOSSIP] Nenhum vizinho para propagar o índice. Ele ficará apenas local.', 'system');
    return;
  }

  logToConsole(`[GOSSIP] Propagando trilha (feromônio) de '${video.title}' para ${peersList.length} vizinhos (Salto: ${hopCount}).`, 'gossip');
  
  peersList.forEach(peerId => {
    sendToPeer(peerId, gossipMsg);
  });
}

function handleGossipIndex(senderId, msg) {
  const video = msg.video;
  const hopCount = msg.hopCount;
  
  // Evitar loops: se fomos nós que criamos o vídeo originalmente
  if (video.owner === myId) return;

  // Calcula o valor do feromônio de acordo com o número de hops (decaimento topológico)
  const pheromoneValue = 1.0 / hopCount;

  // Verifica se já conhecemos esse feromônio de forma melhor
  const existing = pheromones[video.id];
  if (!existing || pheromoneValue > existing.val) {
    pheromones[video.id] = {
      video: video,
      pointer: senderId, // Aponta para quem nos entregou a mensagem (o próximo salto da trilha)
      val: pheromoneValue
    };
    
    logToConsole(`[GOSSIP] Feromônio gravado para '${video.title}' -> Aponta para vizinho '${senderId}' (Força: ${pheromoneValue.toFixed(2)})`, 'gossip');

    // Propaga para os outros vizinhos se o limite de saltos (Hops) não foi atingido
    const MAX_HOPS = 3;
    if (hopCount < MAX_HOPS) {
      const nextGossipMsg = {
        type: 'GOSSIP_INDEX',
        video: video,
        hopCount: hopCount + 1,
        senderId: myId
      };

      Object.keys(connections).forEach(peerId => {
        // Não envia de volta para quem mandou
        if (peerId !== senderId) {
          sendToPeer(peerId, nextGossipMsg);
        }
      });
    }
  }
}

/* ==========================================================================
   BUSCA SEMÂNTICA HÍBRIDA
   ========================================================================== */

let activeSearchQuery = "";
let currentSearchResults = [];
let pendingSearchResponses = new Set();
let searchTimeoutId = null;

function performSearch(query) {
  if (!query || query.trim() === '') return;
  query = query.trim();
  
  activeSearchQuery = query;
  currentSearchResults = [];
  pendingSearchResponses.clear();
  
  logToConsole(`[BUSCA] Iniciando busca híbrida por: "${query}"`, 'info');

  const queryVec = getEmbeddingForText(query);
  
  // 1. BUSCA LOCAL
  localVideos.forEach(vid => {
    const sim = cosineSimilarity(queryVec, vid.embedding);
    if (sim > 0.25) { // Threshold
      currentSearchResults.push({
        video: vid,
        similarity: sim,
        type: 'local',
        route: 'Local (HiveStore Shard)'
      });
    }
  });

  // 2. BUSCA POR FEROMÔNIOS (PONTES DE GOSSIP)
  Object.values(pheromones).forEach(p => {
    const sim = cosineSimilarity(queryVec, p.video.embedding);
    if (sim > 0.25) {
      currentSearchResults.push({
        video: p.video,
        similarity: sim,
        type: 'pheromone',
        route: `Feromônio -> Vizinho ${p.pointer} (Pista: ${p.val.toFixed(2)})`
      });
    }
  });

  // Renderizar resultados parciais imediatamente
  renderSearchResults();

  // 3. BUSCA VETORIAL NA REDE (Fallback/Descoberta Profunda)
  const connectedPeers = Object.keys(connections);
  if (connectedPeers.length > 0) {
    logToConsole(`[BUSCA] Disparando requisições vetoriais paralelas para vizinhos: ${connectedPeers.join(', ')}`, 'info');
    
    const searchMsg = {
      type: 'VECTOR_SEARCH_REQ',
      query: query,
      queryVec: queryVec,
      ttl: 2, // Limite de 2 saltos para busca profunda
      requestId: 'req-' + Math.random().toString(36).substr(2, 9)
    };

    connectedPeers.forEach(peerId => {
      sendToPeer(peerId, searchMsg);
      pendingSearchResponses.add(peerId);
    });

    // Timeout de 1.5 segundos para consolidar buscas em rede lenta
    if (searchTimeoutId) clearTimeout(searchTimeoutId);
    searchTimeoutId = setTimeout(() => {
      logToConsole('[BUSCA] Consolidação de buscas remotas finalizada.', 'system');
      pendingSearchResponses.clear();
      renderSearchResults();
    }, 1500);
  } else {
    logToConsole('[BUSCA] Nenhum peer conectado para busca na rede. Exibindo resultados locais/feromônios.', 'system');
  }
}

function handleVectorSearchReq(senderId, msg) {
  const query = msg.query;
  const queryVec = msg.queryVec;
  const ttl = msg.ttl;
  const requestId = msg.requestId;

  logToConsole(`[BUSCA REQ] Recebida busca por "${query}" de ${senderId}`, 'system');

  // Varre seus vídeos locais
  const results = [];
  localVideos.forEach(vid => {
    const sim = cosineSimilarity(queryVec, vid.embedding);
    if (sim > 0.25) {
      results.push({
        video: vid,
        similarity: sim
      });
    }
  });

  // Se achou, devolve a resposta
  if (results.length > 0) {
    sendToPeer(senderId, {
      type: 'VECTOR_SEARCH_RES',
      requestId: requestId,
      results: results
    });
  }

  // Encaminha para vizinhos se TTL > 1
  if (ttl > 1) {
    const forwardedMsg = {
      ...msg,
      ttl: ttl - 1
    };
    Object.keys(connections).forEach(peerId => {
      // Não reenvia para quem mandou a busca
      if (peerId !== senderId) {
        sendToPeer(peerId, forwardedMsg);
      }
    });
  }
}

function handleVectorSearchRes(senderId, msg) {
  logToConsole(`[BUSCA RES] Respostas de busca recebidas de ${senderId}`, 'success');
  
  msg.results.forEach(res => {
    // Evita duplicados na lista de resultados da busca atual
    const exists = currentSearchResults.some(r => r.video.id === res.video.id);
    if (!exists) {
      currentSearchResults.push({
        video: res.video,
        similarity: res.similarity,
        type: 'vector_network',
        route: `Busca Vetorial -> Swarm Node ${senderId}`
      });
    }
  });

  pendingSearchResponses.delete(senderId);
  renderSearchResults();
}

/* ==========================================================================
   SIMULAÇÃO DE DOWNLOAD (PEERHIVE VS PEERTUBE) & SELF-HEALING
   ========================================================================== */

let isDownloading = false;

function startSimulatedDownload(video, routeType, routeDetail) {
  if (isDownloading) return;
  isDownloading = true;

  logToConsole(`[DOWNLOAD] Iniciando simulação de download do vídeo: '${video.title}'...`, 'info');
  
  // Reseta visual dos cards
  resetPerformanceGraph();
  document.getElementById('comparison-analysis').style.display = 'none';

  // 1. VERIFICAÇÃO SE O PROPRIETÁRIO ESTÁ ONLINE (AUTO-CURA EM AÇÃO)
  let targetNodeId = null;
  
  if (routeType === 'local') {
    // Arquivo local, instantâneo
    logToConsole(`[DOWNLOAD] Vídeo já hospedado localmente no HiveStore. Acesso em microssegundos.`, 'success');
    runHiveDownloadSim(video, 0); // Latência local = 0
    runTubeDownloadSim(video, 0);
    isDownloading = false;
    return;
  } else if (routeType === 'pheromone') {
    // A rota por feromônio aponta para um vizinho intermediário
    // Encontramos o ponteiro correspondente em 'pheromones'
    const ph = pheromones[video.id];
    if (ph) {
      targetNodeId = ph.pointer;
    }
  } else if (routeType === 'vector_network') {
    // Busca profunda identificou o dono direto
    targetNodeId = video.owner;
  }

  // --- MECANISMO DE SELF-HEALING (AUTO-CURA) DE CHURN STORM ---
  const isPeerOnline = connections[targetNodeId] && connections[targetNodeId].open;
  
  if (!isPeerOnline) {
    logToConsole(`[ALERTA CHURN] Tentativa de baixar de '${targetNodeId}', mas o nó está OFFLINE!`, 'error');
    logToConsole(`[SELF-HEALING] Apagando pista de feromônio quebrada para o vídeo '${video.title}'.`, 'healing');
    
    // 1. Apaga a pista local de feromônios
    delete pheromones[video.id];
    
    // 2. Dispara nova busca vetorial de emergência
    logToConsole(`[SELF-HEALING] Disparando busca vetorial na rede por outro seeder ativo...`, 'healing');
    
    const queryVec = getEmbeddingForText(video.title);
    
    // Simula uma busca vetorial síncrona/rápida na lista de conexões atuais
    let foundAlternativePeer = null;
    let alternativeVideo = null;

    Object.keys(connections).forEach(peerId => {
      // Pedimos para os vizinhos conectados se eles têm o vídeo.
      // Em uma rede real, enviamos uma mensagem de busca rápida. Simulamos aqui buscando se o dono é um peer ativo.
      if (video.owner === peerId) {
        foundAlternativePeer = peerId;
        alternativeVideo = video;
      }
    });

    if (foundAlternativePeer) {
      logToConsole(`[SELF-HEALING] Sucesso! Encontrado nó alternativo vivo '${foundAlternativePeer}' contendo o vídeo. Redirecionando download.`, 'success');
      // Atualiza feromônio com a nova rota
      pheromones[video.id] = {
        video: video,
        pointer: foundAlternativePeer,
        val: 1.0 // Pista reconstruída
      };
      
      // Continua download com o novo nó alternativo
      targetNodeId = foundAlternativePeer;
    } else {
      logToConsole(`[FALHA PROTOCOLO] Nenhum nó alternativo vivo hospeda o vídeo '${video.title}'. O vídeo está temporariamente indisponível na rede.`, 'error');
      isDownloading = false;
      return;
    }
  }

  // Peer está online, iniciar os dois fluxos de download para o comparativo
  const simulatedRtt = connections[targetNodeId].rtt || SIM_LATENCY_BASE;
  
  logToConsole(`[SWARM] Conexão ativa com o seeder '${targetNodeId}' (RTT: ${simulatedRtt}ms).`, 'success');
  
  // Dispara os dois simultaneamente para o gráfico
  runHiveDownloadSim(video, simulatedRtt);
  runTubeDownloadSim(video, simulatedRtt);
}

// Fluxo PeerHive: Baixar pesos HNeRV de 10KB
function runHiveDownloadSim(video, rtt) {
  const totalSize = 10; // KB
  let downloaded = 0;
  const startTime = Date.now();
  
  logToConsole(`[PEERHIVE] Baixando payload de pesos HNeRV (10 KB)...`, 'info');
  
  // Pequena simulação de passos rápidos
  const interval = setInterval(() => {
    downloaded += 2.5; // baixa em fatias rápidas
    const pct = Math.min(100, (downloaded / totalSize) * 100);
    
    document.getElementById('hive-progress').style.width = pct + '%';
    document.getElementById('hive-bytes-val').innerText = Math.min(totalSize, downloaded).toFixed(1) + ' KB';
    
    if (pct >= 100) {
      clearInterval(interval);
      const totalTime = Date.now() - startTime + rtt; // soma RTT da sinalização
      document.getElementById('hive-time-val').innerText = totalTime + 'ms';
      logToConsole(`[PEERHIVE] Payload HNeRV recebido e instanciado localmente em ${totalTime}ms! Vídeo pronto para reproduzir.`, 'success');
      checkSimsCompletion();
    }
  }, 40); // Muito rápido
}

// Fluxo PeerTube: Baixar vídeo pesado de 3.0MB (3072 KB)
function runTubeDownloadSim(video, rtt) {
  const totalSize = 3000; // KB (3MB)
  let downloaded = 0;
  const startTime = Date.now();
  
  logToConsole(`[PEERTUBE] Solicitando arquivo de vídeo AV1 tradicional (3.0 MB)...`, 'info');
  
  // Simula velocidade doméstica média no DataChannel (ex: 500KB/s -> leva uns 6s)
  // Para fins do teste não ser exaustivo, aceleramos um pouco (ex: 2.5 segundos de simulação)
  const downloadSpeedPerTick = 120; // KB por tick (a cada 100ms)
  
  const interval = setInterval(() => {
    // Simula variação de latência (jitter)
    const jitter = Math.random() > 0.8 ? 50 : 0;
    
    downloaded += downloadSpeedPerTick;
    const pct = Math.min(100, (downloaded / totalSize) * 100);
    
    document.getElementById('tube-progress').style.width = pct + '%';
    document.getElementById('tube-bytes-val').innerText = (Math.min(totalSize, downloaded) / 1000).toFixed(2) + ' MB';
    
    if (pct >= 100) {
      clearInterval(interval);
      const totalTime = Date.now() - startTime + rtt + 300; // latência de buffer adicional
      document.getElementById('tube-time-val').innerText = (totalTime / 1000).toFixed(2) + 's';
      logToConsole(`[PEERTUBE] Arquivo de 3.0MB recebido completamente e decodificado pelo browser em ${(totalTime/1000).toFixed(2)}s.`, 'success');
      checkSimsCompletion();
    }
  }, 80);
}

let completedSims = 0;
function checkSimsCompletion() {
  completedSims++;
  if (completedSims >= 2) {
    completedSims = 0;
    isDownloading = false;
    
    // Exibe a análise comparativa na UI
    const hiveTimeText = document.getElementById('hive-time-val').innerText;
    const tubeTimeText = document.getElementById('tube-time-val').innerText;
    
    const analysisBox = document.getElementById('comparison-analysis');
    const analysisText = document.getElementById('comparison-text');
    
    analysisText.innerHTML = `<strong>PeerHive</strong> foi aproximadamente <strong>${(parseFloat(tubeTimeText) * 1000 / parseInt(hiveTimeText)).toFixed(0)}x mais rápido</strong> e economizou <strong>99.67% de largura de banda</strong> da rede em relação ao PeerTube!`;
    analysisBox.style.display = 'flex';
  }
}

function resetPerformanceGraph() {
  document.getElementById('hive-progress').style.width = '0%';
  document.getElementById('tube-progress').style.width = '0%';
  document.getElementById('hive-time-val').innerText = '0ms';
  document.getElementById('tube-time-val').innerText = '0s';
  document.getElementById('hive-bytes-val').innerText = '0 KB';
  document.getElementById('tube-bytes-val').innerText = '0 MB';
}

/* ==========================================================================
   UI CONTROLLERS & EVENT LISTENERS
   ========================================================================== */

function initUI() {
  // Conectar Peer Manualmente
  document.getElementById('btn-connect').addEventListener('click', () => {
    const targetId = document.getElementById('target-peer-id').value;
    connectToPeer(targetId);
  });

  // Copiar ID do próprio Nó
  document.getElementById('btn-copy-id').addEventListener('click', () => {
    copyTextToClipboard(myId);
    logToConsole('[SISTEMA] ID copiado para a área de transferência.', 'info');
  });

  // Indexar Vídeo
  document.getElementById('btn-index-video').addEventListener('click', () => {
    const title = document.getElementById('video-title').value;
    const sizeType = document.getElementById('video-size-select').value;
    
    if (!title || title.trim() === '') {
      alert('Digite o título do vídeo para indexar.');
      return;
    }

    indexLocalVideo(title, sizeType);
    document.getElementById('video-title').value = '';
  });

  // Buscar Vídeos
  document.getElementById('btn-search').addEventListener('click', () => {
    const query = document.getElementById('search-query').value;
    performSearch(query);
  });

  document.getElementById('search-query').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      const query = document.getElementById('search-query').value;
      performSearch(query);
    }
  });

  // Limpar Console
  document.getElementById('btn-clear-console').addEventListener('click', () => {
    const consoleArea = document.getElementById('console-log-area');
    consoleArea.innerHTML = `<div class="log-line system">[SISTEMA] Logs limpos. Nó: ${myId}</div>`;
  });
}

function updateStatus(state, text) {
  const dot = document.getElementById('net-status-dot');
  const txt = document.getElementById('net-status-text');
  
  dot.className = `status-indicator ${state}`;
  txt.innerText = text;
}

function generateFrictionlessConnectUI() {
  const qrContainer = document.getElementById('qr-container');
  const qrImg = document.getElementById('qr-code-img');
  
  // Link de conexão direta com base no ID
  const connectUrl = `${window.location.origin}${window.location.pathname}?connect=${myId}`;
  
  // Usa o serviço público e gratuito qrserver.com para gerar o QR code
  qrImg.src = `https://api.qrserver.com/v1/create-qr-code/?size=130x130&data=${encodeURIComponent(connectUrl)}`;
  qrContainer.style.display = 'flex';

  // Configura botão de copiar link
  document.getElementById('btn-copy-link').onclick = () => {
    copyTextToClipboard(connectUrl);
    logToConsole('[SISTEMA] Link de auto-conexão copiado para área de transferência!', 'info');
  };
}

function checkUrlParams() {
  const urlParams = new URLSearchParams(window.location.search);
  const targetId = urlParams.get('connect');
  
  if (targetId) {
    logToConsole(`[FALHA ZERO FRICÇÃO] Detectado parâmetro de auto-conexão para o Peer: ${targetId}`, 'success');
    // Aguarda um pequeno delay para garantir que a sinalização local esteja totalmente pronta
    setTimeout(() => {
      connectToPeer(targetId);
    }, 800);
  }
}

function updatePeersListUI() {
  const listArea = document.getElementById('active-peers-list');
  const countBadge = document.getElementById('active-peers-count');
  
  const peerIds = Object.keys(connections);
  countBadge.innerText = peerIds.length;

  if (peerIds.length === 0) {
    listArea.innerHTML = `
      <div class="empty-state">
        <i data-lucide="wifi-off"></i>
        <p>Nenhum peer conectado no swarm local.</p>
      </div>`;
    lucide.createIcons();
    return;
  }

  listArea.innerHTML = '';
  peerIds.forEach(peerId => {
    const rtt = connections[peerId].rtt || '...';
    
    const row = document.createElement('div');
    row.className = 'peer-row';
    row.innerHTML = `
      <div class="peer-info">
        <div class="peer-avatar">${peerId.charAt(peerId.length - 1).toUpperCase()}</div>
        <span class="peer-name-badge">${peerId}</span>
      </div>
      <div class="peer-controls">
        <span class="peer-ping">${rtt} ms</span>
        <button class="btn-disconnect-sim" onclick="simulatePeerDrop('${peerId}')">Simular Queda (Churn)</button>
      </div>
    `;
    listArea.appendChild(row);
  });
}

// Simulador de queda de nó (permite forçar Churn na UI para ver o Self-Healing agir)
window.simulatePeerDrop = function(peerId) {
  const conn = connections[peerId];
  if (conn) {
    conn.close();
    handlePeerDisconnection(peerId);
  }
};

function updateLocalVideosUI() {
  const list = document.getElementById('local-files-list');
  if (localVideos.length === 0) {
    list.innerHTML = `<li class="empty-list">Nenhum vídeo hospedado neste nó.</li>`;
    return;
  }

  list.innerHTML = '';
  localVideos.forEach(vid => {
    const li = document.createElement('li');
    li.className = vid.sizeType === 'traditional' ? 'traditional-file' : '';
    li.innerHTML = `
      <span>${vid.title}</span>
      <span class="file-meta-tag">${vid.sizeType === 'hnerv' ? '10 KB' : '3.0 MB'}</span>
    `;
    list.appendChild(li);
  });
}

function renderSearchResults() {
  const container = document.getElementById('search-results-list');
  if (currentSearchResults.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <i data-lucide="help-circle"></i>
        <p>Nenhum vídeo compatível encontrado na rede.</p>
      </div>`;
    lucide.createIcons();
    return;
  }

  container.innerHTML = '';
  currentSearchResults.sort((a, b) => b.similarity - a.similarity);

  currentSearchResults.forEach(res => {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    // Adiciona classe visual se for local
    const routeClass = res.type === 'local' ? 'local' : (res.type === 'pheromone' ? 'pheromone' : '');
    
    card.innerHTML = `
      <div class="result-details">
        <div class="result-title">${res.video.title}</div>
        <div class="result-source-row">
          <span class="similarity-badge">${(res.similarity * 100).toFixed(0)}% de similaridade</span>
          <span class="route-path-badge ${routeClass}">${res.route}</span>
        </div>
      </div>
      <button class="btn btn-secondary btn-sm" onclick="triggerDownloadSim('${res.video.id}')">
        Simular Download
      </button>
    `;
    container.appendChild(card);
  });
}

window.triggerDownloadSim = function(videoId) {
  const res = currentSearchResults.find(r => r.video.id === videoId);
  if (res) {
    startSimulatedDownload(res.video, res.type, res.route);
  }
};

function addMockVideos() {
  // Adiciona alguns vídeos para o nó não começar totalmente vazio
  // E cada peer gerará uma lista baseada em seu ID para que tenhamos itens diferentes na rede!
  const isAltNode = myId && myId.includes('-') && parseInt(myId.split('-')[1]) % 2 === 0;

  if (!isAltNode) {
    localVideos.push({
      id: 'vid-mock-1',
      title: 'Tutorial básico de Redes Neurais HNeRV',
      sizeType: 'hnerv',
      sizeBytes: 10240,
      embedding: getEmbeddingForText('Tutorial básico de Redes Neurais HNeRV'),
      owner: myId
    });
    localVideos.push({
      id: 'vid-mock-2',
      title: 'Gato fofo brincando com bola de lã amarela',
      sizeType: 'traditional',
      sizeBytes: 3145728,
      embedding: getEmbeddingForText('Gato fofo brincando com bola de lã amarela'),
      owner: myId
    });
  } else {
    localVideos.push({
      id: 'vid-mock-3',
      title: 'Como treinar Spiking Neural Networks (SNN)',
      sizeType: 'hnerv',
      sizeBytes: 10240,
      embedding: getEmbeddingForText('Como treinar Spiking Neural Networks (SNN)'),
      owner: myId
    });
    localVideos.push({
      id: 'vid-mock-4',
      title: 'Gameplay do clássico Doom rodando em rede neural',
      sizeType: 'hnerv',
      sizeBytes: 10240,
      embedding: getEmbeddingForText('Gameplay do clássico Doom rodando em rede neural'),
      owner: myId
    });
  }
  
  updateLocalVideosUI();
}

/* ==========================================================================
   CONSOLE LOG AREA & CORE HELPERS
   ========================================================================== */

function logToConsole(message, type = 'info') {
  const consoleArea = document.getElementById('console-log-area');
  const logLine = document.createElement('div');
  logLine.className = `log-line ${type}`;
  
  const time = new Date().toLocaleTimeString();
  logLine.innerText = `[${time}] ${message}`;
  
  consoleArea.appendChild(logLine);
  
  // Auto-scroll para a última linha
  consoleArea.scrollTop = consoleArea.scrollHeight;
}

function copyTextToClipboard(text) {
  const dummy = document.createElement("textarea");
  document.body.appendChild(dummy);
  dummy.value = text;
  dummy.select();
  document.execCommand("copy");
  document.body.removeChild(dummy);
}
