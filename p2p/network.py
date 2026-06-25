import numpy as np
import random
from collections import deque

class P2PNetwork:
    """
    Gerencia a rede overlay Peer-to-Peer (PeerHive).
    - Inicializa os nós com diferentes perfis de conexão (Fibra, 4G, 3G).
    - Monta uma topologia de rede descentralizada (Random Regular Graph / Small World).
    - Associa cada nó a um centroid semântico (representando sua 'região' de interesse no DHT).
    - Simula o Roteamento Métrico Voronoi (busca semântica descentralizada).
    - Simula a transferência P2P de mídias (Swarm de upload/download) para HNeRV (10KB) vs AV1 (3MB).
    """
    def __init__(self, dimension):
        self.dimension = dimension
        self.nodes = []
        self.node_map = {}
        
        # Centroids dos nós (usado para Roteamento Métrico estilo CAN/Chord)
        self.centroids = {}

    def add_node(self, node):
        self.nodes.append(node)
        self.node_map[node.node_id] = node
        
        # Gerar um centroid semântico aleatório para este nó
        # Nós serão responsáveis por armazenar vetores próximos ao seu centroid
        rng = np.random.RandomState(node.node_id)
        c = rng.randn(self.dimension).astype(np.float32)
        self.centroids[node.node_id] = c / np.linalg.norm(c)

    def build_topology(self, k_neighbors=4):
        """
        Conecta os nós em uma malha P2P onde cada nó tem pelo menos k_neighbors conexões.
        Garante que a rede é conexa.
        """
        n = len(self.nodes)
        if n <= k_neighbors:
            # Conectar todos entre si se forem poucos
            for i in range(n):
                for j in range(i + 1, n):
                    self.nodes[i].connect_peer(self.nodes[j])
            return

        # Conectar em anel primeiro para garantir conexidade
        for i in range(n):
            self.nodes[i].connect_peer(self.nodes[(i + 1) % n])

        # Adicionar conexões aleatórias adicionais
        for node in self.nodes:
            while len(node.peers) < k_neighbors:
                target = random.choice(self.nodes)
                node.connect_peer(target)

    def get_ping_latency(self, node_a, node_b):
        """Simula a latência de ping base (RTT) entre dois nós baseado no perfil de conexão."""
        # Se forem o mesmo nó, latência é 0
        if node_a.node_id == node_b.node_id:
            return 0.0
            
        # Determinar ping baseado nas velocidades de download (indicativo da tecnologia)
        def get_base_ping(node):
            if node.download_kbps >= 50000:  # Fibra
                return 5.0
            elif node.download_kbps >= 15000: # 4G
                return 30.0
            else:                              # 3G
                return 100.0
                
        rtt = get_base_ping(node_a) + get_base_ping(node_b)
        # Adicionar variação aleatória de jitter (±10%)
        rtt *= random.uniform(0.9, 1.1)
        return rtt / 1000.0 # Converter para segundos

    def route_write(self, vec, video_id, replicate=True):
        """
        Roteia a gravação de um vetor semântico na rede.
        O vetor é inserido no nó cujo centroid é mais próximo do vetor (Voronoi DHT).
        Opcionalmente, replica o índice nos vizinhos diretos (Opção B) para expandir o pool local.
        """
        similarities = {node.node_id: np.dot(self.centroids[node.node_id], vec) for node in self.nodes}
        best_node_id = max(similarities, key=similarities.get)
        target_node = self.node_map[best_node_id]
        
        # Insere o índice no nó alvo
        target_node.insert_index_vector(vec, video_id)
        
        # Replicar o índice para os vizinhos no overlay
        if replicate:
            for peer in target_node.peers:
                if video_id not in peer.global_to_local:
                    peer.insert_index_vector(vec, video_id)
                    
        return best_node_id

    def route_search_decentralized(self, start_node, query_vec, k=3, max_hops=10, top_n_query_nodes=1):
        """
        Roteamento Métrico Ganancioso (Greedy Metric Search) com Busca Multi-Candidato:
        Busca semântica P2P descentralizada sem coordenador central.
        
        1. Começa no nó start_node.
        2. Encaminha a busca para o vizinho mais próximo semânticamente do query_vec.
        3. Se top_n_query_nodes > 1, armazena todos os nós visitados e seleciona os top-N
           que possuem maior similaridade com o query_vec para fazer consultas locais de forma paralela.
        """
        current_node = start_node
        visited_nodes = [current_node]
        visited_ids = {current_node.node_id}
        hops = 0
        total_latency = 0.0
        
        while hops < max_hops:
            # Calcular similaridade com centroid do nó atual
            curr_sim = np.dot(self.centroids[current_node.node_id], query_vec)
            
            # Encontrar melhor vizinho
            best_neighbor = None
            best_sim = curr_sim
            
            for neighbor in current_node.peers:
                if neighbor.node_id in visited_ids:
                    continue
                sim = np.dot(self.centroids[neighbor.node_id], query_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = neighbor
            
            # Se não houver vizinho melhor (mínimo local), paramos o roteamento
            if best_neighbor is None:
                break
                
            # Acumular latência de rede do hop
            total_latency += self.get_ping_latency(current_node, best_neighbor)
            
            # Ir para o próximo nó
            current_node = best_neighbor
            visited_nodes.append(current_node)
            visited_ids.add(current_node.node_id)
            hops += 1
            
        # Determinar quais nós de consulta serão acionados (os top-N mais próximos da query)
        visited_similarities = []
        for node in visited_nodes:
            sim = np.dot(self.centroids[node.node_id], query_vec)
            visited_similarities.append((sim, node))
            
        # Ordenar decrescente por similaridade do centroid
        visited_similarities.sort(key=lambda x: x[0], reverse=True)
        query_targets = [node for _, node in visited_similarities[:top_n_query_nodes]]
        
        # Simular a latência paralela de consulta aos nós selecionados
        # (Latência total = latência do caminho de roteamento + RTT máximo para o nó mais distante consultado)
        if query_targets:
            query_pings = [self.get_ping_latency(start_node, target) for target in query_targets]
            total_latency += max(query_pings)
        
        # Adicionar latência de processamento local (mock 2ms)
        total_latency += 0.002
        
        # Consultar localmente cada um dos nós selecionados e unificar os resultados
        candidates = {}
        for target in query_targets:
            local_hits = target.search_local_knn(query_vec, k=k)
            for glob_id, vec in local_hits:
                # Calcular similaridade exata do vetor com a query
                sim = float(np.dot(query_vec, vec))
                # Se duplicado, manter a maior similaridade
                if glob_id not in candidates or sim > candidates[glob_id][0]:
                    candidates[glob_id] = (sim, vec)
                    
        # Ordenar todos os candidatos unificados
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        
        # Retornar no mesmo formato original: list of (glob_id, vec)
        final_results = [(glob_id, vec) for glob_id, (sim, vec) in sorted_candidates[:k]]
        
        return final_results, hops, total_latency, current_node.node_id

    def simulate_p2p_download(self, client_node, video_id, is_hnerv=True):
        """
        Simula a transferência P2P de mídias (HNeRV vs AV1).
        
        - Payload sizes:
          - HNeRV: 10 KB (redes neurais implícitas super leves)
          - AV1: 3000 KB (3 MB de vídeo convencional curto compactado)
          
        - Localiza todos os nós que hospedam o vídeo (seeders).
        - Calcula a largura de banda efetiva baseada no upload dos seeders e download do cliente.
        - Retorna o tempo de transferência total + latência inicial de conexão.
        """
        payload_size_kb = 10 if is_hnerv else 3000
        
        # Encontrar seeders (nós que hospedam este vídeo fisicamente)
        seeders = [node for node in self.nodes if video_id in node.hosted_videos]
        
        if not seeders:
            # Se ninguém hospeda na rede, baixa do "origin server" (simulado como conexão 3G de fallback)
            # ou assume que algum nó aleatório serve de semente inicial
            seeders = [random.choice(self.nodes)]
            
        # Latência de handshake inicial (RTT para o seeder mais próximo)
        pings = [self.get_ping_latency(client_node, s) for s in seeders]
        min_ping = min(pings)
        
        # Capacidade de upload dos seeders disponíveis
        # Em redes P2P reais, a banda de upload dos seeders é dividida.
        # Aqui, assumimos que cada seeder pode dedicar metade de seu upload para esta transferência.
        total_upload_kbps = sum(s.upload_kbps * 0.5 for s in seeders)
        
        # Limitar pela velocidade de download do cliente
        effective_kbps = min(client_node.download_kbps, total_upload_kbps)
        
        # Garantir velocidade mínima para evitar divisão por zero
        effective_kbps = max(effective_kbps, 50.0) 
        
        # Tempo de transferência de dados (Payload em bits / velocidade em Kbps)
        transfer_time = (payload_size_kb * 8.0) / effective_kbps
        
        # Latência total = handshake + tempo de transferência de dados
        total_time = min_ping + transfer_time
        
        return total_time, len(seeders), effective_kbps / 8.0  # tempo, n_seeders, velocidade em KB/s

    def gossip_pheromones(self):
        """
        Simula o Gossip Protocol de feromônios.
        Periodicamente, os nós espalham seus feromônios locais para seus vizinhos P2P.
        O feromônio recebido é mesclado com decaimento (ex: 80% do valor do vizinho).
        """
        for node in self.nodes:
            if not node.pheromones:
                continue
            # Escolher um vizinho aleatório
            neighbor = random.choice(node.peers)
            # Passar feromônios com decaimento de propagação (gossip loss)
            for vid, val in node.pheromones.items():
                decayed_val = val * 0.8
                # Manter o maior feromônio (local ou recebido)
                curr = neighbor.pheromones.get(vid, 0.0)
                if decayed_val > curr:
                    neighbor.update_pheromone(vid, decayed_val)

    def close(self):
        for node in self.nodes:
            node.close()
