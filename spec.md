# 📐 Especificação Técnica e Dimensionamento do Sistema

Este documento descreve as especificações de dimensionamento (Hot/Warm/Cold), o design da arquitetura distribuída para escala global e as políticas de resiliência e fallback no cliente para o sistema de recomendação e entrega de vídeos curtos.

---

## 📊 1. Dimensionamento para 1 Milhão de Vídeos

A tabela abaixo resume os requisitos de armazenamento calculados com base em uma média de **15 segundos de vídeo a 1.5 Mbps** (AV1 720p/1080p) e representações neurais compactadas.

### 1.1. Distribuição de Armazenamento por Camada

| Camada | Tipo de Dado | Tamanho Unitário | Total (1M de Vídeos) | Tipo de Hardware |
| :--- | :--- | :---: | :---: | :--- |
| **COLD** | Arquivos brutos de vídeo em formato AV1 | `2.8 MB` | **2.80 TB** | Storage de Objetos (S3 / Ceph / HDDs) |
| **WARM** | Previews Neurais (HNeRV) + Thumbnails | `9 KB` | **9.00 GB** | SSD Local (Paginado via `mmap`) |
| **HOT** | Grafo de Embeddings (RAM) + 500 Vídeos Ativos | `Variável` | **3.40 GB** | Memória RAM do Processo |

### 1.2. Decomposição do Uso de RAM (HOT Cache)
*   **Grafo Global de Metadados (100% da Base)**: `2.00 GB`
    *   *Fórmula*: `1.000.000 * 2 KB` (Embedding 256-D + arestas de vizinhos no grafo).
*   **Vídeos Quentes (LRU Cache)**: `1.40 GB`
    *   *Fórmula*: `500 * 2.81 MB` (Segmento AV1 completo carregado em memória física).
*   **Prefetching de Previews Ativos**: `~50 MB` (estimativa para 500 sessões simultâneas de prefetching de 4 vizinhos).

---

## 🌐 2. Arquitetura Distribuída e Escalabilidade (Clustering & Sharding)

Para escalar além de um único servidor físico, o HiveStore adota uma arquitetura de **Sharding Espacial (Voronoi)** para evitar transmissões em broadcast e garantir buscas sub-100ms em escala de bilhões de vetores.

### 2.1. Roteamento Inteligente baseando-se em Sentinelas Globais
1.  **Gateway / Router**:
    *   Mantém na memória RAM um conjunto de **Sentinelas Globais** selecionadas por FPS representando as regiões espaciais do banco.
    *   Divide o espaço vetorial em células de Voronoi distribuídas entre os nós secundários (**Shards**).
2.  **Roteamento de Escrita (Indexing)**:
    *   O embedding do vídeo é comparado com as Sentinelas Globais no Gateway.
    *   O vídeo é encaminhado exclusivamente para o **Shard** responsável pela célula de Voronoi correspondente.
    *   Garante **localidade espacial**: vídeos semelhantes são indexados na mesma máquina física, otimizando o *Waggle Dance* local.
3.  **Roteamento de Busca (Query Routing)**:
    *   A busca grosseira é executada na RAM do Gateway contra as sentinelas.
    *   A query é enviada apenas para os **2 ou 3 nós Shards mais próximos**.
    *   Cada Shard realiza a busca local aproximada no seu grafo via Cython (combinando a similaridade cosseno do embedding com o feromônio de visualizações locais) e retorna o Top-K parcial.
    *   O Gateway junta e deduplica os resultados antes de devolver ao cliente.

### 2.2. Consenso e Replicação de Leitura
*   **Replicação de Arquivos Mapeados (mmap)**: Réplicas de leitura podem copiar os arquivos estáticos de dados e abri-los via `mmap` em modo somente leitura, escalando queries horizontalmente de forma linear.
*   **Metadados e Estado**: Um protocolo de consenso leve (como Raft) gerencia o mapeamento de sentinelas globais e o status do cluster entre os nós Gateway.

### 2.3. Implementação e Resultados da Simulação de Sharding
O comportamento distribuído do sistema foi simulado e testado sob estresse:
* **Balanceamento Dinâmico (Voronoi Split):** Implementado no coordenador, dividindo células espaciais ao passar de 40% de carga. Sob carga viral concentrada, o Shard 1 dividiu-se com sucesso no Shard 5, e posteriormente este no Shard 6.
* **Redução de Carga por Pruning:** A busca roteada espacialmente restringiu-se a apenas 2 shards de 7 disponíveis, reduzindo em **71.4%** a carga computacional global do cluster.
* **Simulação de Latência Física:** Com delay artificial de 10ms por nó de rede física, queries distribuídas retornaram em **22.82 ms**, confirmando baixo overhead do software.
* **Resiliência a Quedas (Failover):** Nós com status offline dispararam roteamento para o próximo vizinho Voronoi ativo de maneira transparente para o cliente.

---

## 🧱 3. Resiliência do Cliente (Fallback Cascade)

Para dispositivos clientes com GPU fraca ou incompatibilidades de driver na decodificação da representação neural (HNeRV), o player executa a seguinte cadeia de degradação graciosa:

1.  **Capa Estática Instantânea (WebP/PNG)**:
    *   Durante a requisição inicial, o player baixa a thumbnail tradicional leve (5 KB) junta com os pesos HNeRV.
    *   Se o decodificador neural falhar ao decodificar os frames em menos de **50ms**, a thumbnail estática é renderizada imediatamente na tela do feed.
2.  **Chaveamento de Decoder (Hardware para Software WASM)**:
    *   O player tenta acelerar via hardware (WebGL/WebGPU).
    *   Em caso de falha de driver, troca automaticamente para um interpretador decodificador compilado em **WebAssembly (WASM)** rodando puramente na CPU através de instruções SIMD.
3.  **Reprodução do Cold AV1 (Fallback Final)**:
    *   Se todo o pipeline de decodificação neural HNeRV falhar, o player ignora as pré-visualizações neurais de baixa latência e inicia o download sequencial padrão do arquivo de vídeo **AV1 completo** do Cold Storage.
    *   Garante que o vídeo sempre seja reproduzido, mesmo que com um pequeno atraso convencional de buffering de rede.

---

## ⚡ 4. Expansão de Hot-Paths em C/Cython (hive_ops)

Para maximizar a eficiência computacional e reduzir o tempo de CPU gasto em loops de alto volume, foram migrados dois caminhos críticos essenciais para C puro via Cython:

### 4.1. Distância Hamming Nativa (`fast_hamming`)
*   **Finalidade**: Projetado para busca rápida no grafo utilizando embeddings binarizados gerados pela BNN (onde cada float assume valores discretizados de sinal $\ge 0$ vs $< 0$).
*   **Implementação**: loop linear em nível C comparando bitwise/sinal diretamente de forma vetorizável, evitando a sobrecarga de memória e alocação dinâmica do Numpy.
*   **Busca no Grafo**: Métodos de busca no grafo (`search_hamming_cython` e `find_neighbors_hamming_cython`) implementados diretamente na camada compilada com ordenação de scores negativos para preservação de similaridade de maior valor.

### 4.2. Merge de Resultados Distribuídos (`merge_results_cython`)
*   **Finalidade**: Otimiza a agregação e fusão de vizinhos candidatos provenientes de múltiplos shards no Gateway do cluster (etapa de agregação no GatewayCoordinator).
*   **Implementação**: Recebe a lista de candidatos agregados dos shards, computa as similaridades de produto escalar em nível de C (utilizando `fast_dot`) e ordena os top-k elementos de forma nativa, retornando a lista final consolidada com overhead mínimo de Python.

