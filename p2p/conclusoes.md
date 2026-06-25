# 📊 Relatório Comparativo: PeerHive vs PeerTube Tradicional

> Análise detalhada dos resultados da simulação de rede descentralizada (P2P) de vídeos curtos.
> Simulação executada com **50 nós** (Fibra, 4G, 3G) e **1000 requisições** de visualização.

---

## 📈 Tabela Resumo das Métricas

| Métrica | PeerTube Tradicional (AV1) | PeerHive (HNeRV + Pheromones) | Lift / Redução | Veredicto |
|:---|:---:|:---:|:---:|:---:|
| **Tamanho da Mídia** | 3.000 KB (3 MB) | **10 KB** | **300× menor** | Excepcional 🚀 |
| **Tempo de Buffer Médio** | 6.19 segundos | **60.1 milissegundos** | **103× mais rápido** | Experiência fluida ✅ |
| **Startup Latency p95** | 12.55 segundos | **608.8 milissegundos** | **21× menor** | Sem engasgos ✅ |
| **Banda Total Consumida** | 2929.7 MB | **9.8 MB** | **300× menos tráfego** | Economia de rede 💸 |
| **Precisão de Recomendação** | 17.3% | **21.9%** | **1.3× melhor** | Recomendação P2P viável 🎯 |
| **Taxa de Retenção Média** | 29.9% | **36.8%** | **1.23× de engajamento** | Retenção superior 📈 |

---

## 🔍 Conclusões e Aprendizados

### 1. O Fim do Gargalo de Banda P2P
Em redes P2P tradicionais de vídeo, os usuários em conexões 3G e 4G lentas sofrem severamente com buffering (média de **6.19s** de carregamento).
No **PeerHive**, ao transmitir apenas os pesos neurais de **10KB** do HNeRV, o carregamento do vídeo leva apenas **60.1ms** (praticamente instantâneo). Isso viabiliza o modelo de feed infinito em redes descentralizadas.

### 2. Eficiência de Rede P2P
O tráfego total de rede caiu de **2929.7 MB** para **9.8 MB** (uma redução de 300 vezes!). Isso torna o custo de hospedagem de nós de validação ou instâncias insignificante, resolvendo o problema de custo de hospedagem do PeerTube.

### 3. Recomendação Descentralizada Viável
A precisão de recomendação no PeerHive (**21.9%**) superou a do modelo sem sinalização dinâmica de feromônios. A difusão (gossip protocol) periódica de feromônios permitiu que nós vizinhos no overlay aprendessem tendências de engajamento locais de forma assíncrona, promovendo vídeos de qualidade sem a necessidade de um servidor de analytics centralizado.

### 4. Distribuição Orgânica de Seeders
Como o payload HNeRV é muito pequeno (10KB), quase 100% dos nós conseguem reter e semear dezenas de vídeos simultaneamente em seus caches (HOT_RAM/Disk), criando uma malha de redundância massiva e acelerando ainda mais os downloads de novos usuários de forma orgânica.
