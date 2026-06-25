# 📊 Relatório Comparativo: PeerHive vs PeerTube Tradicional

> Análise detalhada dos resultados da simulação de rede descentralizada (P2P) de vídeos curtos sob **Tempestade de Churn (40% de nós caem repentinamente)**.
> Simulação executada com **50 nós** (Fibra, 4G, 3G) e **1000 requisições** de visualização.

---

## 📈 Tabela Resumo das Métricas

| Métrica | PeerTube Tradicional (AV1) | PeerHive (HNeRV + Gossip + Self-Healing) | Lift / Redução | Veredicto |
|:---|:---:|:---:|:---:|:---:|
| **Tamanho da Mídia** | 3.000 KB (3 MB) | **10 KB** | **300× menor** | Excepcional 🚀 |
| **Tempo de Buffer Médio** | 6.89 segundos | **68.6 milissegundos** | **100× mais rápido** | Experiência fluida ✅ |
| **Startup Latency p95** | 12.73 segundos | **559.7 milissegundos** | **23× menor** | Sem engasgos ✅ |
| **Banda Total Consumida** | 2903.3 MB | **9.8 MB** | **300× menos tráfego** | Economia de rede 💸 |
| **Precisão de Recomendação** | 18.1% | **22.2%** | **1.2× melhor** | Recomendação P2P viável 🎯 |
| **Taxa de Retenção Média** | 30.0% | **38.0%** | **1.26× de engajamento** | Retenção superior 📈 |
| **Taxa Sucesso Download (Churn)** | 99.1% | **100.0%** | **1.01× de resiliência** | Auto-healing ativo 🛡️ |
| **Downloads Falhados (Perdas)** | 9 | **0** | **9000000000.0× menos quedas** | Sólido sob Churn ✅ |

---

## 🔍 Conclusões e Aprendizados sob Churn de 40%

### 1. Resiliência e Auto-Healing Ativos
Sob uma **Tempestade de Churn** (onde 40% dos nós foram desligados repentinamente na requisição 500), a rede tradicional perdeu a rota para sementes de mídia e teve downloads falhados (**9 falhas** de download, taxa de sucesso de **99.1%**).
No **PeerHive**, graças ao **Self-Healing** (auto-regeneração de ponteiros quebrados com nova busca e atualização do Gossip), a taxa de sucesso de download permaneceu estável em **100.0%** (com apenas **0 falhas**).

### 2. O Fim do Gargalo de Banda P2P
O download do payload HNeRV (10KB) levou apenas **68.6ms**, enquanto o AV1 (3MB) sofria com buffering de **6.89s**, fazendo com que muitos downloads tradicionais demorassem segundos extras após a tempestade de churn devido à escassez de upload dos nós sobreviventes.

### 3. Distribuição e Index Gossip
A disseminação de ponteiros via gossip de índices garantiu que, mesmo que o nó indexador principal de um vídeo tenha caído, o caminho para outros nós seeders que guardavam cópias do vídeo em cache pôde ser localizado rapidamente pelos ponteiros de feromônio espalhados na rede.
