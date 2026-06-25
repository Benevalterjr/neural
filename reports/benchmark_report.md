# 📊 Relatório de Benchmark: FAISS vs HiveStore

Este documento consolida a análise comparativa de performance, acurácia e pegada física entre a **HiveStore** (nosso motor persistente em disco via `mmap` e otimizado com `RWLock` + `HiveSentinels`) e a biblioteca **FAISS** (Flat e HNSW), utilizando embeddings de 256-D extraídos da **BNN (Binary Neural Network)** treinada no MNIST (50.000 imagens indexadas, 5.000 queries de teste).

---

## 📈 Resumo Comparativo de Performance

| Parâmetro / Métrica | HiveStore (Persistente em Disco) | FAISS Flat (RAM Exato) | FAISS HNSW (RAM Grafo) |
| :--- | :--- | :--- | :--- |
| **Acurácia de Classificação** | `93.22%` | `93.48%` (Ground Truth) | `93.56%` |
| **Tempo de Indexação (50k)** | `267.42s` (~4.4 min) | **`0.05s`** | `7.11s` |
| **Velocidade de Busca (QPS)** | `53.27` searches/s | `1,654.04` searches/s | **`12,708.66`** searches/s ⚡ |
| **Pegada Física em Disco** | `77.73 MB` | **`48.83 MB`** | `61.81 MB` |
| **Residência Principal** | Persistido em Disco (`mmap`) | RAM (C++ Puro) | RAM (C++ Puro) |
| **Controle de Concorrência** | **RWLock Embutido** (OLTP) | Lock Manual externo | Lock Manual externo |

---

## 💡 Análise dos Resultados e Tradeoffs

### 1. Acurácia Praticamente Equivalente (Eficiência do Grafo HiveStore)
* A acurácia da HiveStore (**93.22%**) ficou a apenas **0.26%** da acurácia do FAISS Flat exato (**93.48%**).
* Isso comprova empiricamente que o nosso mecanismo de busca coarse-to-fine com **HiveSentinels em RAM + Waggle Dance no disco** alcança uma taxa de recall e classificação extremamente próxima da busca bruta exata, sem precisar ler toda a matriz do banco.

### 2. Velocidade de Busca (QPS): A Diferença RAM vs Disco
* O FAISS HNSW alcançou a incrível marca de **12.708 QPS** por rodar inteiramente compilado em C++ nativo e totalmente residente em RAM, sem qualquer barreira de interpretador.
* A HiveStore operou a **53.27 QPS** (cerca de **18.7 milissegundos por query**). Esse desempenho é decorrente de:
  1. **Acesso ao Disco**: O grafo e os vetores da HiveStore são lidos do arquivo físico mapeado via `mmap`. Embora rápido devido ao page cache, há a latência de verificação do mmap.
  2. **Loop do Grafo em Python**: A busca local (Waggle Dance) do HiveBrain executa o fluxo em Python.

### 3. Cenários de Uso e Decisão de Produção

#### Quando usar FAISS?
* **Sistemas de Alta Performance em Nuvem**: Quando você tem orçamento para provisionar servidores com memória RAM de sobra para manter toda a base carregada.
* **Leitura Exclusiva (Read-Heavy estático)**: Indexação batch rápida offline e carregamento direto em RAM para busca em sub-milissegundos.

#### Quando usar HiveStore?
* **Hardware de Baixo Custo (IoT / Edge / VPS Barata)**: Ideal se o dispositivo tem pouca RAM (ex: 512 MB - 2 GB) e não suporta manter um índice gigante de milhões de vetores na memória física.
* **Escritas OLTP Frequentes e Concorrentes**: Com o **RWLock** integrado diretamente na camada de disco, a HiveStore suporta atualizações simultâneas de banco dinâmico sem crashes de memória por alteração de ponteiros.
* **Custo de Infraestrutura**: Economia extrema em infraestrutura de nuvem, permitindo armazenar bases gigantescas em discos SSD baratos mantendo a RAM livre para outras aplicações.
