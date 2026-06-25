# 🐝 HiveStore & Neural Engine (BNN + SNN)

Este projeto implementa um ecossistema completo e de alta performance que integra Redes Neurais de Pesos Binários (**Binary Neural Networks - BNN**), Redes Neurais Espetaculares Sequenciais baseadas em Spikes (**Spiking RNN - RWKV** baseada no SpikeGPT) e o **HiveStore** — um banco de dados vetorial e em grafo persistido em disco via arquivos mapeados em memória (`mmap`) e otimizado com o algoritmo biomimético de busca espacial inspirado na **Dança do Rebolado (Waggle Dance)** das abelhas.

O projeto conta com controle fino de concorrência com **Read-Write Lock (RWLock)** para segurança multithread em produção (OLTP) e aceleração espacial coarse-to-fine através de **HiveSentinels** em RAM selecionadas por **Farthest Point Sampling (FPS)**.

---

## 📐 Arquitetura do Sistema

O sistema é dividido em três pilares fundamentais:

### 1. Extrator Neural Binário (BNN)
Implementado em [bnn.py](file:///G:/dyad-apps/dyad-apps/neural/bnn.py), o modelo **StableSparseBNN** treina uma rede de pesos binários ($\{-1, 1\}$) no MNIST para extrair embeddings de 256 bits de representação latente.
* **Propagação de Gradiente STE**: Utiliza uma aproximação de gradiente substituto baseada em arcotangente exata no retroprocesso.
* **Backpropagation Personalizado**: Implementação matemática de backpropagation de normalização em lote (Batch Normalization) sem dependências externas de cálculo automático de gradiente, otimizado com o otimizador **Adam**.

### 2. Rede Sequencial Baseada em Pulsos (SNN)
Implementada em [spiking_rwkv.py](file:///G:/dyad-apps/dyad-apps/neural/spiking_rwkv.py), a classe **SpikingRWKVMNIST** simula a dinâmica temporal e eficiência energética do SpikeGPT utilizando neurônios **Leaky Integrate-and-Fire (LIF)**.
* **Token Shift & SRWKV**: Otimiza a dependência temporal processando imagens linha por linha de forma sequencial unidirecional.
* **BPTT (Backpropagation Through Time)**: Otimização temporal com controle de concorrência de CPU (NumPy configurado para rodar em thread único para evitar conflitos no Windows).

### 3. Banco de Dados Vetorial & Grafo (HiveStore)
Implementado em [test_hivestore.py](file:///G:/dyad-apps/dyad-apps/neural/test_hivestore.py), gerencia a persistência binária e a recuperação espacial:
* **GrowableBuffer**: Camada de baixo nível que gerencia o crescimento dinâmico e o truncamento automático de arquivos via `mmap` do SO de forma alinhada a páginas de 4 KB.
* **HiveSentinels (Coarse Quantizer)**: Mantém um mini-codebook espacial em RAM selecionado por *Farthest Point Sampling* (FPS). A query executa uma busca em lote ultra-rápida na RAM contra as sentinelas antes de descer para o disco, descobrindo os melhores portões de entrada e evitando mínimos locais.
* **Waggle Dance (Fine Quantizer)**: Executa uma busca de feixe local (Beam Search) que navega concorrentemente pelas arestas mapeadas em memória do grafo k-NN.
* **Read-Write Lock (RWLock)**: Garante a thread-safety do banco sob carga mista de escrita e leitura simultâneas, impedindo crashes de `ValueError` quando buffers sofrem redimensionamento em tempo de execução.

---

## 📊 Resultados e Métricas (Benchmark de Escala)

Os testes foram executados com **50.000 vetores indexados** (256 dimensões) e **5.000 imagens de teste paralela**, obtendo as seguintes marcas:

* **Acurácia Direta do BNN (Softmax)**: `93.39%`
* **Acurácia por Recuperação no Grafo (BNN + HiveStore)**: **`94.20%`** 🎯
  *(Superando a Softmax do BNN devido à regularização geométrica no espaço de Hamming).*
* **Velocidade de Busca Concorrente**: **`81.53 QPS`** (~12 milissegundos por busca, paralela em CPU multi-core).
* **Consumo de RAM das Sentinelas (RAM)**: `<200 KB` (para 500 sentinelas espaciais).
* **Consistência pós-persistência**: `100.00%` (determinismo binário garantido).
* **Estresse Concorrente**: **100% de Sucesso** (rodando 1 thread de escrita e 4 de leitura simultâneas sob alta carga sem falhas).

---

## 📂 Estrutura do Diretório

```bash
├── G:/dyad-apps/dyad-apps/neural/
│   ├── bnn.py                      # Implementação da BNN (Treinamento & STE)
│   ├── spiking_rwkv.py             # Modelo SNN com dinâmica temporal LIF e Token Shift
│   ├── bnn_hive_pipeline.py        # Script do Pipeline Integrado e Teste de Escala
│   ├── test_hivestore.py           # O motor HiveStore, RWLock e Testes de Concorrência
│   ├── hivestore.ipynb             # Jupyter Notebook interativo atualizado com Sentinelas
│   ├── experiment_results.md       # Relatório de resultados do primeiro teste
│   ├── scale_experiment_results.md # Relatório de resultados do teste de escala e RWLock
│   ├── README.md                   # Este arquivo de documentação
│   └── profiling/                  # Pasta com scripts secundários e micro-benchmarks
│       ├── benchmark.py
│       ├── micro_profile.py
│       ├── micro_profile_threads.py
│       ├── profile_step.py
│       ├── test_eval_speed.py
│       ├── test_run.py
│       └── test_threads.py
```

---

## 🚀 Instalação e Execução

### Pré-requisitos
* Python 3.10 ou superior instalado.
* Compilador C++ (para compilar extensões caso necessário, embora a implementação atual use NumPy otimizado).

### Configuração do Ambiente

1. Crie e ative o ambiente virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Instale as dependências necessárias:
   ```powershell
   pip install numpy scipy tensorflow
   ```

### Executando os Testes

* **Teste do Banco de Dados (Estresse + Persistência + Concorrência Multithread)**:
  ```powershell
  .\.venv\Scripts\python.exe test_hivestore.py
  ```
* **Executar o Pipeline Integrado Completo (Treinamento BNN + Indexação de 50.000 amostras + Teste de Escala com RWLock)**:
  ```powershell
  .\.venv\Scripts\python.exe -u bnn_hive_pipeline.py
  ```

---

## 📄 Licença

Este projeto é licenciado sob a **Licença Apache 2.0**.

```text
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 🐝 Agradecimentos Biomiméticos
A Dança do Rebolado (*Waggle Dance*) é um comportamento biológico extraordinário pelo qual as abelhas operárias informam as suas companheiras de colmeia sobre a distância e a direção de fontes de alimento excelentes. A sua aplicação matemática à busca em grafos aproxima a inteligência descentralizada natural à computação física e resiliente.
