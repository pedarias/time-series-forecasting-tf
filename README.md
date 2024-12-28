# Time Series Forecasting com TensorFlow

Este repositório contém um projeto de estudo para **previsão de séries temporais** usando TensorFlow. Nele, abordamos:

- **Preprocessamento**: subamostragem de dados (de 10min para 1h), limpeza de valores anômalos, engenharia de atributos (vetores de vento, periodicidade diária e anual) e normalização.
- **Criação de janelas** (`WindowGenerator`): organização dos dados em blocos (inputs e labels) respeitando a estrutura temporal.
- **Modelos**:
  1. **Single-step**: onde o objetivo é prever apenas 1 hora à frente.
     - Baseline (retorna valor atual como previsão).
     - Modelos (Linear, Dense, …).
  2. **Multi-step** (24 horas): gerando um perfil de 24h no futuro.
     - Baselines (Last e Repeat).
     - Modelos single-shot (Linear, Dense, CNN, LSTM).
     - Modelo autoregressivo (FeedBack).

## Conteúdo

1. `notebook`
   - Contém o notebook principal (`tsf1.ipynb` ou nome similar) com todo o pipeline:
     - Leitura de dados e subamostragem.
     - Eng. de atributos (vento em Wx/Wy, Day/Year sin/cos).
     - Normalização e *split* em Treino/Val/Teste.
     - Modelos single-step e multi-step, com comparações de métricas.
2. `README.md`
   - Explicações gerais do projeto, instruções de uso, referências.

## Como Executar

1. **Clonar o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/time-series-forecasting-tf.git
   cd time-series-forecasting-tf
