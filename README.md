# Avaliação do Impacto do Aumento de Dados Baseado em Transformers na Classificação de Sinais de EEG para o Transtorno do Espectro Autista

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso (TCC) intitulado **"Avaliação do Impacto do Aumento de Dados Baseado em Transformers na Classificação de Sinais de EEG para o Transtorno do Espectro Autista"**, de autoria de Maycon Borges Costa.

## 📖 Sobre o Trabalho

A aplicação de modelos de aprendizado de máquina para classificação de sinais de eletroencefalograma (EEG) relacionados ao Transtorno do Espectro Autista (TEA) é frequentemente limitada pela escassez de bases de dados rotuladas, aumentando o risco de *overfitting*. 

O objetivo principal desta pesquisa é avaliar se a aplicação de técnicas generativas baseadas em arquiteturas **Transformer** de atenção profunda pode refinar os dados de EEG e servir como método eficaz de aumento de dados (*Data Augmentation*). A metodologia contrasta modelos treinados sem aumento de dados (*baseline*), modelos submetidos a perturbações clássicas no domínio do tempo e modelos auxiliados por dados sintéticos reconstruídos. Dois classificadores discriminativos são utilizados para avaliação: uma arquitetura profunda e híbrida (`ASDClassifier` / CNN-LSTM) e uma rede convolucional compacta (`EEGNet`).

---

## 🔬 Estruturas Avaliadas (Pipelines)

O projeto compara diversas estratégias de treinamento utilizando um protocolo rigoroso de Validação Cruzada Estratificada baseada no Sujeito (4-Folds), garantindo que não haja vazamento de dados (*data leakage*). As configurações incluem:

1. **Baseline Raw**: Classificadores treinados apenas com sinais biológicos brutos organizados em 6 ROIs (Regiões de Interesse).
2. **Jittering / Jittering Combined**: Aplicação isolada e combinada de ruído Gaussiano clássico.
3. **Scaling / Scaling Combined**: Aplicação isolada e combinada de distorção de amplitude não linear.
4. **Sliding Window / Combined**: Multiplicação de dados por translação temporal (janelas sobrepostas).
5. **Standard Transformer / Combined**: Reconstrução dos dados utilizando a arquitetura MAE (Masked Autoencoder).
6. **Slowdown Transformer / Combined**: **Proposta original deste TCC**, utilizando um gargalo temporal focado em extrair assinaturas rítmicas de alta resolução antes do refino.
7. **Mixup**: Interpolação linear convexa de amostras temporais reais e geradas (Standard e Slowdown).

---

## 📁 Estrutura do Projeto

* `paper.tex`: Código-fonte LaTeX contendo o manuscrito acadêmico do TCC.
* `code/config.py`: Arquivo central com definições de diretórios e hiperparâmetros de arquitetura.
* `code/preprocess.py`: Módulo responsável pela filtragem passa-banda (1-40 Hz) e divisão temporal (2s) extraindo janelas de sinais (ROIs) dos arquivos originais do banco de dados (formato `.set` EEGLAB).
* `code/folding.py`: Módulo divisor do *K-Fold* guiado por sujeito.
* `code/models.py`: Implementação PyTorch e Keras/TensorFlow das redes: `EEGNet`, `ASDClassifier`, `Standard Transformer` e `Slowdown Transformer`.
* `code/augmentations.py`: Rotinas matemáticas das perturbações e geração dos modelos.
* `code/classifier.py`: Loops de treinamento da `EEGNet` e de otimização hiperparamétrica.
* `code/pipeline.py`: Orquestrador central que executa todas as etapas, avalia a validação cruzada e injeta as métricas em arquivos de saída (CSV).
* `code/main.py`: Ponto de entrada CLI (Interface de Linha de Comando).
* `requirements.txt`: Relação de dependências para rodar os scripts em Python.

---

## 🚀 Como Reproduzir os Experimentos

### 1. Preparando o Ambiente

É recomendável utilizar um ambiente virtual em Python. Ative seu ambiente (Linux/MacOS) e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Rodando o Pipeline Completo

Para iniciar o treinamento sequencial e reproduzir a extração de métricas de acurácia, precisão, recall e F1-score do zero:

```bash
python code/main.py --raw-dir ./data-set/raw --force-restart
```

> **Nota:** Certifique-se de que os dados originais (em `.set`) encontram-se dentro do diretório `--raw-dir` especificado. O pipeline cuidará do pré-processamento automático, treinamento auto-supervisionado das redes geradoras e, finalmente, do *fine-tuning* dos classificadores finais.

### 3. Recuperação de Sessões Interrompidas

Treinar redes com atenção convolucional e mecanismos temporais em dados de EEG pode exigir longos períodos computacionais. O projeto foi arquitetado com checkpoints estatais (`checkpoints/checkpoint.json`). Caso um experimento crashe por falta de energia térmica/elétrica (ou para interrompê-lo manualmente), reinicie o script utilizando:

```bash
python code/main.py --resume
```

Isso instruirá o orquestrador a pular os *folds* e \textit{stages} já computados com sucesso e adicionar os resultados residuais ao arquivo de destino `experiment_results.csv`.

### 4. Customização da Execução via CLI
* `--raw-dir <path>`: Localização personalizada da base EEGLAB (padrão: `./data-set/raw`).
* `--processed-dir <path>`: Pasta para despejo temporário dos *arrays* de EEG em repouso e pós-filtros (Numpy).
* `--results-file <path>`: Nome customizado do CSV que agregará os resultados empíricos das matrizes de confusão.
* `--folds <int>`: Modifica o total de quebras de validação cruzada (O TCC utiliza e recomenda estritamente **4** para manter balanceamento das partições com 14 pacientes por conjunto).
* `--no-gpu`: Força a execução puramente em Processador de Uso Geral (CPU) desabilitando CUDA/MPS, caso existam incompatibilidades com placas gráficas de hardware instaladas na máquina.
