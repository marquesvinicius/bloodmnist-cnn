# BloodMNIST — Classificação de células sanguíneas com CNNs

Trabalho da disciplina de Redes Neurais (UniRV, Prof. Sandro S. Moreira).
O notebook `bloodmnist.ipynb` contém o pipeline completo e o relatório técnico (seções marcadas com 📄).

- **Dataset:** BloodMNIST (MedMNIST v2), 28×28, 8 classes, splits oficiais train/val/test via biblioteca `medmnist`.
- **Modelos:** CustomCNN residual (do zero) × ResNet18 pré-treinada no ImageNet (fine-tuning, upsampling 28→128).
- **Desbalanceamento:** CrossEntropy ponderada (peso ∝ 1/√n), WeightedRandomSampler e augmentation, comparados por ablação.
- **Otimização:** AdamW × SGD+momentum, Cosine Annealing × ReduceLROnPlateau, weight decay, label smoothing, dropout, early stopping.
- **Métrica oficial:** Macro F1 no teste. Também são reportados a test loss e o tempo de inferência por batch.
- **Reprodutibilidade:** seed fixa (42), algoritmos determinísticos e seleção de modelo apenas pela validação.

## Como executar

### Google Colab (execução oficial)
1. Envie `bloodmnist.ipynb` ao Colab.
2. Escolha o ambiente com GPU T4.
3. Rode tudo. Leva cerca de 30–40 min.

O `medmnist` é instalado automaticamente. Ao final, a pasta `outputs/` é compactada e baixada.

### Local
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\jupyter notebook bloodmnist.ipynb
```
Para um teste rápido do código (poucos dados, 1–2 épocas), defina a variável de ambiente `BLOOD_QUICK=1` antes de abrir o Jupyter.

## Saídas (`outputs/`)
- `figures/`: gráficos (distribuição, curvas, matrizes de confusão, erros)
- `checkpoints/`: pesos do melhor modelo de cada execução
- `history_*.csv`: histórico de cada treino
- `todas_execucoes.csv`, `resultados_teste.csv`, `relatorio_por_classe_submissao.csv`, `resumo.json`
