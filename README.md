# Laboratório 4 — O Transformer Completo "From Scratch"

## Descrição

Implementação completa da arquitetura Transformer (Encoder-Decoder) utilizando NumPy, integrando todos os módulos desenvolvidos nos laboratórios anteriores para realizar um teste de tradução fim-a-fim com uma sequência de brinquedo (toy sequence).

## Estrutura do Projeto

```
Laboratorio-P1-04/
├── transformer.py   # Código principal com toda a implementação
└── README.md        # Este arquivo
```

## Como Executar

### Pré-requisitos

```bash
pip install numpy
```

### Execução

```bash
python transformer.py
```

A saída exibirá o loop auto-regressivo de inferência, mostrando cada token gerado até o símbolo `<EOS>`.

## Arquitetura Implementada

O projeto implementa os seguintes componentes do Transformer:

1. **Scaled Dot-Product Attention** — mecanismo de atenção base com suporte a máscaras
2. **Feed-Forward Network (FFN)** — rede densa position-wise com expansão de dimensão e ReLU
3. **Add & Norm** — conexão residual com normalização de camada (Layer Normalization)
4. **EncoderBlock** — bloco do Encoder com Self-Attention + FFN
5. **DecoderBlock** — bloco do Decoder com Masked Self-Attention + Cross-Attention + FFN
6. **Loop Auto-regressivo** — inferência fim-a-fim com geração token a token

## Nota sobre Uso de IA

Partes geradas/complementadas com IA, revisadas por Vinicius.

## Créditos

Ferramentas de IA (Claude) foram consultadas para auxílio em dúvidas de sintaxe do NumPy durante o desenvolvimento.
