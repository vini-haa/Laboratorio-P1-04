import numpy as np

# =============================================================================
# Laboratório 4 — O Transformer Completo "From Scratch"
# =============================================================================


# =============================================================================
# COMMIT 2 — Scaled Dot-Product Attention
# Equação: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# =============================================================================

def softmax(x):
    """Softmax numericamente estável aplicado na última dimensão."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calcula a atenção escalada por produto escalar.

    Parâmetros:
        Q: matriz de queries  — shape (seq_len_q, d_k)
        K: matriz de keys     — shape (seq_len_k, d_k)
        V: matriz de values   — shape (seq_len_k, d_v)
        mask: máscara binária — shape (seq_len_q, seq_len_k), opcional
              onde 0 indica posições a serem bloqueadas (substituídas por -inf)

    Retorna:
        output: tensor de saída — shape (seq_len_q, d_v)
        weights: pesos de atenção após softmax — shape (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # Passo 1: scores = QK^T / sqrt(d_k)
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # Passo 2: aplicar máscara (posições bloqueadas recebem -inf → zero no softmax)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Passo 3: softmax para obter os pesos de atenção
    weights = softmax(scores)

    # Passo 4: multiplicar pelos values
    output = np.matmul(weights, V)

    return output, weights


# =============================================================================
# COMMIT 3 — Feed-Forward Network (FFN) e Add & Norm
# FFN:      FFN(x) = ReLU(xW1 + b1)W2 + b2
# Add&Norm: Output = LayerNorm(x + Sublayer(x))
# =============================================================================

def feed_forward_network(x, W1, b1, W2, b2):
    """
    Position-wise Feed-Forward Network.

    Parâmetros:
        x:  entrada — shape (seq_len, d_model)
        W1: pesos da 1ª camada — shape (d_model, d_ff)
        b1: bias da 1ª camada  — shape (d_ff,)
        W2: pesos da 2ª camada — shape (d_ff, d_model)
        b2: bias da 2ª camada  — shape (d_model,)

    Retorna:
        saída — shape (seq_len, d_model)
    """
    # Projeção d_model → d_ff com ativação ReLU
    hidden = np.maximum(0, np.dot(x, W1) + b1)
    # Projeção d_ff → d_model
    output = np.dot(hidden, W2) + b2
    return output


def layer_norm(x, eps=1e-6):
    """
    Layer Normalization aplicada na última dimensão.
    Fórmula: (x - mean) / (std + eps)
    """
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def add_and_norm(x, sublayer_output):
    """
    Conexão residual + Layer Normalization.
    Fórmula: LayerNorm(x + Sublayer(x))
    """
    return layer_norm(x + sublayer_output)


# =============================================================================
# COMMIT 4 — EncoderBlock
# Fluxo: X → Self-Attention → Add&Norm → FFN → Add&Norm → Z
# =============================================================================

def init_ffn_weights(d_model, d_ff):
    """Inicializa os pesos da FFN com valores aleatórios pequenos."""
    W1 = np.random.randn(d_model, d_ff) * 0.01
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.01
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


def encoder_block(x, d_model, d_ff):
    """
    Um bloco do Encoder do Transformer.

    Parâmetros:
        x:       tensor de entrada com Positional Encoding somado
                 shape (seq_len, d_model)
        d_model: dimensão do modelo
        d_ff:    dimensão interna da FFN (ex: 4 * d_model)

    Retorna:
        Z: memória contextualizada — shape (seq_len, d_model)
    """
    # --- Sublayer 1: Self-Attention ---
    # Q, K, V são todos derivados de x (Self-Attention bidirecional, sem máscara)
    attn_out, _ = scaled_dot_product_attention(Q=x, K=x, V=x, mask=None)

    # Add & Norm após Self-Attention
    x = add_and_norm(x, attn_out)

    # --- Sublayer 2: Feed-Forward Network ---
    W1, b1, W2, b2 = init_ffn_weights(d_model, d_ff)
    ffn_out = feed_forward_network(x, W1, b1, W2, b2)

    # Add & Norm após FFN
    Z = add_and_norm(x, ffn_out)

    return Z


def encoder_stack(x, d_model, d_ff, n_layers):
    """
    Empilha N blocos de Encoder sequencialmente.

    Parâmetros:
        x:        tensor de entrada — shape (seq_len, d_model)
        d_model:  dimensão do modelo
        d_ff:     dimensão interna da FFN
        n_layers: número de blocos empilhados

    Retorna:
        Z: memória final do Encoder — shape (seq_len, d_model)
    """
    Z = x
    for _ in range(n_layers):
        Z = encoder_block(Z, d_model, d_ff)
    return Z


# =============================================================================
# COMMIT 5 — DecoderBlock
# Fluxo: Y → Masked Self-Attn → Add&Norm → Cross-Attn(Z) → Add&Norm → FFN → Add&Norm → probs
# =============================================================================

def create_causal_mask(seq_len):
    """
    Cria a máscara causal (triangular inferior) para o Masked Self-Attention.
    Posições 1 = permitidas, 0 = bloqueadas (futuro).

    Exemplo para seq_len=3:
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    return np.tril(np.ones((seq_len, seq_len)))


def decoder_block(y, Z, d_model, d_ff, vocab_size):
    """
    Um bloco do Decoder do Transformer.

    Parâmetros:
        y:          tensor alvo (já com Positional Encoding) — shape (seq_len_y, d_model)
        Z:          memória do Encoder                       — shape (seq_len_x, d_model)
        d_model:    dimensão do modelo
        d_ff:       dimensão interna da FFN
        vocab_size: tamanho do vocabulário para projeção final

    Retorna:
        probs: distribuição de probabilidade sobre o vocabulário
               shape (seq_len_y, vocab_size)
    """
    seq_len_y = y.shape[0]

    # --- Sublayer 1: Masked Self-Attention ---
    # Máscara causal impede que cada posição veja tokens futuros
    causal_mask = create_causal_mask(seq_len_y)
    masked_attn_out, _ = scaled_dot_product_attention(Q=y, K=y, V=y, mask=causal_mask)

    # Add & Norm
    y = add_and_norm(y, masked_attn_out)

    # --- Sublayer 2: Cross-Attention ---
    # Q vem da saída anterior do Decoder; K e V vêm da memória Z do Encoder
    cross_attn_out, _ = scaled_dot_product_attention(Q=y, K=Z, V=Z, mask=None)

    # Add & Norm
    y = add_and_norm(y, cross_attn_out)

    # --- Sublayer 3: Feed-Forward Network ---
    W1, b1, W2, b2 = init_ffn_weights(d_model, d_ff)
    ffn_out = feed_forward_network(y, W1, b1, W2, b2)

    # Add & Norm
    y = add_and_norm(y, ffn_out)

    # --- Projeção Linear + Softmax ---
    # Projeta de d_model → vocab_size para obter logits
    W_vocab = np.random.randn(d_model, vocab_size) * 0.01
    logits = np.dot(y, W_vocab)           # shape (seq_len_y, vocab_size)
    probs  = softmax(logits)              # shape (seq_len_y, vocab_size)

    return probs


def decoder_stack(y, Z, d_model, d_ff, vocab_size, n_layers):
    """
    Empilha N blocos de Decoder sequencialmente.
    Apenas o último bloco projeta para o vocabulário.

    Retorna:
        probs: shape (seq_len_y, vocab_size)
    """
    seq_len_y = y.shape[0]

    # Camadas intermediárias: retornam representação em d_model
    for _ in range(n_layers - 1):
        causal_mask = create_causal_mask(seq_len_y)
        attn_out, _ = scaled_dot_product_attention(Q=y, K=y, V=y, mask=causal_mask)
        y = add_and_norm(y, attn_out)
        cross_out, _ = scaled_dot_product_attention(Q=y, K=Z, V=Z, mask=None)
        y = add_and_norm(y, cross_out)
        W1, b1, W2, b2 = init_ffn_weights(d_model, d_ff)
        ffn_out = feed_forward_network(y, W1, b1, W2, b2)
        y = add_and_norm(y, ffn_out)

    # Último bloco: inclui projeção Linear + Softmax
    probs = decoder_block(y, Z, d_model, d_ff, vocab_size)
    return probs


# -----------------------------------------------------------------------------
# Teste rápido
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model, d_ff = 4, 8, 32

    # --- Teste: scaled_dot_product_attention ---
    d_k = d_model
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("=== Teste: scaled_dot_product_attention ===")
    print(f"Shape de Q, K, V : ({seq_len}, {d_k})")
    print(f"Shape da saída   : {output.shape}")
    print(f"Shape dos pesos  : {weights.shape}")
    print(f"Soma dos pesos (deve ser ~1.0 por linha): {weights.sum(axis=-1)}")
    print("Teste passou!\n")

    # --- Teste: feed_forward_network ---
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.zeros(d_model)
    x  = np.random.randn(seq_len, d_model)

    ffn_out = feed_forward_network(x, W1, b1, W2, b2)

    print("=== Teste: feed_forward_network ===")
    print(f"Shape entrada : {x.shape}")
    print(f"Shape saída   : {ffn_out.shape}  (deve ser igual à entrada)")
    print("Teste passou!\n")

    # --- Teste: add_and_norm ---
    sublayer_out = np.random.randn(seq_len, d_model)
    norm_out = add_and_norm(x, sublayer_out)

    print("=== Teste: add_and_norm ===")
    print(f"Shape entrada : {x.shape}")
    print(f"Shape saída   : {norm_out.shape}  (deve ser igual à entrada)")
    print(f"Média por posição (deve ser ~0): {norm_out.mean(axis=-1).round(6)}")
    print("Teste passou!\n")

    # --- Teste: encoder_block ---
    x_enc = np.random.randn(seq_len, d_model)
    Z = encoder_block(x_enc, d_model, d_ff)

    print("=== Teste: encoder_block ===")
    print(f"Shape entrada (X): {x_enc.shape}")
    print(f"Shape saída   (Z): {Z.shape}  (deve ser igual à entrada)")
    print("Teste passou!\n")

    # --- Teste: encoder_stack (2 camadas) ---
    Z_stack = encoder_stack(x_enc, d_model, d_ff, n_layers=2)

    print("=== Teste: encoder_stack (2 camadas) ===")
    print(f"Shape entrada (X): {x_enc.shape}")
    print(f"Shape saída   (Z): {Z_stack.shape}  (deve ser igual à entrada)")
    print("Teste passou!\n")

    # --- Teste: create_causal_mask ---
    mask = create_causal_mask(4)
    print("=== Teste: create_causal_mask (seq_len=4) ===")
    print(mask)
    print("Teste passou!\n")

    # --- Teste: decoder_block ---
    vocab_size = 10
    seq_len_y  = 3
    y_dec = np.random.randn(seq_len_y, d_model)
    Z_enc = encoder_block(x_enc, d_model, d_ff)

    probs = decoder_block(y_dec, Z_enc, d_model, d_ff, vocab_size)

    print("=== Teste: decoder_block ===")
    print(f"Shape entrada (Y)  : {y_dec.shape}")
    print(f"Shape memória (Z)  : {Z_enc.shape}")
    print(f"Shape saída (probs): {probs.shape}  (deve ser ({seq_len_y}, {vocab_size}))")
    print(f"Soma das probs por posição (deve ser ~1.0): {probs.sum(axis=-1).round(6)}")
    print("Teste passou!\n")
