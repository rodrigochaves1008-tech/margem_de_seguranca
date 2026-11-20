"""Rede neural pequena (uma camada oculta, tanh), em numpy, determinística.

- Regularização L2 (alpha, na mesma escala do scikit-learn: 0,5 × alpha × ||W||² ÷ n).
- Early stopping num bloco final do treino, em ordem de tempo, separado por alguns meses;
  depois, retreina com todos os meses de treino pelo número de épocas escolhido.
- Média de várias sementes. O alpha é escolhido pela perda média na validação.
Escrita em numpy (e não com o MLP do scikit-learn) porque o early stopping embutido do
scikit-learn sorteia a validação, o que não serve para séries no tempo.
"""
from __future__ import annotations

import numpy as np


class _Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def passo(self, params, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)


def _iniciar(k: int, h: int, semente: int):
    rng = np.random.default_rng(semente)
    W1 = rng.normal(0.0, np.sqrt(2.0 / (k + h)), size=(k, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0.0, np.sqrt(2.0 / (h + 1)), size=h)
    b2 = np.zeros(1)
    return [W1, b1, W2, b2]


def _prever(params, X):
    W1, b1, W2, b2 = params
    return np.tanh(X @ W1 + b1) @ W2 + b2[0]


def _treinar(X, y, h, alpha, lr, epocas, semente, Xv=None, yv=None, paciencia=30):
    n = X.shape[0]
    params = _iniciar(X.shape[1], h, semente)
    opt = _Adam(params, lr)
    melhor, melhor_ep = np.inf, 0
    for ep in range(1, epocas + 1):
        W1, b1, W2, b2 = params
        H = np.tanh(X @ W1 + b1)
        err = H @ W2 + b2[0] - y
        gW2 = H.T @ err / n + alpha * W2 / n
        gb2 = np.array([err.mean()])
        dH = np.outer(err, W2) * (1.0 - H * H)
        gW1 = X.T @ dH / n + alpha * W1 / n
        gb1 = dH.mean(axis=0)
        opt.passo(params, [gW1, gb1, gW2, gb2])
        if Xv is not None:
            perda = float(np.mean((_prever(params, Xv) - yv) ** 2))
            if perda < melhor - 1e-12:
                melhor, melhor_ep = perda, ep
            elif ep - melhor_ep >= paciencia:
                break
    return params, melhor_ep, melhor


class RedeNeural:
    def __init__(self, ocultos: int = 8, alphas=(0.1, 1.0), taxa: float = 0.01, epocas_max: int = 400,
                 paciencia: int = 30, fracao_validacao: float = 0.2, intervalo_validacao: int = 6,
                 sementes=(11, 23, 37, 53, 71)):
        self.ocultos = ocultos
        self.alphas = tuple(alphas)
        self.taxa = taxa
        self.epocas_max = epocas_max
        self.paciencia = paciencia
        self.fracao_validacao = fracao_validacao
        self.intervalo_validacao = intervalo_validacao
        self.sementes = tuple(sementes)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RedeNeural":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mx, self.sx = X.mean(0), X.std(0)
        self.sx[self.sx == 0] = 1.0
        self.my, self.sy = y.mean(), y.std() or 1.0
        Xs, ys = (X - self.mx) / self.sx, (y - self.my) / self.sy
        n = len(ys)
        n_val = max(6, int(round(self.fracao_validacao * n)))
        gap = self.intervalo_validacao if n - n_val - self.intervalo_validacao >= 18 else 0
        fim_treino = n - n_val - gap
        Xt, yt, Xv, yv = Xs[:fim_treino], ys[:fim_treino], Xs[n - n_val:], ys[n - n_val:]
        resultados = {}
        for a in self.alphas:
            eps, perdas = [], []
            for s in self.sementes:
                _, ep, perda = _treinar(Xt, yt, self.ocultos, a, self.taxa, self.epocas_max, s, Xv, yv, self.paciencia)
                eps.append(max(ep, 1))
                perdas.append(perda)
            resultados[a] = (float(np.mean(perdas)), eps)
        self.alpha = min(self.alphas, key=lambda a: (resultados[a][0], a))
        self.epocas = resultados[self.alpha][1]
        self.params = [
            _treinar(Xs, ys, self.ocultos, self.alpha, self.taxa, ep, s)[0]
            for s, ep in zip(self.sementes, self.epocas)
        ]
        return self

    def prever_por_semente(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, float) - self.mx) / self.sx
        return np.array([_prever(p, Xs) * self.sy + self.my for p in self.params])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.prever_por_semente(X).mean(axis=0)
