# Enfoque Bayesiano para Experimentación

## Descripción General

El enfoque bayesiano trata las tasas de conversión (y otras métricas) como **distribuciones de probabilidad** en lugar de valores fijos. Permite incorporar conocimiento previo y actualizar creencias conforme llegan datos nuevos.

## Características Principales

### Ventajas
- ✅ Permite monitoreo continuo sin inflar errores
- ✅ Incorpora conocimiento previo (priors)
- ✅ Responde preguntas directas: P(A > B)
- ✅ Decisiones más rápidas que frecuentista
- ✅ Más intuitivo: probabilidades vs. p-valores
- ✅ Cuantifica incertidumbre de forma natural

### Desventajas
- ❌ Requiere especificar distribuciones prior
- ❌ Más complejo computacionalmente
- ❌ Puede ser difícil de explicar a stakeholders
- ❌ Sensible a la elección del prior (en muestras pequeñas)
- ❌ Menos estándares de industria establecidos

## Fundamentos Teóricos

### Teorema de Bayes

```
Posterior ∝ Likelihood × Prior

P(θ|data) = P(data|θ) × P(θ) / P(data)
```

Donde:
- **Prior P(θ)**: Creencia antes de ver datos
- **Likelihood P(data|θ)**: Probabilidad de los datos dado θ
- **Posterior P(θ|data)**: Creencia actualizada después de ver datos

### Conjugate Priors

Para métrica binaria (conversión), usamos **Beta-Binomial conjugacy**:

```
Prior: θ ~ Beta(α, β)
Likelihood: x ~ Binomial(n, θ)
Posterior: θ ~ Beta(α + x, β + n - x)
```

Esta conjugación hace el cálculo analítico y rápido.

## Implementación para A/B Testing

### Prior: Beta Distribution

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class BayesianABTest:
    """Implementación de A/B test Bayesiano con Beta-Binomial."""

    def __init__(
        self,
        prior_alpha: float = 1,
        prior_beta: float = 1
    ):
        """
        Inicializa test bayesiano.

        Args:
            prior_alpha: α del prior Beta (default: 1 = uniform prior)
            prior_beta: β del prior Beta (default: 1 = uniform prior)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def posterior(
        self,
        conversions: int,
        trials: int
    ) -> beta:
        """
        Calcula distribución posterior.

        Args:
            conversions: número de conversiones
            trials: número de usuarios

        Returns:
            scipy.stats.beta distribution
        """
        alpha_post = self.prior_alpha + conversions
        beta_post = self.prior_beta + (trials - conversions)

        return beta(alpha_post, beta_post)

    def probability_b_beats_a(
        self,
        conversions_a: int,
        trials_a: int,
        conversions_b: int,
        trials_b: int,
        n_samples: int = 100000
    ) -> float:
        """
        Calcula P(B > A).

        Args:
            conversions_a, trials_a: datos de variante A
            conversions_b, trials_b: datos de variante B
            n_samples: número de muestras Monte Carlo

        Returns:
            Probabilidad de que B sea mejor que A
        """
        # Posteriors
        post_a = self.posterior(conversions_a, trials_a)
        post_b = self.posterior(conversions_b, trials_b)

        # Monte Carlo sampling
        samples_a = post_a.rvs(n_samples)
        samples_b = post_b.rvs(n_samples)

        # P(B > A)
        prob = (samples_b > samples_a).mean()

        return prob

    def expected_loss(
        self,
        conversions_a: int,
        trials_a: int,
        conversions_b: int,
        trials_b: int,
        n_samples: int = 100000
    ) -> dict:
        """
        Calcula pérdida esperada de elegir cada variante.

        Returns:
            dict con 'loss_if_choose_a' y 'loss_if_choose_b'
        """
        post_a = self.posterior(conversions_a, trials_a)
        post_b = self.posterior(conversions_b, trials_b)

        samples_a = post_a.rvs(n_samples)
        samples_b = post_b.rvs(n_samples)

        # Loss = diferencia cuando estamos equivocados
        loss_choose_a = np.maximum(0, samples_b - samples_a).mean()
        loss_choose_b = np.maximum(0, samples_a - samples_b).mean()

        return {
            'loss_if_choose_a': loss_choose_a,
            'loss_if_choose_b': loss_choose_b,
            'relative_loss_a': loss_choose_a / samples_a.mean(),
            'relative_loss_b': loss_choose_b / samples_b.mean()
        }


# Ejemplo de uso
test = BayesianABTest(prior_alpha=1, prior_beta=1)

# Datos observados
conversions_control = 450
trials_control = 5000
conversions_treatment = 520
trials_treatment = 5000

# P(Treatment > Control)
prob = test.probability_b_beats_a(
    conversions_control, trials_control,
    conversions_treatment, trials_treatment
)

print(f"P(Treatment > Control): {prob:.2%}")

# Expected Loss
loss = test.expected_loss(
    conversions_control, trials_control,
    conversions_treatment, trials_treatment
)

print(f"Loss si elegimos Control: {loss['loss_if_choose_a']:.4f}")
print(f"Loss si elegimos Treatment: {loss['loss_if_choose_b']:.4f}")
```

## Elección del Prior

### 1. Uniform Prior (No Informativo)

```python
# Beta(1, 1) = Uniform(0, 1)
prior_alpha = 1
prior_beta = 1

# Significa: todas las tasas de conversión son igualmente probables
```

**Cuándo usar:**
- No tienes información previa
- Quieres que los datos "hablen por sí mismos"
- Primera vez que pruebas algo

### 2. Informative Prior (Con Conocimiento)

```python
def create_informative_prior(
    historical_mean: float,
    historical_std: float
) -> tuple:
    """
    Crea prior Beta basado en datos históricos.

    Args:
        historical_mean: tasa de conversión histórica (ej: 0.10)
        historical_std: desviación estándar histórica (ej: 0.01)

    Returns:
        (alpha, beta) para Beta distribution
    """
    # Method of moments para Beta distribution
    mean = historical_mean
    var = historical_std ** 2

    # Fórmulas de method of moments
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)

    return (alpha, beta)


# Ejemplo
# Históricamente: 10% conversión con std de 1%
alpha, beta_param = create_informative_prior(
    historical_mean=0.10,
    historical_std=0.01
)

print(f"Prior: Beta({alpha:.1f}, {beta_param:.1f})")

# Usar en el test
test = BayesianABTest(prior_alpha=alpha, prior_beta=beta_param)
```

### 3. Weakly Informative Prior (Regularización)

```python
# Beta(5, 5) - centrado en 50%, pero permite flexibilidad
prior_alpha = 5
prior_beta = 5

# Equivalente a ~10 observaciones "sintéticas"
# Útil para regularizar con datos pequeños
```

### Comparación Visual

```python
def plot_prior_comparison():
    """Visualiza diferentes priors."""
    x = np.linspace(0, 1, 1000)

    priors = {
        'Uniform': (1, 1),
        'Informative (10%)': (9, 91),
        'Weakly Informative': (5, 5),
    }

    plt.figure(figsize=(10, 6))

    for name, (a, b) in priors.items():
        prior_dist = beta(a, b)
        plt.plot(x, prior_dist.pdf(x), label=name, linewidth=2)

    plt.xlabel('Conversion Rate')
    plt.ylabel('Probability Density')
    plt.title('Comparison of Different Priors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## Reglas de Decisión

### 1. Threshold de Probabilidad

```python
def make_decision_probability(
    prob_b_beats_a: float,
    threshold: float = 0.95
) -> str:
    """
    Decide basado en threshold de probabilidad.

    Args:
        prob_b_beats_a: P(B > A)
        threshold: umbral de decisión (típicamente 0.95)

    Returns:
        'SHIP_B', 'SHIP_A', o 'CONTINUE'
    """
    if prob_b_beats_a >= threshold:
        return 'SHIP_B'
    elif prob_b_beats_a <= (1 - threshold):
        return 'SHIP_A'
    else:
        return 'CONTINUE'


# Ejemplo
decision = make_decision_probability(prob_b_beats_a=0.97, threshold=0.95)
print(f"Decision: {decision}")
```

### 2. Expected Loss Threshold

```python
def make_decision_expected_loss(
    loss_a: float,
    loss_b: float,
    threshold: float = 0.01  # 1% loss máximo aceptable
) -> str:
    """
    Decide basado en expected loss.

    Criterio más robusto que solo probabilidad.

    Args:
        loss_a: pérdida esperada si elegimos A
        loss_b: pérdida esperada si elegimos B
        threshold: pérdida relativa máxima aceptable

    Returns:
        'SHIP_A', 'SHIP_B', o 'CONTINUE'
    """
    if loss_a < threshold and loss_a < loss_b:
        return 'SHIP_A'
    elif loss_b < threshold and loss_b < loss_a:
        return 'SHIP_B'
    else:
        return 'CONTINUE'
```

### 3. Rope (Region of Practical Equivalence)

```python
def make_decision_rope(
    conversions_a: int,
    trials_a: int,
    conversions_b: int,
    trials_b: int,
    rope: tuple = (-0.01, 0.01),  # ±1pp de diferencia
    n_samples: int = 100000
) -> str:
    """
    Decide usando ROPE: región de equivalencia práctica.

    Si la diferencia está en ROPE, consideramos variantes equivalentes.

    Args:
        rope: (lower, upper) en escala absoluta

    Returns:
        'SHIP_B', 'SHIP_A', 'EQUIVALENT', o 'CONTINUE'
    """
    test = BayesianABTest()
    post_a = test.posterior(conversions_a, trials_a)
    post_b = test.posterior(conversions_b, trials_b)

    samples_a = post_a.rvs(n_samples)
    samples_b = post_b.rvs(n_samples)

    diff = samples_b - samples_a

    # Probabilidades
    prob_in_rope = ((diff > rope[0]) & (diff < rope[1])).mean()
    prob_above_rope = (diff > rope[1]).mean()
    prob_below_rope = (diff < rope[0]).mean()

    # Decisión
    if prob_in_rope > 0.95:
        return 'EQUIVALENT'  # Las variantes son prácticamente iguales
    elif prob_above_rope > 0.95:
        return 'SHIP_B'
    elif prob_below_rope > 0.95:
        return 'SHIP_A'
    else:
        return 'CONTINUE'
```

## Cálculo de Duración

### Enfoque: Simulación de Poder

A diferencia del frecuentista, no hay fórmula cerrada. Usamos simulación:

```python
def bayesian_experiment_duration(
    baseline_rate: float,
    mde: float,
    daily_traffic_per_variant: int,
    decision_threshold: float = 0.95,
    prior_alpha: float = 1,
    prior_beta: float = 1,
    n_simulations: int = 1000
) -> dict:
    """
    Estima duración necesaria via simulación.

    Simula experimentos con el efecto real y determina cuándo
    se alcanza la decisión con el threshold especificado.

    Args:
        baseline_rate: tasa de conversión del control
        mde: efecto que queremos detectar (absoluto)
        daily_traffic_per_variant: usuarios diarios por variante
        decision_threshold: threshold de P(B>A) para decidir
        n_simulations: número de simulaciones

    Returns:
        dict con estadísticas de duración
    """
    treatment_rate = baseline_rate + mde

    test = BayesianABTest(prior_alpha, prior_beta)

    days_to_decision = []

    for _ in range(n_simulations):
        day = 0
        decided = False

        while not decided and day < 100:  # max 100 días
            day += 1

            # Simular datos acumulados hasta este día
            n_control = day * daily_traffic_per_variant
            n_treatment = day * daily_traffic_per_variant

            conversions_control = np.random.binomial(n_control, baseline_rate)
            conversions_treatment = np.random.binomial(n_treatment, treatment_rate)

            # Check decisión
            prob = test.probability_b_beats_a(
                conversions_control, n_control,
                conversions_treatment, n_treatment,
                n_samples=10000
            )

            if prob >= decision_threshold or prob <= (1 - decision_threshold):
                decided = True
                days_to_decision.append(day)

        if not decided:
            days_to_decision.append(100)  # No decidió en 100 días

    return {
        'mean_days': np.mean(days_to_decision),
        'median_days': np.median(days_to_decision),
        'p25_days': np.percentile(days_to_decision, 25),
        'p75_days': np.percentile(days_to_decision, 75),
        'p95_days': np.percentile(days_to_decision, 95),
        'power': (np.array(days_to_decision) < 100).mean()
    }


# Ejemplo
result = bayesian_experiment_duration(
    baseline_rate=0.10,
    mde=0.02,  # 2pp
    daily_traffic_per_variant=2500,
    decision_threshold=0.95,
    n_simulations=500
)

print(f"Duración media: {result['mean_days']:.1f} días")
print(f"Duración mediana: {result['median_days']:.0f} días")
print(f"95% de experimentos deciden en: {result['p95_days']:.0f} días")
print(f"Poder (decidir en <100 días): {result['power']:.2%}")
```

## Monitoreo Continuo

### Dashboard Bayesiano

```python
class BayesianMonitor:
    """Monitor continuo de experimento bayesiano."""

    def __init__(
        self,
        prior_alpha: float = 1,
        prior_beta: float = 1,
        decision_threshold: float = 0.95,
        loss_threshold: float = 0.01
    ):
        self.test = BayesianABTest(prior_alpha, prior_beta)
        self.decision_threshold = decision_threshold
        self.loss_threshold = loss_threshold

    def check_experiment(
        self,
        conversions_a: int,
        trials_a: int,
        conversions_b: int,
        trials_b: int
    ) -> dict:
        """
        Chequea estado del experimento.

        Returns:
            dict con todas las métricas de decisión
        """
        # Probabilidad
        prob_b_beats_a = self.test.probability_b_beats_a(
            conversions_a, trials_a,
            conversions_b, trials_b
        )

        # Expected loss
        loss = self.test.expected_loss(
            conversions_a, trials_a,
            conversions_b, trials_b
        )

        # Posteriors
        post_a = self.test.posterior(conversions_a, trials_a)
        post_b = self.test.posterior(conversions_b, trials_b)

        # Estadísticas
        stats = {
            'prob_b_beats_a': prob_b_beats_a,
            'loss_if_choose_a': loss['loss_if_choose_a'],
            'loss_if_choose_b': loss['loss_if_choose_b'],
            'posterior_mean_a': post_a.mean(),
            'posterior_mean_b': post_b.mean(),
            'posterior_std_a': post_a.std(),
            'posterior_std_b': post_b.std(),
            'credible_interval_a': post_a.interval(0.95),
            'credible_interval_b': post_b.interval(0.95)
        }

        # Decisión
        decision = self._make_decision(
            prob_b_beats_a,
            loss['loss_if_choose_a'],
            loss['loss_if_choose_b']
        )

        stats['decision'] = decision

        return stats

    def _make_decision(
        self,
        prob_b_beats_a: float,
        loss_a: float,
        loss_b: float
    ) -> str:
        """Lógica de decisión combinando probabilidad y loss."""

        # Criterio 1: Probabilidad alta Y loss bajo
        if prob_b_beats_a >= self.decision_threshold and loss_b < self.loss_threshold:
            return 'SHIP_B'
        elif prob_b_beats_a <= (1 - self.decision_threshold) and loss_a < self.loss_threshold:
            return 'SHIP_A'
        else:
            return 'CONTINUE'


# Uso en producción
monitor = BayesianMonitor(
    decision_threshold=0.95,
    loss_threshold=0.01
)

# Cada día/hora, chequear
status = monitor.check_experiment(
    conversions_a=450,
    trials_a=5000,
    conversions_b=520,
    trials_b=5000
)

print(f"P(B > A): {status['prob_b_beats_a']:.2%}")
print(f"Decision: {status['decision']}")
print(f"Posterior B: {status['posterior_mean_b']:.3f} ± {status['posterior_std_b']:.3f}")
```

## Múltiples Variantes

### Enfoque 1: Pairwise Comparisons

```python
def bayesian_multiple_variants(
    data: dict,  # {variant_name: (conversions, trials)}
    control_name: str = 'control',
    threshold: float = 0.95
) -> dict:
    """
    Compara múltiples variantes contra control.

    Args:
        data: dict con datos de cada variante
        control_name: nombre de la variante control
        threshold: threshold de decisión

    Returns:
        dict con resultados para cada variante
    """
    test = BayesianABTest()

    conversions_control, trials_control = data[control_name]

    results = {}

    for variant_name, (conversions, trials) in data.items():
        if variant_name == control_name:
            continue

        prob = test.probability_b_beats_a(
            conversions_control, trials_control,
            conversions, trials
        )

        loss = test.expected_loss(
            conversions_control, trials_control,
            conversions, trials
        )

        results[variant_name] = {
            'prob_beats_control': prob,
            'expected_loss': loss['loss_if_choose_b'],
            'decision': 'WINNER' if prob >= threshold else 'CONTINUE'
        }

    return results


# Ejemplo con 3 variantes
data = {
    'control': (450, 5000),
    'variant_A': (480, 5000),
    'variant_B': (520, 5000),
    'variant_C': (460, 5000)
}

results = bayesian_multiple_variants(data, threshold=0.95)

for variant, metrics in results.items():
    print(f"\n{variant}:")
    print(f"  P(beats control): {metrics['prob_beats_control']:.2%}")
    print(f"  Expected loss: {metrics['expected_loss']:.4f}")
    print(f"  Decision: {metrics['decision']}")
```

### Enfoque 2: Probability of Being Best

```python
def probability_of_being_best(
    data: dict,  # {variant_name: (conversions, trials)}
    n_samples: int = 100000
) -> dict:
    """
    Calcula P(cada variante es la mejor).

    Args:
        data: dict con datos de cada variante

    Returns:
        dict con P(best) para cada variante
    """
    test = BayesianABTest()

    # Muestras de cada posterior
    samples = {}
    for variant_name, (conversions, trials) in data.items():
        posterior = test.posterior(conversions, trials)
        samples[variant_name] = posterior.rvs(n_samples)

    # Crear matriz de muestras
    sample_matrix = np.array([samples[v] for v in data.keys()])

    # Para cada muestra, encontrar cuál variante es mejor
    best_variant_indices = np.argmax(sample_matrix, axis=0)

    # Contar frecuencias
    prob_best = {}
    for i, variant_name in enumerate(data.keys()):
        prob_best[variant_name] = (best_variant_indices == i).mean()

    return prob_best


# Ejemplo
prob_best = probability_of_being_best(data)

print("\nProbability of being best:")
for variant, prob in sorted(prob_best.items(), key=lambda x: x[1], reverse=True):
    print(f"  {variant}: {prob:.2%}")
```

## Velocidad de Decisión vs. Frecuentista

### Comparación Empírica

```python
def compare_bayesian_vs_frequentist_speed(
    baseline_rate: float = 0.10,
    mde: float = 0.02,
    daily_traffic: int = 5000,
    n_simulations: int = 200
):
    """
    Compara tiempo hasta decisión: Bayesiano vs. Frecuentista.
    """
    from scipy.stats import norm

    treatment_rate = baseline_rate + mde

    # Tamaño de muestra frecuentista
    z_alpha = norm.ppf(0.975)
    z_beta = norm.ppf(0.80)
    p = baseline_rate
    n_freq = 2 * ((z_alpha + z_beta)**2 * p * (1-p)) / (mde**2)
    days_freq = n_freq / daily_traffic

    # Simulación bayesiana
    test = BayesianABTest()
    bayesian_days = []

    for _ in range(n_simulations):
        day = 0
        decided = False

        while not decided and day < 100:
            day += 1
            n = day * daily_traffic

            conv_control = np.random.binomial(n, baseline_rate)
            conv_treatment = np.random.binomial(n, treatment_rate)

            prob = test.probability_b_beats_a(
                conv_control, n,
                conv_treatment, n,
                n_samples=10000
            )

            if prob >= 0.95 or prob <= 0.05:
                decided = True
                bayesian_days.append(day)

        if not decided:
            bayesian_days.append(100)

    return {
        'frequentist_days': days_freq,
        'bayesian_median_days': np.median(bayesian_days),
        'bayesian_mean_days': np.mean(bayesian_days),
        'speedup': days_freq / np.median(bayesian_days)
    }


# Ejecutar comparación
comparison = compare_bayesian_vs_frequentist_speed()

print(f"Frecuentista: {comparison['frequentist_days']:.1f} días")
print(f"Bayesiano (mediana): {comparison['bayesian_median_days']:.1f} días")
print(f"Speedup: {comparison['speedup']:.2f}x más rápido")
```

**Típicamente: Bayesiano es 1.2-1.5x más rápido que Frecuentista**

## Best Practices

### 1. Elección de Prior

```python
# Para primera variante de este tipo
prior = (1, 1)  # Uniform

# Si tienes datos históricos
alpha, beta = create_informative_prior(historical_mean=0.10, historical_std=0.01)

# Para regularizar con datos pequeños
prior = (5, 5)  # Weakly informative
```

### 2. Criterios de Decisión

```python
# Conservador (similar a frecuentista)
threshold_prob = 0.975  # 97.5%
threshold_loss = 0.005  # 0.5%

# Balanceado (recomendado)
threshold_prob = 0.95  # 95%
threshold_loss = 0.01  # 1%

# Agresivo (para exploración rápida)
threshold_prob = 0.90  # 90%
threshold_loss = 0.02  # 2%
```

### 3. Mínimo de Datos

Incluso con Bayesiano, establece mínimos:

```python
MIN_SAMPLE_SIZE = 1000  # Por variante
MIN_CONVERSIONS = 50    # Por variante

# No tomar decisiones antes de estos mínimos
```

## Casos de Uso Ideales

### ✅ Úsalo Cuando:

1. **Tienes conocimiento previo** que quieres incorporar
2. **Necesitas monitoreo continuo** sin peeking problem
3. **Quieres respuestas probabilísticas** directas
4. **Stakeholders entienden probabilidades** mejor que p-valores
5. **Necesitas decisiones más rápidas** que frecuentista (pero no tan rápido como SOAT)

### ❌ NO lo Uses Cuando:

1. **Stakeholders solo entienden p-valores**
2. **Regulaciones requieren métodos frecuentistas**
3. **No tienes capacidad computacional** para MCMC/simulaciones
4. **El overhead de explicación** es mayor que esperar más tiempo

## Herramientas

### Python Libraries

```python
# PyMC3 - Para modelos más complejos
import pymc3 as pm

# ArviZ - Visualización y diagnóstico bayesiano
import arviz as az

# scipy.stats - Para casos simples Beta-Binomial
from scipy.stats import beta, binom
```

### Plataformas

- **VWO**: Usa enfoque bayesiano por defecto
- **Google Optimize**: Opción bayesiana disponible
- **AB Tasty**: Soporta análisis bayesiano

## Resumen Ejecutivo

**El enfoque bayesiano es:**
- 🏃 **Más rápido que frecuentista**: 1.2-1.5x speedup
- 🧠 **Más intuitivo**: Responde preguntas probabilísticas directas
- 🔄 **Más flexible**: Permite monitoreo continuo
- 📊 **Más complejo**: Requiere entender distribuciones

**Úsalo cuando:**
- Velocidad es importante (pero no crítica)
- Tienes conocimiento previo valioso
- Quieres monitorear continuamente
- Stakeholders prefieren probabilidades

**NO lo uses cuando:**
- SOAT puede funcionar (más rápido aún)
- Stakeholders no lo entienden
- Simplicidad es más importante que velocidad

## Siguiente: [Enfoque SOAT (Statsig)](./04-enfoque-soat.md)
