"""
Calculadora de Experimentos A/B - Enfoque Bayesiano

Este módulo proporciona funciones para planificar y analizar experimentos A/B
usando el enfoque bayesiano con conjugate priors (Beta-Binomial).
"""

import numpy as np
from scipy.stats import beta
from typing import Dict, Tuple, Optional


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

        Example:
            >>> test = BayesianABTest(prior_alpha=1, prior_beta=1)
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
            conversions: Número de conversiones observadas
            trials: Número de usuarios observados

        Returns:
            scipy.stats.beta distribution

        Example:
            >>> test = BayesianABTest()
            >>> post = test.posterior(conversions=450, trials=5000)
            >>> print(f"Media posterior: {post.mean():.3f}")
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
            conversions_a, trials_a: Datos de variante A
            conversions_b, trials_b: Datos de variante B
            n_samples: Número de muestras Monte Carlo

        Returns:
            Probabilidad de que B sea mejor que A

        Example:
            >>> test = BayesianABTest()
            >>> prob = test.probability_b_beats_a(450, 5000, 520, 5000)
            >>> print(f"P(B > A): {prob:.2%}")
        """
        post_a = self.posterior(conversions_a, trials_a)
        post_b = self.posterior(conversions_b, trials_b)

        samples_a = post_a.rvs(n_samples)
        samples_b = post_b.rvs(n_samples)

        prob = (samples_b > samples_a).mean()

        return prob

    def expected_loss(
        self,
        conversions_a: int,
        trials_a: int,
        conversions_b: int,
        trials_b: int,
        n_samples: int = 100000
    ) -> Dict[str, float]:
        """
        Calcula pérdida esperada de elegir cada variante.

        Args:
            conversions_a, trials_a: Datos de variante A
            conversions_b, trials_b: Datos de variante B
            n_samples: Número de muestras

        Returns:
            Dict con pérdidas esperadas

        Example:
            >>> test = BayesianABTest()
            >>> loss = test.expected_loss(450, 5000, 520, 5000)
            >>> print(f"Loss si elegimos A: {loss['loss_if_choose_a']:.4f}")
        """
        post_a = self.posterior(conversions_a, trials_a)
        post_b = self.posterior(conversions_b, trials_b)

        samples_a = post_a.rvs(n_samples)
        samples_b = post_b.rvs(n_samples)

        # Pérdida = diferencia cuando estamos equivocados
        loss_choose_a = np.maximum(0, samples_b - samples_a).mean()
        loss_choose_b = np.maximum(0, samples_a - samples_b).mean()

        return {
            'loss_if_choose_a': loss_choose_a,
            'loss_if_choose_b': loss_choose_b,
            'relative_loss_a': loss_choose_a / samples_a.mean() if samples_a.mean() > 0 else 0,
            'relative_loss_b': loss_choose_b / samples_b.mean() if samples_b.mean() > 0 else 0
        }

    def credible_interval(
        self,
        conversions: int,
        trials: int,
        credibility: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calcula intervalo de credibilidad.

        Args:
            conversions: Número de conversiones
            trials: Número de trials
            credibility: Nivel de credibilidad (default 0.95)

        Returns:
            Tupla (lower, upper)

        Example:
            >>> test = BayesianABTest()
            >>> ci = test.credible_interval(450, 5000)
            >>> print(f"95% Credible Interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
        """
        post = self.posterior(conversions, trials)
        alpha_level = (1 - credibility) / 2
        lower = post.ppf(alpha_level)
        upper = post.ppf(1 - alpha_level)

        return (lower, upper)

    def make_decision(
        self,
        conversions_a: int,
        trials_a: int,
        conversions_b: int,
        trials_b: int,
        threshold_prob: float = 0.95,
        threshold_loss: float = 0.01
    ) -> Dict:
        """
        Toma decisión basada en probabilidad y expected loss.

        Args:
            conversions_a, trials_a: Datos de variante A
            conversions_b, trials_b: Datos de variante B
            threshold_prob: Umbral de probabilidad para decidir
            threshold_loss: Umbral de pérdida relativa aceptable

        Returns:
            Dict con decisión y métricas

        Example:
            >>> test = BayesianABTest()
            >>> decision = test.make_decision(450, 5000, 520, 5000)
            >>> print(decision['decision'])
        """
        prob_b_beats_a = self.probability_b_beats_a(
            conversions_a, trials_a,
            conversions_b, trials_b
        )

        loss = self.expected_loss(
            conversions_a, trials_a,
            conversions_b, trials_b
        )

        # Decisión combinada
        if prob_b_beats_a >= threshold_prob and loss['relative_loss_b'] < threshold_loss:
            decision = 'SHIP_B'
            reason = f"B wins with {prob_b_beats_a:.1%} probability"
        elif prob_b_beats_a <= (1 - threshold_prob) and loss['relative_loss_a'] < threshold_loss:
            decision = 'SHIP_A'
            reason = f"A wins with {1-prob_b_beats_a:.1%} probability"
        else:
            decision = 'CONTINUE'
            reason = "Insufficient evidence to decide"

        return {
            'decision': decision,
            'reason': reason,
            'prob_b_beats_a': prob_b_beats_a,
            'loss_if_choose_a': loss['loss_if_choose_a'],
            'loss_if_choose_b': loss['loss_if_choose_b']
        }


def create_informative_prior(
    historical_mean: float,
    historical_std: float
) -> Tuple[float, float]:
    """
    Crea prior Beta informativo basado en datos históricos.

    Args:
        historical_mean: Tasa de conversión histórica (ej: 0.10)
        historical_std: Desviación estándar histórica (ej: 0.01)

    Returns:
        Tupla (alpha, beta) para Beta distribution

    Example:
        >>> alpha, beta = create_informative_prior(0.10, 0.01)
        >>> print(f"Prior: Beta({alpha:.1f}, {beta:.1f})")
    """
    mean = historical_mean
    var = historical_std ** 2

    # Method of moments para Beta distribution
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta_param = (1 - mean) * (mean * (1 - mean) / var - 1)

    return (alpha, beta_param)


def simulate_experiment_duration(
    baseline_rate: float,
    mde: float,
    daily_traffic_per_variant: int,
    decision_threshold: float = 0.95,
    prior_alpha: float = 1,
    prior_beta: float = 1,
    n_simulations: int = 500,
    max_days: int = 100
) -> Dict:
    """
    Estima duración de experimento bayesiano via simulación.

    Args:
        baseline_rate: Tasa de conversión del control
        mde: Efecto que queremos detectar (absoluto)
        daily_traffic_per_variant: Usuarios diarios por variante
        decision_threshold: Threshold de P(B>A) para decidir
        prior_alpha, prior_beta: Parámetros del prior
        n_simulations: Número de simulaciones
        max_days: Máximo de días a simular

    Returns:
        Dict con estadísticas de duración

    Example:
        >>> result = simulate_experiment_duration(
        ...     baseline_rate=0.10,
        ...     mde=0.02,
        ...     daily_traffic_per_variant=2500
        ... )
        >>> print(f"Duración media: {result['mean_days']:.1f} días")
    """
    treatment_rate = baseline_rate + mde

    test = BayesianABTest(prior_alpha, prior_beta)

    days_to_decision = []

    for _ in range(n_simulations):
        day = 0
        decided = False

        while not decided and day < max_days:
            day += 1

            n_control = day * daily_traffic_per_variant
            n_treatment = day * daily_traffic_per_variant

            conversions_control = np.random.binomial(n_control, baseline_rate)
            conversions_treatment = np.random.binomial(n_treatment, treatment_rate)

            prob = test.probability_b_beats_a(
                conversions_control, n_control,
                conversions_treatment, n_treatment,
                n_samples=10000
            )

            if prob >= decision_threshold or prob <= (1 - decision_threshold):
                decided = True
                days_to_decision.append(day)

        if not decided:
            days_to_decision.append(max_days)

    days_array = np.array(days_to_decision)

    return {
        'mean_days': np.mean(days_array),
        'median_days': np.median(days_array),
        'p25_days': np.percentile(days_array, 25),
        'p75_days': np.percentile(days_array, 75),
        'p95_days': np.percentile(days_array, 95),
        'power': (days_array < max_days).mean(),
        'all_days': days_array
    }


def multiple_variants_analysis(
    data: Dict[str, Tuple[int, int]],
    control_name: str = 'control',
    threshold: float = 0.95,
    prior_alpha: float = 1,
    prior_beta: float = 1
) -> Dict:
    """
    Analiza múltiples variantes contra control.

    Args:
        data: Dict con {variant_name: (conversions, trials)}
        control_name: Nombre de la variante control
        threshold: Umbral de decisión
        prior_alpha, prior_beta: Parámetros del prior

    Returns:
        Dict con resultados para cada variante

    Example:
        >>> data = {
        ...     'control': (450, 5000),
        ...     'variant_a': (490, 5000),
        ...     'variant_b': (520, 5000)
        ... }
        >>> results = multiple_variants_analysis(data)
        >>> print(results)
    """
    test = BayesianABTest(prior_alpha, prior_beta)

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

        rate_control = conversions_control / trials_control
        rate_variant = conversions / trials
        lift = (rate_variant - rate_control) / rate_control if rate_control > 0 else 0

        ci = test.credible_interval(conversions, trials)

        results[variant_name] = {
            'conversion_rate': rate_variant,
            'lift_relative': lift,
            'prob_beats_control': prob,
            'expected_loss': loss['loss_if_choose_b'],
            'credible_interval_95': ci,
            'decision': 'WINNER' if prob >= threshold else 'CONTINUE'
        }

    return results


def probability_of_being_best(
    data: Dict[str, Tuple[int, int]],
    prior_alpha: float = 1,
    prior_beta: float = 1,
    n_samples: int = 100000
) -> Dict[str, float]:
    """
    Calcula P(cada variante es la mejor).

    Args:
        data: Dict con {variant_name: (conversions, trials)}
        prior_alpha, prior_beta: Parámetros del prior
        n_samples: Número de muestras

    Returns:
        Dict con P(best) para cada variante

    Example:
        >>> data = {
        ...     'control': (450, 5000),
        ...     'variant_a': (490, 5000),
        ...     'variant_b': (520, 5000)
        ... }
        >>> prob_best = probability_of_being_best(data)
        >>> for variant, prob in sorted(prob_best.items(), key=lambda x: x[1], reverse=True):
        ...     print(f"{variant}: {prob:.2%}")
    """
    test = BayesianABTest(prior_alpha, prior_beta)

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


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== Calculadora Bayesiana ===\n")

    # Test simple A/B
    test = BayesianABTest(prior_alpha=1, prior_beta=1)

    print("ANÁLISIS A/B:")
    conversions_a = 450
    trials_a = 5000
    conversions_b = 520
    trials_b = 5000

    prob = test.probability_b_beats_a(
        conversions_a, trials_a,
        conversions_b, trials_b
    )

    loss = test.expected_loss(
        conversions_a, trials_a,
        conversions_b, trials_b
    )

    decision = test.make_decision(
        conversions_a, trials_a,
        conversions_b, trials_b
    )

    print(f"P(B > A): {prob:.2%}")
    print(f"Expected loss if choose A: {loss['loss_if_choose_a']:.4f}")
    print(f"Expected loss if choose B: {loss['loss_if_choose_b']:.4f}")
    print(f"Decision: {decision['decision']}")
    print(f"Reason: {decision['reason']}")

    # Múltiples variantes
    print("\n\nMÚLTIPLES VARIANTES:")
    data = {
        'control': (450, 5000),
        'variant_a': (480, 5000),
        'variant_b': (520, 5000),
        'variant_c': (460, 5000)
    }

    results = multiple_variants_analysis(data)

    for variant, metrics in results.items():
        print(f"\n{variant}:")
        print(f"  Conversion: {metrics['conversion_rate']:.2%}")
        print(f"  Lift: {metrics['lift_relative']:+.1%}")
        print(f"  P(beats control): {metrics['prob_beats_control']:.2%}")
        print(f"  Decision: {metrics['decision']}")

    # Probability of being best
    print("\n\nPROBABILITY OF BEING BEST:")
    prob_best = probability_of_being_best(data)

    for variant, prob in sorted(prob_best.items(), key=lambda x: x[1], reverse=True):
        print(f"  {variant}: {prob:.2%}")
