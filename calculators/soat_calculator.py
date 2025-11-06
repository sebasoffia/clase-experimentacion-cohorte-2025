"""
Calculadora de Experimentos A/B - Enfoque SOAT (Sequential Testing)

Este módulo proporciona implementación simplificada de SOAT (Sequential Open-ended
Adaptive Testing) para análisis de experimentos con monitoreo continuo.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional


class SOATTest:
    """
    Implementación de SOAT para A/B testing.

    SOAT permite monitoreo continuo sin inflar la tasa de falsos positivos.
    Basado en mixture Sequential Probability Ratio Test (mSPRT).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        alternative: str = 'two-sided'
    ):
        """
        Inicializa SOAT test.

        Args:
            alpha: Tasa de falsos positivos deseada
            power: Poder estadístico deseado
            alternative: 'two-sided', 'greater', 'less'

        Example:
            >>> soat = SOATTest(alpha=0.05, power=0.80)
        """
        self.alpha = alpha
        self.power = power
        self.alternative = alternative

        # Thresholds basados en α y β (Wald's SPRT approximation)
        self.threshold_upper = np.log((1 - self.alpha/2) / (1 - self.power))
        self.threshold_lower = np.log((self.alpha/2) / self.power)

    def mixture_log_likelihood_ratio(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int,
        effect_grid: Optional[np.ndarray] = None
    ) -> float:
        """
        Calcula mixture log likelihood ratio (mSPRT).

        Args:
            conversions_control: Conversiones en control
            n_control: Tamaño de muestra control
            conversions_treatment: Conversiones en tratamiento
            n_treatment: Tamaño de muestra tratamiento
            effect_grid: Grid de efectos posibles (default: -50% a +100%)

        Returns:
            Log likelihood ratio

        Example:
            >>> soat = SOATTest()
            >>> llr = soat.mixture_log_likelihood_ratio(450, 5000, 520, 5000)
            >>> print(f"LLR: {llr:.2f}")
        """
        if effect_grid is None:
            # Grid de efectos relativos: -50% a +100%
            effect_grid = np.linspace(-0.5, 1.0, 100)

        # Proporción observada en control (con Laplace smoothing)
        p_control = (conversions_control + 0.5) / (n_control + 1)

        log_likelihoods = []

        for rel_effect in effect_grid:
            p_treatment_alt = p_control * (1 + rel_effect)
            p_treatment_alt = np.clip(p_treatment_alt, 0.0001, 0.9999)

            # Binomial log likelihood bajo esta alternativa
            ll = (
                conversions_treatment * np.log(p_treatment_alt) +
                (n_treatment - conversions_treatment) * np.log(1 - p_treatment_alt)
            )
            log_likelihoods.append(ll)

        # Log likelihood bajo H₀ (p_treatment = p_control)
        ll_null = (
            conversions_treatment * np.log(p_control) +
            (n_treatment - conversions_treatment) * np.log(1 - p_control)
        )

        # Mixture: usar max (simplificación; idealmente sería integral con prior)
        ll_alt_mixture = max(log_likelihoods)

        llr = ll_alt_mixture - ll_null

        return llr

    def sequential_p_value(
        self,
        llr: float
    ) -> float:
        """
        Convierte LLR a sequential p-value.

        Permite interpretación similar a p-valor frecuentista.

        Args:
            llr: Log likelihood ratio

        Returns:
            Sequential p-value

        Example:
            >>> soat = SOATTest()
            >>> llr = 5.2
            >>> p_val = soat.sequential_p_value(llr)
            >>> print(f"Sequential p-value: {p_val:.4f}")
        """
        # Aproximación usando Wilks' theorem
        test_statistic = 2 * llr
        p_value = 1 - norm.cdf(np.sqrt(max(0, test_statistic)))

        return p_value

    def make_decision(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int,
        min_sample_size: int = 1000,
        min_days: int = 7,
        current_day: int = 1
    ) -> Dict:
        """
        Toma decisión basada en SOAT.

        Args:
            conversions_control, n_control: Datos del control
            conversions_treatment, n_treatment: Datos del tratamiento
            min_sample_size: Tamaño mínimo de muestra
            min_days: Duración mínima en días
            current_day: Día actual del experimento

        Returns:
            Dict con decisión y métricas

        Example:
            >>> soat = SOATTest()
            >>> decision = soat.make_decision(
            ...     conversions_control=450,
            ...     n_control=5000,
            ...     conversions_treatment=520,
            ...     n_treatment=5000,
            ...     current_day=10
            ... )
            >>> print(decision['decision'])
        """
        # Validaciones de mínimos
        if n_control < min_sample_size or n_treatment < min_sample_size:
            return {
                'decision': 'CONTINUE',
                'reason': f'Sample size below minimum ({min_sample_size})',
                'llr': None,
                'p_value': None
            }

        if current_day < min_days:
            return {
                'decision': 'CONTINUE',
                'reason': f'Minimum duration not met (day {current_day}/{min_days})',
                'llr': None,
                'p_value': None
            }

        # Calcular LLR
        llr = self.mixture_log_likelihood_ratio(
            conversions_control, n_control,
            conversions_treatment, n_treatment
        )

        # Sequential p-value
        p_value = self.sequential_p_value(llr)

        # Tasas observadas
        p_control = conversions_control / n_control
        p_treatment = conversions_treatment / n_treatment
        lift = (p_treatment - p_control) / p_control if p_control > 0 else 0

        # Decisión basada en thresholds
        if llr > self.threshold_upper:
            if p_treatment > p_control:
                decision = 'SHIP_TREATMENT'
                reason = f'Treatment wins (LLR={llr:.2f}, p={p_value:.4f})'
            else:
                decision = 'SHIP_CONTROL'
                reason = f'Control wins (LLR={llr:.2f}, p={p_value:.4f})'

        elif llr < self.threshold_lower:
            decision = 'NO_DIFFERENCE'
            reason = f'No significant difference (LLR={llr:.2f})'

        else:
            decision = 'CONTINUE'
            reason = f'Inconclusive (LLR={llr:.2f}, threshold={self.threshold_upper:.2f})'

        return {
            'decision': decision,
            'reason': reason,
            'llr': llr,
            'p_value': p_value,
            'threshold_upper': self.threshold_upper,
            'threshold_lower': self.threshold_lower,
            'control_rate': p_control,
            'treatment_rate': p_treatment,
            'lift': lift
        }


def simulate_soat_duration(
    baseline_rate: float,
    true_effect: float,
    daily_traffic_per_variant: int,
    alpha: float = 0.05,
    power: float = 0.80,
    min_days: int = 7,
    n_simulations: int = 500,
    max_days: int = 100
) -> Dict:
    """
    Simula duración de experimento con SOAT.

    Args:
        baseline_rate: Tasa de conversión del control
        true_effect: Efecto verdadero (absoluto)
        daily_traffic_per_variant: Tráfico diario por variante
        alpha: Nivel de significancia
        power: Poder estadístico
        min_days: Mínimo de días
        n_simulations: Número de simulaciones
        max_days: Máximo de días a simular

    Returns:
        Dict con estadísticas de duración

    Example:
        >>> result = simulate_soat_duration(
        ...     baseline_rate=0.10,
        ...     true_effect=0.02,
        ...     daily_traffic_per_variant=2500
        ... )
        >>> print(f"Duración mediana: {result['median_days']:.0f} días")
    """
    treatment_rate = baseline_rate + true_effect

    soat = SOATTest(alpha=alpha, power=power)

    days_to_decision = []

    for _ in range(n_simulations):
        day = 0
        decided = False

        while not decided and day < max_days:
            day += 1
            n = day * daily_traffic_per_variant

            # Simular datos
            conv_control = np.random.binomial(n, baseline_rate)
            conv_treatment = np.random.binomial(n, treatment_rate)

            # Decisión
            result = soat.make_decision(
                conv_control, n,
                conv_treatment, n,
                min_days=min_days,
                current_day=day
            )

            if result['decision'] != 'CONTINUE':
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


def multiple_variants_soat(
    data: Dict[str, Dict[str, int]],
    control_name: str = 'control',
    alpha: float = 0.05,
    power: float = 0.80,
    min_days: int = 7,
    current_day: int = 1,
    bonferroni: bool = False
) -> Dict:
    """
    Analiza múltiples variantes con SOAT.

    Args:
        data: Dict con {variant_name: {'conversions': int, 'trials': int}}
        control_name: Nombre de la variante control
        alpha: Nivel de significancia
        power: Poder estadístico
        min_days: Mínimo de días
        current_day: Día actual
        bonferroni: Si True, aplica corrección Bonferroni

    Returns:
        Dict con resultados para cada variante

    Example:
        >>> data = {
        ...     'control': {'conversions': 450, 'trials': 5000},
        ...     'variant_a': {'conversions': 490, 'trials': 5000},
        ...     'variant_b': {'conversions': 520, 'trials': 5000}
        ... }
        >>> results = multiple_variants_soat(data, current_day=10)
        >>> print(results)
    """
    num_comparisons = len(data) - 1

    if bonferroni and num_comparisons > 1:
        alpha_adj = alpha / num_comparisons
    else:
        alpha_adj = alpha

    soat = SOATTest(alpha=alpha_adj, power=power)

    control = data[control_name]
    results = {}

    for variant_name, variant_data in data.items():
        if variant_name == control_name:
            continue

        result = soat.make_decision(
            control['conversions'], control['trials'],
            variant_data['conversions'], variant_data['trials'],
            min_days=min_days,
            current_day=current_day
        )

        results[variant_name] = result

    # Encontrar mejor variante si hay ganadores
    winners = {name: res for name, res in results.items()
               if res['decision'] == 'SHIP_TREATMENT'}

    if winners:
        best_variant = max(winners.items(), key=lambda x: x[1]['lift'])
        results['__recommendation__'] = f"Ship {best_variant[0]} (+{best_variant[1]['lift']:.1%} lift)"
    else:
        results['__recommendation__'] = "Continue experiment"

    return results


class SOATMonitor:
    """Monitor en tiempo real para experimentos SOAT."""

    def __init__(
        self,
        alpha: float = 0.05,
        power: float = 0.80,
        min_sample_size: int = 1000,
        min_days: int = 7
    ):
        """
        Inicializa monitor SOAT.

        Args:
            alpha: Nivel de significancia
            power: Poder estadístico
            min_sample_size: Tamaño mínimo de muestra
            min_days: Duración mínima en días

        Example:
            >>> monitor = SOATMonitor()
        """
        self.soat = SOATTest(alpha, power)
        self.min_sample_size = min_sample_size
        self.min_days = min_days

    def check_experiment(
        self,
        data: Dict[str, Dict[str, int]],
        current_day: int,
        control_name: str = 'control'
    ) -> Dict:
        """
        Chequea estado del experimento.

        Args:
            data: Dict con datos de cada variante
            current_day: Día actual del experimento
            control_name: Nombre de la variante control

        Returns:
            Dict con estado y recomendaciones

        Example:
            >>> monitor = SOATMonitor()
            >>> data = {
            ...     'control': {'conversions': 450, 'trials': 5000},
            ...     'treatment': {'conversions': 520, 'trials': 5000}
            ... }
            >>> status = monitor.check_experiment(data, current_day=10)
            >>> print(status['recommendation'])
        """
        control = data[control_name]

        # Validaciones
        if control['trials'] < self.min_sample_size:
            return {
                'status': 'CONTINUE',
                'reason': 'Insufficient sample size',
                'recommendation': f"Wait until {self.min_sample_size} samples",
                'day': current_day
            }

        if current_day < self.min_days:
            return {
                'status': 'CONTINUE',
                'reason': 'Minimum duration not met',
                'recommendation': f"Wait until day {self.min_days}",
                'day': current_day
            }

        # Analizar cada variante
        results = {}
        for variant_name, variant_data in data.items():
            if variant_name == control_name:
                continue

            decision = self.soat.make_decision(
                control['conversions'], control['trials'],
                variant_data['conversions'], variant_data['trials'],
                min_sample_size=self.min_sample_size,
                min_days=self.min_days,
                current_day=current_day
            )

            results[variant_name] = decision

        # Generar recomendación
        winners = {name: res for name, res in results.items()
                   if res['decision'] == 'SHIP_TREATMENT'}

        if winners:
            best = max(winners.items(), key=lambda x: x[1]['lift'])
            recommendation = f"✅ Ship {best[0]} with +{best[1]['lift']:.1%} lift"
            status = 'WINNER_FOUND'
        else:
            # Check si hay señal emergente
            max_llr = max((res['llr'] for res in results.values() if res['llr'] is not None), default=0)
            if max_llr > self.soat.threshold_upper * 0.5:
                recommendation = f"📊 Trending positive (max LLR: {max_llr:.2f}), continue monitoring"
            else:
                recommendation = "📊 No clear signal yet, continue experiment"
            status = 'CONTINUE'

        return {
            'status': status,
            'recommendation': recommendation,
            'day': current_day,
            'results': results
        }


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== Calculadora SOAT ===\n")

    # Análisis simple
    soat = SOATTest(alpha=0.05, power=0.80)

    decision = soat.make_decision(
        conversions_control=700,
        n_control=7000,
        conversions_treatment=805,
        n_treatment=7000,
        current_day=10
    )

    print("ANÁLISIS:")
    print(f"Control rate: {decision['control_rate']:.2%}")
    print(f"Treatment rate: {decision['treatment_rate']:.2%}")
    print(f"Lift: {decision['lift']:+.1%}")
    print(f"LLR: {decision['llr']:.2f} (threshold: {decision['threshold_upper']:.2f})")
    print(f"Sequential p-value: {decision['p_value']:.4f}")
    print(f"Decision: {decision['decision']}")

    # Simulación de duración
    print("\n\nSIMULACIÓN DE DURACIÓN:")
    sim_result = simulate_soat_duration(
        baseline_rate=0.10,
        true_effect=0.02,
        daily_traffic_per_variant=2500,
        n_simulations=100
    )

    print(f"Duración mediana: {sim_result['median_days']:.0f} días")
    print(f"P25-P75: {sim_result['p25_days']:.0f} - {sim_result['p75_days']:.0f} días")
    print(f"Poder: {sim_result['power']:.2%}")
