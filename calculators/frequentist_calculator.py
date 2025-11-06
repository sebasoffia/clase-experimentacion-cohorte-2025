"""
Calculadora de Experimentos A/B - Enfoque Frecuentista

Este módulo proporciona funciones para planificar y analizar experimentos A/B
usando el enfoque frecuentista tradicional.
"""

import numpy as np
from scipy.stats import norm, chi2_contingency
from typing import Dict, Tuple, Optional


def sample_size_binary(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> int:
    """
    Calcula tamaño de muestra para métrica binaria (conversión).

    Args:
        baseline_rate: Tasa de conversión del control (ej: 0.10 para 10%)
        mde: Efecto mínimo detectable absoluto (ej: 0.02 para 2pp)
        alpha: Nivel de significancia (típicamente 0.05)
        power: Poder estadístico (típicamente 0.80)
        two_tailed: Si True, test de dos colas

    Returns:
        Tamaño de muestra requerido por variante

    Example:
        >>> n = sample_size_binary(baseline_rate=0.10, mde=0.02)
        >>> print(f"Se requieren {n:,} usuarios por variante")
    """
    if two_tailed:
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    z_beta = norm.ppf(power)

    treatment_rate = baseline_rate + mde
    p_pooled = (baseline_rate + treatment_rate) / 2

    numerator = (z_alpha + z_beta)**2 * 2 * p_pooled * (1 - p_pooled)
    denominator = mde**2

    n = numerator / denominator

    return int(np.ceil(n))


def sample_size_continuous(
    mean: float,
    std: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> int:
    """
    Calcula tamaño de muestra para métrica continua.

    Args:
        mean: Media del control
        std: Desviación estándar
        mde_relative: Efecto mínimo detectable relativo (ej: 0.10 para 10%)
        alpha: Nivel de significancia
        power: Poder estadístico
        two_tailed: Si True, test de dos colas

    Returns:
        Tamaño de muestra requerido por variante

    Example:
        >>> n = sample_size_continuous(mean=25.0, std=50.0, mde_relative=0.10)
        >>> print(f"Se requieren {n:,} usuarios por variante")
    """
    if two_tailed:
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    z_beta = norm.ppf(power)

    mde_absolute = mean * mde_relative
    effect_size = mde_absolute / std

    n = 2 * ((z_alpha + z_beta) / effect_size)**2

    return int(np.ceil(n))


def calculate_duration(
    sample_size_per_variant: int,
    num_variants: int,
    daily_traffic: int,
    traffic_allocation: float = 1.0,
    adjust_to_weeks: bool = True
) -> Dict[str, float]:
    """
    Calcula duración del experimento.

    Args:
        sample_size_per_variant: Tamaño de muestra calculado por variante
        num_variants: Número total de variantes (incluyendo control)
        daily_traffic: Usuarios diarios disponibles
        traffic_allocation: Porcentaje de tráfico asignado (0.0 a 1.0)
        adjust_to_weeks: Si True, ajusta a semanas completas

    Returns:
        Dict con duration_days, duration_weeks, daily_per_variant

    Example:
        >>> duration = calculate_duration(
        ...     sample_size_per_variant=5000,
        ...     num_variants=2,
        ...     daily_traffic=10000,
        ...     traffic_allocation=0.5
        ... )
        >>> print(f"Duración: {duration['duration_weeks']} semanas")
    """
    total_sample = sample_size_per_variant * num_variants
    daily_experiment_traffic = daily_traffic * traffic_allocation
    daily_per_variant = daily_experiment_traffic / num_variants

    duration_days = total_sample / daily_experiment_traffic

    if adjust_to_weeks:
        duration_weeks = int(np.ceil(duration_days / 7))
        adjusted_days = duration_weeks * 7
    else:
        duration_weeks = duration_days / 7
        adjusted_days = duration_days

    return {
        'duration_days': duration_days,
        'duration_weeks': duration_weeks,
        'adjusted_days': adjusted_days,
        'daily_per_variant': daily_per_variant
    }


def bonferroni_correction(
    alpha: float,
    num_comparisons: int
) -> float:
    """
    Aplica corrección Bonferroni para múltiples comparaciones.

    Args:
        alpha: Nivel de significancia original
        num_comparisons: Número de comparaciones

    Returns:
        Alpha ajustado

    Example:
        >>> alpha_adj = bonferroni_correction(0.05, 3)
        >>> print(f"Alpha ajustado: {alpha_adj:.4f}")
    """
    return alpha / num_comparisons


def z_test_proportions(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    alpha: float = 0.05
) -> Dict:
    """
    Realiza test Z de dos proporciones.

    Args:
        conversions_control: Número de conversiones en control
        n_control: Tamaño de muestra control
        conversions_treatment: Número de conversiones en tratamiento
        n_treatment: Tamaño de muestra tratamiento
        alpha: Nivel de significancia

    Returns:
        Dict con resultados del test

    Example:
        >>> result = z_test_proportions(450, 5000, 520, 5000)
        >>> print(f"P-valor: {result['p_value']:.4f}")
        >>> print(f"Significativo: {result['is_significant']}")
    """
    p_control = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment

    # Proporción pooled bajo H₀
    p_pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)

    # Error estándar
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    # Z-score
    z_score = (p_treatment - p_control) / se

    # P-valor (two-tailed)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # Decisión
    is_significant = p_value < alpha

    # Lift
    lift_absolute = p_treatment - p_control
    lift_relative = lift_absolute / p_control if p_control > 0 else 0

    # Intervalo de confianza
    z_crit = norm.ppf(1 - alpha/2)
    se_diff = np.sqrt(
        p_control * (1 - p_control) / n_control +
        p_treatment * (1 - p_treatment) / n_treatment
    )
    ci_lower = lift_absolute - z_crit * se_diff
    ci_upper = lift_absolute + z_crit * se_diff

    return {
        'p_control': p_control,
        'p_treatment': p_treatment,
        'z_score': z_score,
        'p_value': p_value,
        'is_significant': is_significant,
        'lift_absolute': lift_absolute,
        'lift_relative': lift_relative,
        'ci_95': (ci_lower, ci_upper)
    }


def confidence_interval(
    p_control: float,
    n_control: int,
    p_treatment: float,
    n_treatment: int,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calcula intervalo de confianza para diferencia de proporciones.

    Args:
        p_control: Proporción del control
        n_control: Tamaño de muestra control
        p_treatment: Proporción del tratamiento
        n_treatment: Tamaño de muestra tratamiento
        confidence_level: Nivel de confianza (default 0.95)

    Returns:
        Tupla (lower_bound, upper_bound) de la diferencia

    Example:
        >>> ci = confidence_interval(0.09, 5000, 0.104, 5000)
        >>> print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    """
    diff = p_treatment - p_control

    se = np.sqrt(
        p_control * (1 - p_control) / n_control +
        p_treatment * (1 - p_treatment) / n_treatment
    )

    z = norm.ppf(1 - (1 - confidence_level) / 2)

    lower = diff - z * se
    upper = diff + z * se

    return (lower, upper)


def quick_plan(
    baseline_rate: float,
    mde_relative: float,
    daily_traffic: int,
    num_variants: int = 2,
    alpha: float = 0.05,
    power: float = 0.80,
    traffic_allocation: float = 1.0
) -> Dict:
    """
    Planificación rápida de experimento frecuentista.

    Args:
        baseline_rate: Tasa de conversión baseline
        mde_relative: MDE relativo (ej: 0.10 para 10%)
        daily_traffic: Tráfico diario disponible
        num_variants: Número de variantes (incluyendo control)
        alpha: Nivel de significancia
        power: Poder estadístico
        traffic_allocation: % de tráfico asignado

    Returns:
        Dict con plan completo del experimento

    Example:
        >>> plan = quick_plan(
        ...     baseline_rate=0.10,
        ...     mde_relative=0.15,
        ...     daily_traffic=10000,
        ...     num_variants=3
        ... )
        >>> print(plan)
    """
    mde_absolute = baseline_rate * mde_relative

    # Sample size sin corrección
    n_base = sample_size_binary(baseline_rate, mde_absolute, alpha, power)

    # Con corrección Bonferroni si hay múltiples variantes
    if num_variants > 2:
        num_comparisons = num_variants - 1
        alpha_adj = bonferroni_correction(alpha, num_comparisons)
        n_adjusted = sample_size_binary(baseline_rate, mde_absolute, alpha_adj, power)
    else:
        alpha_adj = alpha
        n_adjusted = n_base

    # Duración
    duration = calculate_duration(
        n_adjusted,
        num_variants,
        daily_traffic,
        traffic_allocation
    )

    return {
        'baseline_rate': baseline_rate,
        'mde_relative': mde_relative,
        'mde_absolute': mde_absolute,
        'sample_size_per_variant': n_adjusted,
        'sample_size_per_variant_no_correction': n_base,
        'total_sample_size': n_adjusted * num_variants,
        'alpha': alpha,
        'alpha_adjusted': alpha_adj,
        'power': power,
        'num_variants': num_variants,
        'bonferroni_correction': num_variants > 2,
        'duration_days': duration['duration_days'],
        'duration_weeks': duration['duration_weeks'],
        'daily_per_variant': duration['daily_per_variant']
    }


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== Calculadora Frecuentista ===\n")

    # Planificación
    plan = quick_plan(
        baseline_rate=0.10,
        mde_relative=0.15,
        daily_traffic=20000,
        num_variants=3
    )

    print("PLANIFICACIÓN:")
    print(f"Baseline: {plan['baseline_rate']:.1%}")
    print(f"MDE: {plan['mde_relative']:.0%} relativo ({plan['mde_absolute']:.1%} absoluto)")
    print(f"Variantes: {plan['num_variants']}")
    print(f"Sample size por variante: {plan['sample_size_per_variant']:,}")
    print(f"Duración: {plan['duration_days']:.1f} días ({plan['duration_weeks']} semanas)")

    # Análisis
    print("\n\nANÁLISIS:")
    result = z_test_proportions(
        conversions_control=700,
        n_control=7000,
        conversions_treatment=805,
        n_treatment=7000
    )

    print(f"Control: {result['p_control']:.2%}")
    print(f"Treatment: {result['p_treatment']:.2%}")
    print(f"Lift: {result['lift_relative']:+.1%}")
    print(f"P-valor: {result['p_value']:.4f}")
    print(f"Significativo (α=0.05): {result['is_significant']}")
    print(f"95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
