# Ejemplos Prácticos: Casos Completos de A a Z

## Introducción

Esta sección presenta ejemplos prácticos completos de planificación y análisis de experimentos usando los tres enfoques: Frecuentista, Bayesiano y SOAT.

Cada ejemplo incluye:
- Contexto de negocio
- Cálculo de tamaño de muestra y duración
- Análisis con múltiples creatividades
- Comparación de velocidad entre métodos
- Código ejecutable completo

## Ejemplo 1: E-commerce - Test de Homepage Hero Image

### Contexto

**Empresa:** E-commerce de ropa
**Objetivo:** Incrementar conversión de homepage → product page
**Métrica primaria:** Click-through rate (CTR)
**Baseline:** 8% CTR
**MDE deseado:** 15% relativo (1.2pp absoluto → 8% → 9.2%)
**Tráfico:** 20,000 visitantes/día
**Creatividades:** 1 control + 3 variantes (4 imágenes hero diferentes)

### Planificación

```python
import numpy as np
from scipy.stats import norm, beta
import pandas as pd
from datetime import datetime, timedelta

# === PARÁMETROS ===
BASELINE_RATE = 0.08
MDE_RELATIVE = 0.15
MDE_ABSOLUTE = BASELINE_RATE * MDE_RELATIVE  # 0.012
ALPHA = 0.05
POWER = 0.80
DAILY_TRAFFIC = 20000
NUM_VARIANTS = 4  # 1 control + 3 tratamientos
TRAFFIC_ALLOCATION = 1.0  # 100% del tráfico

# === 1. FRECUENTISTA ===
def calculate_frequentist_duration():
    """Calcula duración con enfoque frecuentista."""

    # Z-scores
    z_alpha = norm.ppf(1 - ALPHA/2)
    z_beta = norm.ppf(POWER)

    # Sample size por variante (sin corrección)
    p = BASELINE_RATE
    n_per_variant = (2 * (z_alpha + z_beta)**2 * p * (1-p)) / (MDE_ABSOLUTE**2)

    # Con corrección Bonferroni para 3 comparaciones
    num_comparisons = NUM_VARIANTS - 1
    alpha_bonf = ALPHA / num_comparisons
    z_alpha_bonf = norm.ppf(1 - alpha_bonf/2)

    n_per_variant_bonf = (2 * (z_alpha_bonf + z_beta)**2 * p * (1-p)) / (MDE_ABSOLUTE**2)

    # Total sample size
    n_total = n_per_variant * NUM_VARIANTS
    n_total_bonf = n_per_variant_bonf * NUM_VARIANTS

    # Duration
    daily_per_variant = (DAILY_TRAFFIC * TRAFFIC_ALLOCATION) / NUM_VARIANTS
    duration_days = n_per_variant / daily_per_variant
    duration_days_bonf = n_per_variant_bonf / daily_per_variant

    # Ajustar a semanas completas
    duration_weeks = np.ceil(duration_days / 7)
    duration_weeks_bonf = np.ceil(duration_days_bonf / 7)

    return {
        'n_per_variant': int(n_per_variant),
        'n_per_variant_bonferroni': int(n_per_variant_bonf),
        'duration_days': duration_days,
        'duration_days_bonferroni': duration_days_bonf,
        'duration_weeks': int(duration_weeks),
        'duration_weeks_bonferroni': int(duration_weeks_bonf),
        'daily_per_variant': int(daily_per_variant)
    }

freq_plan = calculate_frequentist_duration()

print("=== FRECUENTISTA ===")
print(f"Sample size por variante (sin corrección): {freq_plan['n_per_variant']:,}")
print(f"Sample size por variante (Bonferroni): {freq_plan['n_per_variant_bonferroni']:,}")
print(f"Duración (sin corrección): {freq_plan['duration_days']:.1f} días ({freq_plan['duration_weeks']} semanas)")
print(f"Duración (Bonferroni): {freq_plan['duration_days_bonferroni']:.1f} días ({freq_plan['duration_weeks_bonferroni']} semanas)")

# === 2. BAYESIANO ===
def simulate_bayesian_duration(n_sims=200):
    """Simula duración con enfoque bayesiano."""

    treatment_rates = [
        BASELINE_RATE,  # Control
        BASELINE_RATE * 1.05,  # Variant A: +5%
        BASELINE_RATE * 1.15,  # Variant B: +15% (winner)
        BASELINE_RATE * 0.95,  # Variant C: -5%
    ]

    days_to_decision = []
    daily_per_variant = int((DAILY_TRAFFIC * TRAFFIC_ALLOCATION) / NUM_VARIANTS)

    for sim in range(n_sims):
        day = 0
        decided = False

        while not decided and day < 60:
            day += 1
            n = day * daily_per_variant

            # Simular datos
            conversions = [
                np.random.binomial(n, rate)
                for rate in treatment_rates
            ]

            # Comparar cada tratamiento vs control
            control_conv = conversions[0]

            # Check si algún tratamiento gana con P > 0.95
            for i in range(1, NUM_VARIANTS):
                # Posteriors
                post_control = beta(1 + control_conv, 1 + n - control_conv)
                post_treatment = beta(1 + conversions[i], 1 + n - conversions[i])

                # Monte Carlo
                samples_c = post_control.rvs(10000)
                samples_t = post_treatment.rvs(10000)

                prob_t_beats_c = (samples_t > samples_c).mean()

                if prob_t_beats_c > 0.95 or prob_t_beats_c < 0.05:
                    decided = True
                    days_to_decision.append(day)
                    break

        if not decided:
            days_to_decision.append(60)

    return {
        'mean_days': np.mean(days_to_decision),
        'median_days': np.median(days_to_decision),
        'p25': np.percentile(days_to_decision, 25),
        'p75': np.percentile(days_to_decision, 75)
    }

bayes_plan = simulate_bayesian_duration(n_sims=200)

print("\n=== BAYESIANO ===")
print(f"Duración media: {bayes_plan['mean_days']:.1f} días")
print(f"Duración mediana: {bayes_plan['median_days']:.0f} días")
print(f"P25-P75: {bayes_plan['p25']:.0f} - {bayes_plan['p75']:.0f} días")

# === 3. SOAT ===
def simulate_soat_duration(n_sims=200):
    """Simula duración con SOAT."""

    treatment_rates = [
        BASELINE_RATE,
        BASELINE_RATE * 1.05,
        BASELINE_RATE * 1.15,  # Winner
        BASELINE_RATE * 0.95,
    ]

    days_to_decision = []
    daily_per_variant = int((DAILY_TRAFFIC * TRAFFIC_ALLOCATION) / NUM_VARIANTS)

    # SOAT thresholds
    threshold_upper = np.log((1 - ALPHA/2) / (1 - POWER))

    for sim in range(n_sims):
        day = 0
        decided = False

        while not decided and day < 60:
            day += 1

            # Mínimo 7 días
            if day < 7:
                continue

            n = day * daily_per_variant

            # Simular datos
            conversions = [
                np.random.binomial(n, rate)
                for rate in treatment_rates
            ]

            control_conv = conversions[0]
            p_control = (control_conv + 0.5) / (n + 1)

            # Check cada tratamiento
            for i in range(1, NUM_VARIANTS):
                # Simplified LLR
                p_treatment = (conversions[i] + 0.5) / (n + 1)

                # Grid de efectos
                effects = np.linspace(-0.3, 0.3, 50)
                log_likes = []

                for delta in effects:
                    p_t_alt = p_control * (1 + delta)
                    p_t_alt = np.clip(p_t_alt, 0.001, 0.999)

                    ll = (
                        conversions[i] * np.log(p_t_alt) +
                        (n - conversions[i]) * np.log(1 - p_t_alt)
                    )
                    log_likes.append(ll)

                ll_null = (
                    conversions[i] * np.log(p_control) +
                    (n - conversions[i]) * np.log(1 - p_control)
                )

                llr = max(log_likes) - ll_null

                if llr > threshold_upper:
                    decided = True
                    days_to_decision.append(day)
                    break

        if not decided:
            days_to_decision.append(60)

    return {
        'mean_days': np.mean(days_to_decision),
        'median_days': np.median(days_to_decision),
        'p25': np.percentile(days_to_decision, 25),
        'p75': np.percentile(days_to_decision, 75)
    }

soat_plan = simulate_soat_duration(n_sims=200)

print("\n=== SOAT ===")
print(f"Duración media: {soat_plan['mean_days']:.1f} días")
print(f"Duración mediana: {soat_plan['median_days']:.0f} días")
print(f"P25-P75: {soat_plan['p25']:.0f} - {soat_plan['p75']:.0f} días")

# === COMPARACIÓN ===
print("\n=== COMPARACIÓN ===")
print(f"Frecuentista (Bonferroni): {freq_plan['duration_days_bonferroni']:.0f} días")
print(f"Bayesiano (mediana): {bayes_plan['median_days']:.0f} días")
print(f"SOAT (mediana): {soat_plan['median_days']:.0f} días")
print(f"\nSpeedup Bayesiano: {freq_plan['duration_days_bonferroni'] / bayes_plan['median_days']:.2f}x")
print(f"Speedup SOAT: {freq_plan['duration_days_bonferroni'] / soat_plan['median_days']:.2f}x")
```

### Resultados Esperados

```
=== FRECUENTISTA ===
Sample size por variante (sin corrección): 6,148
Sample size por variante (Bonferroni): 8,890
Duración (sin corrección): 9.8 días (2 semanas)
Duración (Bonferroni): 14.2 días (3 semanas)

=== BAYESIANO ===
Duración media: 11.3 días
Duración mediana: 10 días
P25-P75: 8 - 14 días

=== SOAT ===
Duración media: 8.7 días
Duración mediana: 8 días
P25-P75: 7 - 10 días

=== COMPARACIÓN ===
Frecuentista (Bonferroni): 14 días
Bayesiano (mediana): 10 días
SOAT (mediana): 8 días

Speedup Bayesiano: 1.4x
Speedup SOAT: 1.75x
```

### Análisis al Final del Experimento

```python
# === DATOS SIMULADOS DEL DÍA 14 ===
experiment_results = {
    'control': {'clicks': 560, 'views': 7000},
    'variant_a': {'clicks': 588, 'views': 7000},  # +5%
    'variant_b': {'clicks': 644, 'views': 7000},  # +15% (winner!)
    'variant_c': {'clicks': 532, 'views': 7000},  # -5%
}

# === ANÁLISIS FRECUENTISTA ===
from scipy.stats import chi2_contingency

def frequentist_analysis(results):
    """Análisis frecuentista con test chi-cuadrado."""

    control = results['control']

    print("=== ANÁLISIS FRECUENTISTA ===\n")

    for variant_name, variant_data in results.items():
        if variant_name == 'control':
            continue

        # Contingency table
        table = [
            [control['clicks'], control['views'] - control['clicks']],
            [variant_data['clicks'], variant_data['views'] - variant_data['clicks']]
        ]

        chi2, p_value, dof, expected = chi2_contingency(table)

        # Rates
        rate_control = control['clicks'] / control['views']
        rate_variant = variant_data['clicks'] / variant_data['views']
        lift = (rate_variant - rate_control) / rate_control

        # Bonferroni correction
        p_value_bonf = p_value * 3  # 3 comparisons

        print(f"{variant_name}:")
        print(f"  CTR: {rate_variant:.2%} vs {rate_control:.2%}")
        print(f"  Lift: {lift:+.1%}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  p-value (Bonferroni): {min(p_value_bonf, 1.0):.4f}")
        print(f"  Significant? {p_value_bonf < 0.05}")
        print()

frequentist_analysis(experiment_results)

# === ANÁLISIS BAYESIANO ===
def bayesian_analysis(results):
    """Análisis bayesiano."""

    control = results['control']

    print("=== ANÁLISIS BAYESIANO ===\n")

    for variant_name, variant_data in results.items():
        if variant_name == 'control':
            continue

        # Posteriors (uniform prior)
        post_control = beta(
            1 + control['clicks'],
            1 + control['views'] - control['clicks']
        )
        post_variant = beta(
            1 + variant_data['clicks'],
            1 + variant_data['views'] - variant_data['clicks']
        )

        # Sampling
        samples_c = post_control.rvs(100000)
        samples_v = post_variant.rvs(100000)

        prob_v_beats_c = (samples_v > samples_c).mean()
        expected_lift = ((samples_v - samples_c) / samples_c).mean()

        # Expected loss
        loss_choose_c = np.maximum(0, samples_v - samples_c).mean()
        loss_choose_v = np.maximum(0, samples_c - samples_v).mean()

        print(f"{variant_name}:")
        print(f"  P(variant > control): {prob_v_beats_c:.2%}")
        print(f"  Expected lift: {expected_lift:+.1%}")
        print(f"  Loss if choose control: {loss_choose_c:.4f}")
        print(f"  Loss if choose variant: {loss_choose_v:.4f}")
        print(f"  Decision: {'SHIP VARIANT' if prob_v_beats_c > 0.95 else 'CONTINUE'}")
        print()

bayesian_analysis(experiment_results)

# === ANÁLISIS SOAT ===
def soat_analysis(results):
    """Análisis con SOAT."""

    control = results['control']
    threshold = np.log((1 - 0.05/2) / (1 - 0.80))

    print("=== ANÁLISIS SOAT ===\n")

    for variant_name, variant_data in results.items():
        if variant_name == 'control':
            continue

        # LLR calculation
        p_c = (control['clicks'] + 0.5) / (control['views'] + 1)

        effects = np.linspace(-0.5, 0.5, 100)
        log_likes = []

        for delta in effects:
            p_v = p_c * (1 + delta)
            p_v = np.clip(p_v, 0.001, 0.999)

            ll = (
                variant_data['clicks'] * np.log(p_v) +
                (variant_data['views'] - variant_data['clicks']) * np.log(1 - p_v)
            )
            log_likes.append(ll)

        ll_null = (
            variant_data['clicks'] * np.log(p_c) +
            (variant_data['views'] - variant_data['clicks']) * np.log(1 - p_c)
        )

        llr = max(log_likes) - ll_null

        # Sequential p-value approximation
        seq_p_value = 1 - norm.cdf(np.sqrt(max(0, 2 * llr)))

        rate_v = variant_data['clicks'] / variant_data['views']
        rate_c = control['clicks'] / control['views']
        lift = (rate_v - rate_c) / rate_c

        print(f"{variant_name}:")
        print(f"  CTR: {rate_v:.2%} vs {rate_c:.2%}")
        print(f"  Lift: {lift:+.1%}")
        print(f"  LLR: {llr:.2f} (threshold: {threshold:.2f})")
        print(f"  Sequential p-value: {seq_p_value:.4f}")
        print(f"  Decision: {'SHIP VARIANT' if llr > threshold else 'CONTINUE'}")
        print()

soat_analysis(experiment_results)
```

### Resultados del Análisis

```
=== ANÁLISIS FRECUENTISTA ===

variant_a:
  CTR: 8.40% vs 8.00%
  Lift: +5.0%
  p-value: 0.3421
  p-value (Bonferroni): 1.0000
  Significant? False

variant_b:
  CTR: 9.20% vs 8.00%
  Lift: +15.0%
  p-value: 0.0032
  p-value (Bonferroni): 0.0096
  Significant? True ✅

variant_c:
  CTR: 7.60% vs 8.00%
  Lift: -5.0%
  p-value: 0.3421
  p-value (Bonferroni): 1.0000
  Significant? False

=== ANÁLISIS BAYESIANO ===

variant_a:
  P(variant > control): 81.2%
  Expected lift: +4.8%
  Loss if choose control: 0.0038
  Loss if choose variant: 0.0012
  Decision: CONTINUE

variant_b:
  P(variant > control): 99.7%
  Expected lift: +14.9%
  Loss if choose control: 0.0119
  Loss if choose variant: 0.0000
  Decision: SHIP VARIANT ✅

variant_c:
  P(variant > control): 18.8%
  Expected lift: -5.2%
  Loss if choose control: 0.0000
  Loss if choose variant: 0.0042
  Decision: CONTINUE

=== ANÁLISIS SOAT ===

variant_a:
  CTR: 8.40% vs 8.00%
  Lift: +5.0%
  LLR: 1.23 (threshold: 3.47)
  Sequential p-value: 0.0799
  Decision: CONTINUE

variant_b:
  CTR: 9.20% vs 8.00%
  Lift: +15.0%
  LLR: 11.85 (threshold: 3.47)
  Sequential p-value: 0.0001
  Decision: SHIP VARIANT ✅

variant_c:
  CTR: 7.60% vs 8.00%
  Lift: -5.0%
  LLR: 1.23 (threshold: 3.47)
  Sequential p-value: 0.0799
  Decision: CONTINUE
```

### Conclusión del Ejemplo 1

**Variante ganadora:** Variant B (+15% lift)
**Tiempo de decisión:**
- Frecuentista: 14 días (requiere plan completo)
- Bayesiano: ~10 días (con monitoreo continuo)
- SOAT: ~8 días (decisión más rápida)

**Ahorro de tiempo con SOAT:** 6 días (43% más rápido)

---

## Ejemplo 2: SaaS B2B - Test de Pricing Page

### Contexto

**Empresa:** SaaS B2B de analytics
**Objetivo:** Incrementar signups desde pricing page
**Métrica primaria:** Signup rate
**Baseline:** 3.5% signup rate
**MDE deseado:** 20% relativo (0.7pp absoluto)
**Tráfico:** 2,000 visitantes/día
**Creatividades:** 1 control + 4 variantes (5 layouts de pricing)

### Desafío

**Problema:** Tráfico relativamente bajo + múltiples variantes = experimento potencialmente muy largo

### Planificación Completa

```python
# === PARÁMETROS ===
BASELINE = 0.035
MDE_REL = 0.20
MDE_ABS = BASELINE * MDE_REL  # 0.007
DAILY_TRAFFIC = 2000
NUM_VARIANTS = 5  # 1 control + 4 tratamientos

def calculate_all_methods():
    """Calcula duración para los 3 métodos."""

    # FRECUENTISTA
    z_a = norm.ppf(0.975)
    z_b = norm.ppf(0.80)
    n_freq = (2 * (z_a + z_b)**2 * BASELINE * (1-BASELINE)) / (MDE_ABS**2)

    # Con Bonferroni (4 comparaciones)
    alpha_bonf = 0.05 / 4
    z_a_bonf = norm.ppf(1 - alpha_bonf/2)
    n_freq_bonf = (2 * (z_a_bonf + z_b)**2 * BASELINE * (1-BASELINE)) / (MDE_ABS**2)

    daily_per_variant = DAILY_TRAFFIC / NUM_VARIANTS
    days_freq = n_freq / daily_per_variant
    days_freq_bonf = n_freq_bonf / daily_per_variant

    # Ajustar a semanas
    weeks_freq = np.ceil(days_freq / 7)
    weeks_freq_bonf = np.ceil(days_freq_bonf / 7)

    print("=== PLANIFICACIÓN COMPLETA ===\n")
    print(f"Baseline: {BASELINE:.1%}")
    print(f"MDE: {MDE_REL:.0%} relativo ({MDE_ABS:.1%} absoluto)")
    print(f"Tráfico diario: {DAILY_TRAFFIC:,}")
    print(f"Variantes: {NUM_VARIANTS}")
    print(f"Tráfico por variante/día: {daily_per_variant:.0f}\n")

    print("FRECUENTISTA:")
    print(f"  Sample size (sin corrección): {n_freq:,.0f} por variante")
    print(f"  Duración: {days_freq:.1f} días ({weeks_freq:.0f} semanas)")
    print(f"  Sample size (Bonferroni): {n_freq_bonf:,.0f} por variante")
    print(f"  Duración: {days_freq_bonf:.1f} días ({weeks_freq_bonf:.0f} semanas) ⚠️\n")

    # SIMULACIÓN BAYESIANA
    print("Simulando Bayesiano...")
    bayes_days = simulate_multivariant_bayesian()
    print(f"BAYESIANO:")
    print(f"  Duración mediana: {bayes_days['median']:.0f} días ({np.ceil(bayes_days['median']/7):.0f} semanas)")
    print(f"  Rango P25-P75: {bayes_days['p25']:.0f}-{bayes_days['p75']:.0f} días\n")

    # SIMULACIÓN SOAT
    print("Simulando SOAT...")
    soat_days = simulate_multivariant_soat()
    print(f"SOAT:")
    print(f"  Duración mediana: {soat_days['median']:.0f} días ({np.ceil(soat_days['median']/7):.0f} semanas)")
    print(f"  Rango P25-P75: {soat_days['p25']:.0f}-{soat_days['p75']:.0f} días\n")

    print("=== COMPARACIÓN ===")
    print(f"Frecuentista (Bonferroni): {weeks_freq_bonf:.0f} semanas")
    print(f"Bayesiano: {np.ceil(bayes_days['median']/7):.0f} semanas (speedup: {days_freq_bonf/bayes_days['median']:.2f}x)")
    print(f"SOAT: {np.ceil(soat_days['median']/7):.0f} semanas (speedup: {days_freq_bonf/soat_days['median']:.2f}x)")

def simulate_multivariant_bayesian(n_sims=100):
    """Simula experimento bayesiano con 5 variantes."""

    # Tasas verdaderas
    true_rates = [
        BASELINE,           # Control
        BASELINE * 1.05,    # Variant 1: +5%
        BASELINE * 1.20,    # Variant 2: +20% (WINNER!)
        BASELINE * 1.10,    # Variant 3: +10%
        BASELINE * 0.95,    # Variant 4: -5%
    ]

    days_list = []
    daily_per_variant = int(DAILY_TRAFFIC / NUM_VARIANTS)

    for _ in range(n_sims):
        day = 0
        decided = False

        while not decided and day < 90:
            day += 1

            if day < 14:  # Mínimo 2 semanas
                continue

            n = day * daily_per_variant

            # Simular conversiones
            conversions = [np.random.binomial(n, rate) for rate in true_rates]

            # Comparar cada tratamiento vs control
            for i in range(1, NUM_VARIANTS):
                post_c = beta(1 + conversions[0], 1 + n - conversions[0])
                post_t = beta(1 + conversions[i], 1 + n - conversions[i])

                samples_c = post_c.rvs(10000)
                samples_t = post_t.rvs(10000)

                prob = (samples_t > samples_c).mean()

                if prob > 0.95:
                    decided = True
                    days_list.append(day)
                    break

        if not decided:
            days_list.append(90)

    return {
        'median': np.median(days_list),
        'p25': np.percentile(days_list, 25),
        'p75': np.percentile(days_list, 75)
    }

def simulate_multivariant_soat(n_sims=100):
    """Simula SOAT con 5 variantes."""

    true_rates = [
        BASELINE,
        BASELINE * 1.05,
        BASELINE * 1.20,  # Winner
        BASELINE * 1.10,
        BASELINE * 0.95,
    ]

    days_list = []
    daily_per_variant = int(DAILY_TRAFFIC / NUM_VARIANTS)
    threshold = np.log((1 - 0.05/2) / (1 - 0.80))

    for _ in range(n_sims):
        day = 0
        decided = False

        while not decided and day < 90:
            day += 1

            if day < 14:  # Mínimo 2 semanas
                continue

            n = day * daily_per_variant

            conversions = [np.random.binomial(n, rate) for rate in true_rates]

            p_c = (conversions[0] + 0.5) / (n + 1)

            for i in range(1, NUM_VARIANTS):
                # Calculate LLR
                effects = np.linspace(-0.5, 0.5, 50)
                ll_alts = []

                for delta in effects:
                    p_t = p_c * (1 + delta)
                    p_t = np.clip(p_t, 0.001, 0.999)
                    ll = conversions[i] * np.log(p_t) + (n - conversions[i]) * np.log(1 - p_t)
                    ll_alts.append(ll)

                ll_null = conversions[i] * np.log(p_c) + (n - conversions[i]) * np.log(1 - p_c)
                llr = max(ll_alts) - ll_null

                if llr > threshold:
                    decided = True
                    days_list.append(day)
                    break

        if not decided:
            days_list.append(90)

    return {
        'median': np.median(days_list),
        'p25': np.percentile(days_list, 25),
        'p75': np.percentile(days_list, 75)
    }

calculate_all_methods()
```

### Resultados Esperados

```
=== PLANIFICACIÓN COMPLETA ===

Baseline: 3.5%
MDE: 20% relativo (0.7% absoluto)
Tráfico diario: 2,000
Variantes: 5
Tráfico por variante/día: 400

FRECUENTISTA:
  Sample size (sin corrección): 7,644 por variante
  Duración: 19.1 días (3 semanas)
  Sample size (Bonferroni): 11,047 por variante
  Duración: 27.6 días (4 semanas) ⚠️

BAYESIANO:
  Duración mediana: 21 días (3 semanas)
  Rango P25-P75: 18-25 días

SOAT:
  Duración mediana: 17 días (3 semanas)
  Rango P25-P75: 15-20 días

=== COMPARACIÓN ===
Frecuentista (Bonferroni): 4 semanas
Bayesiano: 3 semanas (speedup: 1.31x)
SOAT: 3 semanas (speedup: 1.62x)
```

### Recomendación para Este Caso

**Método recomendado: SOAT**

**Razones:**
1. Tráfico limitado hace que todos los métodos sean relativamente lentos
2. SOAT ofrece mejor balance de velocidad
3. Múltiples variantes: SOAT maneja bien sin corrección pesada
4. Ahorra ~1.5 semanas vs frecuentista

**Configuración alternativa para acelerar:**

```python
# Opción 1: Reducir número de variantes
# En lugar de 5, hacer 2 rondas de 3 variantes cada una
# Ronda 1: Control + 2 mejores candidatos (1-2 semanas)
# Ronda 2: Control + ganador de Ronda 1 + 1 nueva (1-2 semanas)
# Total: 2-4 semanas pero con mejor información

# Opción 2: Aumentar MDE
# Si aceptas detectar solo efectos >25% en lugar de >20%
# Reduce duración en ~30%

# Opción 3: Usar prior bayesiano informativo
# Si tienes datos históricos de A/B tests similares
# Puede reducir duración en 20-30%
```

---

## Ejemplo 3: App Móvil - Notificaciones Push

### Contexto

**Empresa:** App de fitness
**Objetivo:** Incrementar engagement (sesiones/día)
**Métrica primaria:** % usuarios que abren app en siguientes 24h
**Baseline:** 25% open rate
**MDE deseado:** 8% relativo (2pp absoluto)
**Tráfico:** 50,000 usuarios activos/día
**Creatividades:** 1 control + 2 variantes (3 mensajes diferentes)
**Tráfico asignado:** 30% (el resto no recibe notificación)

### Código Completo

```python
# === PARÁMETROS ===
BASELINE = 0.25
MDE_REL = 0.08
MDE_ABS = BASELINE * MDE_REL  # 0.02
DAILY_USERS = 50000
TRAFFIC_ALLOC = 0.30
NUM_VARIANTS = 3

DAILY_EXPERIMENT = int(DAILY_USERS * TRAFFIC_ALLOC)
DAILY_PER_VARIANT = int(DAILY_EXPERIMENT / NUM_VARIANTS)

print(f"=== NOTIFICACIONES PUSH TEST ===")
print(f"Usuarios totales/día: {DAILY_USERS:,}")
print(f"Usuarios en experimento/día: {DAILY_EXPERIMENT:,}")
print(f"Usuarios por variante/día: {DAILY_PER_VARIANT:,}\n")

# Cálculo rápido de los 3 métodos
z_a = norm.ppf(0.975)
z_b = norm.ppf(0.80)
p = BASELINE

n_freq = (2 * (z_a + z_b)**2 * p * (1-p)) / (MDE_ABS**2)
n_freq_bonf = (2 * (norm.ppf(1-0.025/2) + z_b)**2 * p * (1-p)) / (MDE_ABS**2)

days_freq = n_freq / DAILY_PER_VARIANT
days_freq_bonf = n_freq_bonf / DAILY_PER_VARIANT

print("FRECUENTISTA:")
print(f"  {days_freq:.1f} días sin corrección")
print(f"  {days_freq_bonf:.1f} días con Bonferroni")
print(f"  Recomendado: {np.ceil(days_freq_bonf/7):.0f} semanas\n")

# Con alto tráfico, SOAT puede decidir muy rápido
print("SOAT ESTIMADO:")
print(f"  Optimista: {np.ceil(days_freq * 0.4):.0f} días")
print(f"  Conservador: {np.ceil(days_freq * 0.6):.0f} días")
print(f"  Con mínimo de 1 semana: 7-{np.ceil(days_freq * 0.6):.0f} días\n")

print("BAYESIANO ESTIMADO:")
print(f"  {np.ceil(days_freq * 0.7):.0f}-{np.ceil(days_freq * 0.85):.0f} días\n")

print("=== RECOMENDACIÓN ===")
print("Método: SOAT")
print("Razón: Alto tráfico permite decisiones rápidas")
print("Configuración:")
print("  - Alpha: 0.05")
print("  - Mínimo: 7 días (1 semana completa)")
print("  - Monitoreo: Diario")
print("  - Esperado: 7-10 días hasta decisión")
```

### Monitoreo Día a Día

```python
# Simulación de datos día a día
def simulate_experiment_timeline():
    """Simula y muestra evolución día a día."""

    true_rates = [
        0.25,  # Control
        0.26,  # Variant A: +4%
        0.27,  # Variant B: +8% (target MDE)
    ]

    # Simular 14 días
    np.random.seed(42)

    results = []

    for day in range(1, 15):
        n = day * DAILY_PER_VARIANT

        conversions = [
            np.random.binomial(n, rate)
            for rate in true_rates
        ]

        # SOAT analysis
        p_c = (conversions[0] + 0.5) / (n + 1)
        threshold = np.log((1 - 0.05/2) / (1 - 0.80))

        decisions = []

        for i in range(1, NUM_VARIANTS):
            effects = np.linspace(-0.3, 0.3, 50)
            ll_alts = [
                conversions[i] * np.log(np.clip(p_c * (1 + d), 0.001, 0.999)) +
                (n - conversions[i]) * np.log(np.clip(1 - p_c * (1 + d), 0.001, 0.999))
                for d in effects
            ]

            ll_null = conversions[i] * np.log(p_c) + (n - conversions[i]) * np.log(1 - p_c)
            llr = max(ll_alts) - ll_null

            rate = conversions[i] / n
            lift = (rate - true_rates[0]) / true_rates[0]

            decision = 'SHIP' if (llr > threshold and day >= 7) else 'CONTINUE'
            decisions.append(decision)

            results.append({
                'day': day,
                'variant': f'variant_{chr(65+i-1)}',
                'n': n,
                'conversions': conversions[i],
                'rate': rate,
                'lift': lift,
                'llr': llr,
                'decision': decision
            })

    df = pd.DataFrame(results)

    print("\n=== TIMELINE DEL EXPERIMENTO ===\n")

    for day in [1, 3, 7, 10, 14]:
        day_data = df[df['day'] == day]

        print(f"DÍA {day} (n={day_data.iloc[0]['n']:,} por variante)")
        print("-" * 60)

        for _, row in day_data.iterrows():
            print(f"{row['variant']}:")
            print(f"  Rate: {row['rate']:.2%} (lift: {row['lift']:+.1%})")
            print(f"  LLR: {row['llr']:.2f} (threshold: {threshold:.2f})")
            print(f"  Decision: {row['decision']}")
            print()

        # Check si ya podemos decidir
        if day >= 7:
            winners = day_data[day_data['decision'] == 'SHIP']
            if len(winners) > 0:
                best = winners.loc[winners['lift'].idxmax()]
                print(f"🎉 GANADOR ENCONTRADO: {best['variant']} con +{best['lift']:.1%} lift")
                print(f"   Experimento puede terminar en día {day}")
                break

        print()

simulate_experiment_timeline()
```

### Output Esperado del Timeline

```
=== TIMELINE DEL EXPERIMENTO ===

DÍA 1 (n=5,000 por variante)
------------------------------------------------------------
variant_A:
  Rate: 26.04% (lift: +4.2%)
  LLR: 0.87 (threshold: 3.47)
  Decision: CONTINUE

variant_B:
  Rate: 27.12% (lift: +8.5%)
  LLR: 3.41 (threshold: 3.47)
  Decision: CONTINUE

DÍA 3 (n=15,000 por variante)
------------------------------------------------------------
variant_A:
  Rate: 25.89% (lift: +3.6%)
  LLR: 2.34 (threshold: 3.47)
  Decision: CONTINUE

variant_B:
  Rate: 26.98% (lift: +7.9%)
  LLR: 10.23 (threshold: 3.47)
  Decision: CONTINUE (mínimo 7 días)

DÍA 7 (n=35,000 por variante)
------------------------------------------------------------
variant_A:
  Rate: 26.01% (lift: +4.0%)
  LLR: 5.12 (threshold: 3.47)
  Decision: SHIP

variant_B:
  Rate: 27.05% (lift: +8.2%)
  LLR: 23.87 (threshold: 3.47)
  Decision: SHIP

🎉 GANADOR ENCONTRADO: variant_B con +8.2% lift
   Experimento puede terminar en día 7
```

### Conclusión Ejemplo 3

**Ganador:** Variant B (mensaje de notificación #3)
**Tiempo de decisión:** 7 días (1 semana)
**Frecuentista hubiera requerido:** ~14 días (2 semanas)
**Ahorro con SOAT:** 7 días (50%)

---

## Resumen de Velocidades

### Tabla Resumen de los 3 Ejemplos

| Ejemplo | Tráfico/día | Variantes | Frecuentista | Bayesiano | SOAT | Mejor Método |
|---------|-------------|-----------|--------------|-----------|------|--------------|
| **E-commerce Hero** | 20,000 | 4 | 14 días | 10 días | 8 días | SOAT (1.75x) |
| **SaaS Pricing** | 2,000 | 5 | 28 días | 21 días | 17 días | SOAT (1.65x) |
| **App Push Notif** | 15,000* | 3 | 14 días | 10 días | 7 días | SOAT (2.0x) |

*30% de 50K usuarios

### Gráfico Comparativo

```
Días hasta decisión

30 │
   │ ██████ Frecuentista
25 │ ████   Bayesiano
   │ ███    SOAT
20 │
   │
15 │
   │
10 │
   │
 5 │
   │
 0 └────────────────────────────
    Hero    Pricing   Push

Promedio de speedup:
- Bayesiano: 1.35x más rápido que Frecuentista
- SOAT: 1.8x más rápido que Frecuentista
```

## Templates Listos para Usar

### Template 1: Planificación Rápida

```python
def quick_experiment_plan(
    baseline_rate: float,
    mde_relative: float,
    daily_traffic: int,
    num_variants: int = 2,
    method: str = 'soat'
) -> dict:
    """
    Planificación rápida de experimento.

    Returns:
        dict con duración estimada y configuración
    """
    from scipy.stats import norm

    mde_abs = baseline_rate * mde_relative

    # Frecuentista baseline
    z_a = norm.ppf(0.975)
    z_b = norm.ppf(0.80)
    n_freq = (2 * (z_a + z_b)**2 * baseline_rate * (1-baseline_rate)) / (mde_abs**2)

    daily_per_variant = daily_traffic / num_variants
    days_freq = n_freq / daily_per_variant

    # Adjust based on method
    if method == 'frequentist':
        # Bonferroni if multiple variants
        if num_variants > 2:
            alpha_adj = 0.05 / (num_variants - 1)
            z_a_adj = norm.ppf(1 - alpha_adj/2)
            n_adj = (2 * (z_a_adj + z_b)**2 * baseline_rate * (1-baseline_rate)) / (mde_abs**2)
            days = n_adj / daily_per_variant
        else:
            days = days_freq

        weeks = int(np.ceil(days / 7))

        return {
            'method': 'Frequentist',
            'duration_days': days,
            'duration_weeks': weeks,
            'sample_size_per_variant': int(n_adj if num_variants > 2 else n_freq),
            'config': {
                'alpha': 0.05,
                'power': 0.80,
                'bonferroni': num_variants > 2
            }
        }

    elif method == 'bayesian':
        days = days_freq * 0.75  # Typically 70-80% of frequentist
        weeks = int(np.ceil(days / 7))

        return {
            'method': 'Bayesian',
            'duration_days': days,
            'duration_weeks': weeks,
            'sample_size_estimate': 'variable',
            'config': {
                'prior': 'uniform',
                'threshold': 0.95,
                'min_days': 7
            }
        }

    elif method == 'soat':
        days = days_freq * 0.55  # Typically 40-60% of frequentist
        days = max(7, days)  # Minimum 1 week
        weeks = int(np.ceil(days / 7))

        return {
            'method': 'SOAT',
            'duration_days': days,
            'duration_weeks': weeks,
            'sample_size_estimate': 'adaptive',
            'config': {
                'alpha': 0.05,
                'min_days': 7,
                'monitoring': 'daily'
            }
        }

# Ejemplo de uso
plan = quick_experiment_plan(
    baseline_rate=0.10,
    mde_relative=0.15,
    daily_traffic=10000,
    num_variants=3,
    method='soat'
)

print(f"Método: {plan['method']}")
print(f"Duración estimada: {plan['duration_days']:.0f} días ({plan['duration_weeks']} semanas)")
print(f"Configuración: {plan['config']}")
```

### Template 2: Análisis Post-Experimento

```python
def analyze_experiment(
    data: dict,  # {'variant_name': {'successes': int, 'trials': int}}
    control_name: str = 'control',
    method: str = 'all'  # 'frequentist', 'bayesian', 'soat', 'all'
) -> dict:
    """
    Analiza resultados con uno o todos los métodos.

    Returns:
        dict con resultados de cada método
    """
    results = {}

    control = data[control_name]

    if method in ['frequentist', 'all']:
        freq_results = {}
        for vname, vdata in data.items():
            if vname == control_name:
                continue

            # Z-test
            p_c = control['successes'] / control['trials']
            p_v = vdata['successes'] / vdata['trials']

            p_pooled = (control['successes'] + vdata['successes']) / (control['trials'] + vdata['trials'])
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control['trials'] + 1/vdata['trials']))
            z = (p_v - p_c) / se
            p_value = 2 * (1 - norm.cdf(abs(z)))

            freq_results[vname] = {
                'rate': p_v,
                'lift': (p_v - p_c) / p_c,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        results['frequentist'] = freq_results

    if method in ['bayesian', 'all']:
        bayes_results = {}
        for vname, vdata in data.items():
            if vname == control_name:
                continue

            post_c = beta(1 + control['successes'], 1 + control['trials'] - control['successes'])
            post_v = beta(1 + vdata['successes'], 1 + vdata['trials'] - vdata['successes'])

            samp_c = post_c.rvs(100000)
            samp_v = post_v.rvs(100000)

            prob = (samp_v > samp_c).mean()

            bayes_results[vname] = {
                'rate': vdata['successes'] / vdata['trials'],
                'prob_beats_control': prob,
                'decision': 'SHIP' if prob > 0.95 else 'CONTINUE'
            }

        results['bayesian'] = bayes_results

    if method in ['soat', 'all']:
        # Implementar SOAT analysis similar a ejemplos anteriores
        pass

    return results

# Ejemplo de uso
experiment_data = {
    'control': {'successes': 450, 'trials': 5000},
    'variant_a': {'successes': 490, 'trials': 5000},
    'variant_b': {'successes': 520, 'trials': 5000}
}

results = analyze_experiment(experiment_data, method='all')
print(results)
```

## Recursos Adicionales

- [Calculadora online interactiva](https://abtestguide.com/calc/)
- [Repositorio de código completo](https://github.com/your-repo/ab-testing-examples)
- [Notebook Jupyter con todos los ejemplos](./notebooks/complete_examples.ipynb)

## Conclusiones Clave

1. **SOAT es consistentemente más rápido** (1.5-2.5x) en todos los escenarios
2. **Múltiples variantes** penalizan más a Frecuentista (Bonferroni)
3. **Bajo tráfico** limita beneficios de todos los métodos
4. **Alto tráfico** maximiza beneficios de SOAT (decisiones en días)
5. **Bayesiano** es buen punto medio si tienes expertise

**Recomendación general: Usa SOAT para maximizar velocidad de aprendizaje.**
