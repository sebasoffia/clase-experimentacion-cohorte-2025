# Enfoque Frecuentista para Experimentación

## Descripción General

El enfoque frecuentista es el método tradicional de análisis de experimentos A/B. Se basa en la **hipótesis nula** y el concepto de repetir un experimento infinitas veces para evaluar la probabilidad de observar los datos si no hubiera efecto real.

## Características Principales

### Ventajas
- ✅ Ampliamente aceptado y entendido
- ✅ Matemáticamente robusto
- ✅ Controla estrictamente los errores Tipo I (falsos positivos)
- ✅ No requiere conocimiento previo
- ✅ Interpretación clara del p-valor

### Desventajas
- ❌ Requiere tamaño de muestra pre-calculado
- ❌ No permite stop anticipado (peeking problem)
- ❌ Interpretación del p-valor a menudo mal entendida
- ❌ Decisiones binarias (significativo/no significativo)
- ❌ No incorpora conocimiento previo

## Fundamentos Teóricos

### Hipótesis Nula y Alternativa

```
H₀: μ_tratamiento = μ_control  (No hay diferencia)
H₁: μ_tratamiento ≠ μ_control  (Hay diferencia)
```

### Test de Hipótesis

**Proceso:**
1. Definir H₀ y H₁
2. Calcular tamaño de muestra requerido
3. Recolectar datos hasta alcanzar n
4. Calcular estadístico de prueba
5. Obtener p-valor
6. Decisión: Rechazar H₀ si p < α

### P-valor

**Definición Correcta:**
> Probabilidad de observar un efecto igual o más extremo que el observado, asumiendo que H₀ es verdadera.

**NO significa:**
- ❌ Probabilidad de que H₀ sea verdadera
- ❌ Probabilidad de que el resultado sea por azar
- ❌ Tamaño del efecto

## Cálculo de Tamaño de Muestra

### Para Métrica Binaria (Conversión)

```python
import numpy as np
from scipy.stats import norm

def frequentist_sample_size_binary(
    p_control: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> int:
    """
    Calcula tamaño de muestra para test frecuentista de proporción.

    Args:
        p_control: Tasa de conversión del control (ej: 0.10 para 10%)
        mde: Efecto mínimo detectable absoluto (ej: 0.01 para 1pp)
        alpha: Nivel de significancia (típicamente 0.05)
        power: Poder estadístico (típicamente 0.80)
        two_tailed: Si True, test de dos colas

    Returns:
        Tamaño de muestra por variante
    """
    # Z-scores
    if two_tailed:
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    z_beta = norm.ppf(power)

    # Proporciones
    p_treatment = p_control + mde
    p_pooled = (p_control + p_treatment) / 2

    # Fórmula
    numerator = (z_alpha + z_beta)**2 * 2 * p_pooled * (1 - p_pooled)
    denominator = mde**2

    n = numerator / denominator

    return int(np.ceil(n))


# Ejemplo de uso
n = frequentist_sample_size_binary(
    p_control=0.10,
    mde=0.02,  # Detectar diferencia de 2pp
    alpha=0.05,
    power=0.80
)
print(f"Tamaño de muestra requerido por variante: {n:,}")
# Output: 3,841
```

### Para Métrica Continua (Revenue, Time)

```python
def frequentist_sample_size_continuous(
    mean: float,
    std: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True
) -> int:
    """
    Calcula tamaño de muestra para test frecuentista de media.

    Args:
        mean: Media del control
        std: Desviación estándar
        mde_relative: Efecto mínimo detectable relativo (ej: 0.05 para 5%)
        alpha: Nivel de significancia
        power: Poder estadístico
        two_tailed: Si True, test de dos colas

    Returns:
        Tamaño de muestra por variante
    """
    if two_tailed:
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)

    z_beta = norm.ppf(power)

    mde_absolute = mean * mde_relative
    effect_size = mde_absolute / std  # Cohen's d

    n = 2 * ((z_alpha + z_beta) / effect_size)**2

    return int(np.ceil(n))


# Ejemplo de uso
n = frequentist_sample_size_continuous(
    mean=25.0,      # $25 revenue promedio
    std=50.0,       # $50 std
    mde_relative=0.10,  # 10% mejora
    alpha=0.05,
    power=0.80
)
print(f"Tamaño de muestra requerido por variante: {n:,}")
```

## Cálculo de Duración

### Fórmula Básica

```python
def calculate_duration(
    sample_size_per_variant: int,
    num_variants: int,
    daily_traffic: int,
    traffic_allocation: float = 1.0
) -> float:
    """
    Calcula duración del experimento en días.

    Args:
        sample_size_per_variant: n calculado
        num_variants: número total de variantes (incluyendo control)
        daily_traffic: usuarios diarios disponibles
        traffic_allocation: % de tráfico asignado (0.0 a 1.0)

    Returns:
        Duración en días
    """
    total_sample_needed = sample_size_per_variant * num_variants
    daily_experiment_traffic = daily_traffic * traffic_allocation

    duration = total_sample_needed / daily_experiment_traffic

    return duration


# Ejemplo
duration = calculate_duration(
    sample_size_per_variant=3841,
    num_variants=2,  # control + 1 tratamiento
    daily_traffic=10000,
    traffic_allocation=0.5  # 50% del tráfico
)

print(f"Duración: {duration:.1f} días")
print(f"Duración ajustada (semanas completas): {np.ceil(duration/7):.0f} semanas")
```

### Ajuste por Ciclos de Negocio

```python
def adjust_duration_for_business_cycles(
    duration_days: float,
    cycle_type: str = 'weekly'
) -> int:
    """
    Ajusta duración a ciclos completos de negocio.

    Args:
        duration_days: Duración calculada
        cycle_type: 'weekly', 'biweekly', 'monthly'

    Returns:
        Duración ajustada en días
    """
    cycle_days = {
        'weekly': 7,
        'biweekly': 14,
        'monthly': 28
    }

    cycle_length = cycle_days.get(cycle_type, 7)
    num_cycles = np.ceil(duration_days / cycle_length)

    return int(num_cycles * cycle_length)
```

## Análisis al Final del Experimento

### Test Z para Proporciones

```python
from scipy.stats import norm

def z_test_proportions(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    alpha: float = 0.05
) -> dict:
    """
    Realiza test Z de dos proporciones.

    Returns:
        dict con p_valor, z_score, es_significativo, lift
    """
    # Proporciones observadas
    p_control = conversions_control / n_control
    p_treatment = conversions_treatment / n_treatment

    # Proporción pooled (bajo H₀)
    p_pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)

    # Error estándar
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))

    # Z-score
    z_score = (p_treatment - p_control) / se

    # P-valor (two-tailed)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    # Decisión
    is_significant = p_value < alpha

    # Lift relativo
    lift = (p_treatment - p_control) / p_control

    return {
        'p_control': p_control,
        'p_treatment': p_treatment,
        'z_score': z_score,
        'p_value': p_value,
        'is_significant': is_significant,
        'lift_relative': lift,
        'lift_absolute': p_treatment - p_control
    }


# Ejemplo
result = z_test_proportions(
    conversions_control=450,
    n_control=5000,
    conversions_treatment=520,
    n_treatment=5000,
    alpha=0.05
)

print(f"Control: {result['p_control']:.2%}")
print(f"Treatment: {result['p_treatment']:.2%}")
print(f"Lift: {result['lift_relative']:.2%}")
print(f"P-valor: {result['p_value']:.4f}")
print(f"¿Significativo?: {result['is_significant']}")
```

### Intervalo de Confianza

```python
def confidence_interval(
    p_control: float,
    n_control: int,
    p_treatment: float,
    n_treatment: int,
    confidence_level: float = 0.95
) -> tuple:
    """
    Calcula intervalo de confianza para la diferencia de proporciones.

    Returns:
        (lower_bound, upper_bound) de la diferencia
    """
    diff = p_treatment - p_control

    # Error estándar de la diferencia
    se = np.sqrt(
        p_control * (1 - p_control) / n_control +
        p_treatment * (1 - p_treatment) / n_treatment
    )

    # Z-score para nivel de confianza
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    # Intervalo
    lower = diff - z * se
    upper = diff + z * se

    return (lower, upper)


# Ejemplo
ci = confidence_interval(
    p_control=0.09,
    n_control=5000,
    p_treatment=0.104,
    n_treatment=5000,
    confidence_level=0.95
)

print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"95% CI: [{ci[0]*100:.2f}pp, {ci[1]*100:.2f}pp]")
```

## Múltiples Variantes y Correcciones

### Corrección Bonferroni

Cuando comparas múltiples variantes contra el control:

```python
def bonferroni_correction(
    alpha: float,
    num_comparisons: int
) -> float:
    """
    Aplica corrección Bonferroni.

    Args:
        alpha: Nivel de significancia original
        num_comparisons: Número de comparaciones

    Returns:
        Alpha ajustado
    """
    return alpha / num_comparisons


# Ejemplo: 1 control + 3 tratamientos = 3 comparaciones
alpha_original = 0.05
num_comparisons = 3

alpha_adjusted = bonferroni_correction(alpha_original, num_comparisons)
print(f"Alpha ajustado: {alpha_adjusted:.4f}")
# Output: 0.0167

# Recalcular tamaño de muestra con alpha ajustado
n_adjusted = frequentist_sample_size_binary(
    p_control=0.10,
    mde=0.02,
    alpha=alpha_adjusted,  # Usar alpha ajustado
    power=0.80
)
```

### Impacto en Tamaño de Muestra

```python
def compare_sample_sizes():
    """Compara n requerido con y sin corrección."""
    p_control = 0.10
    mde = 0.02
    power = 0.80

    results = []

    for k in [2, 3, 4, 5]:  # número de variantes totales
        num_comparisons = k - 1

        # Sin corrección
        n_no_correction = frequentist_sample_size_binary(
            p_control, mde, 0.05, power
        )

        # Con Bonferroni
        alpha_bonf = 0.05 / num_comparisons
        n_bonferroni = frequentist_sample_size_binary(
            p_control, mde, alpha_bonf, power
        )

        results.append({
            'num_variants': k,
            'n_per_variant_no_correction': n_no_correction,
            'n_per_variant_bonferroni': n_bonferroni,
            'n_total_no_correction': n_no_correction * k,
            'n_total_bonferroni': n_bonferroni * k,
            'increase_pct': (n_bonferroni / n_no_correction - 1) * 100
        })

    return results


# Visualizar
import pandas as pd
df = pd.DataFrame(compare_sample_sizes())
print(df)
```

## El Problema del "Peeking"

### ¿Qué es Peeking?

**Peeking** = Mirar los resultados del experimento antes de alcanzar el tamaño de muestra planeado y tomar decisiones basadas en esa mirada.

### ¿Por qué es un problema?

```python
def simulate_peeking_problem(
    true_effect: float = 0.0,  # H₀ es verdadera
    n_final: int = 5000,
    n_looks: int = 50,
    alpha: float = 0.05
) -> dict:
    """
    Simula el problema de peeking.

    Demuestra que mirar múltiples veces infla la tasa de error Tipo I.
    """
    p_control = 0.10
    false_positives = 0
    num_simulations = 1000

    for _ in range(num_simulations):
        # Simular datos (sin efecto real)
        control = np.random.binomial(1, p_control, n_final)
        treatment = np.random.binomial(1, p_control + true_effect, n_final)

        # Mirar en múltiples puntos
        look_points = np.linspace(500, n_final, n_looks, dtype=int)

        for n in look_points:
            # Test en este punto
            result = z_test_proportions(
                conversions_control=control[:n].sum(),
                n_control=n,
                conversions_treatment=treatment[:n].sum(),
                n_treatment=n,
                alpha=alpha
            )

            if result['is_significant']:
                false_positives += 1
                break  # "Detuvimos" el experimento

    inflated_alpha = false_positives / num_simulations

    return {
        'nominal_alpha': alpha,
        'actual_alpha': inflated_alpha,
        'inflation_factor': inflated_alpha / alpha
    }


# Ejecutar simulación
result = simulate_peeking_problem()
print(f"Alpha nominal: {result['nominal_alpha']:.3f}")
print(f"Alpha real con peeking: {result['actual_alpha']:.3f}")
print(f"Factor de inflación: {result['inflation_factor']:.2f}x")
# Típicamente verás ~2-3x inflación
```

### Solución

**En enfoque frecuentista tradicional: NO MIRES**

Alternativas:
1. ✅ Usar **Sequential Testing** (ver [SOAT](./04-enfoque-soat.md))
2. ✅ Usar **Enfoque Bayesiano** (ver [Bayesiano](./03-enfoque-bayesiano.md))
3. ✅ Aplicar **correcciones** (ej: Pocock, O'Brien-Fleming)

## Reglas de Decisión

### Framework Completo

```python
class FrequentistDecisionFramework:
    """Framework de decisión frecuentista."""

    def __init__(
        self,
        alpha: float = 0.05,
        minimum_detectable_effect: float = 0.02,
        minimum_sample_size: int = 1000
    ):
        self.alpha = alpha
        self.mde = minimum_detectable_effect
        self.min_n = minimum_sample_size

    def make_decision(
        self,
        conversions_control: int,
        n_control: int,
        conversions_treatment: int,
        n_treatment: int
    ) -> dict:
        """
        Toma decisión basada en framework frecuentista.

        Returns:
            dict con decisión y razones
        """
        # Pre-checks
        if n_control < self.min_n or n_treatment < self.min_n:
            return {
                'decision': 'WAIT',
                'reason': f'Sample size too small (min: {self.min_n})'
            }

        # Test estadístico
        result = z_test_proportions(
            conversions_control, n_control,
            conversions_treatment, n_treatment,
            self.alpha
        )

        # Decisión
        if result['is_significant']:
            if abs(result['lift_absolute']) >= self.mde:
                if result['lift_absolute'] > 0:
                    decision = 'SHIP_TREATMENT'
                    reason = f"Significant improvement: {result['lift_relative']:.2%}"
                else:
                    decision = 'SHIP_CONTROL'
                    reason = f"Significant degradation: {result['lift_relative']:.2%}"
            else:
                decision = 'INCONCLUSIVE'
                reason = f"Significant but below MDE ({self.mde:.2%})"
        else:
            decision = 'NO_DIFFERENCE'
            reason = f"Not significant (p={result['p_value']:.4f})"

        return {
            'decision': decision,
            'reason': reason,
            'p_value': result['p_value'],
            'lift': result['lift_relative'],
            'ci': confidence_interval(
                result['p_control'], n_control,
                result['p_treatment'], n_treatment
            )
        }
```

## Velocidad de Decisión

### Comparación con Otros Enfoques

| Métrica | Frecuentista | Bayesiano | SOAT |
|---------|-------------|-----------|------|
| **Tiempo hasta decisión** | 100% | 70-85% | 30-60% |
| **Flexibilidad** | Baja | Alta | Media |
| **Stop anticipado** | ❌ No | ✅ Sí | ✅ Sí |
| **Monitoreo continuo** | ❌ No | ✅ Sí | ✅ Sí |

### Cuándo el Frecuentista es Más Rápido

Prácticamente **nunca** es más rápido que SOAT o Bayesiano, PERO:

✅ **Usa Frecuentista cuando**:
- Regulaciones requieren métodos tradicionales (farmacéutica, medical devices)
- Stakeholders solo entienden p-valores
- No tienes infraestructura para métodos más avanzados
- El overhead de explicar otros métodos es mayor que esperar más tiempo

## Checklist de Implementación

Antes de lanzar un experimento frecuentista:

- [ ] ✅ Calculé el tamaño de muestra requerido
- [ ] ✅ Definí MDE basado en impacto de negocio
- [ ] ✅ Ajusté por múltiples comparaciones (si aplica)
- [ ] ✅ Calculé duración y ajusté por ciclos de negocio
- [ ] ✅ Documenté que NO miraremos resultados antes de n
- [ ] ✅ Configuré alertas para cuando alcancemos n
- [ ] ✅ Definí reglas de decisión pre-experimento
- [ ] ✅ Tengo plan para métricas secundarias (solo exploración)

## Herramientas y Calculadoras

### Online
- [Evan Miller AB Test Calculator](https://www.evanmiller.org/ab-testing/sample-size.html)
- [Optimizely Sample Size Calculator](https://www.optimizely.com/sample-size-calculator/)
- [AB Test Guide Calculator](https://abtestguide.com/calc/)

### Python Libraries
```python
# statsmodels
from statsmodels.stats.power import zt_ind_solve_power

# scipy
from scipy.stats import ttest_ind, chi2_contingency

# pingouin (recomendado)
import pingouin as pg
pg.power_ttest2n(n1, n2, d, alpha=0.05, alternative='two-sided')
```

## Resumen Ejecutivo

**El enfoque frecuentista es:**
- 🐢 **Más lento**: Requiere alcanzar n pre-calculado
- 📊 **Más simple**: Fácil de explicar y entender
- 🎯 **Más conservador**: Controla estrictamente error Tipo I
- ⚠️ **Más rígido**: No permite monitoreo continuo sin correcciones

**Úsalo cuando:**
- La velocidad no es crítica
- Los stakeholders prefieren p-valores
- Las regulaciones lo requieren
- Tienes tiempo para planear y esperar

**NO lo uses cuando:**
- Necesitas decisiones rápidas
- Quieres monitorear continuamente
- Tienes alta presión por resultados tempranos
- Prefieres probabilidades sobre p-valores

## Siguiente: [Enfoque Bayesiano](./03-enfoque-bayesiano.md)
