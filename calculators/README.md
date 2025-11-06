# Calculadoras de Experimentación

Este directorio contiene implementaciones en Python de los tres enfoques de experimentación: Frecuentista, Bayesiano y SOAT.

## Instalación de Dependencias

```bash
pip install numpy scipy
```

## Calculadoras Disponibles

### 1. Frecuentista (`frequentist_calculator.py`)

Implementa el enfoque tradicional de A/B testing.

**Ejemplo de uso:**

```python
from frequentist_calculator import quick_plan, z_test_proportions

# Planificación
plan = quick_plan(
    baseline_rate=0.10,
    mde_relative=0.15,
    daily_traffic=20000,
    num_variants=3
)

print(f"Duración: {plan['duration_weeks']} semanas")
print(f"Sample size: {plan['sample_size_per_variant']:,} por variante")

# Análisis
result = z_test_proportions(
    conversions_control=700,
    n_control=7000,
    conversions_treatment=805,
    n_treatment=7000
)

print(f"P-valor: {result['p_value']:.4f}")
print(f"Significativo: {result['is_significant']}")
```

**Funciones principales:**
- `sample_size_binary()` - Calcula tamaño de muestra para métrica binaria
- `sample_size_continuous()` - Calcula tamaño de muestra para métrica continua
- `calculate_duration()` - Calcula duración del experimento
- `z_test_proportions()` - Test Z para análisis
- `quick_plan()` - Planificación rápida completa

### 2. Bayesiano (`bayesian_calculator.py`)

Implementa enfoque bayesiano con Beta-Binomial conjugacy.

**Ejemplo de uso:**

```python
from bayesian_calculator import BayesianABTest, multiple_variants_analysis

# Test A/B simple
test = BayesianABTest(prior_alpha=1, prior_beta=1)

prob = test.probability_b_beats_a(
    conversions_a=450,
    trials_a=5000,
    conversions_b=520,
    trials_b=5000
)

print(f"P(B > A): {prob:.2%}")

decision = test.make_decision(
    conversions_a=450,
    trials_a=5000,
    conversions_b=520,
    trials_b=5000
)

print(f"Decisión: {decision['decision']}")

# Múltiples variantes
data = {
    'control': (450, 5000),
    'variant_a': (490, 5000),
    'variant_b': (520, 5000)
}

results = multiple_variants_analysis(data)
for variant, metrics in results.items():
    print(f"{variant}: {metrics['prob_beats_control']:.2%}")
```

**Funciones principales:**
- `BayesianABTest` - Clase principal para análisis bayesiano
- `probability_b_beats_a()` - Calcula P(B > A)
- `expected_loss()` - Calcula pérdida esperada
- `make_decision()` - Toma decisión basada en thresholds
- `multiple_variants_analysis()` - Analiza múltiples variantes
- `probability_of_being_best()` - P(cada variante es la mejor)
- `simulate_experiment_duration()` - Estima duración vía simulación

### 3. SOAT (`soat_calculator.py`)

Implementa Sequential Open-ended Adaptive Testing.

**Ejemplo de uso:**

```python
from soat_calculator import SOATTest, SOATMonitor, simulate_soat_duration

# Análisis simple
soat = SOATTest(alpha=0.05, power=0.80)

decision = soat.make_decision(
    conversions_control=700,
    n_control=7000,
    conversions_treatment=805,
    n_treatment=7000,
    current_day=10
)

print(f"LLR: {decision['llr']:.2f}")
print(f"Decisión: {decision['decision']}")

# Monitoreo continuo
monitor = SOATMonitor(min_days=7, min_sample_size=1000)

data = {
    'control': {'conversions': 700, 'trials': 7000},
    'treatment': {'conversions': 805, 'trials': 7000}
}

status = monitor.check_experiment(data, current_day=10)
print(status['recommendation'])

# Simulación de duración
sim = simulate_soat_duration(
    baseline_rate=0.10,
    true_effect=0.02,
    daily_traffic_per_variant=2500
)

print(f"Duración mediana: {sim['median_days']:.0f} días")
```

**Funciones principales:**
- `SOATTest` - Clase principal para SOAT
- `mixture_log_likelihood_ratio()` - Calcula mSPRT
- `sequential_p_value()` - Convierte LLR a p-value
- `make_decision()` - Toma decisión
- `SOATMonitor` - Monitor en tiempo real
- `simulate_soat_duration()` - Estima duración vía simulación
- `multiple_variants_soat()` - Analiza múltiples variantes

## Comparación Rápida

```python
from frequentist_calculator import quick_plan as freq_plan
from bayesian_calculator import simulate_experiment_duration as bayes_sim
from soat_calculator import simulate_soat_duration as soat_sim

# Parámetros comunes
baseline = 0.10
mde = 0.02
daily_traffic = 5000

# Frecuentista
freq = freq_plan(baseline, mde/baseline, daily_traffic)
print(f"Frecuentista: {freq['duration_weeks']} semanas")

# Bayesiano
bayes = bayes_sim(baseline, mde, daily_traffic//2, n_simulations=100)
print(f"Bayesiano: {bayes['median_days']/7:.1f} semanas")

# SOAT
soat = soat_sim(baseline, mde, daily_traffic//2, n_simulations=100)
print(f"SOAT: {soat['median_days']/7:.1f} semanas")
```

## Ejecutar Tests

Cada calculadora incluye ejemplos ejecutables:

```bash
# Frecuentista
python frequentist_calculator.py

# Bayesiano
python bayesian_calculator.py

# SOAT
python soat_calculator.py
```

## Guía de Decisión

**Usa Frecuentista cuando:**
- Regulaciones lo requieren
- Simplicidad es prioritaria
- Stakeholders solo entienden p-valores

**Usa Bayesiano cuando:**
- Tienes conocimiento previo valioso
- Quieres respuestas probabilísticas
- Necesitas velocidad moderada

**Usa SOAT cuando:**
- Velocidad es crítica
- Quieres monitoreo continuo
- Corres muchos experimentos
- No hay restricciones regulatorias

## Documentación Completa

Ver [documentación completa](../docs/) para guías detalladas y ejemplos prácticos.

## Licencia

MIT License
