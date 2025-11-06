# Fundamentos del Cálculo de Duración de Experimentos

## Introducción

La duración de un experimento A/B es uno de los factores más críticos para el éxito de la experimentación. Correr un experimento por muy poco tiempo puede llevar a decisiones incorrectas, mientras que correrlo demasiado tiempo desperdicia recursos y retrasa aprendizajes.

## Componentes Básicos

### 1. Tamaño de Muestra (n)

El tamaño de muestra requerido es la base para calcular la duración del experimento.

**Fórmula General:**

```
n = (Z_α/2 + Z_β)² × 2 × p × (1-p) / (MDE)²
```

Donde:
- `Z_α/2`: Z-score para el nivel de significancia (típicamente 1.96 para α=0.05)
- `Z_β`: Z-score para el poder estadístico (típicamente 0.84 para poder=0.80)
- `p`: Tasa de conversión baseline
- `MDE`: Efecto Mínimo Detectable (Minimum Detectable Effect)

### 2. Efecto Mínimo Detectable (MDE)

El MDE representa el cambio mínimo que queremos detectar en nuestra métrica.

**Consideraciones:**
- **MDE Pequeño** (1-5%): Requiere muestras muy grandes, experimentos largos
- **MDE Mediano** (5-15%): Balance entre precisión y velocidad
- **MDE Grande** (>15%): Muestras pequeñas, decisiones rápidas

**Cómo elegir tu MDE:**

```python
# Basado en impacto de negocio
MDE_business = minimum_revenue_impact / current_revenue

# Basado en viabilidad estadística
MDE_statistical = f(traffic, time_available)

# Usar el mayor de los dos
MDE = max(MDE_business, MDE_statistical)
```

### 3. Nivel de Significancia (α)

Probabilidad de rechazar la hipótesis nula cuando es verdadera (Error Tipo I).

**Estándares Comunes:**
- α = 0.05 (95% confianza) - Estándar en ciencia
- α = 0.01 (99% confianza) - Decisiones críticas
- α = 0.10 (90% confianza) - Exploración rápida

### 4. Poder Estadístico (1-β)

Probabilidad de detectar un efecto cuando realmente existe.

**Estándares Comunes:**
- 0.80 (80%) - Mínimo recomendado
- 0.90 (90%) - Recomendado para decisiones importantes
- 0.95 (95%) - Alto costo, raramente necesario

## Cálculo de Duración

### Fórmula Base

```
Duración (días) = Tamaño de muestra requerido / (Tráfico diario × % asignado al experimento)
```

### Con Múltiples Variantes

```
n_total = n_por_variante × número_de_variantes
Duración = n_total / (Tráfico_diario × % experimento)
```

### Ejemplo Básico

```python
# Parámetros
baseline_conversion = 0.10  # 10%
MDE = 0.02  # 2% absoluto (20% relativo)
alpha = 0.05
power = 0.80
num_variants = 2  # Control + 1 variante

# Cálculo
z_alpha = 1.96
z_beta = 0.84
p = baseline_conversion

n_per_variant = ((z_alpha + z_beta)**2 * 2 * p * (1-p)) / (MDE**2)
# n_per_variant ≈ 3,841

n_total = n_per_variant * num_variants
# n_total ≈ 7,682

# Con 10,000 usuarios/día y 100% tráfico
duracion_dias = n_total / 10000
# ≈ 0.77 días (pero necesitamos considerar ciclos de negocio)
```

## Consideraciones Importantes

### 1. Ciclos de Negocio

**Siempre correr experimentos en múltiplos de ciclos completos:**

- **E-commerce**: Ciclo semanal (lunes a domingo)
- **B2B**: Ciclo de 2-4 semanas
- **Apps móviles**: Considerar fin de semana vs. días laborales

**Duración Mínima Recomendada:**
- 1 semana para patrones semanales
- 2 semanas para mayor robustez
- 4 semanas para cambios de comportamiento

### 2. Varianza en el Tiempo

```python
# Factor de corrección por varianza temporal
variance_factor = 1 + (CV_traffic / sqrt(days))

# Donde CV_traffic es el coeficiente de variación del tráfico
```

### 3. Múltiples Creatividades

Cuando tienes múltiples creatividades/variantes:

**Sin Corrección Bonferroni:**
```
n_total = n_base × k
# k = número de variantes
```

**Con Corrección Bonferroni:**
```
α_ajustado = α / (k-1)  # número de comparaciones
# Recalcular n con α_ajustado
```

### 4. Métricas Múltiples

**Primary Metric**: Define tu tamaño de muestra
**Secondary Metrics**: Solo para exploración, no para decisiones

**Corrección para m métricas:**
```
α_ajustado = α / m  # Bonferroni
# O usar False Discovery Rate (FDR) para menos penalización
```

## Objetivos Mínimos Completados

### Definición

El número mínimo de conversiones/eventos que necesitas para tener poder estadístico.

```python
# Para métrica binaria
min_conversions = n_per_variant × baseline_conversion_rate

# Ejemplo
# Si n = 5000 y baseline = 0.10
min_conversions = 5000 × 0.10 = 500 conversiones
```

### Para Múltiples Objetivos

**Objetivo Primario**: Determina el tamaño de muestra
**Objetivos Secundarios**:

```python
# Calcular por separado
n_secondary = calculate_sample_size(
    baseline=secondary_baseline,
    MDE=secondary_MDE,
    alpha=alpha,
    power=power
)

# Usar el máximo
n_required = max(n_primary, n_secondary)
```

### Regla de Oro

**Mínimo de 350-400 conversiones por variante** para métricas binarias, independiente del cálculo teórico.

## Calculadoras de Tamaño de Muestra

### Fórmulas por Tipo de Métrica

#### Métrica Binaria (Conversión)

```python
def sample_size_binary(p1, MDE, alpha=0.05, power=0.80):
    """
    p1: tasa de conversión baseline
    MDE: efecto mínimo detectable (absoluto)
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    p2 = p1 + MDE
    p_pooled = (p1 + p2) / 2

    n = 2 * ((z_alpha + z_beta)**2) * p_pooled * (1 - p_pooled) / (MDE**2)

    return int(np.ceil(n))
```

#### Métrica Continua (Revenue, Time on Site)

```python
def sample_size_continuous(mean, std, MDE_percent, alpha=0.05, power=0.80):
    """
    mean: media baseline
    std: desviación estándar
    MDE_percent: efecto mínimo detectable (%)
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    MDE_absolute = mean * MDE_percent

    n = 2 * ((z_alpha + z_beta)**2) * (std**2) / (MDE_absolute**2)

    return int(np.ceil(n))
```

#### Métrica de Recuento (Clicks, Page Views)

```python
def sample_size_count(lambda_baseline, MDE_percent, alpha=0.05, power=0.80):
    """
    lambda_baseline: tasa promedio de eventos
    MDE_percent: efecto mínimo detectable (%)
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    lambda_treatment = lambda_baseline * (1 + MDE_percent)

    n = ((z_alpha + z_beta)**2) * (lambda_baseline + lambda_treatment) / ((lambda_treatment - lambda_baseline)**2)

    return int(np.ceil(n))
```

## Plantilla de Cálculo

```python
# 1. DEFINE TUS PARÁMETROS
baseline_metric = 0.15  # 15% conversión
MDE_relative = 0.10  # 10% mejora relativa
MDE_absolute = baseline_metric * MDE_relative  # 0.015

alpha = 0.05
power = 0.80
num_variants = 3  # 1 control + 2 tratamientos

# 2. CALCULA TAMAÑO DE MUESTRA
n_per_variant = sample_size_binary(
    p1=baseline_metric,
    MDE=MDE_absolute,
    alpha=alpha,
    power=power
)

# 3. AJUSTA POR MÚLTIPLES VARIANTES
n_total = n_per_variant * num_variants

# 4. CALCULA DURACIÓN
traffic_per_day = 50000
experiment_traffic_percent = 0.50  # 50% del tráfico

daily_experiment_traffic = traffic_per_day * experiment_traffic_percent

duration_days = n_total / daily_experiment_traffic

# 5. AJUSTA POR CICLOS DE NEGOCIO
weeks_needed = np.ceil(duration_days / 7)
final_duration_days = weeks_needed * 7

print(f"Tamaño de muestra por variante: {n_per_variant:,.0f}")
print(f"Tamaño de muestra total: {n_total:,.0f}")
print(f"Duración calculada: {duration_days:.1f} días")
print(f"Duración ajustada: {final_duration_days:.0f} días ({weeks_needed:.0f} semanas)")
```

## Validaciones Pre-Experimento

Antes de lanzar, verifica:

- [ ] ¿El MDE es realista y vale la pena detectar?
- [ ] ¿Tienes suficiente tráfico para el experimento?
- [ ] ¿La duración cabe en tu timeline de decisión?
- [ ] ¿Has considerado ciclos de negocio?
- [ ] ¿Has ajustado por múltiples comparaciones?
- [ ] ¿Tienes al menos 350-400 conversiones esperadas por variante?

## Referencias

- Kohavi et al. (2020) - Trustworthy Online Controlled Experiments, Cap. 3
- Evan Miller - [Sample Size Calculator](https://www.evanmiller.org/ab-testing/sample-size.html)
- Optimizely - [Stats Accelerator](https://www.optimizely.com/optimization-glossary/statistical-significance/)

## Siguiente: [Enfoque Frecuentista](./02-enfoque-frecuentista.md)
