# Guía de Experimentación: Cálculo de Duración de Experimentos

Esta documentación proporciona una guía completa para calcular la duración óptima de experimentos A/B, considerando diferentes enfoques estadísticos y factores como el número de creatividades y objetivos mínimos.

## Contenido

1. [Fundamentos del Cálculo de Duración](./docs/01-fundamentos-calculo-duracion.md)
2. [Enfoque Frecuentista](./docs/02-enfoque-frecuentista.md)
3. [Enfoque Bayesiano](./docs/03-enfoque-bayesiano.md)
4. [Enfoque SOAT (Statsig)](./docs/04-enfoque-soat.md)
5. [Guía Comparativa y de Decisión](./docs/05-guia-comparativa.md)
6. [Ejemplos Prácticos](./docs/06-ejemplos-practicos.md)

## Resumen Ejecutivo

### ¿Cuándo usar cada enfoque?

| Enfoque | Mejor para | Velocidad de Decisión | Complejidad |
|---------|-----------|----------------------|-------------|
| **Frecuentista** | Experimentos planificados, tamaño de muestra fijo | Media | Baja |
| **Bayesiano** | Incorporar conocimiento previo, actualizaciones continuas | Alta | Media-Alta |
| **SOAT** | Decisiones rápidas, monitoreo continuo, stop anticipado | Muy Alta | Baja |

### Quick Start

```bash
# 1. Determina tus parámetros básicos
- Tasa de conversión actual (baseline)
- Efecto mínimo detectable (MDE)
- Significancia estadística deseada (α)
- Poder estadístico deseado (β)
- Número de variantes/creatividades

# 2. Elige tu enfoque según tu necesidad de velocidad
- ¿Necesitas decisiones rápidas? → SOAT
- ¿Tienes conocimiento previo? → Bayesiano
- ¿Experimento tradicional planificado? → Frecuentista
```

## Estructura del Repositorio

```
.
├── README.md
├── docs/
│   ├── 01-fundamentos-calculo-duracion.md
│   ├── 02-enfoque-frecuentista.md
│   ├── 03-enfoque-bayesiano.md
│   ├── 04-enfoque-soat.md
│   ├── 05-guia-comparativa.md
│   └── 06-ejemplos-practicos.md
└── calculators/
    ├── frequentist_calculator.py
    ├── bayesian_calculator.py
    └── soat_calculator.py
```

## Factores Clave para Calcular Duración

### 1. Tamaño de Muestra Requerido
- Depende del MDE (efecto mínimo detectable)
- Afectado por la varianza de la métrica
- Impactado por el número de variantes

### 2. Número de Creatividades/Variantes
- Más variantes = más tráfico necesario
- Requiere corrección por comparaciones múltiples
- Considera el trade-off entre exploración y explotación

### 3. Objetivos Mínimos Completados
- Define el criterio de "conversión" o éxito
- Puede ser múltiple (primario y secundario)
- Afecta directamente la potencia estadística

### 4. Tráfico Disponible
- Usuarios/día que pueden participar
- Porcentaje de tráfico asignado al experimento
- Distribución entre variantes

## Inicio Rápido por Enfoque

### Frecuentista
```python
# Cálculo básico
n = (2 * (z_α/2 + z_β)² * σ²) / δ²
days = n / (traffic_per_day / num_variants)
```

### Bayesiano
```python
# Con prior informativo
posterior = likelihood * prior
decision = posterior_probability(variant_A > variant_B) > 0.95
```

### SOAT
```python
# Monitoreo continuo con mSPRT
likelihood_ratio = calculate_mSPRT()
if likelihood_ratio > threshold:
    stop_experiment()
```

## Contribuciones

Este repositorio es parte del curso de Experimentación Cohorte 2025.

## Referencias

- Kohavi, R., Tang, D., & Xu, Y. (2020). Trustworthy Online Controlled Experiments
- VWO: [A/B Test Duration Calculator](https://vwo.com/tools/ab-test-duration-calculator/)
- Statsig: [SOAT Documentation](https://docs.statsig.com/stats-engine/methodologies/sequential-testing)
- Optimizely: [Sample Size Calculator](https://www.optimizely.com/sample-size-calculator/)
