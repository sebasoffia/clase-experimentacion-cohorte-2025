# Guía Comparativa: ¿Qué Enfoque Elegir?

## Resumen Ejecutivo

Esta guía te ayudará a elegir entre **Frecuentista**, **Bayesiano** y **SOAT** basándote en tus necesidades específicas de velocidad, rigor y contexto de negocio.

## Comparación Directa

### Tabla Maestra de Comparación

| Criterio | Frecuentista | Bayesiano | SOAT |
|----------|-------------|-----------|------|
| **⏱️ Velocidad** | Lento (baseline: 100%) | Media-Rápida (70-85%) | Rápida (40-60%) |
| **🔄 Monitoreo continuo** | ❌ No | ✅ Sí | ✅ Sí |
| **🛑 Stop anticipado** | ❌ No | ✅ Sí | ✅ Sí |
| **📊 Complejidad teórica** | Baja | Alta | Media |
| **💬 Complejidad explicación** | Baja | Alta | Baja-Media |
| **🎯 Control errores** | Estricto | Configurable | Estricto |
| **📚 Conocimiento previo** | No usa | ✅ Incorpora | Implícito en prior |
| **🔢 Interpretación** | P-valores | Probabilidades | Sequential p-values |
| **🏭 Adopción industria** | Muy alta | Media-Alta | Creciente |
| **🔧 Implementación** | Simple | Compleja | Media |
| **💰 Costo computacional** | Bajo | Medio-Alto | Medio |
| **📖 Regulatorio** | ✅ Ampliamente aceptado | ⚠️ Depende | ⚠️ Emergente |

### Speedup Relativo (Tiempo hasta Decisión)

```
Caso base: Efecto moderado (10% relativo)

Frecuentista: ████████████████████ 100% (20 días)
Bayesiano:    ██████████████       70%  (14 días)
SOAT:         ████████             40%  (8 días)

Caso: Efecto grande (25% relativo)

Frecuentista: ████████             100% (8 días)
Bayesiano:    █████                65%  (5 días)
SOAT:         ███                  35%  (3 días)
```

## Árbol de Decisión

### Paso 1: ¿Hay Restricciones Externas?

```
┌─ ¿Regulaciones requieren frecuentista? (farmacéutica, medical)
│  ├─ SÍ → FRECUENTISTA
│  └─ NO → Continuar
│
└─ ¿Stakeholders solo entienden p-valores tradicionales?
   ├─ SÍ → FRECUENTISTA o SOAT
   └─ NO → Continuar
```

### Paso 2: ¿Qué tan Crítica es la Velocidad?

```
┌─ ¿Necesitas decisiones lo más rápido posible?
│  ├─ CRÍTICA → SOAT
│  ├─ IMPORTANTE → SOAT o Bayesiano
│  └─ NO CRÍTICA → Cualquiera
│
└─ ¿Corres muchos experimentos simultáneos?
   ├─ SÍ (>10/mes) → SOAT (maximizar throughput)
   └─ NO (<5/mes) → Cualquiera
```

### Paso 3: ¿Tienes Conocimiento Previo?

```
┌─ ¿Tienes datos históricos o creencias previas fuertes?
│  ├─ SÍ → BAYESIANO
│  └─ NO → SOAT o Frecuentista
│
└─ ¿Quieres incorporar ese conocimiento?
   ├─ SÍ → BAYESIANO
   └─ NO → SOAT
```

### Paso 4: ¿Capacidad Técnica?

```
┌─ ¿Tu equipo puede implementar métodos avanzados?
│  ├─ SÍ → SOAT o Bayesiano
│  └─ NO → FRECUENTISTA
│
└─ ¿Tienes infraestructura de experimentación?
   ├─ PLATAFORMA (Statsig, etc.) → SOAT
   ├─ CUSTOM → BAYESIANO o FRECUENTISTA
   └─ BÁSICA → FRECUENTISTA
```

## Matriz de Casos de Uso

### Por Industria

| Industria | Recomendado | Razón |
|-----------|-------------|-------|
| **Tech/SaaS** | SOAT | Velocidad crítica, muchos experimentos |
| **E-commerce** | SOAT o Bayesiano | Balance velocidad/complejidad |
| **Fintech** | SOAT | Velocidad importante, stakeholders técnicos |
| **B2B Enterprise** | Frecuentista o Bayesiano | Ciclos largos, menor volumen |
| **Farmacéutica** | Frecuentista | Requerimientos regulatorios |
| **Gaming/Apps** | SOAT | Alta velocidad de iteración |
| **Media/Content** | SOAT o Bayesiano | Cambios frecuentes, datos históricos |

### Por Tipo de Experimento

| Tipo | Recomendado | Configuración |
|------|-------------|---------------|
| **UI/UX pequeños cambios** | SOAT | Agresivo (α=0.10, min_days=3) |
| **Features importantes** | SOAT | Balanceado (α=0.05, min_days=7) |
| **Cambios arquitecturales** | Frecuentista | Conservador (α=0.01, 2+ semanas) |
| **Pricing** | Bayesiano o Frecuentista | Incorporar datos históricos |
| **Algoritmos ML** | Bayesiano | Prior informativo de backtests |
| **Onboarding** | SOAT | Velocidad para iterar rápido |
| **Email campaigns** | SOAT | Múltiples tests paralelos |

### Por Restricciones de Negocio

| Restricción | Enfoque | Ejemplo |
|-------------|---------|---------|
| **Muy poco tráfico** (<1K/día) | Bayesiano | Prior informativo ayuda |
| **Mucho tráfico** (>100K/día) | SOAT | Decisiones en horas/días |
| **Tráfico medio** (1-100K/día) | SOAT o Bayesiano | Según otros factores |
| **Deadline fijo** | SOAT | Maximiza info en tiempo fijo |
| **Presupuesto limitado** | SOAT | Minimiza tiempo de experimento |
| **Risk-averse** | Frecuentista | Estándares tradicionales |

## Escenarios Específicos

### Escenario 1: Startup Tech con Alto Growth

**Contexto:**
- 50K usuarios activos/día
- Corriendo 15+ experimentos/mes
- Equipo técnico sofisticado
- Necesidad de iterar rápido

**Recomendación: SOAT**

**Razones:**
1. ✅ Velocidad maximiza aprendizaje
2. ✅ Alto volumen de experimentos → cada día ahorrado se multiplica
3. ✅ Equipo puede implementar/entender
4. ✅ No hay restricciones regulatorias

**Configuración:**
```python
config = {
    'method': 'SOAT',
    'alpha': 0.05,
    'min_days': 7,
    'min_sample_size': 1000,
    'monitoring': 'daily'
}
```

### Escenario 2: E-commerce Establecido con Datos Históricos

**Contexto:**
- 200K visitantes/día
- 5 años de datos históricos
- 5-10 experimentos/mes
- Equipo con background en analytics

**Recomendación: Bayesiano con Prior Informativo**

**Razones:**
1. ✅ Datos históricos valiosos → mejor prior
2. ✅ Puede acelerar decisiones vs frecuentista
3. ✅ Equipo puede entender probabilidades
4. ✅ No necesita máxima velocidad (SOAT)

**Configuración:**
```python
# Usar datos históricos para prior
historical_conversion = 0.035  # 3.5%
historical_std = 0.002  # 0.2%

alpha, beta = create_informative_prior(
    historical_mean=historical_conversion,
    historical_std=historical_std
)

config = {
    'method': 'Bayesian',
    'prior': (alpha, beta),
    'threshold': 0.95,
    'min_days': 7
}
```

### Escenario 3: B2B SaaS con Poco Tráfico

**Contexto:**
- 500 trials/día
- Ciclos de venta largos (30-90 días)
- Métrica: signup → paid conversion
- Stakeholders no técnicos

**Recomendación: Frecuentista Conservador**

**Razones:**
1. ✅ Poco tráfico → beneficio de SOAT/Bayesiano limitado
2. ✅ Ciclos largos → necesitas correr tiempo completo de todas formas
3. ✅ Stakeholders entienden p-valores
4. ✅ Simplicidad es más valiosa

**Configuración:**
```python
config = {
    'method': 'Frequentist',
    'alpha': 0.05,
    'power': 0.80,
    'min_duration_days': 60,  # 2 ciclos completos
    'traffic_allocation': 0.50
}
```

### Escenario 4: App Móvil con Iteración Rápida

**Contexto:**
- 100K DAU
- Múltiples AB tests paralelos (20+)
- Features nuevas cada sprint (2 semanas)
- Equipo product-driven

**Recomendación: SOAT Agresivo**

**Razones:**
1. ✅ Alto volumen de tests → velocidad = más aprendizaje
2. ✅ Suficiente tráfico para stop anticipado
3. ✅ Contexto permite aceptar más riesgo
4. ✅ Necesita decidir dentro del sprint

**Configuración:**
```python
config = {
    'method': 'SOAT',
    'alpha': 0.10,  # Más agresivo
    'min_days': 3,
    'min_sample_size': 500,
    'early_stop_enabled': True
}
```

### Escenario 5: Experimento Crítico de Negocio

**Contexto:**
- Rediseño completo de checkout
- Impacto potencial: ±$5M/año
- Stakeholders: C-level
- Timeline: No urgente

**Recomendación: Frecuentista Conservador o Bayesiano con Alta Threshold**

**Razones:**
1. ✅ Alto riesgo → necesitas máxima confianza
2. ✅ Decisión única → velocidad no crítica
3. ✅ Justificación a C-level → p-values familiares
4. ✅ Costo de error muy alto

**Configuración:**
```python
# Opción 1: Frecuentista
config_freq = {
    'method': 'Frequentist',
    'alpha': 0.01,  # 99% confianza
    'power': 0.90,
    'min_duration_days': 28  # 4 semanas
}

# Opción 2: Bayesiano conservador
config_bayes = {
    'method': 'Bayesian',
    'threshold': 0.99,  # 99% prob
    'loss_threshold': 0.001,  # 0.1% máximo
    'min_days': 28
}
```

## Análisis de Trade-offs

### Velocidad vs. Rigor

```
Alta Velocidad, Menor Rigor
    │
    ├─ SOAT agresivo (α=0.10)
    │
Media Velocidad, Buen Rigor
    │
    ├─ SOAT balanceado (α=0.05)
    ├─ Bayesiano (threshold=0.95)
    │
Baja Velocidad, Máximo Rigor
    │
    ├─ Frecuentista conservador (α=0.01)
    └─ Bayesiano muy conservador (threshold=0.99)
```

### Complejidad vs. Beneficio

```
                   Alto Beneficio
                        │
        Bayesiano ──────┼──── SOAT
            │           │      │
            │           │      │
Baja ───────┼───────────┼──────┼───── Alta
Complejidad │           │      │   Complejidad
            │           │      │
            │           │      │
    Frecuentista ───────┼──────
                        │
                   Bajo Beneficio
```

**Interpretación:**
- **SOAT**: Mejor balance (alto beneficio, complejidad media)
- **Frecuentista**: Simple pero beneficio limitado
- **Bayesiano**: Alto beneficio si tienes priors, pero más complejo

## Recomendaciones por Volumen de Tráfico

### Muy Bajo Tráfico (<500 usuarios/día)

**Desafío:** Experimentos muy largos

**Mejores opciones:**
1. **Bayesiano con prior informativo** - Aprovecha conocimiento histórico
2. **Frecuentista** - Si no tienes priors

**Configuración recomendada:**
```python
# Bayesiano
config = {
    'method': 'Bayesian',
    'prior': 'informative',  # Crítico con poco tráfico
    'threshold': 0.90,  # Más flexible que 0.95
    'min_conversions': 30
}
```

**Consideraciones:**
- Enfócate en MDE grandes (15-25%)
- Considera métricas más sensibles
- Usa proxies tempranos del funnel

### Tráfico Bajo (500-5K usuarios/día)

**Mejores opciones:**
1. **SOAT** - Balanceado
2. **Bayesiano** - Si tienes priors

**Configuración recomendada:**
```python
config = {
    'method': 'SOAT',
    'alpha': 0.05,
    'min_days': 14,  # Más largo por poco tráfico
    'min_sample_size': 1000
}
```

### Tráfico Medio (5K-50K usuarios/día)

**Sweet spot para todos los métodos**

**Mejores opciones:**
1. **SOAT** - Primera opción
2. **Bayesiano** - Si tienes expertise
3. **Frecuentista** - Si necesitas simplicidad

**Configuración recomendada:**
```python
config = {
    'method': 'SOAT',
    'alpha': 0.05,
    'min_days': 7,
    'min_sample_size': 1000,
    'monitoring': 'daily'
}
```

### Tráfico Alto (50K-500K usuarios/día)

**Beneficio máximo de SOAT**

**Mejores opciones:**
1. **SOAT** - Máxima velocidad
2. **Bayesiano** - Para casos especiales

**Configuración recomendada:**
```python
config = {
    'method': 'SOAT',
    'alpha': 0.05,
    'min_days': 7,
    'min_sample_size': 2000,
    'monitoring': 'daily',
    'early_stop': True  # Puede terminar en 2-3 días con efectos grandes
}
```

### Tráfico Muy Alto (>500K usuarios/día)

**Decisiones en horas/días**

**Mejores opciones:**
1. **SOAT** - Única opción lógica
2. **Bayesiano** - Solo para casos muy específicos

**Configuración recomendada:**
```python
config = {
    'method': 'SOAT',
    'alpha': 0.05,
    'min_days': 3,  # Puede ser más corto
    'min_sample_size': 5000,
    'monitoring': 'hourly',  # Monitoreo más frecuente
    'auto_stop': True
}
```

## Combinaciones de Métodos

### Enfoque Híbrido: Pre-análisis con SOAT, Validación con Frecuentista

**Caso de uso:** Experimentos críticos donde necesitas velocidad Y máxima confianza

**Estrategia:**
```python
# Fase 1: SOAT para señal temprana
soat_result = run_soat_analysis()

if soat_result['decision'] == 'SHIP_TREATMENT':
    # Fase 2: Confirmar con frecuentista
    if current_day >= frequentist_min_days:
        freq_result = run_frequentist_test()

        if freq_result['is_significant']:
            decision = 'SHIP'
        else:
            decision = 'CONTINUE'  # SOAT dijo sí, frecuentista no
```

**Beneficio:**
- Señal temprana de SOAT
- Confirmación rigurosa de frecuentista
- Minimiza falsos positivos

### Enfoque Portfolio: Diferentes Métodos por Tier de Experimento

**Estrategia multi-tier:**

```python
experiment_tiers = {
    'tier_1_low_risk': {
        'method': 'SOAT',
        'alpha': 0.10,
        'min_days': 3,
        'description': 'UI tweaks, copy changes'
    },
    'tier_2_medium_risk': {
        'method': 'SOAT',
        'alpha': 0.05,
        'min_days': 7,
        'description': 'New features, flow changes'
    },
    'tier_3_high_risk': {
        'method': 'Frequentist',
        'alpha': 0.01,
        'min_days': 28,
        'description': 'Core product changes, pricing'
    }
}
```

## Calculadora de Recomendación

```python
def recommend_methodology(
    daily_traffic: int,
    experiments_per_month: int,
    has_historical_data: bool,
    team_sophistication: str,  # 'basic', 'intermediate', 'advanced'
    risk_tolerance: str,  # 'low', 'medium', 'high'
    regulatory_constraints: bool
) -> dict:
    """
    Recomienda metodología basada en contexto.

    Returns:
        dict con metodología recomendada y configuración
    """

    # Hard constraints
    if regulatory_constraints:
        return {
            'method': 'Frequentist',
            'reason': 'Regulatory requirements',
            'confidence': 'high'
        }

    if team_sophistication == 'basic':
        return {
            'method': 'Frequentist',
            'reason': 'Team capability',
            'confidence': 'medium',
            'alternative': 'Consider SOAT platform (Statsig)'
        }

    # Velocity-driven decision
    if experiments_per_month > 10 and daily_traffic > 5000:
        if team_sophistication == 'advanced':
            return {
                'method': 'SOAT',
                'reason': 'High velocity + capability',
                'confidence': 'high',
                'config': {
                    'alpha': 0.05 if risk_tolerance == 'low' else 0.10,
                    'min_days': 7 if risk_tolerance == 'low' else 3
                }
            }

    # Data-driven decision
    if has_historical_data and team_sophistication == 'advanced':
        return {
            'method': 'Bayesian',
            'reason': 'Historical data + expertise',
            'confidence': 'high',
            'config': {
                'prior': 'informative',
                'threshold': 0.99 if risk_tolerance == 'low' else 0.95
            }
        }

    # Low traffic
    if daily_traffic < 1000:
        if has_historical_data:
            return {
                'method': 'Bayesian',
                'reason': 'Low traffic + historical data',
                'confidence': 'medium'
            }
        else:
            return {
                'method': 'Frequentist',
                'reason': 'Low traffic + no priors',
                'confidence': 'medium',
                'warning': 'Consider increasing MDE or accepting longer duration'
            }

    # Default: SOAT
    return {
        'method': 'SOAT',
        'reason': 'Best general-purpose choice',
        'confidence': 'high',
        'config': {
            'alpha': 0.05,
            'min_days': 7,
            'min_sample_size': 1000
        }
    }


# Ejemplo de uso
recommendation = recommend_methodology(
    daily_traffic=25000,
    experiments_per_month=15,
    has_historical_data=True,
    team_sophistication='advanced',
    risk_tolerance='medium',
    regulatory_constraints=False
)

print(f"Recommended method: {recommendation['method']}")
print(f"Reason: {recommendation['reason']}")
print(f"Confidence: {recommendation['confidence']}")
if 'config' in recommendation:
    print(f"Config: {recommendation['config']}")
```

## Checklist Final de Decisión

### Para Elegir Frecuentista

- [ ] Regulaciones lo requieren
- [ ] Equipo no puede implementar métodos avanzados
- [ ] Stakeholders solo entienden p-valores
- [ ] Velocidad no es crítica
- [ ] Simplicidad > eficiencia

### Para Elegir Bayesiano

- [ ] Tienes datos históricos valiosos
- [ ] Equipo tiene expertise estadístico
- [ ] Quieres incorporar conocimiento previo
- [ ] Stakeholders entienden probabilidades
- [ ] Necesitas velocidad > frecuentista pero < SOAT

### Para Elegir SOAT

- [ ] Velocidad es importante
- [ ] Corres muchos experimentos (>5/mes)
- [ ] Tienes tráfico suficiente (>1K/día)
- [ ] Quieres monitorear continuamente
- [ ] No tienes restricciones regulatorias

## Resumen: Regla de Oro

**Para el 80% de casos en tech/e-commerce/SaaS:**

```python
if traffic > 5000 and no_regulatory_constraints:
    USE_SOAT
else:
    USE_FREQUENTIST  # O Bayesiano si tienes expertise
```

**SOAT es el mejor balance de velocidad, rigor y practicidad para la mayoría de casos modernos.**

## Siguiente: [Ejemplos Prácticos](./06-ejemplos-practicos.md)
