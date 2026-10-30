# Sistema experto con factores de certeza

**Pregunta:** ¿Cómo puede un sistema de reglas tomar una decisión con datos inciertos y decir qué tan seguro está?

**Contexto:** ejercicio de la Sesión 6 de Sistemas Expertos (IUSH), con casos ficticios. El sistema decide qué hace un estudiante un viernes en la noche: `quedarse`, `salir` o `casa_de_amigo`.

## Cómo funciona

- **Base de conocimiento:** 7 reglas SI → ENTONCES, cada una con prioridad y factor de certeza (CF).
- **Base de hechos:** los datos de cada caso, con su valor, CF y origen.
- **Motor:** encadenamiento hacia adelante. La prioridad decide el turno, y una regla solo dispara si el CF más bajo de su SI supera +0.20.

| Cuenta | Fórmula |
|---|---|
| Una regla | aporte = CF de la regla × CF más bajo del SI |
| Dos a favor | CF = CF1 + CF2 × (1 − CF1) |
| A favor y en contra | CF = (CF1 + CF2) / (1 − min(\|CF1\|, \|CF2\|)) |

## Resultados

| Caso | Reglas | Recomendación |
|---|---|---|
| Valentina | R4 | `salir` **+0.54** |
| Camilo | R1, R2 y R3 | `quedarse` **+0.90** |
| Mateo | R3, R5 y R7 | `casa_de_amigo` **+0.75** |
| Lucía | R1 y R4 | `quedarse` **+0.80** |

## Hallazgos

- La prioridad cambia el orden, pero no el resultado: el caso de Mateo da +0.75 en cualquier orden.
- Cambiar un dato sin tocar las reglas cambia la certeza: si Valentina confirma con +1.00, `salir` sube de +0.54 a +0.90.
- Una regla puede disparar sin ganar: su conclusión queda como alternativa en la tabla.
- Las reglas solo leen hechos iniciales, así que ninguna conclusión alimenta a otra regla.

## Stack

Python (sin librerías externas)

## Cómo correrlo

Abra [`expert_system.ipynb`](expert_system.ipynb) y corra las celdas de arriba hacia abajo.
