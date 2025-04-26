# Programmentwurf Algorithmen und Verfahren

## Grundlagen

### Netzwerk
**Satz:**  
Aussagenlogische Formel in DNF kann durch zweischichtiges, vorwärtsgetriebenes Netz realisiert werden

- $n$ Aussagenvariablen $x_1, \dots, x_n$ (Eingangsneuronen)
- Werte $-1$ (falsch) und $1$ (wahr)
- $m$ Monome $z_1, \dots, z_m$ (Neuronen in Zwischenschicht)
- Ein Ausgabeneuron $y$
- Aktivierungsfunktion $sgn(x) = \begin{cases} 1 & x \geq 0 \\ -1 & x < 0 \end{cases}$
- Propagation in Vektorform:
  - $\vec{z} = sgn(w\cdot \vec{x} - \vec{v})$
  - $y = sgn(W \cdot \vec{z} - V)$
  - mit Gewichtsmatrizen: 
    - $w \in \mathbb{R}^{m \times n}$
    - $W \in \mathbb{R}^{1 \times m}$
  - und Schwellwerten:
    - $\vec{v} \in \mathbb{R}^{m}$
    - $V \in \mathbb{R}$
- Lernalgorithmus mit Zielmuster $p$ und Lernrate $\eta$:
  - $\Delta W_{1j} = \eta \cdot (p-y)\cdot z_j$
  - $\Delta w_{jk} = \eta \cdot W_{1j}\cdot (p-y) \cdot x_k$
  - $\Delta V = - \eta \cdot (p-y)$
  - $\Delta v_{j} = - \eta \cdot (p-y) \cdot W_{1j}$



### Arithmetisierung der Formel

- DNF: $F = M_1 \lor M_2 \lor \ldots \lor M_m$
- AV: $x_1, \dots, x_n$
- Monom $M_i = l_{i1} \land l_{i2} \land \ldots \land l_{in}$ mit Literalen $l_{ij} \in \{x_j, \neg x_j, w\}$
- $M_i$ wahr, wenn  
  - $a_{i1}x_1 + a_{i2}x_2 + \ldots + a_{in}x_n = b_i$  
  
    mit Koeffizienten  

  - $a_{ij} = \begin{cases} 1 & l_{ij} = x_j \\ 0 & l_{ij} = w,\ \ b_i = \sum_{k=1}^{n}|a_{ik}| \\ -1 & l_{ij} = \neg x_j \end{cases}$
- Realisierung der Monome durch Neuronen $z_j$ in Zwischenschicht
- $F = M_1 \lor M_2 \lor \ldots \lor M_m$  
    ist wahr, wenn  
    $(z_1+1) + (z_2+1) + \ldots + (z_m+1) \geq 1$

## Aufgabe
- Zufällige **DNF**:
  - Mindestens 5 Monome
  - Mindestens 3 Literale je Monom
  - Insgesamt mindestens 10 AV
1. **Implementierung** mit Fehlerrückübertragung
2. **Dokumentation**
3. **Tests**: Gewichte und Bias setzen, sodass korrekt $\longrightarrow$ Inferenz $\longrightarrow$ Sollten statisch bleiben
4. **Experiment**:
   1. Gewichte leicht verändern $\longrightarrow$ Inferenz $\longrightarrow$ Sollten sich ändern
   2. Gewichte zufällig wählen $\longrightarrow$ Inferenz $\longrightarrow$ Sollten sich ändern  

   - **Dokumentation** und **Analyse** der Ergebnisse
