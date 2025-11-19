# Modifications pour heatmaps Monte Carlo (S, K)

## 📋 Résumé des changements

Les heatmaps Monte Carlo ont été modifiées pour afficher les prix en fonction de **S (spot price)** et **K (strike)** pour une **maturité T fixe**, au lieu de K et T pour un S0 fixe.

## ✨ Changements implémentés

### 1. Sidebar - Nouveau paramètre
- **Ajout**: `T_mc` (Maturité T pour heatmaps MC)
  - Valeur par défaut: 1.0 an
  - Range: 0.1 à 5.0 ans
  - Description: "Maturité fixe pour les heatmaps Monte Carlo (S vs K)"

### 2. Paramètres de grille Monte Carlo
**Avant** (colonne Monte Carlo):
- `n_paths`: Nombre de trajectoires
- Caption: "Pas de temps = T × 252"

**Après** (colonne Monte Carlo):
- `n_paths`: Nombre de trajectoires
- `span_S`: Span S (spot) ± (par défaut: 20.0)
- `span_K`: Span K (strike) ± (par défaut: 20.0)
- `n_points_mc`: Points grille MC (par défaut: 21)

### 3. Grilles de calcul
**Avant**:
```python
K_grid = np.arange(S0_ref - span, S0_ref + span + step_strike, step_strike)
T_grid = np.linspace(0.1, years_ahead, n_maturities)
```

**Après**:
```python
S_grid_mc = np.linspace(S0_ref - span_S, S0_ref + span_S, n_points_mc)
K_grid_mc = np.linspace(S0_ref - span_K, S0_ref + span_K, n_points_mc)
T_mc = <valeur fixe depuis sidebar>
```

### 4. Boucle de pricing Monte Carlo
**Avant** (boucle sur T et K):
```python
for i, T_val in enumerate(T_grid):
    for j, K_val in enumerate(K_grid):
        price = heston_mc_pricer(S0_ref, K_val, T_val, ...)
```

**Après** (boucle sur S et K):
```python
for i, S_val in enumerate(S_grid_mc):
    for j, K_val in enumerate(K_grid_mc):
        price = heston_mc_pricer(S_val, K_val, T_mc, ...)
```

### 5. Matrices de prix
**Avant**: `prices_mc[T_index, K_index]`
**Après**: `prices_mc[S_index, K_index]`

### 6. Visualisations Heatmap
**Avant**:
- Axe X: Strike K
- Axe Y: Maturité T
- Titre: "Heatmap Prix ... (MC)"

**Après**:
- Axe X: Strike K
- Axe Y: Spot S
- Titre: "Heatmap Prix ... (MC, T={T_mc:.2f})"

### 7. Surfaces IV 3D Monte Carlo
**Avant**:
```python
KK_mc, TT_mc = np.meshgrid(K_grid, T_grid)
# Surface avec x=K, y=T
```

**Après**:
```python
KK_mc, SS_mc = np.meshgrid(K_grid_mc, S_grid_mc)
# Surface avec x=K, y=S
```

### 8. Section de comparaison
**Avant**:
- Comparaison à une maturité T fixe (milieu de T_grid)
- Variation sur les strikes K

**Après**:
- Comparaison à un spot S fixe (milieu de S_grid_mc)
- Variation sur les strikes K
- Calcul analytique pour ce spot et T_mc spécifiques

## 🎯 Résultat

Les heatmaps Monte Carlo montrent maintenant:
- **Comment les prix varient** quand on change le spot S (axe Y) et le strike K (axe X)
- **Pour une maturité T fixe** choisie dans la sidebar
- **Conformément** au script `heston_mc_heatmap_to_iv.py`

## 📊 Exemple d'usage

1. Dans la sidebar, choisir `T_mc = 1.0` an
2. Configurer `span_S = 20`, `span_K = 20`, `n_points_mc = 21`
3. Lancer l'analyse
4. Résultat: Heatmap 21×21 montrant les prix pour:
   - Spots: 80 à 120 (si S0 = 100)
   - Strikes: 80 à 120
   - Maturité: 1.0 an (fixe)

## ✅ Tests

Tous les tests ont réussi:
- ✓ Grilles S et K créées correctement
- ✓ Matrices de prix ont la bonne dimension (S×K)
- ✓ Boucle de pricing fonctionne sur (S, K)
- ✓ Meshgrid pour visualisation correct
- ✓ Syntaxe Python valide
- ✓ Aucune erreur de compilation
