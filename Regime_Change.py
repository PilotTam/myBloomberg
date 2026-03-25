import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: Generate realistic market data with regime structure
# ============================================================

def generate_regime_market(n_days=500, n_assets=10, seed=42):
  np.random.seed(seed)

  # Three regime patterns
  tech_up = np.array([0.5, 0.5, 0.4, 0.3, -0.2, -0.2, -0.1, 0, 0, 0])
  value_up = np.array([-0.2, -0.2, -0.1, 0.1, 0.5, 0.5, 0.4, 0.2, 0.1, 0])
  crisis = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1])

  # Normalize directions
  tech_up = tech_up / np.linalg.norm(tech_up)
  value_up = value_up / np.linalg.norm(value_up)
  crisis = crisis / np.linalg.norm(crisis)

  returns, labels = [], []
  for t in range(n_days):
      if t < 150:
          base = tech_up * np.random.randn() * 0.02
          labels.append('A: Risk-On')
      elif t < 350:
          base = value_up * np.random.randn() * 0.015
          labels.append('B: Rotation')
      else:
          base = crisis * np.random.randn() * 0.035
          labels.append('C: Crisis')
      returns.append(base + np.random.randn(n_assets) * 0.005)

  return np.array(returns), labels

# ============================================================
# STEP 2: Functions
# ============================================================

def get_singular_values(R):
  return np.linalg.svd(R, full_matrices=False)[1]

def explained_variance_ratios(S):
  return (S ** 2) / np.sum(S ** 2)

def project_all_days(R, directions):
  return R @ directions.T

def distances_from_center(coords):
  return np.linalg.norm(coords - np.mean(coords, axis=0), axis=1)

def angle_between_vectors(v1, v2):
  n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
  if n1 < 1e-10 or n2 < 1e-10: return 0.0
  return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))

def reconstruction_quality(original, reconstructed):
  return np.sum(reconstructed ** 2) / np.sum(original ** 2)

# ============================================================
# STEP 3: Run the full analysis
# ============================================================

R, labels = generate_regime_market()
U, S, Vt = np.linalg.svd(R, full_matrices=False)
ratios = explained_variance_ratios(S)
coords = project_all_days(R, Vt[:2])
distances = distances_from_center(coords)

# Compute angles between consecutive days
angles = [angle_between_vectors(coords[t-1], coords[t]) for t in range(1, len(coords))]
angles = np.array(angles)

# ============================================================
# RESULTS TABLE 1: Variance Explained
# ============================================================

print("=" * 60)
print("TABLE 1: VARIANCE EXPLAINED BY EACH DIRECTION")
print("=" * 60)
print(f"{'Direction':<12} {'Variance %':<12} {'Cumulative %':<12}")
print("-" * 36)
cumulative = 0
for i in range(5):
  cumulative += ratios[i] * 100
  print(f"PC {i+1:<9} {ratios[i]*100:>8.1f}%     {cumulative:>8.1f}%")
print("-" * 36)
print(f"Top 3 directions capture {sum(ratios[:3])*100:.1f}% of all market movement")
print()

# ============================================================
# RESULTS TABLE 2: Reconstruction Quality
# ============================================================

print("=" * 60)
print("TABLE 2: RECONSTRUCTION QUALITY")
print("=" * 60)
print(f"{'Components':<12} {'Quality':<12} {'Interpretation':<30}")
print("-" * 54)
for k in [1, 2, 3, 5, 10]:
  R_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
  q = reconstruction_quality(R, R_approx) * 100
  if q < 50: interp = "Major information loss"
  elif q < 80: interp = "Captures main patterns"
  elif q < 95: interp = "Good approximation"
  else: interp = "Near-perfect reconstruction"
  print(f"{k:<12} {q:>8.1f}%     {interp:<30}")
print()

# ============================================================
# RESULTS TABLE 3: Regime Comparison
# ============================================================

print("=" * 60)
print("TABLE 3: REGIME CHARACTERISTICS")
print("=" * 60)
print(f"{'Regime':<15} {'Days':<8} {'Mean Dist':<12} {'Volatility':<12}")
print("-" * 47)

regime_names = ['A: Risk-On', 'B: Rotation', 'C: Crisis']
for regime in regime_names:
  mask = np.array([l == regime for l in labels])
  d = distances[mask]
  print(f"{regime:<15} {sum(mask):<8} {np.mean(d):>8.4f}     {np.std(d):>8.4f}")
print("-" * 47)
print("Note: Higher distance = more extreme days, Higher volatility = less stable")
print()

# ============================================================
# RESULTS TABLE 4: Transition Detection
# ============================================================

print("=" * 60)
print("TABLE 4: REGIME TRANSITIONS DETECTED")
print("=" * 60)
transitions_90 = np.where(angles > 90)[0] + 1
transitions_120 = np.where(angles > 120)[0] + 1

print(f"Large transitions (>90°):  {len(transitions_90)} days")
print(f"Major transitions (>120°): {len(transitions_120)} days")
print(f"\nKnown regime boundaries: Day 150 (A→B), Day 350 (B→C)")
print(f"\nTop 5 largest angle changes:")
top_5_idx = np.argsort(angles)[-5:][::-1]
for idx in top_5_idx:
  day = idx + 1
  regime_before = labels[idx][:1]
  regime_after = labels[idx+1][:1]
  boundary = "← REGIME BOUNDARY" if day in [150, 350] else ""
  print(f"  Day {day:>3}: {angles[idx]:>6.1f}° ({regime_before}→{regime_after}) {boundary}")
print()

# ============================================================
# VISUALIZATION: 2D Regime Map
# ============================================================

print("=" * 60)
print("VISUALIZATION: Market Days in Reduced Space")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Plot 1: Colored by regime
ax1 = axes[0]
colors = {'A: Risk-On': '#22c55e', 'B: Rotation': '#3b82f6', 'C: Crisis': '#ef4444'}
for regime in regime_names:
  mask = np.array([l == regime for l in labels])
  ax1.scatter(coords[mask, 0], coords[mask, 1], c=colors[regime],
              label=regime, alpha=0.6, s=20)

ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel('Principal Component 1')
ax1.set_ylabel('Principal Component 2')
ax1.set_title('Market Days Colored by Regime')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Colored by time (trajectory)
ax2 = axes[1]
scatter = ax2.scatter(coords[:, 0], coords[:, 1], c=range(len(coords)),
                     cmap='viridis', alpha=0.6, s=20)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

# Mark regime transitions
ax2.scatter(coords[150, 0], coords[150, 1], c='red', s=200, marker='*',
          label='Day 150: A→B', zorder=5)
ax2.scatter(coords[350, 0], coords[350, 1], c='orange', s=200, marker='*',
          label='Day 350: B→C', zorder=5)

ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_title('Market Trajectory Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Day Number')

plt.tight_layout()
plt.show()

print("\n✓ Visualization complete")