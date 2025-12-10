# decision_tree_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle

# --------------------- Parameters (edit these to match your assumptions) ---------------------
opening_cost = 1_000_000  # ₽ initial investment (for reference)

# Probabilities of market demand states (S1)
p_high = 0.30      # High demand
p_mod = 0.50       # Moderate demand
p_low = 0.20       # Low demand

# Local outcome probabilities (conditional on action in each state)
ph_expand = np.array([0.5, 0.3, 0.2])   # High->Expand outcomes probabilities
pm_adjust = np.array([0.4, 0.3, 0.3])   # Moderate->Adjust outcomes probabilities
pl_pivot  = np.array([0.3, 0.7])        # Low->Pivot outcomes probabilities

# Terminal payoffs (net profit after costs, ₽)
T1 = 3_000_000   # High -> Expand -> Big success
T2 =   800_000   # High -> Expand -> Moderate success
T3 =  -600_000   # High -> Expand -> Fail
T4 =   900_000   # High -> Maintain

T5 =   400_000   # Moderate -> Adjust -> Success
T6 =         0   # Moderate -> Adjust -> Partial
T7 =  -500_000   # Moderate -> Adjust -> Fail
T8 =   100_000   # Moderate -> Maintain

T9  = -900_000   # Low -> Close
T10 =  200_000   # Low -> Pivot -> Success
T11 = -800_000   # Low -> Pivot -> Fail

# Hurwicz alpha (optimism)
alpha = 0.6

# ---------------------------------------------------------------------------------------------

# Map of terminals (for Savage)
terminals = {
    "T1_H_Expand_Big": T1,
    "T2_H_Expand_Moderate": T2,
    "T3_H_Expand_Fail": T3,
    "T4_H_Maintain": T4,
    "T5_M_Adjust_Success": T5,
    "T6_M_Adjust_Partial": T6,
    "T7_M_Adjust_Fail": T7,
    "T8_M_Maintain": T8,
    "T9_L_Close": T9,
    "T10_L_Pivot_Success": T10,
    "T11_L_Pivot_Fail": T11
}

# -------------------- Compute local EVs and select local actions by Bayes --------------------
ev_expand = ph_expand @ np.array([T1, T2, T3])
ev_maintain_high = T4

ev_adjust = pm_adjust @ np.array([T5, T6, T7])
ev_maintain_mod = T8

ev_pivot = pl_pivot @ np.array([T10, T11])
ev_close = T9

choice_high = "Expand" if ev_expand >= ev_maintain_high else "Maintain"
choice_mod  = "Adjust" if ev_adjust  >= ev_maintain_mod  else "Maintain"
choice_low  = "Pivot"  if ev_pivot   >= ev_close        else "Close"

ev_open = p_high * (ev_expand if choice_high == "Expand" else ev_maintain_high) + \
          p_mod  * (ev_adjust if choice_mod   == "Adjust" else ev_maintain_mod)  + \
          p_low  * (ev_pivot  if choice_low   == "Pivot"  else ev_close)

ev_not = 0.0

# -------------------- Laplace criterion --------------------
laplace_open = ((ev_expand if choice_high=="Expand" else ev_maintain_high) +
                (ev_adjust if choice_mod=="Adjust" else ev_maintain_mod) +
                (ev_pivot if choice_low=="Pivot" else ev_close)) / 3.0
laplace_not = 0.0

# -------------------- Wald (maximin) --------------------
open_terminals = []
if choice_high == "Expand":
    open_terminals += [T1, T2, T3]
else:
    open_terminals += [T4]
if choice_mod == "Adjust":
    open_terminals += [T5, T6, T7]
else:
    open_terminals += [T8]
if choice_low == "Pivot":
    open_terminals += [T10, T11]
else:
    open_terminals += [T9]

wald_open = min(open_terminals)
wald_not = 0.0

# -------------------- Hurwicz --------------------
gurv_open = alpha * max(open_terminals) + (1 - alpha) * min(open_terminals)
gurv_not = 0.0

# -------------------- Savage (minimax regret) --------------------
payoffs_not_open = np.zeros(len(terminals))
payoffs_open = np.array(list(terminals.values()), dtype=float)
best_per_terminal = np.maximum(payoffs_not_open, payoffs_open)
regret_not = best_per_terminal - payoffs_not_open
regret_open = best_per_terminal - payoffs_open
max_regret_not = regret_not.max()
max_regret_open = regret_open.max()

# -------------------- Prepare results --------------------
results = pd.DataFrame({
    "Критерий": ["Байеса (EV)", "Лапласа (среднее)", "Вальда (максимин)", f"Гурвиц (alpha={alpha})", "Сэвиджа (макс. сожаление)"],
    "Не открывать": [ev_not, laplace_not, wald_not, gurv_not, max_regret_not],
    "Открыть": [ev_open, laplace_open, wald_open, gurv_open, max_regret_open]
})

def prefer(row):
    a = row["Не открывать"]
    b = row["Открыть"]
    if row["Критерий"].startswith("Сэвиджа"):
        return "Не открывать" if a < b else ("Открыть" if b < a else "Обе равны")
    elif row["Критерий"].startswith("Вальда"):
        return "Не открывать" if a > b else ("Открыть" if b > a else "Обе равны")
    else:
        return "Не открывать" if a > b else ("Открыть" if b > a else "Обе равны")

results["Рекомендация"] = results.apply(prefer, axis=1)

# -------------------- PRINT step-by-step --------------------
print("\n=== ПАРАМЕТРЫ МОДЕЛИ ===")
print(f"Стоимость открытия (информативно): {opening_cost:,} ₽")
print(f"Вероятности состояний спроса: High={p_high}, Moderate={p_mod}, Low={p_low}\n")

print("=== ЛОКАЛЬНЫЕ ОЖИДАНИЯ (EV) И ВЫБОРЫ В ЛОКАЛЬНЫХ УЗЛАХ ===")
print(f"EV(High, Expand) = {ev_expand:,.2f} ₽")
print(f"EV(High, Maintain) = {ev_maintain_high:,.2f} ₽ -> выбор: {choice_high}")
print(f"EV(Moderate, Adjust) = {ev_adjust:,.2f} ₽")
print(f"EV(Moderate, Maintain) = {ev_maintain_mod:,.2f} ₽ -> выбор: {choice_mod}")
print(f"EV(Low, Pivot) = {ev_pivot:,.2f} ₽")
print(f"EV(Low, Close) = {ev_close:,.2f} ₽ -> выбор: {choice_low}\n")

print("=== ВЕРХНИЙ УРОВЕНЬ (Байеса) ===")
print(f"EV(Open) = {ev_open:,.2f} ₽")
print(f"EV(Not open) = {ev_not:,.2f} ₽\n")

numeric_cols = ["Не открывать", "Открыть"]

print(results.to_string(
    index=False,
    formatters={col: "{:,.0f}".format for col in numeric_cols}
))


print("\n=== СКОЛЬКО КРИТЕРИЕВ ЗА КАЖДУЮ СТРАТЕГИЮ ===")
print(results["Рекомендация"].value_counts().to_string())

# -------------------- Draw the decision tree diagram --------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

root_x, root_y = 0.5, 0.9
ax.add_patch(FancyBboxPatch((root_x-0.08, root_y-0.03), 0.16, 0.06, boxstyle="round,pad=0.02", edgecolor="black", facecolor="#eef"))
ax.text(root_x, root_y, "D1: Открыть чайную?", ha='center', va='center')

# Left: Not open
nx1, ny1 = 0.15, 0.7
ax.add_patch(FancyBboxPatch((nx1-0.08, ny1-0.03), 0.16, 0.06, boxstyle="round,pad=0.02", edgecolor="black", facecolor="#efe"))
ax.text(nx1, ny1, "Не открывать\nPayoff = 0 ₽", ha='center', va='center')
ax.annotate("", xy=(nx1+0.04, ny1+0.02), xytext=(root_x-0.04, root_y-0.02), arrowprops=dict(arrowstyle="-", color="black"))

# Right: Chance node S1
cx, cy = 0.78, 0.72
ax.add_patch(Circle((cx, cy), 0.03, edgecolor="black", facecolor="#fff"))
ax.text(cx, cy, "S1\nСпрос", ha='center', va='center', fontsize=8)
ax.annotate("", xy=(cx-0.02, cy-0.02), xytext=(root_x+0.04, root_y-0.02), arrowprops=dict(arrowstyle="-", color="black"))

# State nodes
hx, hy = 0.60, 0.52
mx, my = 0.78, 0.52
lx, ly = 0.95, 0.52
for (sx, sy, label) in [(hx, hy, f"High\np={p_high}"), (mx, my, f"Moderate\np={p_mod}"), (lx, ly, f"Low\np={p_low}")] :
    ax.add_patch(Circle((sx, sy), 0.025, edgecolor="black", facecolor="#fff"))
    ax.text(sx, sy, label, ha='center', va='center', fontsize=8)
    ax.annotate("", xy=(sx, sy+0.02), xytext=(cx, cy-0.02), arrowprops=dict(arrowstyle="-", color="black"))

# High decision box and terminals
dhx, dhy = 0.46, 0.36
ax.add_patch(FancyBboxPatch((dhx-0.09, dhy-0.03), 0.18, 0.06, boxstyle="round,pad=0.02", edgecolor="black", facecolor="#ffd"))
ax.text(dhx, dhy, f"D2-H\nChoice: {choice_high}", ha='center', va='center', fontsize=8)
ax.annotate("", xy=(dhx+0.02, dhy+0.02), xytext=(hx, hy-0.02), arrowprops=dict(arrowstyle="-", color="black"))

tx1, ty1 = 0.36, 0.18
tx2, ty2 = 0.46, 0.18
tx3, ty3 = 0.56, 0.18
ax.text(tx1, ty1, f"T1\n{T1:,} ₽\n(p={ph_expand[0]})", ha='center', va='center', fontsize=7)
ax.text(tx2, ty2, f"T2\n{T2:,} ₽\n(p={ph_expand[1]})", ha='center', va='center', fontsize=7)
ax.text(tx3, ty3, f"T3\n{T3:,} ₽\n(p={ph_expand[2]})", ha='center', va='center', fontsize=7)
ax.annotate("", xy=(tx1, ty1+0.02), xytext=(dhx-0.02, dhy-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.annotate("", xy=(tx2, ty2+0.02), xytext=(dhx+0.00, dhy-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.annotate("", xy=(tx3, ty3+0.02), xytext=(dhx+0.02, dhy-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.text(0.46, 0.12, f"T4\n{T4:,} ₽", ha='center', va='center', fontsize=7)
ax.annotate("", xy=(0.46, 0.14), xytext=(dhx, dhy-0.02), arrowprops=dict(arrowstyle="-", color="black"))

# Moderate decision and terminals
dmx, dmy = 0.78, 0.36
ax.add_patch(FancyBboxPatch((dmx-0.09, dmy-0.03), 0.18, 0.06, boxstyle="round,pad=0.02", edgecolor="black", facecolor="#ffd"))
ax.text(dmx, dmy, f"D3-M\nChoice: {choice_mod}", ha='center', va='center', fontsize=8)
ax.annotate("", xy=(dmx, dmy+0.02), xytext=(mx, my-0.02), arrowprops=dict(arrowstyle="-", color="black"))

tx4, ty4 = 0.70, 0.18
tx5, ty5 = 0.78, 0.18
tx6, ty6 = 0.86, 0.18
ax.text(tx4, ty4, f"T5\n{T5:,} ₽\n(p={pm_adjust[0]})", ha='center', va='center', fontsize=7)
ax.text(tx5, ty5, f"T6\n{T6:,} ₽\n(p={pm_adjust[1]})", ha='center', va='center', fontsize=7)
ax.text(tx6, ty6, f"T7\n{T7:,} ₽\n(p={pm_adjust[2]})", ha='center', va='center', fontsize=7)
ax.annotate("", xy=(tx4, ty4+0.02), xytext=(dmx-0.02, dmy-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.annotate("", xy=(tx5, ty5+0.02), xytext=(dmx+0.00, dmy-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.annotate("", xy=(tx6, ty6+0.02), xytext=(dmx+0.02, dmy-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.text(0.78, 0.12, f"T8\n{T8:,} ₽", ha='center', va='center', fontsize=7)
ax.annotate("", xy=(0.78, 0.14), xytext=(dmx, dmy-0.02), arrowprops=dict(arrowstyle="-", color="black"))

# Low decision and terminals
dlx, dly = 0.94, 0.36
ax.add_patch(FancyBboxPatch((dlx-0.09, dly-0.03), 0.18, 0.06, boxstyle="round,pad=0.02", edgecolor="black", facecolor="#ffd"))
ax.text(dlx, dly, f"D4-L\nChoice: {choice_low}", ha='center', va='center', fontsize=8)
ax.annotate("", xy=(dlx, dly+0.02), xytext=(lx, ly-0.02), arrowprops=dict(arrowstyle="-", color="black"))

tx7, ty7 = 0.88, 0.18
tx8, ty8 = 0.96, 0.18
ax.text(tx7, ty7, f"T10\n{T10:,} ₽\n(p={pl_pivot[0]})", ha='center', va='center', fontsize=7)
ax.text(tx8, ty8, f"T11\n{T11:,} ₽\n(p={pl_pivot[1]})", ha='center', va='center', fontsize=7)
ax.annotate("", xy=(tx7, ty7+0.02), xytext=(dlx-0.02, dly-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.annotate("", xy=(tx8, ty8+0.02), xytext=(dlx+0.02, dly-0.02), arrowprops=dict(arrowstyle="-", color="black"))
ax.text(0.95, 0.12, f"T9\n{T9:,} ₽", ha='center', va='center', fontsize=7)
ax.annotate("", xy=(0.95, 0.14), xytext=(dlx, dly-0.02), arrowprops=dict(arrowstyle="-", color="black"))

ax.set_title("Дерево решений: Открыть чайную в СПб — схема ветвлений и вероятности", fontsize=12)
plt.tight_layout()
plt.show()

