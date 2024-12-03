import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from sympy import symbols, Function, Eq, solve

# Matter dominated, numerically, to check ODE method
# Check beat frequency hypothesis
# Vary K, see if lines up with previous limits

# Bartelman: photon density perturbation
# Want to write solution to photon potential with Phi as this
# Green's function

# We have delta --> delta * Poisson = Phi

#------------------------------------
# Manipulate Symbolic ODE
#------------------------------------

t, x, k_sym, H0_sym, cs2_sym, G_sym, pi_sym, a_sym, rho0_sym = symbols('t x k H0 cs2 G pi a rho0', positive=True)
delta = Function('delta')

# Original ODE:
# δ''(t) + (1 / t) δ'(t) = [1 / t²] * [1 - (3 c_s² k²) / (32 π G ρ₀ a²)] δ(t)

# Substitutions:
# t = x / k
# ρ₀ = (3 H₀²) / (8 π G a³)
# x = (2 k a^(3/2)) / (3 H₀)

# Derive a(t) and ρ₀(t) to simplify RHS coefficent
expr = Eq(k_sym * t, (2 * k_sym * a_sym**(3/2)) / (3 * H0_sym))
a_t_expr = solve(expr, a_sym)[0]
a_t_expr = ((3 * H0_sym * t) / 2)**(2 / 3)
rho0_expr = (3 * H0_sym**2) / (8 * pi_sym * G_sym * a_t_expr**3)
rho0_expr = 1 / (6 * pi_sym * G_sym * t**2)
rhs_coefficient_expr = (1 / t**2) * (1 - (3 * cs2_sym * k_sym**2) / (32 * pi_sym * G_sym * rho0_expr * a_t_expr**2))
rhs_coefficient_expr = (1 / t**2) * (1 - (9 * cs2_sym * k_sym**2 * t**2) / (16 * a_t_expr**2))

#------------------------------------
# Implement ODE System
#------------------------------------

# Physical constants
h = 0.7
H0 = 0.0003335 * h          # Hubble constant
k = 0.00528                 # Comoving wavenumber
G = 1.0
pi = np.pi
cs0 = 1e-5                  # Original sound speed squared
delta_cs = 1e-5             # Change during the "kick"

# "Kick" parameters, units of scale factor a
ai = 0.0001489
af = 0.00020

def a(t):
    return ((3 * H0 * t) / 2)**(2 / 3)

def cs2(a):
    if ai <= a <= af:
        return cs0 + delta_cs
    else:
        return cs0

def cs2_t(t):
    a_val = a(t)
    return cs2(a_val)

# ODE system for perturbed case
def ode_system(t, y):
    delta, delta_prime = y
    a_t = a(t)
    cs2_value = cs2_t(t)
    rhs_coefficient = (1 / t**2) * (1 - (9 * cs2_value * k**2 * t**2) / (16 * a_t**2))
    delta_double_prime = - (1 / t) * delta_prime + rhs_coefficient * delta
    return [delta_prime, delta_double_prime]

# ODE system for unperturbed case
def ode_system_unperturbed(t, y):
    delta, delta_prime = y
    a_t = a(t)
    cs2_value = cs0
    rhs_coefficient = (1 / t**2) * (1 - (9 * cs2_value * k**2 * t**2) / (16 * a_t**2))
    delta_double_prime = - (1 / t) * delta_prime + rhs_coefficient * delta
    return [delta_prime, delta_double_prime]

#------------------------------------
# Solve ODE System Numerically
#------------------------------------

# Initial conditions and time range
t_min = 1e-6
t_max = 1e6
delta_0 = 1e-5
delta_prime_0 = 0.0
t_eval = np.logspace(np.log10(t_min), np.log10(t_max), 10000)

sol_unperturbed = solve_ivp(
    ode_system_unperturbed,
    [t_min, t_max],
    [delta_0, delta_prime_0],
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

sol_perturbed = solve_ivp(
    ode_system,
    [t_min, t_max],
    [delta_0, delta_prime_0],
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

delta_perturbed = sol_perturbed.y[0]
delta_unperturbed = sol_unperturbed.y[0]
t_values = sol_unperturbed.t
a_values = a(t_values)
epsilon = 1e-10
fractional_difference = np.abs(delta_perturbed - delta_unperturbed) / (np.abs(delta_unperturbed) + epsilon)

#------------------------------------
# Plot Fractional Difference
#------------------------------------

lw = 1.5
fractional_diff_color = '#7DF9FF'  # Blue
start_kick_color = '#FF69B4'       # Pink
end_kick_color = '#FFFF00'         # Yellow
equation = r'$\ddot{\delta}(\vec{k}) + \frac{1}{t} \dot{\delta}(\vec{k}) - \frac{1}{t^2} \left(1 - \frac{3 c_s^2 k^2}{32 G \pi \bar{\rho} a^2} \right) \delta(\vec{k}) = 0$'

plt.figure(figsize=(12, 8), facecolor='black')
plt.semilogx(a_values, fractional_difference, color=fractional_diff_color, label='Fractional Difference', linewidth=lw)
plt.suptitle(equation, color='white', fontsize=18, y=0.97)
plt.xlabel('Scale Factor $a$', color='white', fontsize=14)
plt.ylabel('Fractional Difference', color='white', fontsize=14)
plt.title('Radiation-Dominated Fractional Difference of $\delta(a)$', color='white', fontsize=16)
ax = plt.gca()
ax.set_facecolor('black')
ax.tick_params(axis='x', colors='white', labelsize=12)
ax.tick_params(axis='y', colors='white', labelsize=12)
for spine in ax.spines.values():
    spine.set_color('white')
plt.axvline(x=ai, color=start_kick_color, linestyle='--', label='Start of Kick', linewidth=lw)
plt.axvline(x=af, color=end_kick_color, linestyle='--', label='End of Kick', linewidth=lw)
plt.grid(True, which="both", ls="--", color='gray')
legend = plt.legend(facecolor='black', framealpha=0.8, fontsize=12)
for text in legend.get_texts():
    text.set_color('white')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fractional-diff-RD.png', dpi=600, facecolor='black')
plt.show()