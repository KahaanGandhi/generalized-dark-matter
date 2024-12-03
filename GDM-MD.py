import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#------------------------------------
# Implement ODE Numerically
#------------------------------------

# Physical constants
h = 0.7
H0 = 0.0003335 * h       # Hubble constant
k = 0.00528              # Comoving wavenumber
G = 1.0

# Kick parameters
ai = 0.01489             # Start of the "kick"
af = 0.020               # End of the "kick"
cs0 = 0.0                # Initial sound speed squared
cs2want = 0.0001         # Desired value for cs^2 * Δa
da = np.log(af / ai)
cs = np.sqrt(cs2want / da)  # Compute cs from cs2want and Δa
Delta_cs = cs            # Change during the kick

def a(t):
    return ((3 * H0 * t) / 2) ** (2 / 3)

def cs2_t(t):
    a_value = a(t)
    if ai <= a_value <= af:
        return cs0 + Delta_cs
    else:
        return cs0  # cs0 = 0

def rhs_coefficient(t):
    a_t = a(t)
    cs2_value = cs2_t(t)
    numerator = 2
    denominator = 3 * t**2
    term = 1 - (2 * a_t * cs2_value * k**2) / (3 * H0**2)
    coefficient = (numerator / denominator) * term
    return coefficient

def rhs_coefficient_unperturbed(t):
    # Since cs0 = 0, simplifies to 2 / (3 t^2)
    coefficient = 2 / (3 * t**2)
    return coefficient

# ODE system for perturbed case
def ode_system(t, y):
    delta, delta_prime = y
    rhs = rhs_coefficient(t)
    delta_double_prime = - (4 / (3 * t)) * delta_prime + rhs * delta
    return [delta_prime, delta_double_prime]

# ODE system for unperturbed case
def ode_system_unperturbed(t, y):
    delta, delta_prime = y
    rhs = rhs_coefficient_unperturbed(t)
    delta_double_prime = - (4 / (3 * t)) * delta_prime + rhs * delta
    return [delta_prime, delta_double_prime]

# Time range and initial conditions
t_min = 1e-6
t_max = 1e6

delta_0 = 1e-5
delta_prime_0 = 0.0

t_eval = np.logspace(np.log10(t_min), np.log10(t_max), 1000)

# Solve ODE for unperturbed case
sol_unperturbed = solve_ivp(
    ode_system_unperturbed,
    [t_min, t_max],
    [delta_0, delta_prime_0],
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

delta_unperturbed = sol_unperturbed.y[0]
t_values = sol_unperturbed.t
a_values = a(t_values)

# Solve ODE for perturbed case
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
epsilon = 1e-10
fractional_difference = np.abs(delta_perturbed - delta_unperturbed) / (np.abs(delta_unperturbed) + epsilon)

#------------------------------------
# Plot Fractional Difference
#------------------------------------

lw = 1.5
fractional_diff_color = '#7DF9FF'  # Blue
start_kick_color = '#FF69B4'       # Pink
end_kick_color = '#FFFF00'         # Yellow
equation_md = r'$\ddot{\delta}(\vec{k}) + \frac{4}{3t} \dot{\delta}(\vec{k}) - \frac{2}{3t^2} \left(1 - \frac{c_s^2 k^2}{4 G \pi \bar{\rho} a^2} \right) \delta(\vec{k}) = 0$'

plt.figure(figsize=(12, 8), facecolor='black')
plt.semilogx(a_values, fractional_difference, color=fractional_diff_color, label='Fractional Difference', linewidth=lw)
plt.suptitle(equation_md, color='white', fontsize=18, y=0.97)
plt.xlabel('Scale Factor $a$', color='white', fontsize=14)
plt.ylabel('Fractional Difference', color='white', fontsize=14)
plt.title('Matter-Dominated Fractional Difference of $\delta(a)$', color='white', fontsize=16)
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
plt.savefig('fractional-diff-MD.png', dpi=600, facecolor='black')
plt.show()

