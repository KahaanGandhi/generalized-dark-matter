import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# MD: matter-dominated era
# RD: radiation-dominated era

# Constants
G = 1.0
epsilon = 1e-10

# RD Era parameters
params_rd = {
    'era_name': 'Radiation-Dominated',
    'h': 0.7,
    'H0': 0.0003335 * 0.7,
    'k': 0.00528,
    'cs0': 1e-5,
    'delta_cs': 1e-5,
    'ai': 0.0001489,
    'af': 0.00020
}

# MD Era parameters
params_md = {
    'era_name': 'Matter-Dominated',
    'H0': params_rd['H0'],
    'k': params_rd['k'],
    'cs0': 0.0,
    'ai': 0.01489,
    'af': 0.020,
    'cs2want': 0.0001,
}

# Compute delta_cs for MD era
da_md = np.log(params_md['af'] / params_md['ai'])
cs_md = np.sqrt(params_md['cs2want'] / da_md)
params_md['delta_cs'] = cs_md

def scale_factor(t, H0):
    return ((3 * H0 * t) / 2) ** (2 / 3)

def cs2(a, cs0, delta_cs, ai, af):
    if ai <= a <= af:
        return cs0 + delta_cs
    else:
        return cs0

def cs2_t(t, a_func, cs2_func):
    a_val = a_func(t)
    return cs2_func(a_val)

def get_ode_system_rd(a_func, cs2_t_func, k):
    def ode_system(t, y):
        delta, delta_prime = y
        a_t = a_func(t)
        cs2_value = cs2_t_func(t)
        rhs_coeff = (1 / t**2) * (1 - (9 * cs2_value * k**2 * t**2) / (16 * a_t**2))
        delta_double_prime = - (1 / t) * delta_prime + rhs_coeff * delta
        return [delta_prime, delta_double_prime]
    return ode_system

def get_ode_system_unperturbed_rd(a_func, cs0, k):
    def ode_system(t, y):
        delta, delta_prime = y
        a_t = a_func(t)
        rhs_coeff = (1 / t**2) * (1 - (9 * cs0 * k**2 * t**2) / (16 * a_t**2))
        delta_double_prime = - (1 / t) * delta_prime + rhs_coeff * delta
        return [delta_prime, delta_double_prime]
    return ode_system

def get_rhs_coefficient_md(t, a_func, cs2_t_func, k, H0):
    a_t = a_func(t)
    cs2_value = cs2_t_func(t)
    term = 1 - (2 * a_t * cs2_value * k**2) / (3 * H0**2)
    return (2 / (3 * t**2)) * term

def get_ode_system_md(a_func, cs2_t_func, k, H0):
    def ode_system(t, y):
        delta, delta_prime = y
        rhs = get_rhs_coefficient_md(t, a_func, cs2_t_func, k, H0)
        delta_double_prime = - (4 / (3 * t)) * delta_prime + rhs * delta
        return [delta_prime, delta_double_prime]
    return ode_system

def get_ode_system_unperturbed_md():
    def ode_system(t, y):
        delta, delta_prime = y
        rhs = 2 / (3 * t**2)
        delta_double_prime = - (4 / (3 * t)) * delta_prime + rhs * delta
        return [delta_prime, delta_double_prime]
    return ode_system

# Solve ODE numerically
def solve_era(ode_system_unperturbed, ode_system_perturbed, a_func, t_min, t_max, delta_0, delta_prime_0, t_eval):
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
        ode_system_perturbed,
        [t_min, t_max],
        [delta_0, delta_prime_0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )

    # Compute fractional difference
    delta_perturbed = sol_perturbed.y[0]
    delta_unperturbed = sol_unperturbed.y[0]
    t_values = sol_unperturbed.t
    a_values = a_func(t_values)
    fractional_difference = np.abs(delta_perturbed - delta_unperturbed) / (np.abs(delta_unperturbed) + epsilon)

    return a_values, fractional_difference

def plot_fractional_difference(a_values_rd, fractional_difference_rd, a_values_md, fractional_difference_md, params_rd, params_md):
    lw = 1.5
    frac_diff_color = '#7DF9FF'   # Blue
    start_kick_color = '#FF69B4'  # Pink
    end_kick_color = '#FFFF00'    # Yellow
    equation_rd = r'$\ddot{\delta}(\vec{k}) + \frac{1}{t} \dot{\delta}(\vec{k}) - \frac{1}{t^2} \left(1 - \frac{3 c_s^2 k^2}{32 G \pi \bar{\rho} a^2} \right) \delta(\vec{k}) = 0$'
    equation_md = r'$\ddot{\delta}(\vec{k}) + \frac{4}{3t} \dot{\delta}(\vec{k}) - \frac{2}{3t^2} \left(1 - \frac{2 a c_s^2 k^2}{3 H_0^2} \right) \delta(\vec{k}) = 0$'

    fig, axs = plt.subplots(1, 2, figsize=(24, 8), facecolor='black')
    plt.subplots_adjust(wspace=0.25, top=0.82)

    # RD Plot
    axs[0].semilogx(a_values_rd, fractional_difference_rd, color=frac_diff_color, label='Fractional Difference', linewidth=lw)
    axs[0].axvline(x=params_rd['ai'], color=start_kick_color, linestyle='--', label='Start of Kick', linewidth=lw)
    axs[0].axvline(x=params_rd['af'], color=end_kick_color, linestyle='--', label='End of Kick', linewidth=lw)
    axs[0].set_xlabel('Scale Factor $a$', color='white', fontsize=14)
    axs[0].set_ylabel('Fractional Difference', color='white', fontsize=14)
    axs[0].set_title('Radiation-Dominated', color='white', fontsize=16)
    axs[0].grid(True, which="both", ls="--", color='gray')
    axs[0].set_facecolor('black')
    axs[0].tick_params(axis='x', colors='white', labelsize=12)
    axs[0].tick_params(axis='y', colors='white', labelsize=12)
    for spine in axs[0].spines.values():
        spine.set_color('white')
    axs[0].text(0.5, 1.08, equation_rd, transform=axs[0].transAxes, fontsize=14, color='white', ha='center')

    # MD Plot
    axs[1].semilogx(a_values_md, fractional_difference_md, color=frac_diff_color, linewidth=lw)
    axs[1].axvline(x=params_md['ai'], color=start_kick_color, linestyle='--', label='Start of Kick', linewidth=lw)
    axs[1].axvline(x=params_md['af'], color=end_kick_color, linestyle='--', label='End of Kick', linewidth=lw)
    axs[1].set_xlabel('Scale Factor $a$', color='white', fontsize=14)
    axs[1].set_ylabel('Fractional Difference', color='white', fontsize=14)
    axs[1].set_title('Matter-Dominated', color='white', fontsize=16)
    axs[1].grid(True, which="both", ls="--", color='gray')
    axs[1].set_facecolor('black')
    axs[1].tick_params(axis='x', colors='white', labelsize=12)
    axs[1].tick_params(axis='y', colors='white', labelsize=12)
    for spine in axs[1].spines.values():
        spine.set_color('white')
    axs[1].text(0.5, 1.08, equation_md, transform=axs[1].transAxes, fontsize=14, color='white', ha='center')

    fig.suptitle('Fractional Difference of $\delta(a)$', fontsize=20, color='white')
    lines_labels = [axs[0].get_legend_handles_labels(), axs[1].get_legend_handles_labels()]
    lines = [line for lines, labels in lines_labels for line in lines]
    labels = [label for lines, labels in lines_labels for label in labels]

    from collections import OrderedDict
    lines_labels_dict = OrderedDict()
    for line, label in zip(lines, labels):
        if label not in lines_labels_dict:
            lines_labels_dict[label] = line

    fig.legend(lines_labels_dict.values(), lines_labels_dict.keys(), loc='lower center', ncol=3, fontsize=12,
               facecolor='black', edgecolor='white', framealpha=0.8, labelcolor='white')
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('fractional-diff-GDM', dpi=600, facecolor='black')
    plt.show()

# Initial conditions and time range
t_min = 1e-6
t_max = 1e6
delta_0 = 1e-5
delta_prime_0 = 0.0
t_eval = np.logspace(np.log10(t_min), np.log10(t_max), 10000)

# Radiation-Dominated Era computations
H0_rd = params_rd['H0']
k_rd = params_rd['k']
cs0_rd = params_rd['cs0']
delta_cs_rd = params_rd['delta_cs']
ai_rd = params_rd['ai']
af_rd = params_rd['af']

def a_rd(t):
    return scale_factor(t, H0_rd)

def cs2_rd(a):
    return cs2(a, cs0_rd, delta_cs_rd, ai_rd, af_rd)

def cs2_rd_t(t):
    return cs2_rd(a_rd(t))

ode_system_unperturbed_rd = get_ode_system_unperturbed_rd(a_rd, cs0_rd, k_rd)
ode_system_perturbed_rd = get_ode_system_rd(a_rd, cs2_rd_t, k_rd)

a_values_rd, fractional_difference_rd = solve_era(
    ode_system_unperturbed_rd,
    ode_system_perturbed_rd,
    a_rd,
    t_min,
    t_max,
    delta_0,
    delta_prime_0,
    t_eval
)

# MD computations
H0_md = params_md['H0']
k_md = params_md['k']
cs0_md = params_md['cs0']
delta_cs_md = params_md['delta_cs']
ai_md = params_md['ai']
af_md = params_md['af']

def a_md(t):
    return scale_factor(t, H0_md)

def cs2_md(a):
    if ai_md <= a <= af_md:
        return cs0_md + delta_cs_md
    else:
        return cs0_md

def cs2_md_t(t):
    return cs2_md(a_md(t))

ode_system_unperturbed_md = get_ode_system_unperturbed_md()
ode_system_perturbed_md = get_ode_system_md(a_md, cs2_md_t, k_md, H0_md)

a_values_md, fractional_difference_md = solve_era(
    ode_system_unperturbed_md,
    ode_system_perturbed_md,
    a_md,
    t_min,
    t_max,
    delta_0,
    delta_prime_0,
    t_eval
)

# Plotting
plot_fractional_difference(a_values_rd, fractional_difference_rd, a_values_md, fractional_difference_md, params_rd, params_md)
