import numpy as np
import matplotlib.pyplot as plt

def f(t, v):
    g = 9.80665 # Aceleração gravitacional
    alpha = 0.1 # Coeficiente de arrasto aerodinâmico
    return g - alpha * v**2

def sol_analitica(t):
    g = 9.80665 # Aceleração gravitacional
    alpha = 0.1 # Coeficiente de arrasto aerodinâmico
    return np.sqrt(g / alpha) * np.tanh(np.sqrt(g * alpha) * t)

# Parâmetros de simulação
t0, tf = 0.0, 5.0 # Intervalo de tempo para simulação
h = 0.1 # Passo de tempo
v0 = 0.0 # Velocidade inicial (m/s)
t_vals = np.arange(t0, tf + h, h)
n_steps = len(t_vals)

# Inicialização de arrays
v_euler = np.zeros(n_steps)
v_rk4 = np.zeros(n_steps)
v_exato = sol_analitica(t_vals)

# Condições iniciais
v_euler[0] = v0
v_rk4[0] = v0

# Loop de integração numérico
for i in range(n_steps - 1):
    t_n = t_vals[i]
    
    # Método de Euler
    v_euler[i+1] = v_euler[i] + h * f(t_n, v_euler[i])
    
    # Método RK4
    k1 = f(t_n, v_rk4[i])
    k2 = f(t_n + h/2, v_rk4[i] + (h/2)*k1)
    k3 = f(t_n + h/2, v_rk4[i] + (h/2)*k2)
    k4 = f(t_n + h, v_rk4[i] + h*k3)
    
    v_rk4[i+1] = v_rk4[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Cálculo de erro absoluto
erro_euler = np.abs(v_euler - v_exato)
erro_rk4 = np.abs(v_rk4 - v_exato)

# Saída de dados no terminal
print(f"{'Tempo (s)':<10} | {'V. Exata':<10} | {'V. Euler':<10} | {'Erro Euler':<12} | {'V. RK4':<10} | {'Erro RK4'}")
print("-" * 80)
for i in range(n_steps):
    print(f"{t_vals[i]:<10.2f} | {v_exato[i]:<10.4f} | {v_euler[i]:<10.4f} | {erro_euler[i]:<12.4e} | {v_rk4[i]:<10.4f} | {erro_rk4[i]:.4e}")

# Plotagem de resultados

plt.figure(figsize=(10, 5))
t_fine = np.linspace(t0, tf, 200)
plt.plot(t_fine, sol_analitica(t_fine), 'k-', label='Solução Analítica')
plt.plot(t_vals, v_euler, 'bo--', label='Euler')
plt.plot(t_vals, v_rk4, 'r^-', label='RK4')
plt.title('Integração Numérica: Arrasto Aerodinâmico')
plt.xlabel('Tempo (s)')
plt.ylabel('Velocidade (m/s)')
plt.legend()
plt.grid(True, ls=':')
plt.tight_layout()
plt.show()
