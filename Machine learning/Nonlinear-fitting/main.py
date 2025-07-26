import pandas as pd
from scipy.linalg import solve # Для решения систем уравнений
import matplotlib.pyplot as plt

# Определение функций
def f(t):
    return np.sin(0.0048 * t) # 0.0048 – частота колебаний

def g(t):
    return np.log((10**14)*t)

def h(t):
    return np.cos(0.0038 * t)

def p(t):
    return np.exp(0.0001 * t) 

def q(t):
    h = len(t) / 2 #центр параболы
    k = ((len(t) / 2)**2)/2.95 #высота параболы
    return -((t - h)**2) + k

# Загрузка данных
file_path_A = "a_A.xlsx"
file_path_B = "a_B.xlsx"

df_A = pd.read_excel(file_path_A).sort_values("data").reset_index(drop=True)
df_B = pd.read_excel(file_path_B).sort_values("data").reset_index(drop=True)

merged = pd.merge(df_A, df_B, on="data", suffixes=("_A", "_B"))

# Подготовка данных
t = np.arange(1, len(merged) + 1)
x = merged["rate_A"].values
y = merged["rate_B"].values

xy_ratio = x / y

# Создание матрицы системы уравнений
N = len(t)
F = np.array([f(t), g(t), h(t), p(t), q(t)])

# Матрица произведений
A_matrix = np.zeros((5, 5)) #матрица коэффициентов
b_vector = np.zeros(5) #вектор правой части уравнений

# Вычисление произведений и заполнение матрицы
for i in range(5):
    for j in range(5):
        # Произведение функции
        A_matrix[i, j] = np.sum(F[i] * F[j])
    # Правая часть уравнения
    b_vector[i] = np.sum(xy_ratio * F[i])

# Решение системы уравнений (нахождение коэффициентов)
coefficients = solve(A_matrix, b_vector)

# Вывод коэффициентов
A, B, C, D, F_coef = coefficients

print(f"A = {A}, B = {B}, C = {C}, D = {D}, F_coef = {F_coef}")

# Аппроксимация
y_fitted = A * f(t) + B * g(t) + C * h(t) + D * p(t) + F_coef * q(t)

# Линейная регрессия
A_l, B_l = np.polyfit(t, xy_ratio, deg=1)
xy_l = A_l * t + B_l

# Регрессия середины
def linear_regression(t, x):
    return np.mean(x)
C_min = linear_regression(t, xy_ratio) # Среднее значение
xy_ratio_reg = np.full(t.shape, C_min) # Заполнение массив средним значением

# Вычисление функций
A1 = np.dot(xy_ratio, f(t)) / np.dot(f(t), f(t))
# np.dot(xy_ratio, f(t)) - скалярное произведение между массивом и функцией
# np.dot(f(t), f(t) - скалярное произведение функции с самой собой
y_A1 = A1 * f(t)
B1 = np.dot(xy_ratio, g(t)) / np.dot(g(t), g(t))
y_B1 = B1 * g(t)
C1 = np.dot(xy_ratio, h(t)) / np.dot(h(t), h(t))
y_C1 = C1 * h(t)
D1 = np.dot(xy_ratio, p(t)) / np.dot(p(t), p(t))
y_D1 = D1 * p(t)
F1 = np.dot(xy_ratio, q(t)) / np.dot(q(t), q(t))
y_F1 = F1 * q(t)

# Начальное значения для выравнивания графиков
y_start = xy_l[0]

plt.figure(figsize=(15, 8))
plt.plot(t, xy_ratio, label="Данные", color='black')
plt.plot(t, y_A1 + (y_start - A1 * f(t)[0]), linestyle='--', color='red', label="Синус")
plt.plot(t, y_B1 + (y_start - B1 * g(t)[0]), linestyle='--', color='orange', label="Логарифм")
plt.plot(t, y_C1 + (y_start - C1 * h(t)[0]), linestyle='--', color='blue', label="Косинус")
plt.plot(t, y_D1 + (y_start - D1 * p(t)[0]), linestyle='--', color='green', label="Экспонента")
plt.plot(t, y_F1 + (y_start - F1 * q(t)[0]), linestyle='--', color='grey', label="Парабола")
plt.legend()
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(t, xy_ratio)
plt.plot(t, y_fitted, label="Аппроксимация", color='red')
#plt.plot(t, xy_l, linestyle='--', color='blue', label="Лин регрессия")
plt.plot(t, xy_ratio_reg, linestyle='--', color='blue')
plt.show()
