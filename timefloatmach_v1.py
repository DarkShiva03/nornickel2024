import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def objective_function(x, pulp_data, constraints):
  """
  Функция цели, которая минимизирует потери руды.
  """
  # Разбивка пульпы на диапазоны
  ranges = np.array(x).reshape(-1, 2)
  
  # Вычисление концентрации руды в каждом диапазоне
  concentrations = []
  for i in range(ranges.shape[0]):
    # Фильтрация данных пульпы по диапазону
    filtered_data = pulp_data[(pulp_data['feature'] >= ranges[i, 0]) & (pulp_data['feature'] <= ranges[i, 1])]
    
    # Обработка исключений для пустых диапазонов
    if len(filtered_data) == 0:
      concentration = 0
    else:
      concentration = filtered_data['ore_concentration'].mean()
    
    concentrations.append(concentration)

  # Вычисление потерь руды
  total_loss = 1 - np.sum(concentrations * (ranges[:, 1] - ranges[:, 0]))
  
  return total_loss

def optimize_ore_extraction(pulp_data, constraints, method='minimize'):
  """
  Функция оптимизации извлечения руды.
  """
  # Инициализация параметров
  n_ranges = len(constraints)
  x0 = np.array([constraints[i]['min'] for i in range(n_ranges)] + 
                 [constraints[i]['max'] for i in range(n_ranges)])

  # Ограничения
  bounds = [(constraints[i]['min'], constraints[i]['max']) for i in range(n_ranges) for _ in range(2)]
  
  # Дополнительные ограничения:
  # 1. Минимальная концентрация руды в диапазоне
  min_concentration = 0.5  # Например, 50%
  cons = (
      {'type': 'ineq', 'fun': lambda x: [np.mean(pulp_data[(pulp_data['feature'] >= x[i]) & (pulp_data['feature'] <= x[i+1])]['ore_concentration']) - min_concentration for i in range(0, len(x), 2)]}
  )
  
  # 2. Максимальная ширина диапазона
  max_width = 2
  cons.append({'type': 'ineq', 'fun': lambda x: [max_width - (x[i+1] - x[i]) for i in range(0, len(x), 2)]})
  
  # 3. Минимальное количество руды в каждом диапазоне
  min_ore_quantity = 100
  cons.append({'type': 'ineq', 'fun': lambda x: [np.sum((pulp_data['feature'] >= x[i]) & (pulp_data['feature'] <= x[i+1])) - min_ore_quantity for i in range(0, len(x), 2)]})

  # Выбор метода оптимизации
  if method == 'minimize':
    result = minimize(objective_function, x0, args=(pulp_data, constraints), bounds=bounds, constraints=cons)
  elif method == 'differential_evolution':
  	  result = differential_evolution(objective_function, bounds=bounds, args=(pulp_data, constraints), constraints=cons)
 elif method == 'basinhopping':
  result = basinhopping(objective_function, x0, minimizer_kwargs={'bounds': bounds, 'constraints': cons}, args=(pulp_data, constraints))
 elif method == 'dual_annealing':
  result = dual_annealing(objective_function, bounds=bounds, args=(pulp_data, constraints), constraints=cons)
 else:
  raise ValueError('Неверный метод оптимизации')

 # Возврат оптимальных диапазонов
 optimal_ranges = result.x.reshape(-1, 2)
 return optimal_ranges

def plot_optimization_results(pulp_data, optimal_ranges):
 """
 Визуализация результатов оптимизации.
 """
 plt.figure(figsize=(10, 6))
 plt.scatter(pulp_data['feature'], pulp_data['ore_concentration'], label='Пульпа')

 for i in range(optimal_ranges.shape[0]):
  plt.axvspan(optimal_ranges[i, 0], optimal_ranges[i, 1], color='lightblue', alpha=0.5, label=f'Диапазон {i+1}')

 plt.xlabel('Характеристика')
 plt.ylabel('Концентрация руды')
 plt.title('Оптимизация извлечения руды')
 plt.legend()
 plt.show()

def predict_optimal_ranges(pulp_data, model):
 """
 Предсказание оптимальных диапазонов с помощью модели машинного обучения.
 """
 # Извлечение характеристик пульпы
 features = pulp_data['feature'].values.reshape(-1, 1)
 
 # Предсказание оптимальных диапазонов
 predicted_ranges = model.predict(features)
 
 # Преобразование предсказанных значений в диапазоны
 predicted_ranges = predicted_ranges.reshape(-1, 2)
 
 return predicted_ranges

def main():
 # Генерация случайных данных пульпы
 pulp_data = pd.DataFrame({'feature': np.random.uniform(0, 10, 1000), 'ore_concentration': np.random.uniform(0, 1, 1000)})

 # Определение ограничений
 constraints = [
   {'feature': 'feature', 'min': 0, 'max': 10, 'width': 2},
   {'feature': 'feature', 'min': 2, 'max': 8, 'width': 1},
   {'feature': 'feature', 'min': 5, 'max': 9, 'width': 1}
 ]

 # Оптимизация с помощью различных методов
 print("Оптимизация с помощью minimize:")
 optimal_ranges_minimize = optimize_ore_extraction(pulp_data, constraints, method='minimize')
 print(f"Оптимальные диапазоны: {optimal_ranges_minimize}")
 plot_optimization_results(pulp_data, optimal_ranges_minimize)

 print("Оптимизация с помощью differential_evolution:")
 optimal_ranges_de = optimize_ore_extraction(pulp_data, constraints, method='differential_evolution')
 print(f"Оптимальные диапазоны: {optimal_ranges_de}")
 plot_optimization_results(pulp_data, optimal_ranges_de)

 print("Оптимизация с помощью basinhopping:")
 optimal_ranges_bh = optimize_ore_extraction(pulp_data, constraints, method='basinhopping')
 print(f"Оптимальные диапазоны: {optimal_ranges_bh}")
 plot_optimization_results(pulp_data, optimal_ranges_bh)

 print("Оптимизация с помощью dual_annealing:")
 optimal_ranges_da = optimize_ore_extraction(pulp_data, constraints, method='dual_annealing')
 print(f"Оптимальные диапазоны: {optimal_ranges_da}")
 plot_optimization_results(pulp_data, optimal_ranges_da)

 # Обучение модели машинного обучения
 model = LinearRegression()
 model.fit(pulp_data[['feature']], pulp_data['ore_concentration'])

 # Предсказание оптимальных диапазонов
 predicted_ranges = predict_optimal_ranges(pulp_data, model)
 print(f"Предсказанные оптимальные диапазоны: {predicted_ranges}")
 plot_optimization_results(pulp_data, predicted_ranges)

 # Анализ чувствительности:
 # 1. Изменить ограничения (min, max, width) и запустить оптимизацию несколько раз.
 # 2. Проанализировать, как меняются оптимальные диапазоны и потери руды при изменении ограничений.
 # 3. Построить графики зависимости оптимальных диапазонов от параметров ограничений.

 # Учет динамики процесса:
 # 1. Сгенерировать новые данные пульпы с измененными характеристиками.
 # 2. Обновить модель машинного обучения с использованием новых данных.
 # 3. Предсказать новые оптимальные диапазоны, используя обновленную модель.

if __name__ == "__main__":
 main()