import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Функция для нахождения контуров объектов на бинарной маске
def find_contours(binary_mask):
    # Создаем пустое изображение (из нулей) такого же размера, как бинарная маска
    contours = np.zeros_like(binary_mask)

    # Проходим по всем пикселям изображения, кроме границ (чтобы не выйти за пределы массива)
    for i in range(1, binary_mask.shape[0] - 1):
        for j in range(1, binary_mask.shape[1] - 1):
            # Если текущий пиксель принадлежит объекту (значение 1)
            if binary_mask[i, j] == 1:
                # Проверяем 4 соседних пикселя (верх, низ, лево, право)
                # Если хотя бы один сосед - фон (0), то это граница объекта
                if (binary_mask[i - 1, j] == 0 or binary_mask[i + 1, j] == 0 or
                        binary_mask[i, j - 1] == 0 or binary_mask[i, j + 1] == 0):
                    contours[i, j] = 1  # помечаем пиксель как границу
    return contours


# Функция для разметки отдельных объектов на изображении (каждому объекту свой номер)
def label_regions(binary_mask):
    labeled_mask = np.zeros_like(binary_mask)  # создаем пустую маску для меток
    label = 1  # начинаем нумерацию с 1 (0 будет фоном)

    # Проходим по всем пикселям изображения
    for i in range(1, binary_mask.shape[0] - 1):
        for j in range(1, binary_mask.shape[1] - 1):
            # Если пиксель принадлежит объекту и еще не помечен
            if binary_mask[i, j] == 1 and labeled_mask[i, j] == 0:
                # Используем волновой алгоритм (заливку) для пометки всей области
                stack = [(i, j)]  # стек для хранения пикселей для обработки
                while stack:
                    x, y = stack.pop()  # берем последний пиксель из стека
                    # Проверяем, что пиксель в пределах изображения
                    if 0 <= x < binary_mask.shape[0] and 0 <= y < binary_mask.shape[1]:
                        # Если пиксель принадлежит объекту и еще не помечен
                        if binary_mask[x, y] == 1 and labeled_mask[x, y] == 0:
                            labeled_mask[x, y] = label  # помечаем текущим номером
                            # Добавляем всех 4-соседей в стек для обработки
                            stack.append((x - 1, y))
                            stack.append((x + 1, y))
                            stack.append((x, y - 1))
                            stack.append((x, y + 1))
                label += 1  # увеличиваем номер для следующего объекта
    return labeled_mask, label - 1  # возвращаем размеченную маску и количество объектов


# Основная функция для выделения объектов на изображении
def highlight_object(image_path):
    # Загружаем изображение и преобразуем в черно-белое (оттенки серого)
    img = Image.open(image_path).convert('L')
    image = np.array(img)  # преобразуем в массив numpy

    # Определяем маску фильтра Гаусса 3x3
    gaussian_mask = 1 / 16 * np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]])

    # Применяем фильтр Гаусса для сглаживания изображения
    blurred_image = np.zeros_like(image, dtype=np.float32)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Применяем свертку с гауссовским фильтром
            blurred_image[i, j] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * gaussian_mask)

    # Преобразуем обратно в целочисленный тип
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)

    # Создаем бинарную маску:пиксели темнее среднего -> 1 (объект), остальные -> 0
    threshold = np.mean(blurred_image)  # порог = средняя яркость изображения
    binary_mask = np.where(blurred_image > threshold, 0, 1)

    # Находим контуры объектов
    contours = find_contours(binary_mask)

    # Размечаем отдельные объекты на изображении
    labeled_mask, num_labels = label_regions(binary_mask)

    # Создаем цветное изображение для контуров (3 канала - RGB)
    contour_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Для каждого объекта выбираем случайный цвет и рисуем его контур
    for label in range(1, num_labels + 1):
        contour_color = np.random.randint(0, 255, 3)  # случайный цвет
        for i in range(1, binary_mask.shape[0] - 1):
            for j in range(1, binary_mask.shape[1] - 1):
                # Если пиксель принадлежит текущему объекту и является контуром
                if labeled_mask[i, j] == label and contours[i, j] == 1:
                    # Рисуем толстый контур (квадрат 12x12 пикселей)
                    for x in range(-6, 6):
                        for y in range(-6, 6):
                            if 0 <= i + x < binary_mask.shape[0] and 0 <= j + y < binary_mask.shape[1]:
                                # Рисуем только близкие пиксели (чтобы контур не был слишком толстым)
                                if np.abs(x) + np.abs(y) <= 2:
                                    contour_image[i + x, j + y] = contour_color

    # Создаем финальное изображение:
    final_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # Сначала делаем бинарную маску белой (все каналы = 255)
    final_image[:, :, 0] = binary_mask * 255
    final_image[:, :, 1] = binary_mask * 255
    final_image[:, :, 2] = binary_mask * 255
    # Затем накладываем цветные контуры
    final_image[contour_image != 0] = contour_image[contour_image != 0]

    # Создаем график с четырьмя изображениями в ряд
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    # 1. Показываем оригинальное изображение
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Оригинальное изображение')
    axs[0].axis('off')  # скрываем оси
    # 2. Показываем бинарную маску (черно-белое)
    axs[1].imshow(binary_mask, cmap='gray')
    axs[1].set_title('Бинарная маска')
    axs[1].axis('off')
    # 3. Показываем контуры объектов
    axs[2].imshow(contours, cmap='gray')
    axs[2].set_title('Контурные линии')
    axs[2].axis('off')
    # 4. Показываем результат - бинарную маску с цветными контурами
    axs[3].imshow(final_image)
    axs[3].set_title('Бинарное изображение с контурами')
    axs[3].axis('off')
    plt.show()
    return binary_mask


image_path = 'image.jpg'
highlight_object(image_path)
