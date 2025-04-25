import streamlit as st
import pandas as pd
import os
from PIL import Image
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import altair as alt
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit import session_state as _state

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    * {
        font-family: 'Roboto', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    pages = {
        "Главная страница": page_first,
        "Загрузка данных": page_second,
        "Редактирование данных": page_third,
        "Визуализация датафрейма": page_forth,
        "Тренировка и проверка модели": page_fifth,
    }

    st.sidebar.title("iQuat⚡")



    page = st.sidebar.radio("Выберите подходящую страницу", tuple(pages.keys()))

    pages[page]()


def page_first():
    col1, col2, col3 = st.columns(3)
    with col2:
        st.title("iQuat Model Trainer")


    st.write(
        """
            Эта платформа предлагает уникальную возможность для аналитиков данных, исследователей и энтузиастов в сфере машинного обучения, 
            предоставляя инструменты для загрузки собственных наборов данных, выбора и обучения моделей машинного обучения, а 
            также визуализации результатов и оценки метрик. Цель платформы - сделать процесс анализа данных и машинного обучения 
            более доступным и интуитивно понятным для пользователей разного уровня подготовки, ускоряя научные исследования и 
            разработку продуктов. 
        """
    )

    st.subheader("Как это работает:")

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    algo_path = Image.open(os.path.join(current_script_dir, "Algorithm.png"))

    st.image(algo_path, caption="Алгоритм платформы")

    st.subheader("Зачем?")
    st.write(
        """
            1. **Доступность:** Снижает барьеры для входа в машинное обучение, делая его доступным широкой аудитории.
            2. **Принятие решений:** Облегчает процесс принятия обоснованных решений в бизнесе и науке, предоставляя глубокий анализ данных.
            3. **Образование:** Помогает в обучении и развитии навыков в области анализа данных и машинного обучения.
            4. **Решение проблем:** Предлагает решения для реальных задач, используя силу данных и машинного обучения.
        """
    )



def page_second():
    st.title("Загрузка данных")
    st.warning("Принимает только файлы формата .csv", icon="⚠️")
    uploaded_file = upload_file()
    if uploaded_file is not None:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension.lower() == ".csv":
            try:
                df = pd.read_csv(uploaded_file, index_col=0)
                if "original_data" not in st.session_state:
                    st.session_state["original_data"] = df
                else:
                    st.session_state["original_data"] = df
                if "uploaded" not in st.session_state:
                    st.session_state["uploaded"] = True
                st.dataframe(df)
            except Exception as e:
                st.error(f"Произошла ошибка при чтении файла: {e}", icon="🚨")
        else:
            st.error("Загруженный файл не соответствует формату CSV", icon="🚨")

    with st.expander(
        "Что такое `CSV` и почему сайт поддерживает только файлы этого формата?"
    ):
        st.write(
            """
            **CSV (Comma-Separated Values)** — это формат текстового файла, предназначенный для хранения табличных данных. 
            Каждая строка файла представляет собой одну строку таблицы, а столбцы разделяются запятыми или другими разделителями, такими как точка с запятой. 
            Пример содержимого CSV-файла:   
        """
        )
        small_code = """
        name,age,city
        John,34,New York
        Jane,28,Los Angeles
        """
        st.code(small_code, language="python")
        st.write(
            """
            В этом примере данные о людях хранятся в формате, где первая строка — это заголовки столбцов, а последующие строки — это записи данных.
            
            **CSV файлы являются одним из самых популярных форматов для работы с данными в области `Data Science` по нескольким причинам:**
            
            **1.Простота и универсальность:** Формат CSV прост для понимания и может быть легко прочитан как людьми, так и компьютерными программами. Его можно открыть в любом текстовом редакторе, таблицах Excel или Google Sheets, а также обрабатывать с помощью большинства языков программирования и инструментов анализа данных.
            
            **2.Совместимость:** CSV файлы поддерживаются практически всеми инструментами анализа данных и базами данных, что делает их удобным выбором для обмена данными между различными системами и приложениями.
            
            **3.Эффективность:** Хотя CSV не самый эффективный формат с точки зрения использования дискового пространства (по сравнению, например, с бинарными форматами), его простота компенсирует это за счет легкости чтения и написания без необходимости специализированных парсеров или библиотек.
            
            **4.Гибкость:** CSV файлы легко модифицировать и обрабатывать, позволяя добавлять, удалять или изменять строки и столбцы без сложностей. Это делает их идеальными для различных этапов работы с данными, от первоначальной очистки и предобработки до более сложного анализа и визуализации.
            
            Из-за этих причин CSV файлы остаются одним из самых распространенных форматов для работы с данными в области Data Science, несмотря на наличие альтернативных форматов, таких как `JSON`, `XML` и бинарные форматы (например, `Parquet` и `HDF5`), которые могут быть более подходящими в определенных случаях использования. 
        """
        )

    st.write(
        """
            Отлично, теперь поработаем с данными и приведём их в порядок. :sunglasses:
        """
    )


def page_third():
    if "uploaded" not in st.session_state:
        st.info("Сначала вы должны загрузить данные")
    else:
        if "data" not in st.session_state:
            st.session_state["data"] = st.session_state["original_data"].copy()

        df = st.session_state["original_data"].copy()

        st.title("Редактирование данных")

        st.subheader("Оригинальный датафрейм")
        with st.expander("Что такое `датафрейм`?"):
            st.write(
                """DataFrame — это термин, используемый в программировании и анализе данных, особенно в таких языках и библиотеках, как Python (Pandas), R, и других. Это двумерная, изменяемая структура данных, которая может хранить данные различных типов (например, целые числа, строки, плавающие точки), и это делает её очень удобной для работы с табличными данными. DataFrame обычно включает в себя строки и столбцы, где строки обозначают наблюдения, а столбцы — переменные.

Основные возможности DataFrame включают в себя:
- Чтение и запись данных из различных источников, таких как CSV, Excel файлы, базы данных и другие.
- Манипуляции с данными: фильтрация, сортировка, группировка, объединение (join) и другие операции.
- Управление пропущенными данными, возможность их замещения или удаления.
- Агрегирование данных, позволяющее выполнять различные статистические расчеты.
- Визуализация данных с помощью графиков и диаграмм."""
            )
        st.dataframe(df, hide_index=True)
        st.write(
            "Размер датафрейма =", df.shape[0], "данных и ", df.shape[1], "параметров"
        )
        with st.container():
            st.warning(
                "Перед загрузкой другого датафрейма обязательно нужно нажать на кнопку!"
            )
            if st.button("Изменение датафрейма", use_container_width=True):
                reset_application_state_with_data()
        st.subheader("1. Общая информация")
        col1, col2 = st.columns(2)
        show_describe, show_unique = False, False
        with col1:
            if st.button("describe( )", use_container_width=True):
                show_describe = True
        if show_describe == True:
            st.dataframe(df.describe())
        with col2:
            if st.button("nunique( )", use_container_width=True):
                show_unique = True
        if show_unique == True:
            st.dataframe(pd.DataFrame(df.nunique(), columns=["num of unique values"]))

        with st.expander("что делает функция `describe()`?"):
            st.write(
                """
        Функция `describe()` в библиотеке pandas в Python используется для получения описательной статистики по данным в `DataFrame` или `Series`. Она возвращает статистику, которая включает в себя:

        - **count**: количество ненулевых значений
        - **mean**: среднее значение числовых данных
        - **std**: стандартное отклонение, которое измеряет разброс числовых данных    
        - **min**: минимальное значение      
        - **25%**: первый квартиль (медиана первой половины данных)      
        - **50%**: медиана или второй квартиль (медиана всего набора данных)
        - **75%**: третий квартиль (медиана второй половины данных)
        - **max**: максимальное значение
                
        По умолчанию `describe()` возвращает описательную статистику только для колонок с числовыми данными. 
        Однако, её можно использовать и для колонок с нечисловыми данными (например, тип `object` или `categorical`), передав параметр include. 
        В этом случае функция может возвращать такую статистику, как количество уникальных значений, самое частое значение и его частоту.

        Пример использования функции `describe()`:
        """
            )

            small_code = """
            # Создание простого DataFrame
            df = pd.DataFrame({
            'числовая_колонка': [1, 2, 3, 4, 5],
            'категориальная_колонка': ['A', 'B', 'A', 'B', 'C']
        })

        # Применение функции describe к DataFrame
        описание = df.describe()

        # Для включения описательной статистики по нечисловым колонкам
        описание_все = df.describe(include='all')
            """
            st.code(small_code, language="python")
            st.write(
                """
        В данном примере описание будет содержать статистику только для `числовая_колонка`, в то время как описание_все включает статистику по всем колонкам, включая `категориальная_колонка`.        
        """
            )
        with st.expander("Что делает функция `nunique()`?"):
            st.write(
                """
        Функция `nunique()` используется в библиотеке pandas в Python для подсчета количества уникальных значений в столбце `DataFrame` или `Series`. 

        Эта функция очень полезна при анализе данных, поскольку позволяет быстро понять, сколько уникальных элементов содержится в определенном наборе данных или его части.

        **Пример использования**

        Допустим, у вас есть DataFrame df, который содержит данные о клиентах, включая их имена и города, в которых они проживают. 
        Вы можете использовать `nunique()` для определения количества уникальных имен и городов:
                    """
            )
            small_code = """
            import pandas as pd

            # Создание DataFrame
            data = {
            'name': ['John', 'Jane', 'Mary', 'John', 'Mary'],
            'city': ['New York', 'Los Angeles', 'Los Angeles', 'New York', 'San Francisco']
        }
            df = pd.DataFrame(data)
            
            # Подсчет уникальных имен
            unique_names = df['name'].nunique()
            print(f"Уникальных имен: {unique_names}")
            
            # Подсчет уникальных городов, включая пропущенные значения
            unique_cities = df['city'].nunique(dropna=False)
            print(f"Уникальных городов: {unique_cities}")    
        })"""
            st.code(small_code, language="python")
            st.write(
                """
        В этом примере df['name'].nunique() вернет `3`, так как в столбце `name` содержится три уникальных имени (John, Jane, Mary). 

        Аналогично, df['city'].nunique() вернет `3`, поскольку имеется три уникальных города.

        Функция `nunique()` является мощным инструментом для исследования и анализа данных, позволяя быстро оценить разнообразие содержимого в наборах данных.
                    """
            )
        change_df = df
        st.subheader("2.Работа с null значениями")
        with st.expander("Что такое `null` значение?"):
            st.write(
                """Null значение — это специальный маркер в программировании и базах данных, используемый для обозначения отсутствия какого-либо значения или данных. В различных языках программирования и системах управления базами данных это может обозначаться по-разному, например, `null` в Java, `None` в Python, `NULL` в SQL.

Основные особенности null значений:
- Они используются для указания на то, что переменная не имеет присвоенного значения.
- В базах данных null может обозначать отсутствие значения в поле.
- При арифметических операциях или сравнениях с участием null результаты могут быть неочевидными, так как null не является ни истиной, ни ложью, ни нулем, ни любым конкретным значением. Например, в SQL выражение `SELECT * FROM table WHERE column = NULL;` не вернет строки, где column содержит null, поскольку null не равен ничему, даже самому себе. Вместо этого используется `IS NULL`.

Обработка null значений требует особого внимания при программировании и работе с данными, чтобы избежать ошибок в логике программы или анализе данных."""
            )
        st.write(
            "Ты можешь удалить столбец с null значениями или можешь заполнить его другими значениями автоматически. Выбор за тобой."
        )
        with st.expander("Как выбрать, какие переменные удалить, а какие оставить?"):
            st.write(
                """Во время предварительной обработки данных (препроцессинга) из DataFrame обычно удаляются или трансформируются определенные параметры (столбцы) на основе их важности для анализа или моделирования и качества данных. Выбор столбцов для удаления или сохранения зависит от многих факторов, включая цели анализа, качество и тип данных. Ниже приведены общие критерии для решения, какие параметры удалить и какие оставить:

### Параметры для удаления

1. **Нерелевантные параметры**: Столбцы, которые не имеют значимости для анализа или построения модели. Например, идентификаторы, имена, метки времени могут быть нерелевантными в зависимости от контекста задачи.
2. **Параметры с высоким процентом пропущенных значений**: Если в столбце отсутствует значительное количество данных, и это не может быть адекватно восстановлено или заменено, такой столбец часто удаляют.
3. **Параметры с одним уникальным значением (или очень низким разнообразием)**: Столбцы, которые содержат только одно уникальное значение или большинство значений одинаковы, не несут полезной информации для анализа.
4. **Сильно коррелированные параметры**: Пары или группы параметров, которые сильно коррелированы друг с другом, могут быть избыточными. Обычно сохраняют один параметр из группы коррелированных параметров, чтобы избежать мультиколлинеарности.

### Параметры для сохранения

1. **Целевая переменная**: В контексте обучения с учителем, целевая переменная (или зависимая переменная), которую модель должна предсказать, всегда сохраняется.
2. **Параметры с высокой предиктивной способностью**: Параметры, которые считаются важными для предсказания целевой переменной или имеют сильную связь с исследуемым явлением.
3. **Параметры, необходимые для дальнейшего анализа или построения признаков**: Некоторые параметры могут понадобиться для создания новых признаков (фичи инжиниринг) или для проведения специфического анализа.

Важно отметить, что решение о том, какие параметры удалять или оставлять, должно основываться на тщательном анализе данных и знании предметной области. Иногда применяются статистические методы и методы машинного обучения для оценки важности параметров перед их удалением или преобразованием."""
            )

        delete_column = create_toggle("delete_column_toggle", "Удалить")
        if delete_column:
            options = st.multiselect(
                "Выберите какие параметры хотите удалить:",
                change_df.columns,
                key=persist("delete_columns_multiselect"),
            )
            change_df = delete_columns(change_df, options)
        st.write(
            "Теперь выберите категориальные и числовые параметры из своего датафрейма."
        )

        with st.expander(
            "В чем разница между категориальными и числовыми параметрами?"
        ):
            st.write(
                """Категориальные и числовые параметры — это два основных типа данных, с которыми вы встретитесь при работе с датасетами в анализе данных и машинном обучении. Каждый из них играет уникальную роль в анализе и требует различных методов предобработки. Вот подробное объяснение:

### Категориальные параметры
Категориальные параметры представляют собой тип данных, который описывает группы или категории. Они могут быть как численными, так и текстовыми значениями, но эти значения не имеют математического смысла. Категориальные данные подразделяются на два основных типа:

- **Номинальные**: Данные, которые описывают категории без какого-либо порядка или ранжирования. Примеры включают цвета, типы жилья, названия марок автомобилей.
- **Порядковые**: Данные, которые включают категории с некоторым порядком или иерархией. Примеры включают рейтинги обслуживания (хорошо, удовлетворительно, плохо), уровень образования (высшее, среднее, низшее).

### Числовые параметры
Числовые параметры представляют собой данные, которые измеряются численно. Эти данные могут быть использованы для выполнения математических операций. Числовые данные также делятся на два типа:

- **Дискретные**: Данные, которые принимают только определенные значения. Эти значения обычно представляют собой счетные величины. Примеры включают количество автомобилей в домохозяйстве, количество детей в семье.
- **Непрерывные**: Данные, которые могут принимать любое значение в пределах диапазона. Примеры включают вес, рост, температуру.

### Псевдо-числовые параметры
Псевдо-числовые параметры — это категориальные данные, которые кодируются числами, но эти числа не имеют собственно математического значения или порядка. Примером может служить почтовый индекс. Хотя почтовый индекс и представлен числами, арифметические операции (сложение, умножение и т.д.) над этими числами не имеют смысла с точки зрения их категориального значения. Такие данные часто требуют специальной обработки перед использованием в аналитических моделях, например, преобразования в one-hot encoding или другие формы категориального представления, чтобы модель могла корректно интерпретировать информацию."""
            )

        options_categorical = st.multiselect(
            "Выберите категориальные параметры",
            change_df.columns,
            key=persist("select_categorical_columns_multiselect"),
        )
        categorical_data = df[options_categorical]

        options_numerical = st.multiselect(
            "Выберите числовые параметры",
            change_df.columns,
            key=persist("select_numerical_columns_multiselect"),
        )
        numerical_data = df[options_numerical]

        fill_numerical = create_toggle(
            "fill_numerical_missing_toggle", "Заполнить числовые параметры"
        )
        if fill_numerical:
            options = st.multiselect(
                "Выберите числовую колонку которую хотите заполнить",
                numerical_data.columns,
                key=persist("select_to_fill_numerical_columns_multiselect"),
            )
            change_df = fill_numerical_data(change_df, options)

        fill_categorical = create_toggle(
            "fill_categorical_missing_toggle", "Заполнить категориальные параметры"
        )
        if fill_categorical:
            options = st.multiselect(
                "Выберите категориальную колонку которую хотите заполнить",
                categorical_data.columns,
                key=persist("select_to_fill_categorical_columns_multiselect"),
            )
            change_df = fill_categorical_data(change_df, options)

        with st.expander(
            "Каким образом происходит заполнение числовых и категориальных параметров?"
        ):
            st.write(
                """
### Заполнение пропущенных значений в числовых параметрах с помощью медианы

Когда мы сталкиваемся с пропущенными значениями в числовых данных, одним из наиболее распространенных и эффективных методов их заполнения является использование медианы. Медиана — это значение, которое делит распределение данных на две равные части. Другими словами, это средний элемент в упорядоченном списке чисел. Если количество элементов четное, медиана будет средним арифметическим двух центральных чисел.

Использование медианы в качестве метода заполнения пропущенных значений предпочтительно, потому что она менее чувствительна к выбросам в данных по сравнению со средним арифметическим. Это делает медиану надежным выбором для заполнения пропущенных значений, особенно в случаях, когда распределение данных имеет тяжелые хвосты или не является симметричным.

### Заполнение пропущенных значений в категориальных параметрах с помощью моды

Для категориальных данных, где значения представляют собой группы или категории, пропущенные значения часто заполняются модой. Мода — это наиболее часто встречающееся значение в наборе данных. Этот метод особенно полезен, когда есть категории, которые значительно чаще встречаются, чем другие, и, следовательно, представляют собой "типичные" случаи.

Использование моды для заполнения пропущенных категориальных данных позволяет сохранить распределение категорий в наборе данных и обеспечивает простой, но эффективный способ обработки пропущенных значений. Это особенно актуально в случаях, когда нет явного порядка или иерархии среди категорий, что делает использование медианы или среднего неприменимым.

Оба эти метода — заполнение медианой для числовых данных и модой для категориальных — являются примерами "одномерной импутации", где каждый параметр заполняется независимо от других, на основе его собственного распределения значений. Эти подходы позволяют эффективно обрабатывать пропущенные значения, минимизируя искажение в данных и сохраняя их статистические свойства."""
            )
        nulls_change = check_nulls(change_df)
        st.dataframe(nulls_change, width=500)
        st.dataframe(change_df, hide_index=True)

        st.subheader("3.Работа с категориальными данными")

        with st.expander(
            "Какие функции выполняют `one hot encoding`, `label encoding`, `ordinal encoding`?"
        ):
            st.write(
                """One Hot Encoding, Label Encoding и Ordinal Encoding — это техники преобразования категориальных данных в числовой формат, чтобы их можно было использовать в алгоритмах машинного обучения, большинство из которых требуют числового ввода. Вот обзор каждой техники и их польза:

### One Hot Encoding
One Hot Encoding преобразует каждую категорию в новый столбец и присваивает 1 или 0 в каждом из них в зависимости от того, присутствует ли категория в записи. Например, если у нас есть признак "цвет" с тремя категориями (Красный, Зеленый, Синий), One Hot Encoding создаст три новых столбца, по одному для каждого цвета, и поместит 1 в соответствующий столбец, если этот цвет присутствует, и 0 в другие.

**Польза**: Этот метод идеально подходит для номинальных данных, где нет явного порядка между категориями, поскольку он не вносит искусственный порядок и позволяет модели точно интерпретировать наличие или отсутствие каждой категории.

### Label Encoding
Label Encoding присваивает каждой категории уникальное целочисленное значение. Например, если у нас есть категории (Красный, Зеленый, Синий), Label Encoding может присвоить Красный = 1, Зеленый = 2, Синий = 3. 

**Польза**: Этот метод эффективно уменьшает размерность данных, преобразуя категориальные данные в один столбец чисел, что может быть полезно для определенных типов моделей. Однако он вносит искусственный порядок, который может не соответствовать данным, особенно для номинальных переменных.

### Ordinal Encoding
Ordinal Encoding присваивает уникальные числовые значения категориям согласно их порядку или рангу. Это подходит для порядковых данных, где категории имеют явный порядок или иерархию. Например, для рейтинга услуг (плохо, удовлетворительно, хорошо) можно присвоить плохо = 1, удовлетворительно = 2, хорошо = 3.

**Польза**: Ordinal Encoding сохраняет информацию о порядке между категориями, что может быть полезно для моделей, чтобы уловить и использовать эту порядковую связь.

### Зачем это нужно делать?
Преобразование категориальных данных в числовые необходимо, поскольку многие алгоритмы машинного обучения могут эффективно работать только с числовыми данными. Эти методы преобразования позволяют интегрировать важную информацию о категориальных признаках в модели машинного обучения, улучшая их способность к обучению и предсказанию. Выбор метода зависит от конкретного признака и контекста задачи, и правильное применение этих техник может значительно повысить качество моделирования."""
            )
        updated_categorical_data = change_df[options_categorical]
        one_hot_encoding = create_toggle("one_hot_encoding_toggle", " One Hot encoding")
        if one_hot_encoding:
            one_hot_encoding_choose = st.radio(
                "Выберите параметры для использования one hot encoding:",
                ["Не использовать", "Выбрать вручную", "Все параметры"],
                key=persist("radio_for_one_hot_encoding"),
            )
            if one_hot_encoding_choose == "Все параметры":
                change_df = apply_one_hot_encoder(
                    change_df, updated_categorical_data.columns
                )

            elif one_hot_encoding_choose == "Выбрать вручную":
                options = st.multiselect(
                    "Выберите параметр one hot encoder: ",
                    updated_categorical_data.columns,
                    key=persist("select_manually_one_hot_encoding"),
                )
                change_df = apply_one_hot_encoder(change_df, options)
                updated_categorical_data = updated_categorical_data.drop(
                    options, axis=1
                )
            else:
                pass

        label_encoding = create_toggle("label_encoding_toggle", "Label encoding")
        if label_encoding:
            label_encoding_choose = st.radio(
                "Выберите параметры для использования label encoding:",
                ["Не использовать", "Выбрать вручную", "Все параметры"],
                key=persist("radio_for_label_encoding"),
            )
            if label_encoding_choose == "Все параметры":
                change_df = apply_label_encoder(
                    change_df, updated_categorical_data.columns
                )
            elif label_encoding_choose == "Выбрать вручную":
                options = st.multiselect(
                    "Выберите параметр label encoder: ",
                    updated_categorical_data.columns,
                    key=persist("select_manually_label_encoding"),
                )
                change_df = apply_label_encoder(change_df, options)
                updated_categorical_data = updated_categorical_data.drop(
                    options, axis=1
                )
            else:
                pass

        ordinal_encoding = create_toggle("ordinal_encoding_toggle", "Ordinal encoding")
        if ordinal_encoding:
            ordinal_encoding_choose = st.radio(
                "Выберите параметры для использования ordinal encoding:",
                ["Не использовать", "Выбрать вручную", "Все параметры"],
                key=persist("radio_for_ordinal_encoding"),
            )
            if ordinal_encoding_choose == "Все параметры":
                change_df = apply_ordinal_encoder(
                    change_df, updated_categorical_data.columns
                )
            elif ordinal_encoding_choose == "Выбрать вручную":
                options = st.multiselect(
                    "Выберите параметр ordinal encoder: ",
                    updated_categorical_data.columns,
                    key=persist("select_manually_ordinal_encoding"),
                )
                change_df = apply_ordinal_encoder(change_df, options)
                updated_categorical_data = updated_categorical_data.drop(
                    options, axis=1
                )
            else:
                pass

        st.subheader("4. Работа с числовыми данными")
        with st.expander(
            "Что такое `нормализация`, `стандартизация`, и `логарифмическая трансформация` и зачем оно нужно?"
        ):
            st.write(
                """Нормализация, стандартизация и логарифмическая трансформация — это методы предварительной обработки данных, используемые для приведения числовых признаков к общему масштабу. Эти методы помогают улучшить процесс обучения машинных моделей, делая его более эффективным и улучшая качество предсказаний. Давайте разберемся подробнее:

### Нормализация
Нормализация — это процесс масштабирования данных таким образом, чтобы их значения лежали в определенном диапазоне, часто между 0 и 1. Это достигается с помощью функции min-max scaling.

**Зачем это нужно**: Нормализация особенно полезна, когда данные содержат атрибуты с различными масштабами, и вы используете алгоритмы, чувствительные к величине признаков, такие как k-NN, нейронные сети, или алгоритмы, использующие меры расстояния."""
            )
            st.write(
                """
### Стандартизация
Стандартизация — это процесс приведения данных к виду, где среднее значение каждого признака равно 0, а стандартное отклонение — 1. Это достигается с помощью метода, иногда называемого Z-score normalization.

**Зачем это нужно**: Стандартизация полезна, когда данные распределены приблизительно нормально, и особенно важна для алгоритмов, предполагающих нормальное распределение данных входных признаков, таких как логистическая регрессия, линейные и радиальные SVM."""
            )
            st.write(
                """
### Логарифмическая трансформация
Логарифмическая трансформация — это преобразование данных с помощью логарифма, часто натурального, что помогает уменьшить скос распределения данных. Преобразование применяется по формуле:"""
            )
        combined_object_string_cols = change_df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()
        col1, col2, col3 = st.columns(3)
        if len(combined_object_string_cols) > 0:
            status = True
        with col1:
            normalization = st.toggle(
                "Нормализация", key=persist("normalization_toggle")
            )
            if normalization:
                change_df = apply_min_max_scaling(change_df, change_df.columns)
        with col2:
            standartization = st.toggle(
                "Стандартизация", key=persist("standartization_toggle")
            )
            if standartization:
                change_df = apply_standardization(change_df, change_df.columns)
        with col3:
            log_transformation = st.toggle(
                "Логарифмическая трансформация",
                key=persist("log_transformation_toggle"),
            )
            if log_transformation:
                change_df = apply_log_transformation(change_df, change_df.columns)

        st.dataframe(change_df, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Сохранить изменения"):
                st.session_state["data"] = change_df
                st.success("Изменения сохранены.")
        with col2:
            if st.button("Очистить изменения"):
                st.session_state["data"] = st.session_state["original_data"].copy()
                reset_application_state()
                st.warning("Изменения сброшены")


def page_forth():
    if "uploaded" not in st.session_state:
        st.info("Сначала вы должны загрузить данные")
    else:    
        st.title("Визуализация датафрейма")
        complete_df = st.session_state["data"]
        st.subheader("Сохраненный датафрейм")
        st.dataframe(complete_df, hide_index=True)

        st.subheader("Корреляция параметров")
        color_theme_list = [
            "Blues",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "Purples",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ]
        color_for_corr = st.selectbox(
            "Выберите цвет для графика",
            color_theme_list,
            key=persist("select_color_for_corr"),
        )
        correlation_matrix_visualize(complete_df, color_for_corr)

        st.subheader("1.Визуализация датафрейма")
        plots = ["Line Plot", "Scatter Plot", "Histogram Plot", "Box Plot", "Density Plot"]
        visualizations = st.selectbox(
            "Выберите график который вы хотите увидеть",
            plots,
            key=persist("select_type_of_visualization_for_dataset"),
        )
        x_axis = st.selectbox(
            "Выберите параметр который будет расположен по оси X",
            complete_df.columns,
            key=persist("select_column_for_x_axis"),
        )
        y_axis = st.selectbox(
            "Выберите параметр который будет расположен по оси Y",
            complete_df.columns,
            key=persist("select_column_for_y_axis"),
        )
        color_theme_list = [
            "blues",
            "cividis",
            "greens",
            "inferno",
            "magma",
            "plasma",
            "reds",
            "turbo",
            "viridis",
        ]

        use_hue = create_toggle(
            "use_hue_toggle", "Включить группировку точек данных по цвету"
        )

        if use_hue:
            hue = st.selectbox(
                "Выберите параметр, с помощью которого будут группироваться точки данных по цвету",
                complete_df.columns,
                key="select_hue",  # Assuming persist() function is defined elsewhere to handle session state
            )
            selected_color_theme = st.selectbox(
                "Select a color theme", color_theme_list, key=persist("select_color_theme")
            )
        else:
            hue = None
            selected_color_theme = st.color_picker(
                "Pick A Color", "#4bdbff", key=persist("color")
            )

        # Then, depending on the type of visualization selected by the user:
        if visualizations == "Line Plot":
            line_chart(
                complete_df, x_axis, y_axis, input_color_theme=selected_color_theme, hue=hue
            )
        elif visualizations == "Scatter Plot":
            scatter_plot(
                complete_df, x_axis, y_axis, input_color_theme=selected_color_theme, hue=hue
            )
        elif visualizations == "Histogram Plot":
            histogram(complete_df, x_axis, input_color_theme=selected_color_theme, hue=hue)
        elif visualizations == "Box Plot":
            box_plot(
                complete_df, x_axis, y_axis, input_color_theme=selected_color_theme, hue=hue
            )
        elif visualizations == "Density Plot":
            density_plot(
                complete_df, x_axis, input_color_theme=selected_color_theme, hue=hue
            )


def page_fifth():
    if "uploaded" not in st.session_state:
        st.info("Сначала вы должны загрузить данные")
    else:
        st.title("Тренировка и проверка модели")
        regression_models = [
            "Linear Regression",
            "Decision Tree Regression",
            "Random Forest Regression",
            "Support Vector Machine Regression",
            "Gradient Boosting Regression",
            "Multi Layer Perceptron Regression"
        ]
        classification_models = [
            "Logistic Regression",
            "Decision Tree CLassifier",
            "Random Forest Classification",
            "Support Vector Machine CLassification",
            "Gradient Boosting Classification",
            "Multi Layer Perceptron Classifier"
        ]

        complete_df = st.session_state["data"]
        st.subheader("Сохраненный датафрейм")
        st.dataframe(complete_df, hide_index=True)
        st.subheader("1.Разделение на тренировочный и тестовый датафрейм")
        y_column = st.selectbox(
            "Выберите целевой параметр, значение которого будут спрогнозированы:",
            complete_df.columns,
            key=persist("select_y_column"),
        )
        X = delete_columns(complete_df, y_column)
        y = complete_df[y_column]
        col1, col2, col3 = st.columns([5, 0.3, 1])
        with col1:
            st.subheader("X:")
            st.dataframe(X, hide_index=True)
            st.write("Размер датафрейма :", X.shape)
        with col3:
            st.subheader("y:")
            st.dataframe(y, hide_index=True)
            st.write("Размер датафрейма :", y.shape[0])

        size = st.slider(
            "Выберите размер тестового датафрейма:",
            0.0,
            1.0,
            0.2,
            step=0.01,
            key=persist("slider_size_of_test_dataset"),
        )
        st.write(f"Размер тестового датафрейма {size * 100} % от общего датафрейма")
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, size=size)

        col1, col2, col3 = st.columns([5, 0.3, 1])
        with col1:
            st.subheader("X_train:")
            st.dataframe(X_train, hide_index=True)
            st.write("Размер датафрейма :", X_train.shape)
        with col3:
            st.subheader("y_train:")
            st.dataframe(y_train, hide_index=True)
            st.write("Размер датафрейма :", y_train.shape[0])
        st.divider()
        col4, col5, col6 = st.columns([5, 0.3, 1])
        with col4:
            st.subheader("X_test:")
            st.dataframe(X_test, hide_index=True)
            st.write("Размер датафрейма :", X_test.shape)
        with col6:
            st.subheader("y_test:")
            st.dataframe(y_test, hide_index=True)
            st.write("Размер датафрейма :", y_test.shape[0])

        st.subheader("2.Выбор алгоритма машинного обучения")

        model = None

        type_of_task_radio = st.radio(
            "Выберите тип задачи:",
            ["Регрессия", "Классификация"],
            key=persist("type_of_task_radio"),
        )
        if type_of_task_radio == "Регрессия":
            options_regression = st.selectbox(
                "Выберите один из алгоритмов Регрессии: ",
                regression_models,
                key=persist("select_regression_model"),
            )
            if options_regression == "Linear Regression":
                model = custom_linear_regression()
            elif options_regression == "Decision Tree Regression":
                options_decision_tree_regression = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_decision_tree_regression"),
                )
                if options_decision_tree_regression == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        max_depth = st.number_input(
                            "`max_depth` Максимальная глубина дерева",
                            1,
                            100,
                            3,
                            key=persist("number_for_decision_tree_regression_max_depth"),
                        )
                    with col2:
                        min_samples_split = st.number_input(
                            "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                            2,
                            100,
                            2,
                            key=persist(
                                "number_for_decision_tree_regression_min_samples_split"
                            ),
                        )
                    col3, col4 = st.columns(2)
                    with col3:
                        min_samples_leaf = st.number_input(
                            "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                            1,
                            100,
                            1,
                            key=persist(
                                "number_for_decision_tree_regression_min_samples_leaf"
                            ),
                        )
                    with col4:
                        ccp_alpha = st.slider(
                            "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                            0.0,
                            1.0,
                            1.0,
                            step=0.01,
                            key=persist("slider_decision_tree_regression_ccp_alpha"),
                        )
                    criterion_for_decision_tree_regression = [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                    criterion = st.selectbox(
                        "`criterion` Функция для измерения качества разделения",
                        criterion_for_decision_tree_regression,
                        key=persist("select_criterion_decision_tree_regression"),
                    )

                    model = custom_decision_tree_regression(
                        criterion_c=criterion,
                        max_depth_c=max_depth,
                        min_samples_split_c=min_samples_split,
                        min_samples_leaf_c=min_samples_leaf,
                        ccp_alpha_c=ccp_alpha,
                    )
                else:
                    model = DecisionTreeRegressor()

            elif options_regression == "Random Forest Regression":
                options_random_forest_regression = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_random_forest_regression"),
                )
                if options_random_forest_regression == "Выбрать параметры вручную":
                    n_estimators = st.slider(
                        "`n_estimators` Количество деревьев в лесу",
                        50,
                        1000,
                        100,
                        key=persist("number_for_random_forest_regression_n_estimators"),
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        max_depth = st.number_input(
                            "`max_depth` Максимальная глубина каждого дерева",
                            1,
                            100,
                            3,
                            key=persist("number_for_random_forest_regression_max_depth"),
                        )
                    with col2:
                        min_samples_split = st.number_input(
                            "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                            2,
                            100,
                            2,
                            key=persist(
                                "number_for_random_forest_regression_min_samples_split"
                            ),
                        )
                    col3, col4 = st.columns(2)
                    with col3:
                        min_samples_leaf = st.number_input(
                            "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                            1,
                            100,
                            1,
                            key=persist(
                                "number_for_random_forest_regression_min_samples_leaf"
                            ),
                        )
                    with col4:
                        ccp_alpha = st.slider(
                            "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                            0.0,
                            1.0,
                            0.0,
                            step=0.01,
                            key=persist("slider_random_forest_regression_ccp_alpha"),
                        )
                    criterion_for_random_forest_regression = [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ]
                    criterion = st.selectbox(
                        "`criterion` Функция для измерения качества разделения",
                        criterion_for_random_forest_regression,
                        key=persist("select_criterion_random_forest_regression"),
                    )

                    model = custom_random_forest_regression(
                        n_estimators,
                        criterion,
                        max_depth,
                        min_samples_split,
                        min_samples_leaf,
                        ccp_alpha,
                    )
                else:
                    model = RandomForestRegressor()

            elif options_regression == "Support Vector Machine Regression":
                options_svm_regression = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_svm_regression"),
                )
                if options_svm_regression == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        kernel = st.selectbox(
                            "`kernel` Указывает тип ядра, который будет использоваться в алгоритме.",
                            ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                            key=persist("select_kernel_svm_regression"),
                        )
                    with col2:
                        degree = st.number_input(
                            "`degree` Степень полиномиальной функции ядра ('poly')",
                            1,
                            10,
                            3,
                            key=persist("number_for_svm_regression_degree"),
                        )
                    model = custom_svr(kernel, degree)
                else:
                    model = SVR()
            elif options_regression == "Gradient Boosting Regression":
                options_gbc_regression = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_gbc_regression"),
                )
                if options_gbc_regression == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        loss = st.selectbox(
                            "`loss` Функция потерь, которую необходимо оптимизировать",
                            ["squared_error", "absolute_error", "huber", "quantile"],
                            key=persist("select_loss_gbc_regression"),
                        )
                    with col2:
                        learning_rate = st.slider(
                            "`learning_rate` Скорость обучения уменьшает вклад каждого дерева на величину learning_rate",
                            0.0,
                            1.0,
                            0.1,
                            step=0.01,
                            key=persist("number_for_gbc_regression_learning_rate"),
                        )
                    n_estimators = st.slider(
                        "`n_estimators` Количество деревьев в лесу",
                        50,
                        1000,
                        100,
                        key=persist("number_for_gbc_regression_n_estimators"),
                    )
                    col3, col4 = st.columns(2)
                    with col3:
                        max_depth = st.number_input(
                            "`max_depth` Максимальная глубина каждого дерева",
                            1,
                            100,
                            3,
                            key=persist("number_for_gbc_regression_max_depth"),
                        )
                    with col4:
                        min_samples_split = st.number_input(
                            "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                            2,
                            100,
                            2,
                            key=persist("number_for_gbc_regression_min_samples_split"),
                        )
                    col5, col6 = st.columns(2)
                    with col5:
                        min_samples_leaf = st.number_input(
                            "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                            1,
                            100,
                            1,
                            key=persist("number_for_gbc_regression_min_samples_leaf"),
                        )
                    with col6:
                        ccp_alpha = st.slider(
                            "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                            0.0,
                            1.0,
                            0.0,
                            step=0.01,
                            key=persist("slider_gbc_regression_ccp_alpha"),
                        )
                    criterion_for_random_forest_classification = [
                        "friedman_mse",
                        "squared_error",
                    ]
                    criterion = st.selectbox(
                        "`criterion` Функция для измерения качества разделения",
                        criterion_for_random_forest_classification,
                        key=persist("select_criterion_gbc_regression"),
                    )

                    model = custom_gbr(
                        loss_c=loss,
                        learning_rate_c=learning_rate,
                        n_estimators_c=n_estimators,
                        criterion_c=criterion,
                        max_depth_c=max_depth,
                        min_samples_split_c=min_samples_split,
                        min_samples_leaf_c=min_samples_leaf,
                        ccp_alpha_c=ccp_alpha,
                    )
                else:
                    model = GradientBoostingRegressor()
            else:
                options_mlp_regression = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_mlpc"),
                )
                if options_mlp_regression == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        activation = st.selectbox("`activation` Функция активации для скрытого слоя", ["relu", "logistic", "tanh", "identity"], key=persist("select_activation_mlpc"))
                    with col2:
                        solver = st.selectbox("`solver` Решающая программа для оптимизации веса", ["adam", "lbfgs", "sgd"],  key=persist("select_solver_mlpc"))
                    model = custom_mlp_regressor(activation_func_c=activation, solver_c=solver)
                else:
                    model = MLPRegressor()
                
        else:
            options_clf = st.selectbox(
                "Выберите один из алгоритмов Классификации: ",
                classification_models,
                key=persist("select_classification_model"),
            )
            if options_clf == "Logistic Regression":
                options_logistic_regression = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_logistic_regression"),
                )
                if options_logistic_regression == "Выбрать параметры вручную":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        C = st.slider(
                            "`C` Обратное значение силы регуляризации",
                            0.0,
                            1.0,
                            1.0,
                            step=0.01,
                            key=persist("slider_for_logistic_regression_C"),
                        )
                    with col2:
                        max_iter = st.number_input(
                            "` max_iter` Максимальное количество итераций, необходимое для сходимости решателей",
                            10,
                            1000,
                            100,
                            key=persist("number_for_logistic_regression_max_iter"),
                        )
                    with col3:
                        l1_ratio = st.slider(
                            "`l1_ratio` Параметр смешивания Elastic-Net",
                            0.0,
                            1.0,
                            1.0,
                            step=0.01,
                            key=persist("slider_for_logistic_regression_l1_ratio"),
                        )

                    model = custom_logistic_regression(
                        C_c=C, max_iter_c=max_iter, l1_ratio_c=l1_ratio
                    )
                else:
                    model = LogisticRegression()
            if options_clf == "Decision Tree CLassifier":
                options_decision_tree_classification = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_decision_tree_classification"),
                )
                if options_decision_tree_classification == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        max_depth = st.number_input(
                            "`max_depth` Максимальная глубина дерева",
                            1,
                            100,
                            3,
                            key=persist(
                                "number_for_decision_tree_classification_max_depth"
                            ),
                        )
                    with col2:
                        min_samples_split = st.number_input(
                            "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                            2,
                            100,
                            2,
                            key=persist(
                                "number_for_decision_tree_classification_min_samples_split"
                            ),
                        )
                    col3, col4 = st.columns(2)
                    with col3:
                        min_samples_leaf = st.number_input(
                            "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                            1,
                            100,
                            1,
                            key=persist(
                                "number_for_decision_tree_classification_min_samples_leaf"
                            ),
                        )
                    with col4:
                        ccp_alpha = st.slider(
                            "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                            0.0,
                            1.0,
                            0.0,
                            step=0.01,
                            key=persist("slider_decision_tree_classification_ccp_alpha"),
                        )
                    criterion_for_decision_tree_classification = [
                        "gini",
                        "entropy",
                        "log_loss",
                    ]
                    criterion = st.selectbox(
                        "`criterion` Функция для измерения качества разделения",
                        criterion_for_decision_tree_classification,
                        key=persist("select_criterion_decision_tree_classification"),
                    )

                    model = custom_decision_tree_classification(
                        criterion_c=criterion,
                        max_depth_c=max_depth,
                        min_samples_split_c=min_samples_split,
                        min_samples_leaf_c=min_samples_leaf,
                        ccp_alpha_c=ccp_alpha,
                    )
                else:
                    model = DecisionTreeClassifier()

            elif options_clf == "Random Forest Classification":
                options_random_forest_classification = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_random_forest_classification"),
                )
                if options_random_forest_classification == "Выбрать параметры вручную":
                    n_estimators = st.slider(
                        "`n_estimators` Количество деревьев в лесу",
                        50,
                        1000,
                        100,
                        key=persist("number_for_random_forest_classification_n_estimators"),
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        max_depth = st.number_input(
                            "`max_depth` Максимальная глубина каждого дерева",
                            1,
                            100,
                            3,
                            key=persist(
                                "number_for_random_forest_classification_max_depth"
                            ),
                        )
                    with col2:
                        min_samples_split = st.number_input(
                            "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                            2,
                            100,
                            2,
                            key=persist(
                                "number_for_random_forest_classification_min_samples_split"
                            ),
                        )
                    col3, col4 = st.columns(2)
                    with col3:
                        min_samples_leaf = st.number_input(
                            "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                            1,
                            100,
                            1,
                            key=persist(
                                "number_for_random_forest_classification_min_samples_leaf"
                            ),
                        )
                    with col4:
                        ccp_alpha = st.slider(
                            "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                            0.0,
                            1.0,
                            0.0,
                            step=0.01,
                            key=persist("slider_random_forest_classification_ccp_alpha"),
                        )
                    criterion_for_random_forest_classification = [
                        "gini",
                        "entropy",
                        "log_loss",
                    ]
                    criterion = st.selectbox(
                        "`criterion` Функция для измерения качества разделения",
                        criterion_for_random_forest_classification,
                        key=persist("select_criterion_random_forest_classification"),
                    )

                    model = custom_random_forest_classification(
                        n_estimators,
                        criterion,
                        max_depth,
                        min_samples_split,
                        min_samples_leaf,
                        ccp_alpha,
                    )
                else:
                    model = RandomForestClassifier()

            elif options_clf == "Support Vector Machine CLassification":
                options_svm_classification = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_svm_classification"),
                )
                if options_svm_classification == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        kernel = st.selectbox(
                            "`kernel` Указывает тип ядра, который будет использоваться в алгоритме.",
                            ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                            key=persist("select_kernel_svm_classification"),
                        )
                    with col2:
                        degree = st.number_input(
                            "`degree` Степень полиномиальной функции ядра ('poly')",
                            1,
                            10,
                            3,
                            key=persist("number_for_svm_classification_degree"),
                        )
                    model = custom_svc(kernel, degree)
                else:
                    model = SVC()
            elif options_clf == "Gradient Boosting Classification":
                options_gbc_classification = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_gbc_classification"),
                )
                if options_gbc_classification == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        loss = st.selectbox(
                            "`loss` Функция потерь, которую необходимо оптимизировать",
                            ["log_loss", "exponential"],
                            key=persist("select_loss_gbc_classification"),
                        )
                    with col2:
                        learning_rate = st.slider(
                            "`learning_rate` Скорость обучения уменьшает вклад каждого дерева на величину learning_rate",
                            0.0,
                            1.0,
                            0.1,
                            step=0.01,
                            key=persist("number_for_gbc_classification_learning_rate"),
                        )
                    n_estimators = st.slider(
                        "`n_estimators` Количество деревьев в лесу",
                        50,
                        1000,
                        100,
                        key=persist("number_for_gbc_classification_n_estimators"),
                    )
                    col3, col4 = st.columns(2)
                    with col3:
                        max_depth = st.number_input(
                            "`max_depth` Максимальная глубина каждого дерева",
                            1,
                            100,
                            3,
                            key=persist("number_for_gbc_classification_max_depth"),
                        )
                    with col4:
                        min_samples_split = st.number_input(
                            "`min_samples_split` Минимальное количество образцов, необходимое для разбиения внутреннего узла",
                            2,
                            100,
                            2,
                            key=persist("number_for_gbc_classification_min_samples_split"),
                        )
                    col5, col6 = st.columns(2)
                    with col5:
                        min_samples_leaf = st.number_input(
                            "`min_samples_leaf` Минимальное количество образцов, необходимое для нахождения в узле листа",
                            1,
                            100,
                            1,
                            key=persist("number_for_gbc_classification_min_samples_leaf"),
                        )
                    with col6:
                        ccp_alpha = st.slider(
                            "`ccp_alpha` Параметр сложности, используемый для обрезки по принципу минимальная стоимость - сложность",
                            0.0,
                            1.0,
                            0.0,
                            step=0.01,
                            key=persist("slider_gbc_classification_ccp_alpha"),
                        )
                    criterion_for_random_forest_classification = [
                        "friedman_mse",
                        "squared_error",
                    ]
                    criterion = st.selectbox(
                        "`criterion` Функция для измерения качества разделения",
                        criterion_for_random_forest_classification,
                        key=persist("select_criterion_gbc_classification"),
                    )

                    model = custom_gbc(
                        loss_c=loss,
                        learning_rate_c=learning_rate,
                        n_estimators_c=n_estimators,
                        criterion_c=criterion,
                        max_depth_c=max_depth,
                        min_samples_split_c=min_samples_split,
                        min_samples_leaf_c=min_samples_leaf,
                        ccp_alpha_c=ccp_alpha,
                    )
                else:
                    model = GradientBoostingClassifier()
            else:
                options_mlp_classification = st.radio(
                    "Сделайте выбор:",
                    ["Выбрать параметры вручную", "Выбрать дефолтную модель"],
                    key=persist("choose_parameters_for_mlpc"),
                )
                if options_mlp_classification == "Выбрать параметры вручную":
                    col1, col2 = st.columns(2)
                    with col1:
                        activation = st.selectbox("`activation` Функция активации для скрытого слоя", ["relu", "logistic", "tanh", "identity"], key=persist("select_activation_mlpc"))
                    with col2:
                        solver = st.selectbox("`solver` Решающая программа для оптимизации веса", ["adam", "lbfgs", "sgd"],  key=persist("select_solver_mlpc"))
                    col1, col2 = st.columns(2)
                    with col1:
                        number_layers = st.slider("`hidden_layer` Это количество скрытых нейронов в слое", 32, 256, 1, key=persist("select_layers_mlpc"))
                    model = custom_mlp_classifier(activation_func_c=activation, solver_c=solver, number_layers_c=number_layers)
                else:
                    model = MLPClassifier()
                
                
        st.write(model)

        color_theme_list = [
            "Blues",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "Purples",
            "Greens",
            "Oranges",
            "Reds",
            "YlOrBr",
            "YlOrRd",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
        ]
        show_results_visualizations = False

        if st.checkbox("Обучить и проверить модель", key=persist("checkbox_model_see")):
            show_results_visualizations = True
            model, pred = train_and_predict(model, X_train, y_train, X_test)
            if type_of_task_radio == "Регрессия":
                r2 = r2_score(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                regression_score = pd.DataFrame(
                    {"R_2": r2, "Mean Absolute Error": mae, "Mean Squared Error": mse},
                    index=[0],
                )
                st.dataframe(regression_score, hide_index=True)
            else:
                acc = accuracy_score(y_test, pred)
                if y_test.nunique()!=2:
                    f1 = f1_score(y_test, pred, average='micro')
                    precision = precision_score(y_test, pred, average='micro')
                    recall = recall_score(y_test, pred, average='micro')
                else:
                    f1 = f1_score(y_test, pred)
                    precision = precision_score(y_test, pred)
                    recall = recall_score(y_test, pred)
                classification_score = pd.DataFrame(
                    {
                        "Accuracy": acc,
                        "F1_score": f1,
                        "Precision": precision,
                        "Recall": recall,
                    },
                    index=[0],
                )
                st.write("Результаты:")
                st.dataframe(classification_score, hide_index=True)

        # Assuming your model is named `model` and is already trained
        filename = "Completed_model.joblib"
        joblib.dump(model, filename)  # Save the model to a file

        # Open the file in binary mode to pass to the download button
        with open(filename, "rb") as file:
            st.download_button(
                label="Cкачать готовую модель",
                data=file,
                file_name=filename,
                mime="application/octet-stream",
            )

        st.subheader("3.Визуализация результатов")
        if show_results_visualizations:
            if type_of_task_radio == "Регрессия":
                color = st.color_picker(
                    "Выберите цвет", "#00f900", key=persist("color_for_result")
                )
                regression_result_visualization(y_test, pred, color)
            else:
                color = st.selectbox(
                    "Выберите схему цветов для визуализации", color_theme_list
                )
                confusion_matrix_visualization(y_test, pred, color)
        else:
            st.info("Сначала нужно обучить модель и получить результаты")


_PERSIST_STATE_KEY = f"{__name__}_PERSIST"


def persist(key: str) -> str:
    """Mark widget state as persistent."""
    if _PERSIST_STATE_KEY not in _state:
        _state[_PERSIST_STATE_KEY] = set()

    _state[_PERSIST_STATE_KEY].add(key)

    return key


initial_state = {"uploaded": True}

initial_state_data = {"uploaded": False}


def load_widget_state():
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in _state:
        _state.update(
            {
                key: value
                for key, value in _state.items()
                if key in _state[_PERSIST_STATE_KEY]
            }
        )


def reset_application_state():
    global _state
    # Сохранение значений, которые не должны быть удалены
    preserved_values = {
        key: _state[key] for key in ["data", "original_data"] if key in _state
    }

    # Очистка текущего состояния
    _state.clear()

    # Загрузка исходного состояния
    _state.update(
        initial_state.copy()
    )  # Используем .copy() для избежания изменений в исходном словаре

    # Восстановление сохраненных значений
    _state.update(preserved_values)
    st.rerun()


def reset_application_state_with_data():
    global _state

    _state.update(
        initial_state_data.copy()
    )  # Используем .copy() для избежания изменений в исходном словаре

    _state.clear()

    st.rerun()


def upload_file():
    """
    Presents a file uploader widget and returns the uploaded file object.

    This function creates a file uploader in a Streamlit app with the specified prompt message.
    Users can upload a file through the UI. If a file is uploaded, the function returns the
    file object provided by Streamlit, allowing further processing of the file. If no file is
    uploaded, the function returns None.

    Returns:
    - uploaded_file (UploadedFile or None): The file object uploaded by the user through the
      file uploader widget. The object contains methods and attributes to access and read the file's
      content. Returns None if no file has been uploaded.

    Example usage:
    ```
    uploaded_file = upload_file()
    if uploaded_file is not None:
        # Process the file
        st.write("File uploaded:", uploaded_file.name)
    else:
        st.write("No file uploaded.")
    ```
    """
    uploaded_file = st.file_uploader("Загрузите файл")
    if uploaded_file is not None:
        return uploaded_file
    return None


def check_nulls(df):
    """
    Checks and summarizes the number and percentage of missing (null) values in each column of a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame to be analyzed for null values.

    Returns:
    - pd.DataFrame: A DataFrame with two columns: 'amount' which represents the count of null values per column,
      and 'percentage' which shows the percentage of null values per column, rounded to four decimal places and
      multiplied by 100 to convert to percentage format.
    """

    df_null = pd.DataFrame(
        {
            "Missing Values": df.isna().sum().values,
            "% of Total Values": 100 * df.isnull().sum() / len(df),
        },
        index=df.isna().sum().index,
    )
    df_null = df_null.sort_values("% of Total Values", ascending=False).round(2)
    # print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"
    # "There are " + str(df_null.shape[0]) + " columns that have missing values.")
    return df_null


def delete_columns(data, columns):
    """
    Removes specified columns from a DataFrame.

    This function takes a DataFrame and a list of column names to be removed. It returns a new DataFrame with the specified
    columns removed, leaving the original DataFrame unchanged.

    Parameters:
    - data (pd.DataFrame): The original DataFrame from which columns will be removed.
    - columns (list of str): A list of strings representing the names of the columns to be removed.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    dataframe = data.drop(columns, axis=1)
    return dataframe


def fill_numerical_data(data, columns):
    """
    Fills missing values in specified numerical columns with the median of each column.

    This function iterates over a list of columns in a DataFrame, filling in missing (NaN) values in those columns with
    the median of each respective column. The changes are made in place, but the DataFrame is also returned for convenience
    and chaining operations.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the columns to be filled.
    - columns (list of str): A list of column names (strings) in the DataFrame that are to be filled. These should be columns with numerical data.

    Returns:
    - pd.DataFrame: The original DataFrame with missing values in the specified columns filled with their respective medians.
    """
    for col in columns:
        data[col].fillna(data[col].median(), inplace=True)
    return data


def fill_categorical_data(data, columns):
    """
    Fills missing values in specified categorical columns with the most frequent value of each column.

    Utilizes the SimpleImputer class from sklearn.impute to fill missing (NaN) values in the DataFrame's categorical columns
    with the most frequent value (mode) found in each column. This operation updates the DataFrame in place and returns it
    for potential chaining of operations or further modifications.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the columns to be filled. It's expected to be modified in place.
    - columns (list of str): A list of column names (strings) that are to be treated, which should correspond to categorical data.

    Returns:
    - pd.DataFrame: The original DataFrame with missing values in the specified columns filled with their most frequent value.

    Note:
    - This function requires the sklearn library for the SimpleImputer class. Ensure sklearn is installed and imported correctly.
    """
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="most_frequent")
    for col in columns:
        data[col] = imputer.fit_transform(data[[col]]).ravel()
    return data


def create_toggle(session_name: str, name: str, status=False) -> bool:
    """
    Creates a toggle switch in a Streamlit app and manages its state across reruns.

    This function displays a toggle switch with the label specified by the 'name' parameter.
    It uses Streamlit's session state to keep track of the toggle's position (on/off) across reruns.
    The current state of the toggle is stored in `st.session_state` using a key provided by the
    'session_name' parameter.

    Parameters:
    - session_name (str): The key used to store the toggle's state in `st.session_state`.
                          This key should be unique to each toggle to prevent state conflicts.
    - name (str): The label displayed next to the toggle switch in the UI.

    Returns:
    - bool: The current state of the toggle (True for on, False for off).

    Example:
    ```
    delete_column = create_toggle('delete_status', 'Удалить')
    if delete_column:
        st.write("Toggle is ON - deletion logic can be placed here.")
    else:
        st.write("Toggle is OFF - no deletion occurs.")
    ```
    """
    # Initialize toggle status in session_state if it doesn't exist
    if session_name not in st.session_state:
        st.session_state[session_name] = False

    # Display the toggle and assign its current value based on session_state
    toggle = st.toggle(name, value=st.session_state[session_name], disabled=status)

    # Update session state based on the toggle's position
    if toggle != st.session_state[session_name]:
        st.session_state[session_name] = toggle
        st.rerun()
    return toggle


def apply_one_hot_encoder(df, columns):
    dataframe = pd.get_dummies(df, columns=columns, drop_first=True)
    return dataframe


def apply_label_encoder(df, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df


def apply_ordinal_encoder(df, columns):
    ordinal_encoder = OrdinalEncoder()
    for column in columns:
        # Ensure the column is in a 2D array format and fit_transform
        # Also, handling the case where column data might not be numeric
        df[column] = ordinal_encoder.fit_transform(df[[column]])
    return df


def reset_all_parameters_selection():
    st.session_state["all_parameters_selected"] = False


def apply_min_max_scaling(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def apply_standardization(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def apply_log_transformation(df, columns):
    # Apply log transformation with a small constant to avoid log(0)
    for column in columns:
        df[column] = np.log(df[column] + 1)
    return df


# def line_chart(data, x_axis, y_axis, plot_title=None, hue=None):
#     fig = px.line(data, x=x_axis, y=y_axis, title=plot_title, color=hue)
#     st.plotly_chart(fig, use_container_width=True)

# def scatter_plot(data, x_axis, y_axis, plot_title=None, hue=None):
#     fig = px.scatter(data, x=x_axis, y=y_axis, title=plot_title, color=hue)
#     st.plotly_chart(fig, use_container_width=True)


def line_chart(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the circles based on the hue column and the input color theme.
    # If hue is None, all circles will be colored red.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    c = (
        alt.Chart(data)
        .mark_line()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(c, use_container_width=True)


def scatter_plot(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the circles based on the hue column and the input color theme.
    # If hue is None, all circles will be colored red.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    c = (
        alt.Chart(data)
        .mark_circle()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(c, use_container_width=True)


import altair as alt
import streamlit as st


def histogram(data, x_axis, input_color_theme, hue=None):
    # If hue is provided, color the bars based on the hue column and the input color theme.
    # If hue is None, all bars will be colored with the input color theme.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(x=alt.X(x_axis, bin=True), y="count()", color=color_encoding)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def box_plot(data, x_axis, y_axis, input_color_theme, hue=None):
    # If hue is provided, color the box plot based on the hue column and the input color theme.
    # If hue is None, all plots will be colored with the input color theme.
    if hue:
        color_encoding = alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
    else:
        color_encoding = alt.value(
            input_color_theme
        )  # Directly set the color to red if hue is not provided

    chart = (
        alt.Chart(data)
        .mark_boxplot()
        .encode(x=x_axis, y=y_axis, color=color_encoding)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def density_plot(data, x_axis, input_color_theme, hue=None):
    # Start with the base chart
    chart = alt.Chart(data)

    # Apply the density transformation conditionally based on hue
    if hue:
        # If hue is provided, calculate density with grouping
        chart = chart.transform_density(
            density=x_axis,
            as_=[x_axis, "density"],
            groupby=[hue],  # Ensure this is a list
        )
    else:
        # If hue is not provided, calculate density without grouping
        chart = chart.transform_density(density=x_axis, as_=[x_axis, "density"])

    # Apply the rest of the encoding
    chart = chart.mark_area().encode(
        x=f"{x_axis}:Q",
        y="density:Q",
        color=(
            alt.Color(hue, scale=alt.Scale(scheme=input_color_theme))
            if hue
            else alt.value(input_color_theme)
        ),
    )

    st.altair_chart(chart, use_container_width=True)


def custom_linear_regression():
    model = LinearRegression()
    return model


def custom_logistic_regression(C_c=1, n_jobs_c=-1, max_iter_c=100, l1_ratio_c=None):
    model = LogisticRegression(
        C=C_c, n_jobs=n_jobs_c, max_iter=max_iter_c, l1_ratio=l1_ratio_c
    )
    return model


def custom_decision_tree_regression(
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeRegressor(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_decision_tree_classification(
    criterion_c="gini",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeClassifier(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_train_test_split(X, y, size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    return X_train, X_test, y_train, y_test


def confusion_matrix_visualization(y_true, y_pred, input_color="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap=input_color, cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    st.pyplot(fig)


def regression_result_visualization(y_true, y_pred, input_color):
    # Create a lmplot with seaborn. 'palette' should be a dictionary mapping levels of the hue variable to matplotlib colors.
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, c=input_color)

    st.pyplot(fig)


def correlation_matrix_visualize(data, color):
    fig = plt.figure(figsize=(15, 15))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap=color, fmt=".2f")
    plt.title("Correlation of Features")

    st.pyplot(fig)


def train_and_predict(model, X_train, y_train, X_test):
    """
    Trains a model and predicts outcomes for the given test data.
    This function uses Streamlit's memoization to cache the model training and prediction steps,
    improving performance for repeated calls with unchanged data.

    Args:
        model: The machine learning model to be trained. Must be hashable by Streamlit.
        X_train (pd.DataFrame or np.ndarray): Training data features.
        y_train (pd.Series or np.ndarray): Training data labels/targets.
        X_test (pd.DataFrame or np.ndarray): Test data features.

    Returns:
        A tuple containing:
        - model: The trained machine learning model.
        - pred: Predictions made by the model on X_test.
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred


def custom_random_forest_regression(
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_random_forest_classification(
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_svr(kernel_c="rbf", degree_c=3):
    model = SVR(kernel=kernel_c, degree=degree_c)
    return model


def custom_svc(kernel_c="rbf", degree_c=3):
    model = SVC(kernel=kernel_c, degree=degree_c)
    return model


def custom_gbc(
    loss_c="log_loss",
    learning_rate_c=0.1,
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = GradientBoostingClassifier(
        loss=loss_c,
        learning_rate=learning_rate_c,
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_gbr(
    loss_c="squared_error",
    learning_rate_c=0.1,
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=2,
    ccp_alpha_c=0.0,
):
    model = GradientBoostingRegressor(
        loss=loss_c,
        learning_rate=learning_rate_c,
        n_estimators=n_estimators_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model


def custom_mlp_classifier(
    activation_func_c="relu",
    solver_c="adam",
    number_layers_c=100
):
    model = MLPClassifier(
        activation=activation_func_c,
        solver=solver_c,
        hidden_layer_sizes=number_layers_c,
    )
    return model

def custom_mlp_regressor(
    activation_func_c="relu",
    solver_c="adam",
):
    model = MLPRegressor(
        activation=activation_func_c,
        solver=solver_c
    )
    return model


if __name__ == "__main__":
    load_widget_state()
    main()
