import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sn

columns = ["SibSp", "Parch", "Fare", "Age"]


def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)

    for col in columns:
        data[col].fillna(data[col].median(), inplace=True)

    data.Embarked.fillna("U", inplace=True)
    return data


def my_get_dummies(data):
    sex_dummies = pd.get_dummies(data.Sex)
    sex_dummies = sex_dummies.drop(["female"], axis=1)
    embarked_dummies = pd.get_dummies(data.Embarked)
    contains_U = False
    for name in embarked_dummies:
        if name == "U":
            contains_U = True
    if contains_U:
        embarked_dummies = embarked_dummies.drop(["U"], axis=1)
    data = data.drop(["Sex", "Embarked"], axis=1)
    return data, sex_dummies, embarked_dummies


if __name__ == "__main__":
    data_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    data_df = clean(data_df)
    test_df = clean(test_df)

    data_df, data_df_sex_dummies, data_df_embarked_dummies = my_get_dummies(data_df)
    test_df, test_df_sex_dummies, test_df_embarked_dummies = my_get_dummies(test_df)

    final_data_df = pd.concat([data_df, data_df_sex_dummies, data_df_embarked_dummies], axis=1)
    final_test_df = pd.concat([test_df, test_df_sex_dummies, test_df_embarked_dummies], axis=1)

    print(final_data_df)
    print(final_test_df)

    column_names = list(final_data_df.columns)
    divisors = {'Survived': 1.01, 'Pclass': 3, 'Age': 80, 'SibSp': 8, 'Parch': 9, 'Fare': 512, 'male': 1, 'C': 1,
                'Q': 1, 'S': 1}
    for name in column_names:
        if divisors[name] != 1:
            final_data_df[name] = final_data_df[name] / divisors[name]
    print(final_data_df.head(10))

    column_names.remove('Survived')
    for name in column_names:
        if divisors[name] != 1:
            final_test_df[name] = final_test_df[name] / divisors[name]

    x_train = final_data_df.drop("Survived", axis=1)
    y_train = final_data_df.Survived
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_test = final_test_df
    x_test = x_test.to_numpy()

    print('Creating model')
    model = keras.Sequential([
        keras.layers.Dense(100, input_shape=(9,), activation="sigmoid"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    print('Specifying model parameters')
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["accuracy", "categorical_accuracy"]
    )
    model.fit(x_train, y_train, epochs=10)

    y_predicted = model.predict(x_test)
    y_sharpened = []
    answer = pd.DataFrame(range(892, 1310), columns=["PassengerId"], dtype="int")
    for i in y_predicted:
        if i < 0.5:
            y_sharpened.append(0)
        else:
            y_sharpened.append(1)

    y_sharpened_df = pd.DataFrame(y_sharpened, columns=['Survived'])

    answer_df = pd.concat([answer, y_sharpened_df], axis=1)
    print(answer_df)
    answer_df.to_csv("answer.csv", index=False)