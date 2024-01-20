import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

# تولید داده با استفاده از یک معادله درجه چهارم
np.random.seed(0)
X = np.sort(15 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# افزودن توان دوم، سوم و چهارم به داده‌ها
X_poly = np.column_stack((X, X**2, X**3, X**4))

# تقسیم داده به دو بخش train و test
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

# مدل‌های مختلف با توان‌های مختلف
degrees = [1, 2, 4, 8, 16]
train_errors = []
test_errors = []

plt.figure(figsize=(15, 5))

# آموزش و ارزیابی مدل‌ها
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    # محاسبه خطا برای داده train
    y_train_pred = model.predict(X_train)
    train_error = mean_squared_error(y_train, y_train_pred)
    train_errors.append(train_error)

    # محاسبه خطا برای داده test
    y_test_pred = model.predict(X_test)
    test_error = mean_squared_error(y_test, y_test_pred)
    test_errors.append(test_error)

    # رسم نمودار تطابق مدل با داده train
    plt.subplot(1, len(degrees), i + 1)
    plt.scatter(X_train[:, 0], y_train, color='blue', s=30, label='Training Data')
    plt.scatter(X_test[:, 0], y_test, color='cyan', s=30, label='Testing Data')
    plt.plot(X[:, 0], model.predict(X_poly), color='red', label='Fitted Curve')
    plt.title(f'Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.show()

# نمودار خطا در مراحل مختلف
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='o', label='Testing Error')
plt.title('Training and Testing Errors for Different Polynomial Degrees')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.legend()
plt.show()

# نمودار learning curve
plt.figure(figsize=(10, 6))
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    train_sizes, train_scores, test_scores = learning_curve(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    train_mse = -np.mean(train_scores, axis=1)
    test_mse = -np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_mse, marker='o', label=f'Training (Degree {degree})')
    plt.plot(train_sizes, test_mse, marker='o', label=f'Testing (Degree {degree})')

plt.title('Learning Curve for Different Polynomial Degrees')
plt.xlabel('Training Size')
plt.ylabel('Mean Squared Error')
plt.yscale('log')
plt.legend()
plt.show()
