from packages import *


def DataPreprocessing():
    # Data input and Log10 Transformation
    df1 = pd.read_csv(r'Data\regime_I.csv', names=['x', 'y'], skiprows=1)
    df1l = df1.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
    df1l['Regime'] = 1

    df2 = pd.read_csv(r'Data\regime_II.csv', names=['x', 'y'], skiprows=1)
    df2l = df2.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
    df2l['Regime'] = 2

    df3 = pd.read_csv(r'Data\regime_III.csv', names=['x', 'y'], skiprows=1)
    df3l = df3.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
    df3l['Regime'] = 3

    df4 = pd.read_csv(r'Data\regime_IV.csv', names=['x', 'y'], skiprows=1)
    df4l = df4.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
    df4l['Regime'] = 4

    df5 = pd.read_csv(r'Data\regime_V.csv', names=['x', 'y'], skiprows=1)
    df5l = df5.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)
    df5l['Regime'] = 5

    dfl = pd.concat([df1l, df2l, df3l, df4l, df5l])
    return dfl


def ModelPrediction():
    # Shuffle DataInput
    dfs = shuffle(DataPreprocessing())


    # Splitting DataInput  20%Test/80%Train
    train, test = train_test_split(dfs, test_size=0.20)

    # Set TrainData and TestData
    train_x = train[['x', 'y']]
    train_y = train[['Regime']]

    test_x = test[['x', 'y']]
    test_y = test[['Regime']]

    # Train the Model
    classifier = svm.SVC(kernel="rbf", C=10000, gamma=0.8)
    classifier.fit(train_x, train_y)

    # Get support vector indices
    support_vector_indices = classifier.support_
    print(support_vector_indices)

    # Get number of support vectors per class
    support_vectors_per_class = classifier.n_support_
    print(support_vectors_per_class)

    # Prediction Model
    y_pred = classifier.predict(test_x)
    y_pred = y_pred.reshape(21, 1)

    # Prediction Results
    print(confusion_matrix(test_y, y_pred))
    print(classification_report(test_y, y_pred))
    print("A total of %d samples out of %d have been assigned to a wrong class." % (
    (test_y != y_pred).sum(), len(test_y)), end='  ')
    print("Accuracy = %.1f%%" % ((100.0 * (test_y == y_pred).sum()) / len(test_y)))

    # Plot the decision boundary
    h = .02

    x_min, x_max = train_x.values[:, 0].min() - 1, train_x.values[:, 0].max() + 1
    y_min, y_max = train_x.values[:, 1].min() - 1, train_x.values[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SVC with rbf Kernel')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.show()

    return plt.show()
