from sklearn.svm import SVC
from data import generate_xor_data
from sklearn.metrics import accuracy_score
from visual import plot_2d_data
def main():
    #Generate XOR data
    X,y = generate_xor_data(n=200)
    plot_2d_data(X,y,title="XOR Data(Original Space)")

    #Linear model without feature transformation
    linear_svm=SVC(kernel='linear')
    linear_svm.fit(X,y)
    y_pred_linear = linear_svm.predict(X)
    plot_2d_data(X,y_pred_linear,title="Linear SVM Predictions (Fails)")

    #polynomial kenel svm
    poly_svm = SVC(kernel='poly',degree=2)
    poly_svm.fit(X,y)
    y_pred_poly=poly_svm.predict(X)
    plot_2d_data(X,y_pred_poly,title="Polynomial Kernel SVM predictions(succeeds)")

    #RBF Kernel SVM
    rbf_svm = SVC(kernel='rbf',gamma='scale')
    rbf_svm.fit(X,y)
    y_pred_rbf = rbf_svm.predict(X)
    plot_2d_data(X,y_pred_rbf,title="RBF Kernel SVM Predictions (Succeeds)")

    #print accuracies
    from sklearn.metrics import accuracy_score
    print('Linear SVM accuracy:',accuracy_score(y,y_pred_linear))
    print('POlynomial Kernel SVM Accuracy:',accuracy_score(y,y_pred_poly))
    print('RBF Kernel SVM Accuracy:',accuracy_score(y,y_pred_rbf))
if __name__=="__main__":
    main()