import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, classification_report


def train_logreg_aux_emissions(train_inputs, test_inputs, train_aux_emissions, test_aux_emissions, train_stateseq, test_stateseq, num_z, input_mask_by_auxemission):
    print('Fitting LogReg on aux emissions...')

    def train_logreg(x, y):
        model = LogisticRegression(solver='lbfgs', C=0.1, max_iter=1000, class_weight='balanced')
        model.fit(x, y)
        weights = model.coef_[0]  # shape: (D,)
        bias = model.intercept_[0]
        return model, weights, bias

    def predict_logreg(model, x, y):
        probs = model.predict_proba(x)[:, 1]
        preds = probs >= 0.5
        f1score = f1_score(y, preds)
        print(classification_report(y, preds))


        # import matplotlib.pyplot as plt
        # from sklearn.metrics import precision_recall_curve, average_precision_score
        # precision, recall, thresholds = precision_recall_curve(y, probs)
        # ap = average_precision_score(y, probs)
        # # Plot
        # plt.figure(figsize=(6, 5))
        # plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        return probs, f1score, preds

    train_inputs_ = np.concatenate(train_inputs, axis=0)
    test_inputs_ = np.concatenate(test_inputs, axis=0)
    train_aux_emissions_ = np.concatenate(train_aux_emissions, axis=0)
    test_aux_emissions_ = np.concatenate(test_aux_emissions, axis=0)
    train_z_seq_ = np.concatenate(train_stateseq, axis=0)
    test_z_seq_ = np.concatenate(test_stateseq, axis=0)

    aux_emission_dim = train_aux_emissions_.shape[-1]
    input_dim = train_inputs_.shape[-1]

    loss, train_probs, test_probs, train_true, test_true = {}, {}, {}, {}, {}
    train_f1score, train_preds, test_f1score, test_preds = {}, {}, {}, {}
    w = np.zeros((num_z, aux_emission_dim, input_dim))
    b = np.zeros((num_z, aux_emission_dim))
    for z in range(num_z):
        print("z =", z, "samples", train_aux_emissions_[train_z_seq_ == z][:, 0].shape, test_aux_emissions_[test_z_seq_ == z][:, 0].shape)
        loss[z] = {}
        train_probs[z] = {}
        train_f1score[z] = {}
        train_preds[z] = {}
        train_true[z] = {}
        test_probs[z] = {}
        test_f1score[z] = {}
        test_preds[z] = {}
        test_true[z] = {}
        for o in range(aux_emission_dim):
            print("o = ", o)
            mask_o = input_mask_by_auxemission[o]
            model, w[z, o, mask_o==1], b[z, o] = train_logreg(train_inputs_[train_z_seq_ == z][:, mask_o==1], train_aux_emissions_[train_z_seq_ == z][:, o])
            print('train')
            train_probs[z][o], train_f1score[z][o], train_preds[z][o] = predict_logreg(model, train_inputs_[train_z_seq_ == z][:, mask_o==1], train_aux_emissions_[train_z_seq_ == z][:, o])
            print('test')
            test_probs[z][o], test_f1score[z][o], test_preds[z][o] = predict_logreg(model, test_inputs_[test_z_seq_ == z][:, mask_o==1], test_aux_emissions_[test_z_seq_ == z][:, o])
            train_true[z][o] = train_aux_emissions_[train_z_seq_ == z][:, o]
            test_true[z][o] = test_aux_emissions_[test_z_seq_ == z][:, o]

    print('Done fitting LogReg.')
    return w, b, train_probs, test_probs, train_f1score, test_f1score, train_preds, test_preds, train_true, test_true
