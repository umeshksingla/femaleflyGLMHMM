import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score


def train_aux_emissions(inputs, aux_emissions, z_seqs, num_z):

    def train_logreg(x, y):
        # print(x.shape, y.shape)
        model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
        model.fit(x, y)
        probs = model.predict_proba(x)[:, 1]  # probability of class 1
        preds = (probs >= 0.5).astype(int)
        loss = log_loss(y, probs)
        acc = accuracy_score(y, preds)
        weights = model.coef_[0]  # shape: (D,)
        bias = model.intercept_[0]
        return loss, probs, acc, weights, bias

    inputs_ = np.concatenate(inputs, axis=0)
    aux_emissions_ = np.hstack([np.concatenate(aux_emissions, axis=0), np.concatenate(aux_emissions, axis=0)])
    z_seq_ = np.concatenate(z_seqs, axis=0)

    aux_emission_dim = aux_emissions_.shape[-1]
    input_dim = inputs_.shape[-1]
    # print(inputs_.shape, aux_emissions_.shape, z_seq_.shape)

    loss, probs, accuracy, counts = {}, {}, {}, {}
    w = np.empty((num_z, aux_emission_dim, input_dim))
    b = np.empty((num_z, aux_emission_dim))
    # print(w.shape, b.shape)
    for z in range(num_z):
        loss[z] = {}
        probs[z] = {}
        accuracy[z] = {}
        counts[z] = {}
        z_mask = (z_seq_ == z)
        for o in range(aux_emission_dim):
            counts[z][o] = np.sum(aux_emissions_[z_mask][:, o])/len(aux_emissions_[z_mask][:, o])
            loss[z][o], probs[z][o], accuracy[z][o], w[z, o], b[z, o] = train_logreg(inputs_[z_mask], aux_emissions_[z_mask][:, o])

    # print(loss, probs, accuracy)
    # print(w, b, w.shape, b.shape)
    return accuracy, counts, w, b


def predict_aux_emissions(weights, inputs, aux_emissions, z_seqs, num_z):

    inputs_ = np.concatenate(inputs, axis=0)
    aux_emissions_ = np.hstack([np.concatenate(aux_emissions, axis=0), np.concatenate(aux_emissions, axis=0)])
    z_seq_ = np.concatenate(z_seqs, axis=0)
    aux_emission_dim = aux_emissions_.shape[-1]
    W, B = weights['w'], weights['b']

    def predict(x, y_true, w, b):
        # print(x.shape, y_true.shape, w.shape, b.shape)
        logits = x @ w + b  # shape (N,)
        # print(logits.shape)
        probs = expit(logits)  # sigmoid to get probabilities
        # print(probs, probs.shape)
        preds = (probs >= 0.5).astype(int)  # threshold at 0.5
        # print(preds, preds.shape)
        acc = np.mean(preds == y_true)  # compute accuracy
        return acc

    accuracy = {}
    counts = {}
    for z in range(num_z):
        accuracy[z] = {}
        counts[z] = {}
        z_mask = (z_seq_ == z)
        for o in range(aux_emission_dim):
            counts[z][o] = np.sum(aux_emissions_[z_mask][:, o]) / len(aux_emissions_[z_mask][:, o])
            accuracy[z][o] = predict(inputs_[z_mask], aux_emissions_[z_mask][:, o], W[z, o], B[z, o])
    # print("accuracy", accuracy)
    return accuracy, counts
