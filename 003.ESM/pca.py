from insert_embeddings import embeddings_main
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np



X_train, X_val, X_test = embeddings_main()


X_train = X_train[:,179:,:]
X_val = X_val[:,179:,:]
X_test = X_test[:,179:,:]

emb_train = X_train[:,:,54:]
emb_val = X_val[:,:,54:]
emb_test = X_test[:,:,54:]


print(emb_train.shape)
print(emb_val[:10,:10, :2])

emb_train = emb_train.reshape((emb_train.shape[0]*emb_train.shape[1], -1))
emb_val = emb_val.reshape((emb_val.shape[0]*emb_val.shape[1], -1))
emb_test = emb_test.reshape((emb_test.shape[0]*emb_test.shape[1], -1))

print(emb_train.shape)
print(emb_val[:100, :2])

model = PCA(100)
scaler = StandardScaler()

emb_train = scaler.fit_transform(emb_train)
emb_val = scaler.transform(emb_val)
emb_test = scaler.transform(emb_test)

emb_train = model.fit_transform(emb_train)
emb_val = model.transform(emb_val)
emb_test = model.transform(emb_test)


print(emb_train.shape)
print(emb_val[:100, :2])
print(emb_val.shape)
print(emb_test.shape)

emb_train = emb_train.reshape((X_train.shape[0], X_train.shape[1], 100))
emb_val = emb_val.reshape((X_val.shape[0], X_val.shape[1] , 100))
emb_test = emb_test.reshape((X_test.shape[0], X_test.shape[1] , 100))

print(emb_val[:10,:10, :2])
print(emb_val.shape)
print(emb_train.shape)
print(emb_test.shape)

# How much variance is explained
print(np.sum(model.explained_variance_ratio_))

X_train = np.concatenate([X_train[:,:,:54], emb_train], 2)
X_val = np.concatenate([X_val[:,:,:54], emb_val], 2)
X_test = np.concatenate([X_test[:,:,:54], emb_test], 2)

np.savez_compressed('data/X_train_pca', X_train)
np.savez_compressed('data/X_val_pca', X_val)
np.savez_compressed('data/X_test_pca', X_test)

# Check that dimensions are sensible
print(X_train.shape, X_val.shape, X_test.shape)



