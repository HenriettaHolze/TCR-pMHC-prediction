from insert_embeddings import embeddings_main
from sklearn.decomposition import PCA
import numpy as np



X_train, X_val = embeddings_main()

X_train = X_train[:,179:,:]
X_val = X_val[:,179:,:]

emb_train = X_train[:,:,54:]
print(emb_train.shape)

emb_val = X_val[:,:,54:]

print(emb_val[:10,:10, :2])
emb_train = emb_train.reshape((emb_train.shape[0]*emb_train.shape[1], -1))
emb_val = emb_val.reshape((emb_val.shape[0]*emb_val.shape[1], -1))
print(emb_train.shape)
print(emb_val[:100, :2])
model = PCA(50, whiten=True)

emb_train = model.fit_transform(emb_train)

emb_val = model.transform(emb_val)


print(emb_train.shape)
print(emb_val[:100, :2])
print(emb_val.shape)

emb_val = emb_val.reshape((X_val.shape[0], X_val.shape[1] , 50))
emb_train = emb_train.reshape((X_train.shape[0], X_train.shape[1], 50))
print(emb_val[:10,:10, :2])
print(emb_val.shape)
print(emb_train.shape)

print(np.sum(model.explained_variance_))

X_train = np.concatenate([X_train[:,:,:54], emb_train], 2)
X_val = np.concatenate([X_val[:,:,:54], emb_val], 2)

np.savez_compressed('data/X_train_pca', X_train)
np.savez_compressed('data/X_val_pca', X_val)

print(X_train.shape, X_val.shape)



