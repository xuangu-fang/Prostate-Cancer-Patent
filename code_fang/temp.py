def model_test(model): # avg on the indicator
    N_collect = len(model.W_collect)
    if N_collect > 0:
        W_samples = np.zeros((N_collect,model.N_feature))
        for i in range(N_collect):
            W_samples[i,:-1] = np.concatenate((model.W_collect[i]))
            W_samples[i,-1] = model.b_collect[i]

        W_avg = np.mean(W_samples,axis=0).reshape(-1,1)


        # compute the avg indicators
        s_mean = []
        for k_id in range(len(model.z)):
            s_sample = np.stack([model.s_collect[i][k_id] for i in range(len(model.s_collect))])
            s_mean_sub = np.mean(s_sample,0)
            s_mean_sub = np.where(s_mean_sub>0.9,1,0)
            s_mean.append(s_mean_sub)

        z_sample = np.stack(model.z_collect)
        z_mean = np.mean(z_sample,0)
        z_mean = np.where(z_mean>0.9,1,0)

        indicators = np.zeros((model.N_feature))
        indicators[:-1] = np.concatenate([s_mean[i] * z_mean[i] for i in range(len(z_mean))])
        indicators[-1] = 1.0

        sparse_W = indicators.reshape(-1,1) * W_avg

        predict = np.matmul(model.X_test, sparse_W).squeeze()
        print("\n\n final test rmse = %.5f"%(np.sqrt(np.mean((predict-model.Y_test.squeeze())**2))))

        model.z_mean = z_mean
        model.s_mean = s_mean
        model.W_mean = sparse_W.squeeze() 