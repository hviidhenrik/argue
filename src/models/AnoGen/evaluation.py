

"""
Experiment with the anomaly threshold - this could be an average of the three datasets or simply the max
"""
# mse_test = np.mean(np.power(x_test_scaled - ae.predict(x_test_scaled), 2), axis=1)
# mse_list = [mse_train, mse_val, mse_test]
# for mse in mse_list:
#     print(pd.DataFrame(mse).describe())
