import joblib
s = joblib.load('scaler.joblib')
print('type:', type(s))
print('n_features_in_', getattr(s, 'n_features_in_', None))
mn = getattr(s, 'mean_', None)
print('mean_len', len(mn) if mn is not None else None)
print('feature_names_in_', getattr(s, 'feature_names_in_', None))
print('mean_:', getattr(s, 'mean_', None))
