from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, LeaveOneOut, RepeatedStratifiedKFold, StratifiedShuffleSplit

def holdout(X, y, test_size: float, random_state: int, stratify: bool = True):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

def make_kfold(n_splits: int, stratified: bool, shuffle: bool, random_state: int):
    if stratified:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

def make_loo():
    return LeaveOneOut()

def make_repeated_kfold(n_splits: int, n_repeats: int, random_state: int):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

def make_repeated_holdout(test_size: float, n_splits: int, random_state: int, stratified: bool = True):
    if stratified:
        return StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)