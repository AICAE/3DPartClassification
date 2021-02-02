#!/usr/bin/python3
"""
https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/

"""

def my_split(df, images, class_col_name, train_folder_name = None, ratio=0.8, shuffling=True):
    """
    parameters:
        df: pandas.DataFrame
        images: numpy.ndaray, has the same order as df
        class_col_name: df column name for the target in classification
        train_folder_name: None,  can be "train", subcategories (file path) contain this word is train samples
        ratio: 0.8 

    return:
        train, train_images, test, test_images
    """
    import pandas as pd
    #import numpy as np

    # add Index to dataframe, then group, get the unique classes
    df = df.reset_index()
    df['orig_index'] = df.index   # images has the same order as df
    df.groupby([class_col_name])
    group_values = df[class_col_name].unique()  # return value list? 
    group_count = len(group_values)

    train = pd.DataFrame()
    test = pd.DataFrame()
    gi = 0
    for val in group_values:
        group_data = df[df[class_col_name] == val]
        # shuffling?
        group_size = group_data.shape[0]
        if train_folder_name:
            # conversion from dtype of object of list
            train_g = []
            test_g = []
            for i, v in group_data["subcategories"].iteritems():
                if v == [train_folder_name]:
                    train_g.append(i)
                else:
                    test_g.append(i)
            train = pd.concat([train, df.iloc[train_g, :]])
            test = pd.concat([test, df.iloc[test_g, :]])
        else: # using ratio is split
            train_size = int(group_size * ratio)
            print("group name = {}, group size = {}, train_size={}".format(val, group_size, train_size))
            if train_size >= 1 and group_size > 10:
                if gi == 0:
                    train = group_data.iloc[:train_size]
                    test = group_data.iloc[train_size:]
                else:
                    train = train.append(group_data.iloc[:train_size])  # generate a new DF,  python list
                    test = test.append(group_data.iloc[train_size:])  # append will generat a new DF
                gi += 1
            else:
                print("WARN: group size is too small to split, skip this group")
    
    print(train.columns, train.shape)
    if shuffling:
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        # drop=True prevents .reset_index from creating a column containing the old index entries.
    train_images = images[train["orig_index"]]
    test_images = images[test["orig_index"]]
    return train,  test, train_images, test_images


def stratify(data, classes, ratios, one_hot=False):
    """Stratifying procedure.

    data is a list of lists: a list of labels, for each sample.
        Each sample's labels should be ints, 
        if they are one-hot encoded, use one_hot=True
    
    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    """
    # one-hot decoding
    if one_hot:
        temp = [[] for _ in range(len(data))]
        indexes, values = np.where(np.array(data).astype(int) == 1)
        for k, v in zip(indexes, values):
            temp[k].append(v)
        data = temp

    # Organize data per label: for each label l, per_label_data[l] contains the list of samples
    # in data which have this label
    per_label_data = {c: set() for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].add(i)

    # number of samples
    size = len(data)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios]
    target_subset_sizes = deepcopy(subset_sizes)
    per_label_subset_sizes = {
        c: [r * len(per_label_data[c]) for r in ratios]
        for c in classes
    }

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))]

    # For each sample in the data set
    while size > 0:
        # Compute |Di|
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        }
        try:
            # Find label of smallest |Di|
            label = min(
                {k: v for k, v in lengths.items() if v > 0}, key=lengths.get
            )
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error. 
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to format your data accordingly.
            break
        current_length = lengths[label]

        # For each sample with label `label`
        while per_label_data[label]:
            # Select such a sample
            current_id = per_label_data[label].pop()

            subset_sizes_for_label = per_label_subset_sizes[label]
            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(
                subset_sizes_for_label == np.amax(subset_sizes_for_label)
            ).flatten()

            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need
            # of any label
            else:
                largest_subsets = np.argwhere(
                    subset_sizes == np.amax(subset_sizes)
                ).flatten()
                if len(largest_subsets) == 1:
                    subset = largest_subsets[0]
                else:
                    # If there is more than one such subset, choose at random
                    subset = np.random.choice(largest_subsets)

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is one fewer sample to distribute
            size -= 1
            # The selected subset needs one fewer sample
            subset_sizes[subset] -= 1

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
            
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for l, label_data in per_label_data.items():
                if current_id in label_data:
                    label_data.remove(current_id)

    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    stratified_data = [
        [data[i] for i in strat] for strat in stratified_data_ids
    ]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset
    return stratified_data_ids, stratified_data