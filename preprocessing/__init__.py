import os

from sklearn.model_selection import train_test_split

def load(main_args, sub_args, seed=42, test_size=.1):
    if not os.path.isfile(os.path.join(sub_args['etc']['result_root'], 
                                       '{}_trainset.txt'.format(sub_args['task']['subtask']))):

        if main_args.task == 'CDSS_Liver':
            if sub_args['task']['subtask'] == 'multi_organ':
                whole_cases = [c for c in os.listdir(os.path.join(sub_args['etc']['data_root'],
                                                                  sub_args['task']['subtask'],
                                                                  'image')) if 'hdr' in c]

            elif sub_args['task']['subtask'] == 'Liver':
                pass

            elif sub_args['task']['subtask'] == 'HCC':
                pass

            elif sub_args['task']['subtask'] == 'Vessel':
                whole_cases = [c[:-11] for c in os.listdir(os.path.join(sub_args['etc']['data_root'],
                                                                        sub_args['task']['subtask'],
                                                                        'imagesTr')) if 'hdr' in c]

            train_idx, test_idx = train_test_split(range(1, len(whole_cases)+1), test_size=test_size, random_state=seed)
            trainset = [whole_cases[x-1] for x in train_idx]
            testset = [whole_cases[x-1] for x in test_idx]
            print(testset)

        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_trainset.txt'.format(sub_args['task']['subtask'])), 'w') as f:
            for t in trainset:
                f.write(t+'\n')

        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_testset.txt'.format(sub_args['task']['subtask'])), 'w') as f:
            for t in testset:
                f.write(t+'\n')

    else:
        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_trainset.txt'.format(sub_args['task']['subtask'])), 'r') as f:
            trainset = [t.rstrip() for t in f.readlines()]
        
        with open(os.path.join(sub_args['etc']['result_root'], 
                               '{}_testset.txt'.format(sub_args['task']['subtask'])), 'r') as f:
            testset = [t.rstrip() for t in f.readlines()]

    valset = trainset[:len(testset)]
    trainset = trainset[len(testset):]
    print('# of training data :', len(trainset), '/ # of validation data :', len(valset), '/ # of test data :', len(testset))
    return trainset, valset, testset