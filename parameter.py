# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir='/home/phan/Dataset/DCASE2020/SELD/',
        # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATH
        feat_label_dir='./feat_label/',  # Directory to dump extracted features and labels
        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        # DNN MODEL PARAMETERS
        label_sequence_length=60,        # Feature sequence length
        batch_size=64,              # Batch size

        rnn_nb_layer=1,             # number of rrn layers
        rnn_hidden_size = 256,      # size of hidden layer
        attention_size = 64,
        dnn_size = 512,
        loss_weights=[1., 1.],      # [sed, doa] weight for scaling the DNN outputs, accounting for doa range [-1, 1] while sed range [0,1] and doa has 3 times more elements

        l2_reg_lambda= 0,
        dropout_keep_prob_rnn=0.9,
        dropout_keep_prob_cnn=0.5,  # stronger augmentation
        dropout_keep_prob_dnn=0.75,
        
        #METRIC PARAMETERS
        lad_doa_thresh=20
       
    )
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling

    params['unique_classes'] = {
            'alarm': 0,
            'baby': 1,
            'crash': 2,
            'dog': 3,
            'engine': 4,
            'female_scream': 5,
            'female_speech': 6,
            'fire': 7,
            'footsteps': 8,
            'knock': 9,
            'male_scream': 10,
            'male_speech': 11,
            'phone': 12,
            'piano': 13
        }


    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '3':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '4':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '5':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
