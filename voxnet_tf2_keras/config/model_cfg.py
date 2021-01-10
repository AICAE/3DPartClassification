class_id_to_name_modelnet10 = {
    "1": "bathtub",
    "2": "bed",
    "3": "chair",
    "4": "desk",
    "5": "dresser",
    "6": "monitor",
    "7": "night_stand",
    "8": "sofa",
    "9": "table",
    "10": "toilet"
}
class_name_to_id_modelnet10 = { v : k for k, v in class_id_to_name_modelnet10.items() }
class_names_modelnet10 = set(class_id_to_name_modelnet10.values())


class_id_to_name_modelnet40 = {
'1':'airplane',
'2':'bookshelf',
'3':'chair',
'4':'desk',
'5':'glass_box',
'6':'laptop',
'7':'person',
'8':'range_hood',
'9':'stool',
'10':'tv_stand',
'11':'bathtub',
'12':'bottle',
'13':'cone',
'14':'door',
'15':'guitar',
'16':'mantel',
'17':'piano',
'18':'sink',
'19':'table',
'20':'vase',
'21':'bed',
'22':'bowl',
'23':'cup',
'24':'dresser',
'25':'keyboard',
'26':'monitor',
'27':'plant',
'28':'sofa',
'29':'tent',
'30':'wardrobe',
'31':'bench',
'32':'car',
'33':'curtain',
'34':'flower_pot',
'35':'lamp',
'36':'night_stand',
'37':'radio',
'38':'stairs',
'39':'toilet',
'40':'xbox',
}
class_name_to_id_modelnet40 = { v : k for k, v in class_id_to_name_modelnet40.items() }
class_names_modelnet40 = set(class_id_to_name_modelnet40.values())

class_id_to_name_testing = {
    "1": "piano",
    "2": "lamp",
    '3':'bathtub',
}
class_name_to_id_testing = { v : k for k, v in class_id_to_name_testing.items() }
class_names_testing = set(class_id_to_name_testing.values())



lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }

voxnet_cfg = {'batch_size' : 32,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 64,
       'max_epochs' : 80,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 4000,
       }
