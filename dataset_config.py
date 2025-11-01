"""
数据集配置文件
统一管理所有支持的数据集配置信息
支持数据集：AID, NaSC, NWPU, PatternNet, UCM
"""

# 数据集配置字典
# 格式: dataset_name: {
#     'num_classes': int,        # 类别数
#     'default_input_size': int,  # 默认输入尺寸
#     'train_folder': str,        # 训练集文件夹名称
#     'val_folder': str,          # 验证集文件夹名称（如果使用固定划分）
#     'use_random_split': bool,   # 是否使用随机划分（True表示使用build_dataset_new）
#     'default_data_path': str    # 默认数据路径（可选，如果未指定则使用--data-path参数）
# }
DATASET_CONFIG = {
    'AID': {
        'num_classes': 30,
        'default_input_size': 600,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': False,
        'default_data_path': None,  # 使用--data-path参数
    },
    'NaSC': {
        'num_classes': 10,
        'default_input_size': 128,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': True,  # NaSC使用随机划分
        'default_data_path': None,
    },
    'NWPU': {
        'num_classes': 45,
        'default_input_size': 256,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': False,
        'default_data_path': None,
    },
    'PatternNet': {
        'num_classes': 38,
        'default_input_size': 256,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': False,
        'default_data_path': None,
    },
    'UCM': {
        'num_classes': 21,
        'default_input_size': 256,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': False,
        'default_data_path': None,
    },
}


def get_dataset_config(dataset_name):
    """
    获取数据集配置
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        dict: 数据集配置字典，如果数据集不存在则返回None
    """
    return DATASET_CONFIG.get(dataset_name.upper(), None)


def get_num_classes(dataset_name):
    """获取数据集的类别数"""
    config = get_dataset_config(dataset_name)
    return config['num_classes'] if config else None


def get_default_input_size(dataset_name):
    """获取数据集的默认输入尺寸"""
    config = get_dataset_config(dataset_name)
    return config['default_input_size'] if config else 224


def should_use_random_split(dataset_name):
    """判断数据集是否应该使用随机划分"""
    config = get_dataset_config(dataset_name)
    return config['use_random_split'] if config else False

