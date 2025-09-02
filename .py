"""
AutoGen配置文件
用于设置AutoGen的LLM配置和环境变量
"""

import os
import json
from typing import List, Dict, Any

# 确保 OAI_CONFIG_LIST 文件存在或环境变量已设置
# 建议将 OAI_CONFIG_LIST 放在项目根目录
# 或者将其内容作为环境变量 OAI_CONFIG_LIST 的值

def create_oai_config_list():
    """创建OAI_CONFIG_LIST配置"""
    config_list = [
        {
            "model": "gpt-4-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "openai",
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "api_version": None
        },
        {
            "model": "gpt-4-vision-preview",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "openai", 
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "api_version": None
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "openai",
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "api_version": None
        }
    ]
    
    # 过滤掉没有API密钥的配置
    valid_configs = [config for config in config_list if config["api_key"]]
    
    if not valid_configs:
        raise ValueError("未找到有效的OpenAI API密钥。请设置OPENAI_API_KEY环境变量。")
    
    return valid_configs

def save_oai_config_list(config_list: List[Dict[str, Any]], file_path: str = "OAI_CONFIG_LIST"):
    """保存OAI_CONFIG_LIST到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_list, f, indent=2, ensure_ascii=False)
    print(f"OAI_CONFIG_LIST已保存到 {file_path}")

def load_oai_config_list(file_path: str = "OAI_CONFIG_LIST") -> List[Dict[str, Any]]:
    """从文件加载OAI_CONFIG_LIST"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_list = json.load(f)
        print(f"成功从 {file_path} 加载OAI_CONFIG_LIST")
        return config_list
    except FileNotFoundError:
        print(f"未找到 {file_path} 文件，将创建默认配置")
        config_list = create_oai_config_list()
        save_oai_config_list(config_list, file_path)
        return config_list
    except json.JSONDecodeError as e:
        raise ValueError(f"OAI_CONFIG_LIST文件格式错误: {e}")

def validate_config():
    """验证配置是否正确"""
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"缺少必需的环境变量: {', '.join(missing_vars)}")
    
    print("配置验证通过")

# 初始化配置
def initialize_autogen_config():
    """初始化AutoGen配置"""
    try:
        validate_config()
        config_list = load_oai_config_list()
        print(f"AutoGen配置初始化完成，共加载 {len(config_list)} 个模型配置")
        return config_list
    except Exception as e:
        print(f"AutoGen配置初始化失败: {e}")
        raise

if __name__ == "__main__":
    # 如果直接运行此文件，将创建或验证配置
    try:
        initialize_autogen_config()
        print("AutoGen配置验证成功！")
    except Exception as e:
        print(f"配置验证失败: {e}")
        print("\n请确保设置了以下环境变量:")
        print("export OPENAI_API_KEY='your_openai_api_key_here'")
        print("export OPENAI_API_BASE='https://api.openai.com/v1'  # 可选")



