
import json
from typing import Any, Dict, List
import json
# from utils import get_Qwen_8b, read_json_file, save_json_file
from langchain_openai import ChatOpenAI
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import re
import os
import pandas as pd


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Save a dictionary to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def chunk_text(text, max_length=50000):
    '''Split text into chunks of a specified maximum length, ensuring no chunk exceeds the limit.
    Args:
        text (str): The text to be split into chunks.
        max_length (int): The maximum length of each chunk.
    Returns:
        List[str]: A list of text chunks, each not exceeding the specified maximum length.'''
    chunks = []
    while len(text) > max_length:
        split_pos = text.rfind(',', 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        chunks.append(text[:split_pos])
        text = text[split_pos:]
    if text:
        chunks.append(text)
    return chunks


def get_Qwen_8b(messages):
    model_name = "Qwen3-8b"
    DOC_PATH = r"data2/document_data"

    llm = ChatOpenAI(model=model_name, api_key=GITEE_AI_API_KEY, base_url=base_url, streaming=True, temperature=0.1,
                    presence_penalty=1.05, top_p=0.9,
                    extra_body={
                        "guided_json": """{
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "用户的姓名"
                                },
                                "age": {
                                    "type": "integer",
                                    "description": "用户的年龄"
                                },
                                "city": {
                                    "type": "string",
                                    "description": "用户的城市"
                                }
                            },
                            "required": ["name", "city"]
                        }"""
                    })
    response = llm(messages)
    return response

def get_figure():
    from langchain_openai import ChatOpenAI

    model_name = "Qwen2.5-72B-Instruct"

    llm = ChatOpenAI(model=model_name, api_key=GITEE_AI_API_KEY, base_url=base_url, streaming=True, temperature=0.1,
                    presence_penalty=1.05, top_p=0.9,
                    extra_body={
                        "guided_json": """{
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "用户的姓名"
                                },
                                "age": {
                                    "type": "integer",
                                    "description": "用户的年龄"
                                },
                                "city": {
                                    "type": "string",
                                    "description": "用户的城市"
                                }
                            },
                            "required": ["name", "city"]
                        }"""
                    })

    prompt = [{"role": "system", "content": "你是聪明的助手，以 json 格式输出数据，如果无法判断年龄，则 age 为 0"},
            {"role": "user", "content": """
                在一个风和日丽的春日午后，小马走在了北京的街头。时间是 2023年的4月15日，
                正值樱花盛开的季节。作为一位热爱摄影的年轻人，他带着相机，希望能捕捉到这个季节最美的瞬间。
                北京的春天总是短暂而美丽，每一处公园、每一条街道都充满了生机与活力。
            """}]
    
    content = ""
    save_name_tmp = f"{os.path.dirname(__file__)}//output_tmp.json"

    for response in llm.stream(prompt):
        if (response.content):
            content += response.content

            print(response.content, end="")
    with open(save_name_tmp, "w+", encoding="utf-8") as f:
        f.write(content)

def parse_text_to_json1(text):
        # 初始化结果字典
        result = {
            "宏领域技术领域": "",
            "细粒度技术分类": "",
            "特征词结果": [],
            "分类逻辑说明": {},
            "特别关注点": [],
            "补充说明": ""
        }
        
        # 分割文本为不同部分
        sections = re.split(r'###', text.replace('\n', ''))
        if sections[0].strip() == "":
            sections.pop(0)  # 移除第一个空字符串部分
        
        # 解析第一部分：技术领域分类结果和特征词汇总
        part1 = sections[0].strip()
        match = re.search(r'宏领域技术领域[:：]?\s*(.*?)\s*(?:\n|$)', part1)
        if match:
            result["宏领域技术领域"] = match.group(1).strip().replace('*', '')
        else:
            result["宏领域技术领域"] = ""
        #细粒度技术分类
        part1 = sections[1].strip()
        match = re.search(r'细粒度技术分类[:：]?\s*(.*?)\s*(?:\n|$)', part1)
        if match:
            result["细粒度技术分类"] = match.group(1).strip().replace('*', '')
        else:
            result["细粒度技术分类"] = ""
        # 提取特征词汇总
        part1 = sections[2].strip()
        features_section = re.search(r'特征词结果[:：]?\s*(.*?)\s*(?:\n|$)', part1)
        if features_section:
            # 支持多种分隔符（如逗号、斜杠、空格等），并去除多余空格
            features_raw = features_section.group(1)
            features = re.split(r'[,/，、|]', features_raw)
            result["特征词结果"] = [f.strip().strip('*').strip().replace("*","").replace("-","") for f in features if f.strip()]
        else:
            result["特征词结果"] = []

        # 解析第二部分：分类逻辑说明
        part2 = sections[3].strip()
        logic_items = re.search(r'详细判断逻辑(.*?)(?=\n-|\Z)', part2, re.DOTALL)
        if logic_items:
            result["分类逻辑说明"] = logic_items.group(1).strip().replace('；', '')
        else:
            result["分类逻辑说明"] = ""

        # # 解析第三部分：特别关注点
        # part3 = sections[3].strip()
        # focus_points = re.findall(r'- (.*?)(?=\n-|\Z)', part3, re.DOTALL)
        # result["特别关注点"] = [point.strip().replace('；', '') for point in focus_points]
        
        # # 解析第四部分：补充说明
        # part4 = sections[4].strip()
        # result["补充说明"] = part4.replace('（注：', '注：').replace('）', '')  # 清理括号
        
        return result


def get_decision_feature():
    # 输入文本
    input_text = """
    ### 技术领域分类结果：非AI技术领域  
    #### 所有特征词汇总（去重后）：  
    1. **Android应用开发**  
    2. **Java语言**  
    3. **MainActivity.java**  
    4. **ActionPanelController.java**  
    5. **FreezeController.java**  
    6. **XML布局文件**  
    7. **资源文件**  

    ---

    ### 分类逻辑说明：  
    - **核心框架/平台**：基于 `Android` 开发平台构建应用程序；  
    - **编程语言**：主要使用 `Java` 作为实现语言；  
    - **关键组件/模块**：包含典型 Android 应用结构中的核心类（如 `MainActivity` 作为主界面入口、`ActionPanelController` 和 `FreezeController` 作为功能模块控制器）；  
    - **界面与交互设计**：涉及大量 `XML布局文件` 和 `资源文件` 的设计与优化；  

    ---

    ### 特别关注点：  
    - 典型 Android 应用架构特性：Activity 控制器 + 布局定义 + 资源管理；  
    - 非 AI 相关功能聚焦于相机模式切换与界面交互优化；  

    --- 

    ### 补充说明：
    该领域的核心技术栈围绕 Android SDK 的 API 使用展开（如 Camera API、UI 组件），而非依赖 AI 框架或算法模型。（注：若存在其他潜在关联性需进一步分析）
    """

    # 转换为JSON
    parsed_data = parse_text_to_json(input_text)
    json_output = json.dumps(parsed_data, ensure_ascii=False, indent=2)

    # 打印结果
    print(json_output)

    # 可选：保存到文件
    with open("tech_domain_classification.json", "w", encoding="utf-8") as f:
        f.write(json_output)

def hong_division_precision_recall():
    """根据关键词过滤内容"""
    src_path = r"Qwen-8b-ans1"
    ground_truth = {}
    test = {}
    keywords = ["ai", "langraph", "gpt", "llm", "chat", "openai","agent","qwen","bert","transformer","tensorflow","mindspore-lab","paddlepaddle_","llama","gemini","nlp","textclassifier","torch"]
    for i in os.listdir(src_path):

        try:
            # if "PaddlePaddle" in i:
            #     ground_truth[i] = "yes"

            # ground_truth[i] = "no"

            # for keyword in keywords:
            #     if keyword in i.lower():
            #         ground_truth[i] = "yes"


            file_path = os.path.join(src_path, i)
            file_path = os.path.join(file_path, "output.txt")
            with open(file_path, "r", encoding="utf-8") as f:
                feature_content = f.read()
                parsed_data = parse_text_to_json(feature_content)
                domain_output = parsed_data.get("技术领域分类结果", "")
                if "AI" in domain_output and "非 AI" not in domain_output and "非AI" not in domain_output:
                    test[i] = "yes"
                else:
                    test[i] = "no"
        except Exception as e:
            print(f"Error processing {i}: {e}")
            continue


    # ground_truth1 = {}
    df1 = pd.read_csv(r"qxy_zyx.csv",header=None)
    for index, row in df1.iterrows():
        # print(row)
        if "用户" in row[0]:
            continue
        ground_truth[row[0]] = row[1]



    ground_truth["FlagOpen_FlagPerf"] = "yes"
    ground_truth["EdisonLeeeee_GraphGallery"] = "yes"
    ground_truth["TideDra_lmm-r1"] = "yes"
    ground_truth["amazon-science_fmcore"] = "yes"
    ground_truth["jeffffffli_HybrIK"] = "yes"
    ground_truth["APIJSON_apijson-milvus"] = "yes"
    ground_truth["scikit-learn-contrib_imbalanced-learn"]  = "yes"
    ground_truth["fake-useragent_fake-useragent"] = "no"
    # ground_truth["aleju_imgaug"] = "no"
    ground_truth["langchain-ai_langchain"] = "yes"
    ground_truth["facebook_rebound"] = "no"
    ground_truth["oddfar_campus-imaotai"] = "no"
    # print(ground_truth["aleju_imgaug"])

    #ground_truth的value和test的value进行比较，不同存储到csv，只存储分类不一致的
    # 创建DataFrame
    # 只保留分类不一致的行
    test1 = {key: value for key, value in test.items() if key in ground_truth}
    ground_truth1 = {key: value for key, value in ground_truth.items() if key in test}
    # df = pd.DataFrame(list(test1.items()), columns=['项目名称', '预测分类'])
    # df['实际分类'] = df['项目名称'].map(ground_truth)
    # df = df[df['预测分类'] != df['实际分类']]  # 只保留分类不一致的行
    # # 保存到CSV文件
    # df.to_csv("分类结果1.csv", index=False)

    precision = sum(1 for key in test1 if test1[key] == ground_truth1[key]) / len(test1) if test1 else 0
    recall = sum(1 for key in test1 if test1[key] == ground_truth1[key]) / len(ground_truth1) if ground_truth1 else 0
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1 Score: {f1:.2f}")
    df = pd.DataFrame(list(test1.items()), columns=['项目名称', '预测分类'])
    df['实际分类'] = df['项目名称'].map(ground_truth1)
    df = df[df['预测分类'] != df['实际分类']]  # 只保留分类不一致的行
    # 保存到CSV文件
    df.to_csv("分类结果123.csv", index=False)




    

    




    df = pd.DataFrame(list(ground_truth.items()), columns=['项目名称', '实际分类'])
    df['预测分类'] = df['项目名称'].map(test)
    df.to_csv("分类结果.csv", index=False)

    
   

           


        

    return ground_truth


if __name__ == "__main__":
    # get_decision_feature()
    # # Example usage
    # content = ""
    # #  filter_content_by_keyword(content: str)
    # a,b = filter_content_by_keyword(content)
    # with open("sample_content.txt", "w+") as file:
    #     file.write(str(a)+"\n\n" + str(b))
    # print()  # Print the response from the model
    # get_figure()
    # with open("output_tmp.json", "r", encoding="utf-8") as f:
    #     content = json.load(f)
    # # print(content)
    # import matplotlib.font_manager as fm
    # # 搜索所有支持中文的字体
    # for font in fm.findSystemFonts():
    #     if "wqy" in font or "Noto" in font:  # 文泉驿/Noto字体
    #         print(font)  # 输出可用路径

    # hong_division_precision_recall()
    text = """


   

### 宏领域技术领域：**AI技术领域**  
### 细粒度技术分类：**生成式人工智能 (AIGC)**, **AI智能体 (Agents)**, **工作流管理 (Workflow Management)**  
### 特征词结果：**Rust编程语言**, **自动化工具有关模块**, **外部API集成**, **JSON Schema**, **structured output**, **tool execution**, **async executors**, **provider abstraction**, **flow customization**

---

### 详细判断逻辑  

#### 1. 宏领域修正依据  
- 初始分类认为"非AI域"存在偏差：尽管未直接实现大模型训练/推理能力（如不包含Transformer架构代码），但其核心定位为"构建并运行与大型语言模型(LLMs)交互的 AI 代理(agents)"——这是典型 AIGC 工具链应用场景；  
- TPL 数据中明确提及 `openai/anthropic/ollama` 等 LLM 服务调用接口及 `MCP servers` 模型上下文协议支持——符合 AIGC 工具链特征；  
- 功能注解强调 "解决多模型服务商切换痛点" 和 "抽象 LLM 接口差异"——这是 AIGC 领域标准化工具开发的核心需求  

#### 2. 细粒度分类推导过程  
- **生成式人工智能(AIGC)**: 核心定位为 "统一 API 调用不同 LLMs 实现内容创作自动化" 直接服务于文本/代码等内容生产场景；关键依赖项 `schemars` 支持结构化输出规范符合 AIGC 输出要求；  
- **AI智能体(Agents)**: 明确声明 "构建 AI agent 抽象多提供商细节" 并提供自定义流程控制能力 (`custom_flow_closure/manual_tool_execution`)——这是 Agent 架构标准特征；  
- **工作流管理(Workflow Management)**: 核心模块 `plan_and_execute` 实现异步执行器 (`async executors`) 和动态模板引擎 (`TemplateDataSource`) 支持复杂任务编排——符合 AI 工程化流水线需求  

#### 
    """
    print(parse_text_to_json1(text))