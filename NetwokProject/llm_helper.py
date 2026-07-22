# llm_helper.py
import os
import openai
from dotenv import load_dotenv
load_dotenv()

# 1. 从系统的环境变量中读取密匙
api_key = os.environ.get("DEEPSEEK_API_KEY")

# 安全检查
if not api_key:
    raise ValueError("❌ 未检测到环境变量 'DEEPSEEK_API_KEY'，请先在系统中配置该环境变量！")

# 2. 按照官方教材初始化客户端
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"  # 对应教材中的 base_url (OpenAI)
)

def _call_llm_api(prompt: str) -> str:
    """内部私有函数：严格按照官方最新 curl 样例构建请求"""
    try:
        # 使用官方最新推荐的 deepseek-v4-pro 模型
        # 并加入 extra_body 传入官方新增的思考参数 (thinking 和 reasoning_effort)
        response = client.chat.completions.create(
            model="deepseek-v4-pro",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant. 你是一个严谨的学术分析助手，擅长分析论文、研究方向以及作者之间的合作与差异。请用清晰、结构化的中文回答。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,  # 对应教材样例
            extra_body={
                "thinking": {"type": "enabled"}, # 开启深度思考模式
                "reasoning_effort": "high"       # 思考强度设置为高，适合学术对比
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI 响应出错: {str(e)}"


def handle_ai_compare(node1: dict, node2: dict) -> str:
    """对外接口：处理两篇论文或两位作者的对比逻辑"""
    if node1['type'] == 'paper' and node2['type'] == 'paper':
        prompt = f"""
        请对比以下两篇论文的研究方向、核心方法、潜在关联及差异：
        
        论文 1 标题: {node1['name']}
        论文 1 摘要: {node1['info']}
        
        --------------------------------------------------
        
        论文 2 标题: {node2['name']}
        论文 2 摘要: {node2['info']}
        
        请从以下三个维度全面深入地对比分析：
        1. 共同的科学背景/研究痛点是什么？
        2. 两者采取的技术路线或研究视角有何本质不同？
        3. 如果将两者的成果进行结合，有哪些潜在的创新研究方向？
        """
    elif node1['type'] == 'author' and node2['type'] == 'author':
        prompt = f"""
        在学术网络中，以下两位学者具有密切的学术联系（或在同一聚类）：
        学者 1: {node1['name']}
        学者 2: {node2['name']}
        
        请基于此，合理推导并对比两者的研究方向：
        1. 分析他们可能共同涉足的学术领域。
        2. 探讨两位学者在研究风格、关注细分方向上的潜在差异。
        3. 预测他们如果开展学术跨界合作，可能诞生什么方向的创新成果。
        """
    else:
        return "❌ 暂不支持跨类型对比（例如一篇论文与一位作者对比），请选择两篇论文或两位作者。"
        
    return _call_llm_api(prompt)


def handle_ai_question(selected_nodes: list, user_question: str) -> str:
    """对外接口：处理基于当前选中节点的自由提问"""
    if not user_question or len(user_question.strip()) == 0:
        return "❌ 请在文本框中输入你想问的具体问题！"
        
    context = ""
    for i, n in enumerate(selected_nodes):
        prefix = "📄 论文" if n['type'] == 'paper' else "👤 作者"
        context += f"{prefix} [{i+1}]: {n['name']}\n上下文/摘要: {n['info']}\n\n"
        
    prompt = f"""
    基于以下学术实体的上下文信息，回答用户的问题。
    
    【学术上下文】
    {context}
    
    【用户问题】
    {user_question}
    
    请结合给出的上下文，给出详尽、富有学术洞察力的回答。若问题与上下文无关，请在回答开头予以说明。
    """
    return _call_llm_api(prompt)