# -*- coding: utf-8 -*-
from django.shortcuts import render
import re
from zhipuai import ZhipuAI
from toolkit.pre_load import neo_con

api_key = "1d974cd01f386a6ff37fefd14cf07722.QZs3S0NTQKBABluU"
if api_key == "":
    raise Exception("Please set the api_key in the ner_view.py file !")
client = ZhipuAI(api_key=api_key)

ENTITY_EXTRACT_PROMPT = """
Please extract the name of the entity.
Entities follow the original text in its entirety without any modifications.

Notes: 
1. Use the char <<entity_name>> to highlight the entity's name.
2. Only extract the entity name, not answer the question, entity name must the same as the question
3. One sentence may has more than one entity, you should highlight all the entities and keep the origional sentence.

Example:
---
Question: 鹦鹉的头部比较大，脑瓜饱满，头顶凸起，嘴巴宽大，有着可爱的仰面姿势，黑眼睛，全身有着妩媚的色彩。鹦鹉的胸前部位，有一圈羽毛覆盖，色彩斑斓，多彩多姿。
Response: <<鹦鹉>>的头部比较大，脑瓜饱满，头顶凸起，嘴巴宽大，有着可爱的仰面姿势，黑眼睛，全身有着妩媚的色彩。<<鹦鹉>>的胸前部位，有一圈羽毛覆盖，色彩斑斓，多彩多姿。
---
Question: {query_str}
"""


def ask_llm(text: str):
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {
                    "role": "user",
                    "content": ENTITY_EXTRACT_PROMPT.format(
                        query_str=text,
                    )
                },
            ],
            temperature=0.01,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return text


def replace_names(text: str):
    # 匹配 <名字> 的正则表达式
    pattern = re.compile(r"<<(.*?)>>")

    # 替换逻辑
    def replacer(match):
        name = match.group(1)  # 提取名字
        if 'error' not in search_entity(name):
            describe = search_entity(name)['entity']
            return (
                    f"<a href='../detail?title={name}' data-original-title='{name}' "
                    f"data-placement='top' data-trigger='hover' data-content='{describe}' "
                    "class='popovers'>" + name + "</a>"
            )
        else:
            return (
                    f"<a href='#'  data-original-title='{name}' "
                    f"data-placement='top' data-trigger='hover' data-content='(暂无资料)' "
                    f"class='popovers'>" + name + "</a>"
            )

    # 使用正则表达式进行替换
    return pattern.sub(replacer, text)


def ner_llm(request):
    ctx = {}
    if request.POST:
        text = request.POST['user_text']
        highlight = ask_llm(text)
        ctx['rlt'] = replace_names(highlight)

    return render(request, "index.html", ctx)


def search_entity(entity: str):
    # 根据传入的实体名称搜索出关系
    db = neo_con
    entityRelation = db.matchItembyTitle(entity)
    if len(entityRelation) == 0:
        # 若数据库中无法找到该实体，则返回数据库中无该实体
        ctx = {'error': '数据库中暂未添加该实体'}
    else:
        descirbe = (entityRelation[0]['n'].get('describe'))
        ctx = {'entity': descirbe}
    return ctx
