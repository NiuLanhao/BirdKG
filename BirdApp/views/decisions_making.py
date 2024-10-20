import base64
from io import BytesIO
from PIL import Image
from django.shortcuts import render
import json
import os
import sys

# 将项目路径添加到 Python 解释器的搜索路径中
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from toolkit.pre_load import neo_con
from vision.ResNet import resnet_predict

relationCountDict = {}
# filePath = os.path.abspath(os.path.join(os.getcwd(), "."))
# with open(filePath + "/toolkit/relationStaticResult.txt", "r", encoding='utf8') as fr:
#     for line in fr:
#         relationNameCount = line.split(",")
#         relationName = relationNameCount[0][2:-1]
#         relationCount = relationNameCount[1][1:-2]
#         relationCountDict[relationName] = int(relationCount)


def sortDict(relationDict):
    for i in range(len(relationDict)):
        relationName = relationDict[i]['rel']['type']
        relationCount = relationCountDict.get(relationName)
        if (relationCount is None):
            relationCount = 0
        relationDict[i]['relationCount'] = relationCount

    relationDict = sorted(relationDict, key=lambda item: item['relationCount'], reverse=True)

    return relationDict


def base64_to_image(base64_str):
    # 去掉 Base64 编码字符串的头部信息
    base64_str = base64_str.split('base64,')[-1]
    # 解码 Base64 编码字符串为字节数据
    image_bytes = base64.b64decode(base64_str)
    # 将字节数据转换为 PIL Image
    image = Image.open(BytesIO(image_bytes))
    return image


# 调用图像分类训练好的模型进行预测
def vision_predict(img):
    result_list = resnet_predict.predict(img, confidence_threshold=0.1)
    sorted_results = sorted(result_list, key=lambda x: x['prob'], reverse=True)
    return sorted_results


def decisions_making(request):  # index页面需要一开始就加载的内容写在这里
    ctx = {}
    if request.POST:
        img_base64 = request.POST['img_base64']
        img = base64_to_image(img_base64)
        entity_list = vision_predict(img)

        best_match = ''
        best_match += '<div class="row"> <div class="col-md-6"> ' \
                      '<a href="../detail?title=' + entity_list[0]['class'] + '">'
        best_match += '<div class="col-md-9"><h1>' + entity_list[0]['class'] + '</h1>'
        best_match += '</h4> 置信度：{:.5f}% </h4>'.format(entity_list[0]['prob'] * 100)
        best_match += '<div class="progress"> <div class="progress-bar progress-bar-success" role="progressbar" style="width: '
        best_match += str(entity_list[0]['prob'] * 100) + '%;" aria-valuemin="0" aria-valuemax="100">'

        ctx['best_match'] = best_match

        other_match = ''
        for i in range(1, len(entity_list)):
            other_match += '<div class="row"> <div class="col-md-5"> <h5>'
            other_match += '<div class="col-md-7"><h4>' + entity_list[i]['class'] + '</h4>'
            other_match += '</h5> 置信度：{:.3f}% </h5>'.format(entity_list[i]['prob'] * 100)
            other_match += '<div class="progress"> <div class="progress-bar progress-bar-info" role="progressbar" style="width: '
            other_match += str(entity_list[i]['prob'] * 100) + '%;" aria-valuemin="0" aria-valuemax="100">'
            other_match += '<h5><a href="../detail?title=' + entity_list[i]['class'] + '">[查看详细]</a></h5></div></div>'

        ctx['other_match'] = other_match

        entity = entity_list[0]['class']
        ctx['best_match_title'] = entity

        # 连接数据库
        db = neo_con
        entityRelation = db.getEntityRelationbyEntity(entity)
        if len(entityRelation) == 0:
            # 若数据库中无法找到该实体，则返回数据库中无该实体
            ctx_graph = {'title': '<h1>数据库中暂未添加该实体</h1>'}
            ctx['graph'] = json.dumps(ctx_graph, ensure_ascii=False)
        else:
            # 返回查询结果
            # 将查询结果按照"关系出现次数"的统计结果进行排序
            entityRelation = sortDict(entityRelation)
            ctx['graph'] = json.dumps(entityRelation, ensure_ascii=False)
            print(ctx['graph'])

    return render(request, 'decisions_making.html', ctx)
