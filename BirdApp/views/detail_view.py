# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.views.decorators import csrf
import thulac
import sys
import os

# 将项目路径添加到 Python 解释器的搜索路径中
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from toolkit.pre_load import neo_con


def table_html(data):
    table_html = '<table class="table table-striped table-advance table-hover"> <tbody>'

    for key, value in data.items():
        table_html += f"<tr>" \
                      f"<td><strong>{key}</strong></td>" \
                      f"<td>{value}</td></tr>"

    table_html += "</tbody></table>"
    return table_html


# 接收GET请求数据
def showdetail(request):
    ctx = {}
    if 'title' in request.GET:
        # 连接数据库
        db = neo_con

        title = request.GET['title']
        answer = db.matchHudongItembyTitle(title)
        if answer == None:
            return render(request, "404.html", ctx)

        if len(answer) > 0:
            answer = answer[0]['n']
        else:
            ctx['title'] = '实体条目出现未知错误'
            return

        ctx['habit'] = answer['habit']
        ctx['name'] = answer['name']

        ctx['baseInfoTable'] = table_html(answer)
    else:
        return render(request, "404.html", ctx)

    return render(request, "detail.html", ctx)

#	
## -*- coding: utf-8 -*-
# from django.http import HttpResponse
# from django.shortcuts import render_to_response
# import thulac
# 
# import sys
# sys.path.append("..")
# from neo4jModel.models import Neo4j
#
# def search_detail(request):
#	return render_to_response('detail.html')
#
## 接收GET请求数据
# def showdetail(request):
#	request.encoding = 'utf-8'
#	if 'title' in request.GET:
#		# 连接数据库
#		db = Neo4j()
#		db.connectDB()
#		title = request.GET['title']
#		answer = db.matchItembyTitle(title)
#		message = answer['detail']
#				
#	return HttpResponse(message)
