# -*- coding: utf-8 -*-
import nltk
import ssl
# 取消 SSl 认证
ssl._create_default_https_context = ssl._create_unverified_context
# 下载 nltk 数据包
nltk.download()
    
