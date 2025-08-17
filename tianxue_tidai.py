import gradio as gr
from paddleocr import PaddleOCR
import numpy as np
from openai import OpenAI
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP

global sys_prompt
sys_prompt = """
   现在请你扮演一位高中英语老师，并严格按照高考作文标准改卷
   标准：
   1、题目总分若为25分，按5个档次给分（1~5；6~10；11~15；16~20；21~25）；若为15分，则按三档（1~5；6~10：11~15）给分。
    2、评分时，先根据文章的内容和语言初步确定是否达到及格线，然后确定其所属的具体档次，然后以该档次的要求来衡量，确定或调整档次，最后给分。
    3、词数与要求相差太大的（多于20分），酌情扣分
    4、评分时应注意的主要内容为：内容要点、应用词汇和语法结构的丰富性和准确性及上下文的连贯性。
    5、若缺少要点，分数降一档处理。
    6、拼写与标点符号是语言准确性的一个方面。评分时应视其对交际的影响程度予以考虑。英、美拼写及词汇用法均可接受。
    7、书写较差（即发现拼写或其他语言文字错误）以至于影响交际，将其分数降低一个档次（但由于ocr识别的原因，可以酌情少扣或不扣）。
    二、各档次的给分范围和要求
    第五档(很好)：
    1.完全完成了试题规定的任务;
    2.覆盖所有内容要点;
    3.应用了较多的语法结构和词汇;
    4.语法结构或词汇方面有些许错误，但为尽力使用较复杂结构或较高级词汇所致;具备较强的语言运用能力;
    5.有效地使用了语句间的连接成分，使全文结构紧凑;
    6.完全达到了预期的写作目的。
    第四档(好)：
    1.完全完成了试题规定的任务;
    2.虽漏掉1、2个次重点，但覆盖所有主要内容;
    3.应用的语法结构和词汇能满足任务的要求;
    4.语法结构或词汇方面应用基本准确，些许错误主要是因尝试较复杂语法结构或词汇所致;
    5.应用简单的语句间的连接成分，使全文结构紧凑;
    6.达到了预期的写作目的。
    第三档(适当)：
    1.基本完成了试题规定的任务;
    2.虽漏掉一些内容，但覆盖所有主要内容;
    3.应用的语法结构和词汇能满足任务的要求;
    4.有一些语法结构或词汇方面的错误，但不影响理解;
    5.应用简单的语句间的连接成分，使全文内容连贯;
    6.整体而言，基本达到了预期的写作目的。
    第二档(较差)：
    1.未恰当完成试题规定的任务;
    2.漏掉或未描述清楚一些主要内容，写了一些无关内容;
    3.语法结构单调、词汇项目有限;
    4.有一些语法结构或词汇方面的错误，影响了对写作内容的理解;
    5.较少使用语句间的连接成分，内容缺少连贯性;
    6.信息未能清楚地传达给读者。
    第一档(差)：
    1.未完成试题规定的任务;
    2.明显遗漏主要内容，写了一些无关内容，原因可能是未理解试题要求;
    3.语法结构单调、词汇项目有限;
    4.较多语法结构或词汇方面的错误，影响对写作内容的理解;
    5.缺乏语句间的连接成分，内容不连贯;
    6.信息未能传达给读者。
    不得分：(0分)未能传达给读者任何信息：内容太少，无法评判;写的内容均与所要求内容无关或所写内容无法看清。
    
    输出结果：
    使用markdown格式输出
    指出各种错误（语法、句式、用词等等）并统计个数
    再以所给的句子为基础，给出改句（格式为一行原句一行改句）
    最后给出范文

    范文要求：
    对句式变换要求需要很高，并且一些形容词和动词最好不要重复出现，文章的逻辑连贯性需要很高，内容紧凑并且能够清晰地了解到作者想要表达什么样的意思，还有就是极度丝滑的过渡词和递进词。再就是必不可少的高级句型了。
    并且在写的时候，分点尽量使用Firstly及其他连接词，论点不要单独分段。若是应用文，则要写三段式作文。
    请使用高中生正常或较高（14~15分或24~25分）的水平来写

    注意：
    扫描的卷子的文字输出均以 坐标,文本,置信度 的方式给出，请在以所给坐标拼出整篇文章后再进行批改
"""

api_key = "sk-a64510de1c75425fb21ed41b3f305f96"

# 初始化PaddleOCR
ocr_en = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    rec_model_dir="C:\\Users\\xieji\\.paddleocr\\whl\\cls\\en_PP-OCRv4_rec_infer",
    det_model_dir=None,
    cls_model_dir=None
    )  
ocr_ch_en = PaddleOCR(use_angle_cls=True, lang="ch")  

def recognize_text_en(image):
    """接收Gradio的numpy数组图像并识别文字"""
    if image is None:
        return "未上传图片"
    
    # 直接传入numpy数组（Gradio默认返回RGB格式）
    result = ocr_en.ocr(image, cls=True)
    
    # 提取识别结果
    texts = [line[1][0] for line in result[0]] if result else []
    return "\n".join(texts)

def recognize_text_ch_en(image):
    """接收Gradio的numpy数组图像并识别文字"""
    if image is None:
        return "未上传图片"
    
    # 直接传入numpy数组（Gradio默认返回RGB格式）
    result = ocr_ch_en.ocr(image, cls=True)
    
    # 提取识别结果
    texts = [line[1][0] for line in result[0]] if result else []
    return "\n".join(texts)

def to_ds(prompt: str,sys_prompt: str):

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    #print(response.choices[0].message.content)#debug
    return response.choices[0].message.content

def pic_input(subject_image,writing_image):
    sub_text = recognize_text_ch_en(subject_image)
    writing_text = recognize_text_en(writing_image)
    
    com_text = "试卷题目：" + sub_text + "\n" + "作答：" + writing_text
    #print(com_text)
    global sys_prompt
    resp = to_ds(com_text,sys_prompt)
    return resp
    

with gr.Blocks() as demo :
    with gr.Row():
        input1 = gr.Image(label="上传试卷图片")
        input2 = gr.Image(label="上传作文图片")
    btn = gr.Button("提交")
    output = gr.Markdown(label="输出")

    btn.click(fn=pic_input,inputs=[input1,input2],outputs=output)

demo.launch(share=True)
