import os
import shutil
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TextGenerationTransformersPreprocessor
from modelscope.utils.constant import Tasks
from pydantic import BaseModel, Field



class TextRequest(BaseModel):
  """文本处理请求模型"""
  text: str = Field(..., description="待处理的文本内容")
  task_type: str = Field(..., description="任务类型: summarize/qa/sentiment等")
  options: Optional[str] = Field(None, description="可选参数，用于某些特定任务")

  class Config:
    schema_extra = {
      "example": {
        "text": "这个产品的做工非常好，使用很方便",
        "task_type": "sentiment",
        "options": None
      }
    }


class ModelLoader:
  @staticmethod
  def load_model(model_path: str, model_id: str, revision: str = 'v0.1'):
    """
    安全加载模型，支持自动下载和本地缓存

    :param model_path: 本地模型路径
    :param model_id: 模型ID
    :return: tokenizer和model
    """

    def is_model_complete(path):
      """检查模型文件是否完整"""
      required_files = [
        'config.json',
        'pytorch_model.bin',
        'tokenizer_config.json',
      ]
      return all(
          os.path.exists(os.path.join(path, file)) for file in required_files)

    try:
      # 检查本地模型是否存在且完整
      if not os.path.exists(model_path) or not is_model_complete(model_path):
        print(f"本地模型不存在或不完整，开始下载模型到: {model_path}")
        # 清理不完整的模型目录
        if os.path.exists(model_path):
          shutil.rmtree(model_path)

        os.makedirs(model_path, exist_ok=True)

        print(f"开始下载模型: {model_id}")

        try:
          # 下载模型
          model = T5ForConditionalGeneration.from_pretrained(
              model_id,
              revision=revision,
              # local_files_only=False,  # 允许从网络下载
              # cache_dir=None  # 不使用默认缓存
          )
          model.save_pretrained(model_path,
                save_checkpoint_names='model')  # 指定检查点名称
          print(f"模型下载完成: {model_path}")
        except Exception as download_error:
          print(f"模型下载失败: {download_error}")
          if os.path.exists(model_path):
            shutil.rmtree(model_path)
          raise

      # # 加载本地模型
      model = T5ForConditionalGeneration.from_pretrained(model_path,
              revision=revision)

      # 创建预处理器和pipeline
      preprocessor = TextGenerationTransformersPreprocessor(model.model_dir)
      pipe = pipeline(
          task=Tasks.text2text_generation,
          model=model,
          preprocessor=preprocessor)

      # model = T5ForConditionalGeneration.from_pretrained('ClueAI/PromptCLUE',
      #                                                    revision='v0.1')
      # preprocessor = TextGenerationTransformersPreprocessor(model.model_dir)
      # pipe = pipeline(task=Tasks.text2text_generation, model=model,
      #                         preprocessor=preprocessor)

      return pipe

    except Exception as e:
      print(f"模型加载失败: {e}")
      raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


class PromptCLUEService:
  def __init__(self):
    # 加载配置
    load_dotenv()

    # 加载模型配置
    model_path = os.getenv("MODEL_FILE_PATH")
    model_id = os.getenv("MODEL_ID")
    model_version = os.getenv("MODEL_VERSION")

    # 加载模型
    try:
      self.pipeline_t2t = ModelLoader.load_model(model_path, model_id,
                                                 model_version)
      print("PromptCLUE模型加载成功")
    except Exception as e:
      print(f"模型加载失败: {e}")
      raise

    # 定义任务模板
    self.prompt_templates = {
      "summarize": "为下面的文章生成摘要：\n{}\n",
      "qa": "问答：\n问题：{}\n答案：",
      "sentiment": "情感分析：\n{}\n选项：积极，消极",
      "rewrite": "生成与下列文字相同意思的句子：\n{}\n答案：",
      "semantic_similar": "下面句子是否表示了相同的语义：\n文本1：{}\n文本2：{}\n选项：相似，不相似\n答案：",
      "classification": "这是关于哪方面的新闻：\n{}\n选项：故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏",
      "extract": "阅读文本抽取关键信息：\n{}\n问题：机构，人名，职位，籍贯，专业，国籍，学历，种族\n答案：",
      "translate_en": "翻译成英文：\n{}\n答案：",
      "correction": "文本纠错：\n{}\n答案：",
      "inference": "推理关系判断：\n前提：{}\n假设：{}\n选项：矛盾，蕴含，中立\n答案："
    }

  async def process_text(self, request: TextRequest) -> str:
    """处理文本请求"""
    try:
      if request.task_type not in self.prompt_templates:
        raise ValueError(f"不支持的任务类型: {request.task_type}")

      # 根据任务类型构建提示语
      template = self.prompt_templates[request.task_type]

      # 对于需要两段文本的任务进行特殊处理
      if request.task_type in ["semantic_similar", "inference"]:
        if not request.options:
          raise ValueError(
              f"{request.task_type}任务需要提供第二段文本在options中")
        prompt = template.format(request.text, request.options)
      else:
        prompt = template.format(request.text)

      # 使用模型处理
      result = self.pipeline_t2t(
          prompt,
          do_sample=True,
          top_p=0.8
      )

      return result['text']

    except Exception as e:
      print(f"处理失败: {e}")
      raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
  """
  创建FastAPI应用
  """
  # 设置CUDA设备
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"

  # 创建应用
  app = FastAPI(
      title="PromptCLUE文本处理服务",
      description="基于PromptCLUE的中文文本理解与生成服务",
      version="1.0.0"
  )

  # 配置跨域
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # 初始化服务
  service = PromptCLUEService()

  @app.post("/process")
  async def process_text(request: TextRequest):
    """
    处理文本接口
      "summarize": "为下面的文章生成摘要：\n{}\n",
      "qa": "问答：\n问题：{}\n答案：",
      "sentiment": "情感分析：\n{}\n选项：积极，消极",
      "rewrite": "生成与下列文字相同意思的句子：\n{}\n答案：",
      "semantic_similar": "下面句子是否表示了相同的语义：\n文本1：{}\n文本2：{}\n选项：相似，不相似\n答案：",
      "classification": "这是关于哪方面的新闻：\n{}\n选项：故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏",
      "extract": "阅读文本抽取关键信息：\n{}\n问题：机构，人名，职位，籍贯，专业，国籍，学历，种族\n答案：",
      "translate_en": "翻译成英文：\n{}\n答案：",
      "correction": "文本纠错：\n{}\n答案：",
      "inference": "推理关系判断：\n前提：{}\n假设：{}\n选项：矛盾，蕴含，中立\n答案："
      """
    result = await service.process_text(request)
    return {"result": result}

  return app


# 创建应用
app = create_app()

if __name__ == '__main__':
  uvicorn.run(
      'app:app',
      host="0.0.0.0",
      port=19990,
      # reload=True # 关闭热重载
  )
