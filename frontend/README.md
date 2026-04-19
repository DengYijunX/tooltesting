# 编程题目生成平台前端

这是一个原生 HTML/CSS/JavaScript 前端，用于调用 FastAPI 后端并展示当前 RAG 编程题生成工作流。

## 启动方式

先安装依赖：

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

再启动 FastAPI 服务：

```powershell
.\venv\Scripts\uvicorn.exe app.server:app --reload --host 127.0.0.1 --port 8000
```

然后访问：

```text
http://127.0.0.1:8000/
```

## 接口

前端会调用：

```text
POST /api/generate-problem
```

请求体：

```json
{
  "topic": "牛顿第二定律",
  "subject": "physics",
  "mode": "verified",
  "algorithm": "auto",
  "notes": "题面尽量简洁，必须通过一致性检查和沙箱执行。"
}
```

## 页面展示

- 主题与学科输入
- 生成模式选择
- 九阶段工作流进度
- 标准题面
- 参考代码
- 知识支撑、一致性、沙箱、审核状态
- 审计日志路径
