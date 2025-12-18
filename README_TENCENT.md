# Nano Banana Tencent - ComfyUI Plugin

使用腾讯云 VOD AIGC API (GEM 模型) 在 ComfyUI 中生成和编辑图片的插件。

## 功能特性

- **图片生成**: 根据文本提示词生成高质量图片
- **图片编辑**: 编辑现有图片的特定区域
- **风格迁移**: 将参考图片的风格应用到新图片
- **对象插入**: 在现有图片中插入新对象
- **多图片支持**: 支持最多 5 张参考图片
- **灵活配置**: 支持多种分辨率 (1K/2K/4K) 和宽高比

## 系统要求

- ComfyUI (已安装)
- Python 3.8+
- 腾讯云账号（需要开通 VOD 服务）

## 安装步骤

### 1. 安装依赖

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-NanoBanano
pip install -r requirements_tencent.txt
```

### 2. 获取腾讯云凭证

1. 访问 [腾讯云控制台](https://console.cloud.tencent.com/cam/capi)
2. 创建或获取 API 密钥 (SecretId 和 SecretKey)
3. 开通 [云点播 VOD 服务](https://console.cloud.tencent.com/vod)
4. 获取 SubAppID (在 VOD 控制台的应用管理中查看)

### 3. 配置凭证

有两种方式配置腾讯云凭证：

#### 方式 1: 环境变量 (推荐)

```bash
export TENCENTCLOUD_SECRET_ID="your_secret_id"
export TENCENTCLOUD_SECRET_KEY="your_secret_key"
export TENCENTCLOUD_SUB_APP_ID="your_sub_app_id"
```

#### 方式 2: 在 ComfyUI 界面中输入

在节点的参数中直接填写：
- `secret_id`: 腾讯云 SecretId
- `secret_key`: 腾讯云 SecretKey
- `sub_app_id`: 云点播 SubAppID

配置会自动保存到 `config_tencent.json` 文件中。

## 使用方法

### 基本工作流

1. 在 ComfyUI 中添加 "Nano Banana Tencent" 节点
2. 填写提示词 (prompt)
3. 选择操作类型 (operation):
   - `generate`: 生成新图片
   - `edit`: 编辑现有图片
   - `style_transfer`: 风格迁移
   - `object_insertion`: 对象插入
4. (可选) 连接参考图片输入
5. 配置其他参数
6. 执行工作流

### 参数说明

#### 必需参数

- **prompt** (文本): 描述你想要生成或编辑的内容
- **operation** (下拉菜单): 操作类型
  - `generate`: 生成新图片
  - `edit`: 编辑图片 (需要参考图片)
  - `style_transfer`: 风格迁移 (需要参考图片)
  - `object_insertion`: 对象插入 (需要参考图片)

#### 可选参数

##### 参考图片
- **reference_image_1 ~ 5** (IMAGE): 最多 5 张参考图片

##### 腾讯云凭证
- **secret_id** (文本): 腾讯云 SecretId
- **secret_key** (文本): 腾讯云 SecretKey
- **sub_app_id** (整数): 云点播 SubAppID
- **region** (文本): 腾讯云区域，默认 "ap-guangzhou"

##### 模型配置
- **model_version** (下拉菜单): GEM 模型版本 (2.0 或 3.0)
- **aspect_ratio** (下拉菜单): 图片宽高比
  - 1:1, 3:2, 2:3, 3:4, 4:3, 16:9, 9:16, 21:9, 4:5, 5:4
- **image_size** (下拉菜单): 图片分辨率
  - 1K (最快，成本最低)
  - 2K (平衡)
  - 4K (最高质量，成本最高)
- **enable_safety** (布尔): 启用内容安全过滤 (默认: True)

##### 任务控制
- **poll_interval** (整数): 任务状态轮询间隔，单位秒 (默认: 5)
- **timeout** (整数): 最大等待时间，单位秒 (默认: 600)

### 输出

- **generated_images** (IMAGE): 生成的图片张量
- **operation_log** (STRING): 操作日志，包含详细的处理过程

## 使用示例

### 示例 1: 生成图片

```
Prompt: "A beautiful sunset over mountains with a lake in the foreground"
Operation: generate
Image Size: 2K
Aspect Ratio: 16:9
```

### 示例 2: 编辑图片

```
Prompt: "Change the sky to a starry night sky"
Operation: edit
Reference Image 1: [连接你的图片]
Image Size: 2K
```

### 示例 3: 风格迁移

```
Prompt: "A modern cityscape"
Operation: style_transfer
Reference Image 1: [连接风格参考图]
Image Size: 2K
```

## 工作流程

1. **图片上传**: 将参考图片上传到腾讯云 VOD
   - 申请上传 (ApplyUpload)
   - 上传到 COS (使用临时凭证)
   - 确认上传 (CommitUpload)

2. **创建任务**: 调用 CreateAigcImageTask API 创建 AIGC 任务

3. **轮询状态**: 定期查询任务状态 (DescribeTaskDetail)
   - PROCESSING: 处理中
   - FINISH: 完成
   - FAIL: 失败

4. **下载结果**: 从返回的 URL 下载生成的图片

5. **转换输出**: 将图片转换为 ComfyUI 张量格式

## 费用说明

腾讯云 VOD AIGC API 按照以下方式计费：

- **图片生成**: 按生成的图片数量计费
- **存储**: 临时存储 7 天免费
- **流量**: 下载生成的图片会产生流量费用

具体价格请参考 [腾讯云 VOD 定价](https://cloud.tencent.com/document/product/266/2838)

## 故障排除

### 问题 1: "创建腾讯云客户端失败"

**原因**: 凭证未正确配置

**解决方法**:
- 检查环境变量是否设置正确
- 或在节点参数中填写正确的 SecretId、SecretKey 和 SubAppID

### 问题 2: "上传图片到 VOD 失败"

**原因**: 可能是网络问题或临时凭证过期

**解决方法**:
- 检查网络连接
- 重试操作
- 检查 VOD 服务是否已开通

### 问题 3: "任务超时"

**原因**: AIGC 任务处理时间过长

**解决方法**:
- 增加 `timeout` 参数值
- 降低图片分辨率 (使用 1K 而不是 4K)
- 简化提示词

### 问题 4: "任务失败"

**原因**: 可能是提示词违反内容安全策略

**解决方法**:
- 修改提示词，避免敏感内容
- 检查操作日志中的错误信息
- 如果是合法内容，可以尝试禁用安全过滤 (enable_safety=False)

## 技术细节

### SDK 依赖

- `tencentcloud-sdk-python`: 腾讯云官方 SDK
- `cos-python-sdk-v5`: 腾讯云对象存储 SDK
- `Pillow`: 图片处理
- `torch`: 张量处理 (ComfyUI 依赖)

### API 调用流程

```
用户输入 → 上传参考图片到 VOD → 创建 AIGC 任务 → 轮询任务状态 → 下载结果 → 转换为张量 → 输出
```

### 文件说明

- `nano_banana_tencent.py`: 主插件文件
- `config_tencent.json`: 保存的配置文件 (自动生成)
- `requirements_tencent.txt`: Python 依赖列表
- `README_TENCENT.md`: 本文档

## 与 Gemini 版本的区别

| 特性 | Gemini 版本 | Tencent 版本 |
|------|-------------|--------------|
| API 提供商 | Google Gemini | 腾讯云 VOD |
| 模型 | Gemini 2.5 Flash | GEM (Gemini) 3.0 |
| 凭证类型 | API Key | SecretId + SecretKey |
| 图片上传 | Base64 直接传输 | 上传到 VOD/COS |
| 任务模式 | 流式响应 | 异步任务 + 轮询 |
| 存储 | 无需存储 | 临时存储 7 天 |
| 国内访问 | 可能需要代理 | 国内直连 |

## 注意事项

1. **凭证安全**: 不要将 SecretId 和 SecretKey 硬编码在代码中或提交到版本控制
2. **成本控制**: 每次生成都会产生费用，建议设置费用告警
3. **网络要求**: 需要稳定的网络连接到腾讯云
4. **内容合规**: 生成的内容需要符合相关法律法规
5. **临时存储**: 生成的图片在 VOD 中只保存 7 天

## 许可证

与 ComfyUI-NanoBanano 主项目相同

## 贡献

欢迎提交 Issue 和 Pull Request!

## 支持

如有问题，请访问:
- [腾讯云文档](https://cloud.tencent.com/document/product/266)
- [VOD AIGC API 文档](https://cloud.tencent.com/document/product/266/73993)
- [ComfyUI 社区](https://github.com/comfyanonymous/ComfyUI)

## 更新日志

### v1.0.0 (2025-12-17)
- 初始版本
- 支持图片生成、编辑、风格迁移、对象插入
- 支持最多 5 张参考图片
- 支持 1K/2K/4K 分辨率
- 支持多种宽高比配置
