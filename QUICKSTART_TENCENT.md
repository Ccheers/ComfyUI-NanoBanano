# 快速入门 - Nano Banana Tencent

5 分钟开始使用腾讯云 GEM 模型生成图片

## 1. 安装依赖 (1 分钟)

```bash
cd ComfyUI/custom_nodes/ComfyUI-NanoBanano
pip install -r requirements_tencent.txt
```

## 2. 获取凭证 (2 分钟)

1. 访问 [腾讯云 API 密钥](https://console.cloud.tencent.com/cam/capi)
2. 点击"新建密钥"获取 `SecretId` 和 `SecretKey`
3. 访问 [云点播控制台](https://console.cloud.tencent.com/vod/overview) 开通服务
4. 在应用管理中获取 `SubAppID` (默认应用 ID 通常显示在页面上)

## 3. 配置凭证 (1 分钟)

### 方式 A: 环境变量 (推荐)

```bash
export TENCENTCLOUD_SECRET_ID="your_secret_id_here"
export TENCENTCLOUD_SECRET_KEY="your_secret_key_here"
export TENCENTCLOUD_SUB_APP_ID="xxx"  # 替换为你的 SubAppID
```

### 方式 B: 在 ComfyUI 界面输入

在节点参数中填写凭证，会自动保存到配置文件。

## 4. 测试凭证 (可选)

```bash
python test_tencent_credentials.py
```

如果看到 "✓ 所有检查通过!"，说明配置成功。

## 5. 在 ComfyUI 中使用 (1 分钟)

1. 启动 ComfyUI
2. 在节点库中搜索 "Nano Banana Tencent"
3. 添加节点并填写提示词
4. 运行工作流

### 最简单的例子

```
节点: Nano Banana Tencent
Prompt: "A cute cat playing with yarn"
Operation: generate
Image Size: 1K
Aspect Ratio: 1:1

然后连接到 SaveImage 节点保存图片
```

## 常见问题

**Q: 如何知道我的 SubAppID?**
A: 登录 [云点播控制台](https://console.cloud.tencent.com/vod/overview)，在"应用管理"中查看

**Q: 会产生多少费用?**
A: 每生成一张图片约 0.04 元人民币（具体以腾讯云官网价格为准）

**Q: 生成速度慢怎么办?**
A:
- 使用 1K 分辨率（最快）
- 降低轮询间隔（但不要设置太低）
- 检查网络连接

**Q: 报错 "上传图片到 VOD 失败"?**
A:
- 检查凭证是否正确
- 确保已开通云点播服务
- 检查网络连接

## 下一步

- 查看 [完整文档](README_TENCENT.md) 了解所有功能
- 尝试图片编辑、风格迁移等高级功能
- 调整参数优化生成效果

## 获取帮助

- 腾讯云文档: https://cloud.tencent.com/document/product/266
- Issues: 在项目仓库提交问题

---

🎉 现在开始创作吧！
