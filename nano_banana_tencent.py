# -*- coding: utf-8 -*-
import os
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

# 腾讯云 SDK
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.vod.v20180717 import vod_client, models

# 腾讯云 COS SDK
from qcloud_cos import CosConfig, CosS3Client

class ComfyUI_NanoBanana_Tencent:
    def __init__(self):
        # 只从环境变量读取默认凭证，客户端在每次任务执行时动态创建
        self.default_secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID")
        self.default_secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY")
        env_sub_app_id = os.environ.get("TENCENTCLOUD_SUB_APP_ID")
        self.default_sub_app_id = int(env_sub_app_id) if env_sub_app_id else None

    @classmethod
    def IS_CHANGED(self, **kwargs) -> str:
        prompt = kwargs.get("prompt")
        md5_hash = hash(prompt)
        return str(md5_hash) + ":" + str(datetime.now().time())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "Generate a high-quality, photorealistic image",
                    "multiline": True,
                    "tooltip": "Describe what you want to generate or edit"
                }),
                "operation": (["generate", "edit", "style_transfer", "object_insertion"], {
                    "default": "generate",
                    "tooltip": "Choose the type of image operation"
                }),
            },
            "optional": {
                "reference_image_1": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Primary reference image for editing/style transfer"
                }),
                "reference_image_2": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Second reference image (optional)"
                }),
                "reference_image_3": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Third reference image (optional)"
                }),
                "reference_image_4": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fourth reference image (optional)"
                }),
                "reference_image_5": ("IMAGE", {
                    "forceInput": False,
                    "tooltip": "Fifth reference image (optional)"
                }),
                "secret_id": ("STRING", {
                    "default": "",
                    "tooltip": "Tencent Cloud Secret ID"
                }),
                "secret_key": ("STRING", {
                    "default": "",
                    "tooltip": "Tencent Cloud Secret Key"
                }),
                "sub_app_id": ("STRING", {
                    "default": "",
                    "tooltip": "Tencent Cloud VOD SubAppID"
                }),
                "region": ("STRING", {
                    "default": "ap-guangzhou",
                    "tooltip": "Tencent Cloud Region"
                }),
                "aspect_ratio": (["", "1:1", "3:2", "2:3", "3:4", "4:3", "16:9", "9:16", "21:9", "4:5", "5:4"], {
                    "default": "",
                    "tooltip": "Image aspect ratio"
                }),
                "image_size": (["1K", "2K", "4K"], {
                    "default": "1K",
                    "tooltip": "Image resolution size"
                }),
                "enable_safety": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable content safety filters"
                }),
                "poll_interval": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Task status polling interval (seconds)"
                }),
                "timeout": ("INT", {
                    "default": 600,
                    "min": 60,
                    "max": 3600,
                    "tooltip": "Maximum wait time (seconds)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("generated_images", "operation_log")
    FUNCTION = "nano_banana_generate"
    CATEGORY = "Nano Banana Tencent (Tencent Cloud GEM)"
    DESCRIPTION = "Generate and edit images using Tencent Cloud GEM model via VOD AIGC API."

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.cpu()
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor[0]

        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        return Image.fromarray(image_np, mode='RGB')

    def upload_image_to_vod(self, pil_image, vod_client_instance, sub_app_id, operation_log):
        """Upload PIL image to Tencent Cloud VOD and return FileID"""
        try:
            # 1. 将图片保存为临时文件
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            # 2. 申请上传
            file_name = f"image_{int(time.time() * 1000)}.jpg"
            apply_req = models.ApplyUploadRequest()
            apply_req.MediaType = "JPG"
            apply_req.SubAppId = int(sub_app_id)
            apply_req.MediaName = file_name
            # 设置过期时间为 1 小时后
            expire_time = datetime.now().timestamp() + 3600
            apply_req.ExpireTime = datetime.fromtimestamp(expire_time).strftime("%Y-%m-%dT%H:%M:%S+08:00")

            operation_log.append(f"申请上传: {file_name}")
            apply_resp = vod_client_instance.ApplyUpload(apply_req)

            storage_bucket = apply_resp.StorageBucket
            storage_region = apply_resp.StorageRegion
            media_storage_path = apply_resp.MediaStoragePath
            temp_cert = apply_resp.TempCertificate

            operation_log.append(f"申请上传成功: Bucket={storage_bucket}, Region={storage_region}")

            # 3. 上传到 COS（使用临时凭证）
            object_key = media_storage_path.lstrip('/')

            # 创建 COS 配置（使用临时凭证）
            cos_config = CosConfig(
                Region=storage_region,
                SecretId=temp_cert.SecretId,
                SecretKey=temp_cert.SecretKey,
                Token=temp_cert.Token,
                Scheme='https'
            )
            cos_client = CosS3Client(cos_config)

            operation_log.append(f"上传到 COS: {object_key}")

            # 上传文件
            cos_client.put_object(
                Bucket=storage_bucket,
                Key=object_key,
                Body=img_bytes,
                ContentType='image/jpeg'
            )

            operation_log.append(f"上传到 COS 成功")

            # 4. 确认上传
            commit_req = models.CommitUploadRequest()
            commit_req.VodSessionKey = apply_resp.VodSessionKey
            commit_req.SubAppId = int(sub_app_id)

            operation_log.append(f"确认上传")
            commit_resp = vod_client_instance.CommitUpload(commit_req)

            file_id = commit_resp.FileId
            operation_log.append(f"确认上传成功: FileID={file_id}")

            return file_id

        except TencentCloudSDKException as e:
            raise RuntimeError(f"上传图片到 VOD 失败: {e.message}")
        except Exception as e:
            raise RuntimeError(f"上传图片到 VOD 失败: {str(e)}")

    def prepare_images_for_api(self, vod_client_instance, sub_app_id, operation_log,
                                img1=None, img2=None, img3=None, img4=None, img5=None):
        """Convert up to 5 tensor images and upload to VOD, return FileIDs"""
        file_ids = []

        # Process all provided images (up to 5)
        for i, img in enumerate([img1, img2, img3, img4, img5], 1):
            if img is not None:
                # Handle both single images and batched images
                if isinstance(img, torch.Tensor):
                    if len(img.shape) == 4:  # Batch of images
                        # Take only the first image from batch to avoid confusion
                        pil_image = self.tensor_to_image(img[0])
                    else:  # Single image
                        pil_image = self.tensor_to_image(img)

                    operation_log.append(f"处理参考图片 #{i}")
                    file_id = self.upload_image_to_vod(pil_image, vod_client_instance, sub_app_id, operation_log)
                    file_ids.append(file_id)

        return file_ids

    def build_prompt_for_operation(self, prompt, operation, has_references=False):
        """Build optimized prompt based on operation type"""

        base_quality = "Generate a high-quality, photorealistic image"

        if operation == "generate":
            if has_references:
                final_prompt = f"{base_quality} inspired by the style and elements of the reference images. {prompt}."
            else:
                final_prompt = f"{base_quality} of: {prompt}."

        elif operation == "edit":
            if not has_references:
                raise ValueError("Edit operation requires reference images")
            final_prompt = f"Edit the provided reference image(s). {prompt}. Maintain the original composition and quality while making the requested changes."

        elif operation == "style_transfer":
            if not has_references:
                raise ValueError("Style transfer requires reference images")
            final_prompt = f"Apply the style from the reference images to create: {prompt}. Blend the stylistic elements naturally."

        elif operation == "object_insertion":
            if not has_references:
                raise ValueError("Object insertion requires reference images")
            final_prompt = f"Insert or blend the following into the reference image(s): {prompt}. Ensure natural lighting, shadows, and perspective."

        return final_prompt

    def create_aigc_task(self, vod_client_instance, sub_app_id, file_ids, prompt,
                         aspect_ratio, image_size, enable_safety, operation_log):
        """Create AIGC image task and return TaskID"""
        try:
            request = models.CreateAigcImageTaskRequest()
            request.SubAppId = int(sub_app_id)
            request.ModelName = "GEM"  # Gemini
            request.ModelVersion = "3.0"

            # 设置输入图片（FileID 列表）
            request.FileInfos = []
            for file_id in file_ids:
                file_info = models.AigcImageTaskInputFileInfo()
                file_info.Type = "File"
                file_info.FileId = file_id
                request.FileInfos.append(file_info)

            # 设置提示词
            request.Prompt = prompt
            request.EnhancePrompt = "Disabled"

            # 设置输出配置
            output_config = models.AigcImageOutputConfig()
            output_config.StorageMode = "Temporary"  # 临时存储（7天）
            output_config.PersonGeneration = "AllowAdult"
            output_config.Resolution = image_size
            output_config.AspectRatio = aspect_ratio if aspect_ratio else None
            output_config.InputComplianceCheck = "Enabled" if enable_safety else "Disabled"
            output_config.OutputComplianceCheck = "Disabled"

            request.OutputConfig = output_config

            operation_log.append(f"创建 AIGC 任务...")
            response = vod_client_instance.CreateAigcImageTask(request)

            task_id = response.TaskId
            operation_log.append(f"AIGC 任务已创建: TaskID={task_id}")

            return task_id

        except TencentCloudSDKException as e:
            raise RuntimeError(f"创建 AIGC 任务失败: {e.message}")

    def poll_task_status(self, vod_client_instance, sub_app_id, task_id, poll_interval, timeout, operation_log):
        """Poll task status until completion and return result FileURL"""
        start_time = time.time()
        last_progress = -1

        while True:
            # 检查超时
            if time.time() - start_time > timeout:
                raise RuntimeError(f"任务超时（{timeout}秒）")

            # 查询任务详情
            try:
                request = models.DescribeTaskDetailRequest()
                request.TaskId = task_id
                request.SubAppId = int(sub_app_id)

                response = vod_client_instance.DescribeTaskDetail(request)

                # 检查任务状态
                aigc_task = response.AigcImageTask
                if aigc_task is None:
                    raise RuntimeError("任务不存在或非 AIGC 任务")

                status = aigc_task.Status
                progress = aigc_task.Progress

                # 仅当进度变化时记录日志
                if progress != last_progress:
                    operation_log.append(f"任务状态: {status}, 进度: {progress}%")
                    last_progress = progress

                if status == "FINISH":
                    if aigc_task.ErrCode and aigc_task.ErrCode != 0:
                        raise RuntimeError(f"任务失败，错误码: {aigc_task.ErrCode}，错误信息: {aigc_task.Message}")

                    # 任务完成，获取文件 URL
                    if not aigc_task.Output or not aigc_task.Output.FileInfos:
                        raise RuntimeError("任务完成但没有输出文件")

                    file_url = aigc_task.Output.FileInfos[0].FileUrl
                    operation_log.append(f"任务完成: FileURL={file_url}")
                    return file_url

                elif status == "FAIL":
                    err_msg = aigc_task.Message or "未知错误"
                    raise RuntimeError(f"任务失败: {err_msg}")

                elif status == "PROCESSING":
                    # 继续等待
                    time.sleep(poll_interval)

                else:
                    operation_log.append(f"未知的任务状态: {status}")
                    time.sleep(poll_interval)

            except TencentCloudSDKException as e:
                operation_log.append(f"查询任务状态失败: {e.message}")
                time.sleep(poll_interval)

    def download_and_convert_image(self, file_url, operation_log):
        """Download image from URL and convert to tensor"""
        try:
            operation_log.append(f"下载图片: {file_url}")

            # 下载图片（带重试）
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(file_url, timeout=60)
                    response.raise_for_status()
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    operation_log.append(f"下载失败，重试 {attempt + 1}/{max_retries}: {str(e)}")
                    time.sleep(2)

            # 转换为 PIL Image
            image = Image.open(BytesIO(response.content))

            # Ensure it's RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array and normalize
            img_np = np.array(image).astype(np.float32) / 255.0

            # Create tensor with correct dimensions
            img_tensor = torch.from_numpy(img_np)[None,]

            operation_log.append(f"图片下载完成: {image.size}")

            return img_tensor

        except Exception as e:
            raise RuntimeError(f"下载图片失败: {str(e)}")

    def nano_banana_generate(self, prompt, operation,
                             reference_image_1=None, reference_image_2=None,
                             reference_image_3=None, reference_image_4=None, reference_image_5=None,
                             secret_id="", secret_key="", sub_app_id="", region="ap-guangzhou",
                             model_version="3.0", aspect_ratio="", image_size="1K",
                             enable_safety=True, poll_interval=5, timeout=600):

        operation_log = []

        # 使用传入的参数，如果为空则使用默认值
        actual_secret_id = secret_id.strip() or self.default_secret_id
        actual_secret_key = secret_key.strip() or self.default_secret_key
        actual_sub_app_id = sub_app_id if sub_app_id != "" else self.default_sub_app_id
        actual_sub_app_id = int(actual_sub_app_id)

        # 验证凭证
        if not actual_secret_id or not actual_secret_key:
            raise ValueError(
                "NANO BANANA TENCENT ERROR: 未提供腾讯云凭证!\n\n"
                "需要设置 TENCENTCLOUD_SECRET_ID、TENCENTCLOUD_SECRET_KEY 和 TENCENTCLOUD_SUB_APP_ID。\n"
                "请访问: https://console.cloud.tencent.com/cam/capi"
            )

        if not actual_sub_app_id:
            raise ValueError(
                "NANO BANANA TENCENT ERROR: 未提供 SubAppID!\n\n"
                "请设置 TENCENTCLOUD_SUB_APP_ID 或在节点参数中填写 sub_app_id。"
            )

        # 动态创建 VOD 客户端（每次任务独立创建）
        try:
            cred = credential.Credential(actual_secret_id, actual_secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "vod.tencentcloudapi.com"
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            vod_client_instance = vod_client.VodClient(cred, region, clientProfile)
        except Exception as e:
            raise ValueError(f"创建腾讯云客户端失败: {str(e)}")

        try:
            # Log operation start
            operation_log.append(f"NANO BANANA TENCENT 操作日志")
            operation_log.append(f"操作类型: {operation.upper()}")
            operation_log.append(f"模型版本: {model_version}")
            operation_log.append(f"分辨率: {image_size}")
            operation_log.append(f"宽高比: {aspect_ratio}")
            operation_log.append(f"安全过滤: {enable_safety}")
            operation_log.append(f"轮询间隔: {poll_interval}秒")
            operation_log.append(f"超时时间: {timeout}秒")
            operation_log.append("")

            # Process reference images (up to 5)
            file_ids = self.prepare_images_for_api(
                vod_client_instance, actual_sub_app_id, operation_log,
                reference_image_1, reference_image_2, reference_image_3,
                reference_image_4, reference_image_5
            )
            has_references = len(file_ids) > 0

            operation_log.append(f"参考图片数量: {len(file_ids)}")
            operation_log.append("")

            # Build optimized prompt
            final_prompt = self.build_prompt_for_operation(prompt, operation, has_references)
            operation_log.append(f"最终提示词: {final_prompt[:150]}...")
            operation_log.append("")

            # Create AIGC task
            task_id = self.create_aigc_task(
                vod_client_instance, actual_sub_app_id, file_ids, final_prompt,
                aspect_ratio, image_size, enable_safety, operation_log
            )
            operation_log.append("")

            # Poll task status
            file_url = self.poll_task_status(
                vod_client_instance, actual_sub_app_id, task_id, poll_interval, timeout, operation_log
            )
            operation_log.append("")

            # Download and convert image
            img_tensor = self.download_and_convert_image(file_url, operation_log)

            operation_log.append("")
            operation_log.append("=== 操作成功完成 ===")

            # 返回结果
            return (img_tensor, "\n".join(operation_log))

        except Exception as e:
            operation_log.append("")
            operation_log.append(f"=== 操作失败 ===")
            operation_log.append(f"错误: {str(e)}")
            # Re-raise the error instead of returning placeholder
            raise RuntimeError(f"NANO BANANA TENCENT ERROR: {str(e)}\n\n日志:\n" + "\n".join(operation_log))


# Node registration
NODE_CLASS_MAPPINGS = {
    "ComfyUI_NanoBanana_Tencent": ComfyUI_NanoBanana_Tencent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI_NanoBanana_Tencent": "Nano Banana Tencent (Tencent Cloud GEM)",
}
