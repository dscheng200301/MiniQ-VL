"""
下载评估数据集脚本
自动下载 COCO Val2017 图像用于评估
"""
import os
import urllib.request
import zipfile
from pathlib import Path

def download_coco_val():
    """下载 COCO Val2017 数据集"""
    output_dir = Path(__file__).parent.parent / "dataset" / "eval_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"下载目录: {output_dir}")
    
    zip_path = output_dir / "val2017.zip"
    
    if (output_dir / "val2017").exists():
        print("COCO Val2017 已存在，跳过下载")
        return
    
    url = "http://images.cocodataset.org/zips/val2017.zip"
    print(f"正在下载 COCO Val2017... (约 600MB)")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成，正在解压...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # 移动文件到根目录
        extracted_dir = output_dir / "val2017"
        if extracted_dir.exists():
            for item in extracted_dir.iterdir():
                item.rename(output_dir / item.name)
            extracted_dir.rmdir()
        
        # 删除 zip 文件
        zip_path.unlink()
        print(f"解压完成！")
        print(f"图像保存在: {output_dir}")
        
        # 统计图像数量
        images = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"共 {len(images)} 张图像")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载: https://cocodataset.org")


if __name__ == "__main__":
    download_coco_val()
