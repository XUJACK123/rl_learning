"""
导出 PPO 策略为 ONNX 格式，用于部署到小车

输入维度: 3  (bbox_cx, bbox_cy, z)
输出维度: 3  (vx, vy, omega)
"""
import torch
from stable_baselines3 import PPO


def export_to_onnx():
    model_path = "final_car_tracking_policy.zip"
    onnx_output_path = "car_tracking_policy.onnx"

    print(f"Loading model: {model_path}...")
    model = PPO.load(model_path)

    # 提取 Actor（策略网络）
    actor_net = model.policy.policy_net

    # Dummy input: batch=1, 3 维观测（检测模型输出）
    dummy_input = torch.randn(1, 3, dtype=torch.float32)

    torch.onnx.export(
        actor_net,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["detection_output"],          # 3维：bbox中心xy + 深度z
        output_names=["velocity_command"],         # 3维：vx, vy, omega
        dynamic_axes={
            "detection_output": {0: "batch"},
            "velocity_command": {0: "batch"},
        },
    )

    print(f"✅ ONNX 模型已导出至: {onnx_output_path}")
    print(f"   输入: detection_output (batch, 3)")
    print(f"   输出: velocity_command (batch, 3)")


if __name__ == "__main__":
    export_to_onnx()
