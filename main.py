# main.py
import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime

from config import Config
from data.dataloader import create_dataloaders
from models.convlstm import AirQualityModel
from utils.trainer import Trainer
from utils.predictor import Predictor
from utils.evaluation import Evaluator
from utils.visualization import Visualizer
import matplotlib.pyplot as plt
import matplotlib as mpl
from models.transformer import SpaceTimeTransformer
from torch.utils.data import DataLoader, Subset
from models.swin_transformer import SwinTransformerForAirQuality
from models.st_transformer import SpatioTemporalTransformer
from models.dilated_attention_convlstm import DilatedAttentionConvLSTM

# 设置全局字体为文泉驿微米黑
mpl.rcParams['font.family'] = 'WenQuanYi Micro Hei'
# 解决负号显示问题
mpl.rcParams['axes.unicode_minus'] = False


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='空气质量网格预测')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'predict'],
                        help='运行模式: train, test, predict')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径')
    parser.add_argument('--device', type=str, default=None,
                        help='设备: cuda, cpu')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--year', type=str, default="2019",
                        help='要处理的年份，如"2019"或"2018,2019"')
    parser.add_argument('--input-hours', type=int, default=None,
                        help='输入时间长度（小时），默认使用配置文件中的值')
    parser.add_argument('--forecast-horizon', type=int, default=None,
                        help='预测时间长度（小时），默认使用配置文件中的值')
    parser.add_argument('--vis-step', type=int, default=None,
                        help='可视化哪个预测时间步，默认为0（第一个时间步）')
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['convlstm', 'transformer', 'swin', 'st_transformer', 'dilated_attention_convlstm', 'improved_st_transformer'],
                        help='模型类型: convlstm, transformer, swin, st_transformer, dilated_attention_convlstm, improved_st_transformer')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='使用缓存数据加快加载(默认:开启)')
    parser.add_argument('--no-cache', action='store_true',
                        help='禁用数据缓存')
    parser.add_argument('--cache-dir', type=str, default='./data/cache',
                        help='缓存数据目录(默认:./data/cache)')
    parser.add_argument('--fast-dev', action='store_true',
                        help='使用小数据集进行快速开发测试')
    # 空间评估参数
    parser.add_argument('--spatial-eval', action='store_true',
                        help='执行空间性能评估，分析每个网格点的预测性能')


    return parser.parse_args()


def train(config, checkpoint_path=None, use_cache=True, cache_dir='./data/cache', fast_dev=False):
    """训练模型"""
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        config, use_cache=use_cache, cache_dir=cache_dir
    )

    # 使用Config中已定义的MODEL_DIR路径
    config_path = os.path.join(config.MODEL_DIR, 'training_config.txt')
    Config.save(config_path)
    print(f"训练配置已保存至: {config_path}")

    # 可以同时保存命令行参数
    args_path = os.path.join(config.MODEL_DIR, 'command_args.txt')
    with open(args_path, 'w') as f:
        import sys
        f.write(' '.join(sys.argv) + '\n')
    print(f"命令行参数已保存至: {args_path}")

    # 快速开发模式处理 - 修复版本
    if fast_dev:
        print("⚡ 快速开发模式: 使用小数据集")

        # 只使用少量样本
        train_indices = list(range(min(20, len(train_loader.dataset))))
        val_indices = list(range(min(10, len(val_loader.dataset))))
        test_indices = list(range(min(10, len(test_loader.dataset))))

        # 创建子集数据集
        train_subset = Subset(train_loader.dataset, train_indices)
        val_subset = Subset(val_loader.dataset, val_indices)
        test_subset = Subset(test_loader.dataset, test_indices)

        # 调整批次大小
        batch_size = min(config.BATCH_SIZE, 2)

        # 重新创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    print("创建模型...")
    if config.MODEL_TYPE == "transformer":
        model = SpaceTimeTransformer(config)
        print(f"使用 Transformer 模型进行训练")
    elif config.MODEL_TYPE == "swin":
        model = SwinTransformerForAirQuality(config)
        print(f"使用 Swin Transformer 模型进行训练")
    elif config.MODEL_TYPE == "st_transformer":
        model = SpatioTemporalTransformer(config)
        print(f"使用 ST-Transformer 模型进行训练")
    elif config.MODEL_TYPE == "dilated_attention_convlstm":  # 添加新的模型类型
        model = DilatedAttentionConvLSTM(config)
        print(f"使用 带膨胀卷积和时间注意力的ConvLSTM 模型进行训练")
    elif config.MODEL_TYPE == "improved_st_transformer":
        from models.improved_st_transformer import ImprovedSTTransformer
        model = ImprovedSTTransformer(config)
        print(f"使用 改进的ST-Transformer 模型进行训练")
    else:
        model = AirQualityModel(config)
        print(f"使用 ConvLSTM 模型进行训练")

    print("创建训练器...")
    trainer = Trainer(model, config, norm_params)

    # 加载检查点(如果有)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载检查点 {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)

    # 训练模型
    print("开始训练...")
    history = trainer.train(train_loader, val_loader)

    # 可视化训练历史
    visualizer = Visualizer(config)
    visualizer.plot_training_history(history)

    # 评估模型
    print("模型评估...")
    evaluator = Evaluator(model, config, norm_params)
    metrics = evaluator.evaluate(test_loader)

    # 预测和可视化
    print("预测和可视化...")
    predictor = Predictor(model, config, norm_params)

    # 加载最佳模型
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model_weights.pt')
    if os.path.exists(best_model_path):
        predictor.load_model(best_model_path)

    # 获取一个批次用于可视化
    X, y, mask = next(iter(test_loader))

    # 预测
    with torch.no_grad():
        pred = predictor.predict(X, mask)

    # 可视化样本
    for i in range(min(2, X.size(0))):
        # 可视化第一个和最后一个预测时间步
        visualizer.plot_sample(X, y, mask, pred, sample_idx=i, time_idx=0)
        if config.FORECAST_HORIZON > 1:
            visualizer.plot_sample(X, y, mask, pred, sample_idx=i, time_idx=config.FORECAST_HORIZON - 1)

        # 可视化完整预测序列
        for p_idx in range(len(config.PREDICTION_POLLUTANTS)):
            visualizer.plot_forecast_sequence(X, y, pred, mask, sample_idx=i, pollutant_idx=p_idx)

    # 绘制误差分布
    evaluator.plot_error_distribution(test_loader, predictor)

    print(f"训练完成，模型保存在 {config.MODEL_DIR}")
    return model, history, metrics


def test(config, checkpoint_path, use_cache=True, cache_dir='./data/cache', fast_dev=False, spatial_eval=False):
    """测试模型"""
    print("创建数据加载器...")
    _, _, test_loader, norm_params = create_dataloaders(
        config, use_cache=use_cache, cache_dir=cache_dir
    )

    # 快速开发模式处理 - 修复版本
    if fast_dev:
        print("⚡ 快速开发模式: 使用小数据集")

        # 只使用少量样本
        test_indices = list(range(min(10, len(test_loader.dataset))))

        # 创建子集数据集
        test_subset = Subset(test_loader.dataset, test_indices)

        # 调整批次大小
        batch_size = min(config.BATCH_SIZE, 2)

        # 重新创建数据加载器
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    print("创建模型...")
    if config.MODEL_TYPE == "transformer":
        model = SpaceTimeTransformer(config)
        print(f"使用 Transformer 模型进行训练")
    elif config.MODEL_TYPE == "swin":
        model = SwinTransformerForAirQuality(config)
        print(f"使用 Swin Transformer 模型进行训练")
    elif config.MODEL_TYPE == "st_transformer":
        model = SpatioTemporalTransformer(config)
        print(f"使用 ST-Transformer 模型进行训练")
    elif config.MODEL_TYPE == "dilated_attention_convlstm":  # 添加新的模型类型
        model = DilatedAttentionConvLSTM(config)
        print(f"使用 带膨胀卷积和时间注意力的ConvLSTM 模型进行训练")
    elif config.MODEL_TYPE == "improved_st_transformer":
        from models.improved_st_transformer import ImprovedSTTransformer
        model = ImprovedSTTransformer(config)
        print(f"使用 改进的ST-Transformer 模型进行训练")
    else:
        model = AirQualityModel(config)
        print(f"使用 ConvLSTM 模型进行训练")

    print(f"加载检查点 {checkpoint_path}...")
    predictor = Predictor(model, config, norm_params)
    predictor.load_model(checkpoint_path)

    print("开始评估...")
    evaluator = Evaluator(model, config, norm_params)
    metrics = evaluator.evaluate(test_loader, predictor)

    # 执行空间评估（如果需要）
    if spatial_eval:
        print("\n开始空间性能评估...")
        from utils.spatial_evaluation import SpatialEvaluator
        spatial_evaluator = SpatialEvaluator(model, config, norm_params)
        spatial_metrics = spatial_evaluator.evaluate_spatial_performance(test_loader, predictor)
        print("空间性能评估完成")

    # 可视化
    visualizer = Visualizer(config)

    # 获取一个批次用于可视化
    X, y, mask = next(iter(test_loader))

    # 预测
    with torch.no_grad():
        pred = predictor.predict(X, mask)

    # 可视化样本
    for i in range(min(2, X.size(0))):
        # 可视化第一个和最后一个预测时间步
        visualizer.plot_sample(X, y, mask, pred, sample_idx=i, time_idx=0)
        if config.FORECAST_HORIZON > 1:
            visualizer.plot_sample(X, y, mask, pred, sample_idx=i, time_idx=config.FORECAST_HORIZON - 1)

        # 可视化完整预测序列
        for p_idx in range(len(config.PREDICTION_POLLUTANTS)):
            visualizer.plot_forecast_sequence(X, y, pred, mask, sample_idx=i, pollutant_idx=p_idx)

    # 保存第一个样本的预测结果
    predictor.save_prediction_to_nc(
        pred[0], y[0], mask[0],
        os.path.join(config.RESULT_DIR, 'sample_prediction.nc')
    )

    print(f"测试完成，结果保存在 {config.RESULT_DIR}")
    return metrics


def predict(config, checkpoint_path, vis_step=None, use_cache=True, cache_dir='./data/cache', fast_dev=False):
    """预测单个样本"""
    print("创建数据加载器...")
    _, _, test_loader, norm_params = create_dataloaders(
        config, use_cache=use_cache, cache_dir=cache_dir
    )

    # 快速开发模式处理 - 修复版本
    if fast_dev:
        print("⚡ 快速开发模式: 使用小数据集")

        # 只使用少量样本
        test_indices = list(range(min(10, len(test_loader.dataset))))

        # 创建子集数据集
        test_subset = Subset(test_loader.dataset, test_indices)

        # 调整批次大小
        batch_size = min(config.BATCH_SIZE, 2)

        # 重新创建数据加载器
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    print("创建模型...")
    if config.MODEL_TYPE == "transformer":
        model = SpaceTimeTransformer(config)
        print(f"使用 Transformer 模型进行训练")
    elif config.MODEL_TYPE == "swin":
        model = SwinTransformerForAirQuality(config)
        print(f"使用 Swin Transformer 模型进行训练")
    elif config.MODEL_TYPE == "st_transformer":
        model = SpatioTemporalTransformer(config)
        print(f"使用 ST-Transformer 模型进行训练")
    elif config.MODEL_TYPE == "dilated_attention_convlstm":  # 添加新的模型类型
        model = DilatedAttentionConvLSTM(config)
        print(f"使用 带膨胀卷积和时间注意力的ConvLSTM 模型进行训练")
    elif config.MODEL_TYPE == "improved_st_transformer":
        from models.improved_st_transformer import ImprovedSTTransformer
        model = ImprovedSTTransformer(config)
        print(f"使用 改进的ST-Transformer 模型进行训练")
    else:
        model = AirQualityModel(config)
        print(f"使用 ConvLSTM 模型进行训练")

    print(f"加载检查点 {checkpoint_path}...")
    predictor = Predictor(model, config, norm_params)
    predictor.load_model(checkpoint_path)

    # 获取一个批次
    X, y, mask = next(iter(test_loader))

    # 预测
    with torch.no_grad():
        pred = predictor.predict(X, mask)

    # 可视化
    visualizer = Visualizer(config)

    # 设置要可视化的时间步
    time_step = 0 if vis_step is None else min(vis_step, config.FORECAST_HORIZON - 1)

    # 可视化多个样本
    for i in range(min(3, X.size(0))):
        # 可视化指定时间步
        visualizer.plot_sample(X, y, mask, pred, sample_idx=i, time_idx=time_step)

        # 也可视化完整预测序列
        for p_idx in range(len(config.PREDICTION_POLLUTANTS)):
            visualizer.plot_forecast_sequence(X, y, pred, mask, sample_idx=i, pollutant_idx=p_idx)

    # 保存预测结果
    for i in range(min(3, X.size(0))):
        predictor.save_prediction_to_nc(
            pred[i], y[i], mask[i],
            os.path.join(config.RESULT_DIR, f'prediction_sample_{i}.nc')
        )
        predictor.visualize_prediction(
            pred[i], y[i], mask[i],
            os.path.join(config.PLOT_DIR, f'prediction_sample_{i}.png')
        )

    # 创建动画
    visualizer.create_animation(X[0], mask[0], pollutant_idx=0)

    print(f"预测完成，结果保存在 {config.RESULT_DIR}")


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    config = Config()

    # 设置年份
    if args.year:
        config.YEARS = args.year.split(',')

    # 设置输入和预测时长（如果提供）
    if args.input_hours:
        config.INPUT_HOURS = args.input_hours
        print(f"使用自定义输入时长: {config.INPUT_HOURS}小时")

    if args.forecast_horizon:
        config.FORECAST_HORIZON = args.forecast_horizon
        print(f"使用自定义预测时长: {config.FORECAST_HORIZON}小时")

    if args.model_type:
        config.MODEL_TYPE = args.model_type
        print(f"使用模型类型: {config.MODEL_TYPE}")

    print(f"输入变量: {config.ALL_FEATURES}")
    print(f"预测变量: {config.PREDICTION_POLLUTANTS}")

    # 设置设备
    if args.device:
        config.DEVICE = torch.device(args.device)

    # 设置随机种子
    if args.seed:
        config.SEED = args.seed

    set_seed(config.SEED)

    # 处理缓存设置
    use_cache = args.use_cache and not args.no_cache
    cache_dir = args.cache_dir

    # 创建缓存目录(如果不存在)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"使用数据缓存，缓存目录: {cache_dir}")
    else:
        print("数据缓存已禁用")

    # 打印配置信息
    print(f"运行设备: {config.DEVICE}")
    print(f"随机种子: {config.SEED}")
    print(f"输入时长: {config.INPUT_HOURS}小时")
    print(f"预测时长: {config.FORECAST_HORIZON}小时")
    print(f"输出目录: {config.OUTPUT_ROOT}")

    # 是否使用快速开发模式
    fast_dev = args.fast_dev

    # 根据模式运行
    if args.mode == 'train':
        train(config, args.checkpoint, use_cache=use_cache, cache_dir=cache_dir, fast_dev=fast_dev)
    elif args.mode == 'test':
        if not args.checkpoint:
            raise ValueError("测试模式需要提供检查点路径")
        test(config, args.checkpoint, use_cache=use_cache, cache_dir=cache_dir,
             fast_dev=fast_dev, spatial_eval=args.spatial_eval)

    elif args.mode == 'predict':
        if not args.checkpoint:
            raise ValueError("预测模式需要提供检查点路径")
        predict(config, args.checkpoint, vis_step=args.vis_step, use_cache=use_cache,
                cache_dir=cache_dir, fast_dev=fast_dev)


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"程序运行时间: {end_time - start_time}")