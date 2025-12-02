"""
========================================
FILE 2: model_training_optimized.py
========================================
Training tối ưu cho BALANCED DATASET
Data: ~20,000 train + ~20,000 val, 80 classes (balanced)
Target: 250 images/class (balanced)
Mục tiêu: mAP 0.78-0.82
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import yaml
import time
from datetime import datetime

class OptimizedTrainer:
    def __init__(self, yaml_config_path, model_size='n'):
        """
        model_size: 
            'n' = nano (KHUYẾN NGHỊ - phù hợp với 80 classes balanced)
            's' = small (nếu muốn accuracy cao hơn)
        """
        self.yaml_config = yaml_config_path
        self.model_size = model_size
        self.model = None
        
        print("="*70)
        print("🔧 KIỂM TRA PHẦN CỨNG")
        print("="*70)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Device: {self.device.upper()}")
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 VRAM: {vram:.2f} GB")
            
            # Auto-detect P100 and optimize
            is_p100 = 'P100' in gpu_name or 'Tesla P100' in gpu_name
            if is_p100:
                print(f"\n🚀 P100 DETECTED - OPTIMIZING FOR 12H LIMIT")
                print(f"   ⚡ Auto-tuning batch size and config...")
            
            print(f"\n💡 CẤU HÌNH CHO BALANCED DATASET:")
            print(f"   📊 ~20,000 train + ~20,000 val (250 images/class)")
            print(f"   🎯 80 classes (balanced)")
            print(f"   ⚖️  Balanced: ~250 images per class")
            print(f"\n   KHUYẾN NGHỊ:")
            print(f"   - Model: 'n' (nano - đủ cho balanced data)")
            print(f"   - Epochs: 100 (data tốt cần nhiều epochs)")
            
            # Optimize batch size based on GPU
            if is_p100:
                print(f"   - Batch: 40-48 (P100 optimized)")
                print(f"   - Workers: 12-16 (faster data loading)")
                print(f"   - Cache: Enabled (faster training)")
            elif vram >= 15:
                print(f"   - Batch: 40-48 (optimal)")
            elif vram >= 12:
                print(f"   - Batch: 32-40")
            else:
                print(f"   - Batch: 24-32")
            
            print(f"   - LR: 0.002 (cao hơn cho convergence nhanh)")
            print(f"   - Augmentation: VỪA PHẢI (data đã balance)")
        
        print("="*70)
    
    def load_model(self):
        """Load model"""
        model_name = f'yolov8{self.model_size}.pt'
        
        print(f"\n📦 Đang tải: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            print(f"✓ Loaded!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            raise
        
        try:
            with open(self.yaml_config, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"\n📊 Dataset Info:")
            print(f"   - Path: {config['path']}")
            print(f"   - Classes: {config['nc']}")
            
            # Handle names (can be dict or list)
            if isinstance(config['names'], dict):
                names_list = [config['names'][i] for i in sorted(config['names'].keys())[:5]]
            else:
                names_list = config['names'][:5]
            print(f"   - Names: {', '.join(names_list)} ...")
            
            # Try to count actual images if possible
            train_path = Path(config['path']) / config['train']
            val_path = Path(config['path']) / config['val']
            train_count = len(list(train_path.glob('*.jpg'))) if train_path.exists() else 0
            val_count = len(list(val_path.glob('*.jpg'))) if val_path.exists() else 0
            
            if train_count > 0 or val_count > 0:
                print(f"   - Train images: {train_count:,}")
                print(f"   - Val images: {val_count:,}")
                if train_count > 0:
                    print(f"   - Avg per class: ~{train_count // config['nc']:.0f} images")
            
            print(f"\n   ✨ BALANCED DATA = BETTER TRAINING!")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            raise
        
        return self.model
    
    def train(self, epochs=100, imgsz=640, batch=None, patience=40, save_period=5):
        """
        Training tối ưu cho BALANCED dataset
        Auto-optimize batch size for P100
        """
        
        # Auto-optimize batch size for P100
        if batch is None and self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            is_p100 = 'P100' in gpu_name or 'Tesla P100' in gpu_name
            
            if is_p100:
                # For accuracy: use smaller batch for more gradient updates
                # Model 's' needs smaller batch due to higher VRAM usage
                if self.model_size == 's':
                    batch = 28  # Smaller batch for model 's' (VRAM limit)
                else:
                    batch = 36  # Optimal for model 'n'
                print(f"\n⚡ P100 detected - Auto-setting batch size: {batch} (model '{self.model_size}', accuracy-optimized)")
            elif vram >= 15:
                batch = 40
            elif vram >= 12:
                batch = 32
            else:
                batch = 24
        elif batch is None:
            batch = 32
        
        print("\n" + "="*70)
        print("🚀 TRAINING (OPTIMIZED FOR BALANCED DATA)")
        print("="*70)
        
        print(f"\n⚙️  CONFIG:")
        print(f"   Model: YOLOv8{self.model_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Image: {imgsz}")
        print(f"   Batch: {batch}")
        print(f"   Patience: {patience}")
        print(f"   Optimizer: SGD (better than AdamW for balanced data)")
        print(f"   LR: 0.002 (higher for faster convergence)")
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            is_p100 = 'P100' in gpu_name or 'Tesla P100' in gpu_name
            
            if is_p100:
                # P100 time estimate based on ACTUAL training logs
                # From log: 120 epochs in 7.689h = 0.064h/epoch (model 's', batch 28)
                # Với augmentation mạnh hơn: +5-8% thời gian
                base_time_per_epoch = 0.064  # Actual from training log (model 's', batch 28)
                if self.model_size == 's':
                    model_multiplier = 1.0  # Đã là model 's'
                    batch_multiplier = 1.0  # Batch 28 đã được tính
                    aug_multiplier = 1.08   # Augmentation mạnh hơn +8%
                else:
                    model_multiplier = 0.56  # Model 'n' nhanh hơn ~1.8x
                    batch_multiplier = 0.95 if batch >= 36 else 1.0
                    aug_multiplier = 1.05   # Augmentation nhẹ hơn cho model 'n'
                
                est_time_per_epoch = base_time_per_epoch * model_multiplier * batch_multiplier * aug_multiplier
                total_time = epochs * est_time_per_epoch
                total_time_max = epochs * est_time_per_epoch * 1.1  # 10% buffer
                est_time = f"{total_time:.1f}-{total_time_max:.1f}h"
                print(f"\n⏱️  Estimated Time: {est_time} (improved accuracy-optimized)")
                
                if total_time_max <= 12:
                    print(f"   ✅ Safe within 12h limit! ({total_time_max:.1f}h / 12h)")
                    print(f"   🎯 Accuracy focus: {epochs} epochs, Batch {batch}, More training")
                elif total_time_max <= 13:
                    print(f"   ⚠️  Close to limit ({total_time_max:.1f}h / 12h) - should be OK")
                else:
                    print(f"   ❌ Exceeds 12h limit ({total_time_max:.1f}h) - consider reducing epochs")
            else:
                est_time = f"{epochs * 0.07:.1f}-{epochs * 0.09:.1f}h"
                print(f"\n⏱️  Time: {est_time}")
        else:
            est_time = "N/A"
            print(f"\n⏱️  Time: {est_time}")
        
        print(f"\n🎯 EXPECTED RESULTS (0.8+ ACCURACY TARGET):")
        print(f"   Baseline (imbalanced):  mAP50 = 0.69")
        model_boost = "Model 's' (+3-5%)" if self.model_size == 's' else "Model 'n'"
        if epochs >= 180:
            print(f"   Target (balanced, {epochs} epochs, {model_boost}): mAP50 = 0.80-0.85")
            print(f"   Improvement:            +16-23%")
            print(f"\n   🎯 ACCURACY OPTIMIZATIONS FOR 0.8+ TARGET:")
            print(f"   ✅ Model '{self.model_size}' (larger capacity)")
            print(f"   ✅ {epochs} epochs (maximum training)")
            print(f"   ✅ Batch {batch} (more gradient updates)")
            print(f"   ✅ Optimized augmentation")
            print(f"   ✅ Fine-tuned learning rate schedule")
            print(f"   ✅ Extended fine-tuning phase")
        elif epochs >= 120:
            print(f"   Target (balanced, {epochs} epochs, {model_boost}): mAP50 = 0.78-0.83")
            print(f"   Improvement:            +13-20%")
        else:
            print(f"   Target (balanced, {model_boost}): mAP50 = 0.78-0.82")
            print(f"   Improvement:            +13-19%")
        print(f"\n   Why better:")
        print(f"   ✅ Balanced data (250 images/class)")
        print(f"   ✅ Smart sampling & augmentation")
        print(f"   ✅ Model '{self.model_size}' ({'larger capacity' if self.model_size == 's' else 'efficient'})")
        print(f"   ✅ SGD optimizer (stable)")
        
        print(f"\n❓ Start? (y/n): ", end="")
        confirm = input().strip().lower()
        
        if confirm != 'y':
            print("❌ Cancelled.")
            return None, None
        
        print("\n" + "="*70)
        print("🏃 TRAINING...")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        try:
            results = self.model.train(
                data=self.yaml_config,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self.device,
                
                # Data loading - Optimized for P100
                workers=20 if self.device == 'cuda' and ('P100' in torch.cuda.get_device_name(0) or 'Tesla P100' in torch.cuda.get_device_name(0)) else 8,  # Increased workers
                cache=True if self.device == 'cuda' and ('P100' in torch.cuda.get_device_name(0) or 'Tesla P100' in torch.cuda.get_device_name(0)) else False,  # Cache in RAM for P100
                
                # Early stopping
                patience=patience,
                
                # Save
                save=True,
                save_period=save_period,
                project='runs/detect',
                name='animal_balanced',
                exist_ok=True,
                
                # Model
                pretrained=True,
                optimizer='SGD',      # SGD tốt hơn cho balanced data
                verbose=True,
                seed=42,
                deterministic=False,
                single_cls=False,
                
                # Training - CẢI THIỆN NÂNG CAO: Tối ưu cho accuracy cao hơn
                rect=False,
                cos_lr=True,          # Cosine LR schedule
                close_mosaic=12,      # Tăng từ 10 lên 12 - Turn off mosaic 12 epochs before end (epoch 128/140)
                resume=False,
                amp=True,             # Mixed precision training
                fraction=1.0,
                profile=False,
                
                # Validation
                val=True,
                plots=True,
                save_json=False,
                conf=None,
                iou=0.7,
                max_det=300,
                half=False,
                dnn=False,
                
                # AUGMENTATION - CẢI THIỆN NÂNG CAO: Tăng augmentation để tăng đa dạng dữ liệu
                hsv_h=0.02,           # Tăng từ 0.015 lên 0.02 để đa dạng hue hơn
                hsv_s=0.7,
                hsv_v=0.5,            # Tăng từ 0.4 lên 0.5 để đa dạng brightness hơn
                degrees=20.0,         # Tăng từ 15.0 lên 20.0 để rotation đa dạng hơn
                translate=0.2,       # Tăng từ 0.15 lên 0.2 để translation mạnh hơn
                scale=0.8,            # Giảm từ 0.85 xuống 0.8 để scale range lớn hơn (0.8-1.0) - tăng đa dạng
                shear=10.0,           # Tăng từ 8.0 lên 10.0 để shear mạnh hơn
                perspective=0.001,   # Tăng từ 0.0005 lên 0.001 để perspective đa dạng hơn
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.25,           # Tăng từ 0.2 lên 0.25 để mixup mạnh hơn
                copy_paste=0.2,       # Tăng từ 0.15 lên 0.2 để copy-paste mạnh hơn
                
                # OPTIMIZER - CẢI THIỆN NÂNG CAO: Điều chỉnh để tăng recall và mAP50
                lr0=0.0015,           # Giảm từ 0.002 xuống 0.0015 để training ổn định hơn, tránh overshooting
                lrf=0.000005,         # Giảm xuống 0.000005 cho 140 epochs để fine-tuning tốt hơn ở cuối
                momentum=0.937,
                weight_decay=0.001,   # Tăng từ 0.0008 lên 0.001 để tăng regularization mạnh hơn
                warmup_epochs=10.0,   # Tăng từ 8.0 lên 10.0 để warmup tốt hơn với LR thấp và nhiều epochs
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                
                # LOSS - CẢI THIỆN NÂNG CAO: Tăng cls loss để cải thiện recall và classification
                box=7.5,
                cls=0.9,              # Tăng từ 0.75 lên 0.9 để cải thiện classification và recall (an toàn hơn 1.0)
                dfl=1.5,
                pose=12.0,
                kobj=1.0,
                # Note: label_smoothing is deprecated in newer YOLO versions
                # YOLO handles regularization internally
                
                # Batch
                nbs=64,
                overlap_mask=True,
                mask_ratio=4,
                dropout=0.12,         # Tăng từ 0.1 lên 0.12 để tăng regularization (an toàn hơn 0.15, tránh underfitting)
            )
            
            end_time = time.time()
            hours = int((end_time - start_time) // 3600)
            minutes = int(((end_time - start_time) % 3600) // 60)
            
            print("\n" + "="*70)
            print("🎉 TRAINING DONE!")
            print("="*70)
            print(f"⏱️  Time: {hours}h {minutes}m")
            
            best_path = Path('runs/detect/animal_balanced/weights/best.pt')
            last_path = Path('runs/detect/animal_balanced/weights/last.pt')
            results_path = Path('runs/detect/animal_balanced')
            
            print(f"\n📁 SAVED:")
            print(f"   - Best: {best_path}")
            print(f"   - Last: {last_path}")
            print(f"   - Plots: {results_path}")
            
            return results, best_path
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopped by user")
            last_path = Path('runs/detect/animal_balanced/weights/last.pt')
            return None, last_path if last_path.exists() else None
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            
            if "memory" in str(e).lower():
                print("\n💡 FIX:")
                print("   - Reduce batch: 24 or 16")
                print("   - Reduce imgsz: 512")
            
            raise
    
    def validate(self, model_path=None):
        """Validation"""
        if model_path:
            print(f"\n📦 Load: {model_path}")
            self.model = YOLO(model_path)
        
        print("\n📊 Validating...")
        
        try:
            metrics = self.model.val(data=self.yaml_config)
            
            print("\n" + "="*70)
            print("📈 FINAL RESULTS")
            print("="*70)
            print(f"📊 mAP50:     {metrics.box.map50:.4f}")
            print(f"📊 mAP50-95:  {metrics.box.map:.4f}")
            print(f"🎯 Precision: {metrics.box.mp:.4f}")
            print(f"🎯 Recall:    {metrics.box.mr:.4f}")
            
            # Compare
            baseline = 0.6925  # From previous run
            improvement = ((metrics.box.map50 - baseline) / baseline) * 100
            
            print(f"\n📊 COMPARISON:")
            print(f"   Imbalanced data:  {baseline:.4f}")
            print(f"   Balanced data:    {metrics.box.map50:.4f}")
            
            if improvement > 0:
                print(f"   Improvement:      +{improvement:.1f}% 🎉")
            else:
                print(f"   Change:           {improvement:.1f}%")
            
            print(f"\n💬 EVALUATION:")
            if metrics.box.map50 >= 0.80:
                print("   🌟 EXCELLENT! Target achieved!")
            elif metrics.box.map50 >= 0.75:
                print("   ✅ VERY GOOD! Close to target.")
                print("   💡 To reach 0.80+:")
                print("      - Train +20 epochs")
                print("      - Try model 's'")
            elif metrics.box.map50 >= 0.72:
                print("   ✅ GOOD! Better than baseline.")
                print("   💡 To improve:")
                print("      - Epochs 120-150")
                print("      - Model 's'")
            else:
                print("   ⚠️  Check training logs")
            
            print("="*70)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING WITH BALANCED DATASET")
    print("="*70)
    print(f"Dataset: ~40,000 samples (balanced)")
    print(f"Train: ~20,000 | Val: ~20,000")
    print(f"Target: 250 images/class (balanced)")
    print(f"Goal: mAP50 = 0.78-0.82")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ========================================
    # CONFIG - OPTIMIZED FOR BALANCED DATA
    # ========================================
    
    # FIX: Update path to match data_preparation_pro.py OUTPUT_DIR
    YAML_CONFIG = "/kaggle/working/yolo_balanced_data/data.yaml"
    
    # Alternative: Use relative path if running locally
    # YAML_CONFIG = "yolo_balanced_data/data.yaml"
    
    MODEL_SIZE = 's'  # Small model for better accuracy (0.8+ target) - 'n' for faster training
    
    # OPTIMAL CONFIG - CẢI THIỆN: Tăng epochs và tối ưu để đạt accuracy cao hơn
    # NOTE: Model 's' is ~1.8x slower than 'n', batch 28 is slower
    # Với augmentation mạnh hơn: +5-8% thời gian/epoch
    # Tính toán: 150 epochs × 0.064h/epoch × 1.08 (aug) × 1.1 buffer = 11.4h (GẦN GIỚI HẠN)
    # Hoặc: 140 epochs × 0.064h/epoch × 1.08 × 1.1 = 10.6h (AN TOÀN HƠN)
    # CHỌN 140 để an toàn hơn, vẫn đủ để training tốt với LR thấp
    EPOCHS = 140      # Tăng từ 120 lên 140 để training tốt hơn với LR thấp hơn, an toàn trong 12h
    IMAGE_SIZE = 640  # Giữ 640 để đảm bảo thời gian, có thể tăng lên 768 nếu muốn accuracy cao hơn (chậm hơn ~1.5x)
    BATCH_SIZE = None  # Auto-detect (will use smaller batch for more updates)
    
    PATIENCE = 0      # Disable early stopping - run full epochs for maximum accuracy
    SAVE_PERIOD = 10  # Save less frequently to save time
    
    # ========================================
    
    if not Path(YAML_CONFIG).exists():
        print(f"\n❌ Not found: '{YAML_CONFIG}'")
        exit(1)
    
    print(f"\n⚙️  CONFIG:")
    print(f"   Model: YOLOv8{MODEL_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image: {IMAGE_SIZE}")
    print(f"   Batch: Auto (will optimize for your GPU)")
    
    # Check if P100
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        is_p100 = 'P100' in gpu_name or 'Tesla P100' in gpu_name
        if is_p100:
            # Time estimate based on ACTUAL training logs
            # From log: 120 epochs in 7.689h = 0.064h/epoch (model 's', batch 28)
            # Với augmentation mạnh hơn: +8% thời gian
            base_time = 0.064  # Actual from training log (model 's', batch 28)
            if MODEL_SIZE == 's':
                model_multiplier = 1.0  # Đã là model 's'
                aug_multiplier = 1.08   # Augmentation mạnh hơn +8%
            else:
                model_multiplier = 0.56  # Model 'n' nhanh hơn ~1.8x
                aug_multiplier = 1.05   # Augmentation nhẹ hơn
            est_time = EPOCHS * base_time * model_multiplier * aug_multiplier
            est_time_max = EPOCHS * base_time * model_multiplier * aug_multiplier * 1.1  # 10% buffer
            print(f"   Estimated Time: ~{est_time:.1f}-{est_time_max:.1f}h (P100, với cải thiện)")
            if est_time_max <= 12:
                print(f"   ✅ Safe within 12h limit! ({est_time_max:.1f}h / 12h)")
            elif est_time_max <= 12.5:
                print(f"   ⚠️  Close to limit ({est_time_max:.1f}h / 12h) - should be OK")
            else:
                print(f"   ❌ May exceed 12h ({est_time_max:.1f}h) - consider reducing epochs to 140")
        else:
            model_multiplier = 2.0 if MODEL_SIZE == 's' else 1.0
            print(f"   Time: ~{EPOCHS * 0.08 * model_multiplier:.1f}h")
    else:
        model_multiplier = 2.0 if MODEL_SIZE == 's' else 1.0
        print(f"   Time: ~{EPOCHS * 0.08 * model_multiplier:.1f}h")
    
    print(f"\n🎯 WHY THIS WORKS FOR 0.8+ TARGET:")
    print(f"   ✅ Balanced data (250 images/class)")
    print(f"   ✅ Model '{MODEL_SIZE}' ({'larger capacity for better accuracy' if MODEL_SIZE == 's' else 'efficient'})")
    print(f"   ✅ SGD (stable for balanced)")
    print(f"   ✅ {EPOCHS} epochs (full convergence)")
    print(f"   ✅ Optimized augmentation & learning rate")
    print(f"   → Expected: mAP50 = 0.80-0.85")
    
    # Init
    trainer = OptimizedTrainer(
        yaml_config_path=YAML_CONFIG,
        model_size=MODEL_SIZE
    )
    
    # Load
    print("\n" + "="*70)
    print("📥 LOAD MODEL")
    print("="*70)
    trainer.load_model()
    
    # Train
    results, best_model = trainer.train(
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        save_period=SAVE_PERIOD
    )
    
    if best_model:
        # Validate
        print("\n" + "="*70)
        print("🎯 VALIDATION")
        print("="*70)
        trainer.validate(best_model)
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"📁 Model: {best_model}")
        print(f"\n📌 NEXT:")
        print(f"   Test inference on real images")
        print("="*70)
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
