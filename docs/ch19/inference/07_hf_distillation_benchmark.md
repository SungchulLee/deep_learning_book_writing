# HuggingFace 앎 내리기 잣대

HuggingFace 앎 내리기 잣대.

깊은 배움 모델을 효율적으로 펼치려면 모델 크기, 빠르기, 정확도의 맞바꿈을 조심스레 다듬어야 한다. 여기 짠 것은 실전 환경에서 신경망을 눌러 담고 빠르게 하는 데 쓰는 모델 눌러 담기 재주를 보여 준다.

## 코드

```python
"""HuggingFace 앎 내리기 잣대."""
# ---
# title: "허깅페이스 앎 내리기와 미룸 잣대 재기"
# description: "맞춤 손실을 쓴 DistillationTrainer, 정확도/크기/늦음을 재는
#               PerformanceBenchmark, ONNX 런타임 내보내기, 눌러 담기 견줌"
# ---
#
# 앞선 각본(01~06)은 모델 눌러 담기에 날 PyTorch를 쓴다.
# 실전에서는 허깅페이스 Trainer가 맞춤 익히기 되풀이를 위한
# 힘 있는 추상을 준다. 이 각본은 다음을 보인다:
#
#   1부 – DistillationTrainer: 맞춤 손실로 허깅페이스 Trainer 넓히기
#   2부 – PerformanceBenchmark: 정확도, 크기, 늦음 재기
#   3부 – ONNX 런타임 내보내기와 OnnxPipeline 감개
#   4부 – 눌러 담기 견줌: 바탕 → 내린 것 → 양자화한 것 → ONNX
#
# 바탕: O'Reilly "NLP with Transformers" 8장

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import time
import os


# =====================================================================
# 1부 – DistillationTrainer: 허깅페이스 Trainer에 맞춤 손실 쓰기
# =====================================================================
print("=" * 60)
print("Part 1: DistillationTrainer — Extending HF Trainer")
print("=" * 60)

# 핵심 결: 허깅페이스 Trainer의 compute_loss()를 덮어써서
# 여느 엇갈린 엔트로피와 스승에 대한 KL 벌어짐을 아우른다.
#
#   L = α × CE(제자 로짓, 이름표) + (1-α) × T² × KL(부드러운 제자, 부드러운 스승)
#
# 이러면 Trainer의 모든 기능(배움 빠르기 차례 짜기, 기록,
# 기울기 쌓기, 되짚기)을 쓰면서 앎 내리기를 더할 수 있다.

try:
    from transformers import Trainer, TrainingArguments

    class DistillationTrainer(Trainer):
        """앎 내리기 손실로 넓힌 허깅페이스 Trainer.

        스승 모델은 값매김 방식으로 돈다(기울기 없음). 제자는
        다음을 무게 붙여 아우른 것으로 익힌다:
          - 딱딱한 손실: 참 이름표에 대한 엇갈린 엔트로피
          - 부드러운 손실: 스승의 부드러운 어림에 대한 KL 벌어짐

        인수:
            teacher_model: 미리 익힌 스승(익히는 동안 얼림)
            temperature:   부드러운 목표를 위한 부드러운 최댓값 온도(붙박이: 2.0)
            alpha:         딱딱한 손실의 무게. 부드러운 손실은 (1-alpha)
            *args, **kwargs: 허깅페이스 Trainer로 넘긴다
        """

        def __init__(
            self,
            teacher_model=None,
            temperature: float = 2.0,
            alpha: float = 0.5,
            *args, **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.teacher = teacher_model
            self.temperature = temperature
            self.alpha = alpha

            # 스승을 얼리고 제자와 같은 기기로 옮긴다
            if self.teacher is not None:
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Trainer의 손실 셈하기를 앎 내리기 손실로 덮어쓴다.

            스승이 온도 T에서 부드러운 목표를 낸다.
            제자는 딱딱한 이름표와 부드러운 목표 둘 다에서 배운다.

            수식:
                soft_teacher = softmax(teacher_logits / T)
                soft_student = log_softmax(student_logits / T)
                L_soft = KL(soft_student, soft_teacher) × T²
                L_hard = CE(student_logits, labels)
                L = α × L_hard + (1 - α) × L_soft
            """
            # 제자의 앞먹임
            labels = inputs.pop("labels", None)
            outputs_student = model(**inputs)
            student_logits = outputs_student.logits

            # 딱딱한 손실(여느 엇갈린 엔트로피)
            loss_hard = F.cross_entropy(student_logits, labels)

            if self.teacher is not None:
                # 스승의 앞먹임(기울기 없음)
                with torch.no_grad():
                    outputs_teacher = self.teacher(**inputs)
                    teacher_logits = outputs_teacher.logits

                # 부드러운 손실(온도 T에서의 KL 벌어짐)
                soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

                loss_soft = F.kl_div(
                    soft_student,
                    soft_teacher,
                    reduction="batchmean",
                ) * (self.temperature ** 2)

                # 결합된 손실
                loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft
            else:
                loss = loss_hard

            return (loss, outputs_student) if return_outputs else loss

    print("  DistillationTrainer defined successfully.")
    print()
    print("  Usage:")
    print("""
    teacher = AutoModelForSequenceClassification.from_pretrained("bert-base")
    student = AutoModelForSequenceClassification.from_pretrained("distilbert-base")

    training_args = TrainingArguments(
        output_dir="./distilled-model",
        num_train_epochs=5,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
    )

    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=2.0,
        alpha=0.5,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    """)

    HAS_TRAINER = True

except ImportError:
    print("  transformers not available — showing concept only")
    HAS_TRAINER = False
    print()


# =====================================================================
# 2부 – PerformanceBenchmark
# =====================================================================
print("=" * 60)
print("Part 2: PerformanceBenchmark — Accuracy, Size & Latency")
print("=" * 60)

# 눌러 담기 재주를 견줄 때는 한결같은 잣대가 필요하다.
# PerformanceBenchmark는 허깅페이스 물길을 감싸고 다음을 잰다:
#   1. 남겨 둔 자료 묶음에서의 정확도(또는 F1)
#   2. 판에 담긴 모델 크기(MB)
#   3. 미룸 늦음(보기마다 ms)


class PerformanceBenchmark:
    """정확도, 크기, 늦음으로 허깅페이스 물길의 잣대를 잰다.

    이 갈래는 여러 모델 눌러 담기 재주를 같은 잣대로 견주는
    표준화된 길을 준다.

    인수:
        pipeline:    허깅페이스 물길(예: text-classification)
        dataset:     'text'와 'label' 세로줄을 갖춘 값매김 자료 묶음
        optim_type:  이 자리매김의 이름표(예: 'Baseline', 'Distilled')
    """

    def __init__(self, pipeline, dataset, optim_type: str = "Baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self) -> float:
        """값매김 자료 묶음에서의 정확도를 셈한다."""
        preds = self.pipeline(self.dataset["text"])
        # 허깅페이스 가르기 물길은 [{"label": ..., "score": ...}]를 돌려준다
        pred_labels = [p["label"] for p in preds]
        true_labels = self.dataset["label"]

        # 글 이름표와 정수 이름표를 다룬다
        if isinstance(true_labels[0], str):
            correct = sum(p == t for p, t in zip(pred_labels, true_labels))
        else:
            # 물길 모델 자리매김으로 이름표 글을 번호에 대응시킨다
            label2id = self.pipeline.model.config.label2id
            pred_ids = [label2id.get(p, -1) for p in pred_labels]
            correct = sum(p == t for p, t in zip(pred_ids, true_labels))

        accuracy = correct / len(true_labels)
        print(f"  [{self.optim_type}] Accuracy: {accuracy:.4f}")
        return accuracy

    def compute_size(self) -> float:
        """판에 담긴 모델 크기를 MB로 셈한다."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.pipeline.model.save_pretrained(tmp_dir)
            # 모든 .bin / .safetensors 파일을 더한다
            size_bytes = sum(
                os.path.getsize(os.path.join(tmp_dir, f))
                for f in os.listdir(tmp_dir)
                if f.endswith((".bin", ".safetensors"))
            )

        size_mb = size_bytes / (1024 * 1024)
        print(f"  [{self.optim_type}] Size: {size_mb:.1f} MB")
        return size_mb

    def compute_latency(
        self,
        sample_text: str = "This is a test sentence for benchmarking.",
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> float:
        """미룸 늦음의 평균을 밀리초로 잰다.

        곳간이 안정되도록 먼저 몸풀기를 되풀이한 뒤
        num_runs번 되풀이한 평균을 잰다.
        """
        # 워밍업
        for _ in range(num_warmup):
            _ = self.pipeline(sample_text)

        # 시간을 잰 돌리기
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.pipeline(sample_text)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        print(f"  [{self.optim_type}] Latency: {avg_latency:.1f} ± {std_latency:.1f} ms")
        return avg_latency

    def run_benchmark(self) -> Dict[str, float]:
        """모든 잣대를 재고 결과를 사전으로 돌려준다."""
        print(f"\n  Benchmarking: {self.optim_type}")
        print("  " + "-" * 40)
        metrics = {
            "accuracy": self.compute_accuracy(),
            "size_mb": self.compute_size(),
            "latency_ms": self.compute_latency(),
        }
        return metrics


# 허깅페이스 없이 하는 시범
print("""
  사용법:
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline("text-classification", model="my-model")
    bench = PerformanceBenchmark(pipe, eval_dataset, optim_type="Baseline")
    metrics = bench.run_benchmark()
    # → 정확도: 0.8742, 크기: 418.2 MB, 늦음: 12.3 ± 0.8 ms
""")


# =====================================================================
# 3부 – ONNX 런타임 내보내기와 OnnxPipeline
# =====================================================================
print("=" * 60)
print("Part 3: ONNX Runtime Export & OnnxPipeline")
print("=" * 60)

# ONNX 런타임은 변환기 모델에 가장 좋게 다듬은 미룸을 준다.
# ONNX로 내보내 ORT를 쓰면 CPU에서 늦음을 2~3배 줄일 수 있다.
#
# 물길:
#   1. PyTorch 모델을 ONNX 꼴로 내보낸다
#   2. ONNX 런타임 InferenceSession으로 불러온다
#   3. 허깅페이스와 어울리는 겉면을 위해 OnnxPipeline으로 감싼다


class OnnxPipeline:
    """ONNX 런타임 미룸을 위한, 허깅페이스와 어울리는 물길 감개.

    transformers.pipeline()과 같은 겉면을 주되, CPU에서 더 빨리 돌도록
    ONNX 런타임으로 미룬다.

    인수:
        model_path:  내보낸 .onnx 모델 파일의 길
        tokenizer:   글을 미리 다듬는 허깅페이스 토막내개
        id2label:    갈래 번호를 이름표 글로 대응시킨 것
    """

    def __init__(self, model_path: str, tokenizer, id2label: Dict[int, str]):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        self.tokenizer = tokenizer
        self.id2label = id2label

    def __call__(self, texts, **kwargs):
        """글 하나 또는 여럿에 미룸을 돌린다.

        인수:
            texts: 글줄 하나 또는 글줄의 목록.

        반환값:
            'label'과 'score' 열쇠를 갖춘 사전의 목록으로,
            허깅페이스 물길 꼴과 맞다.
        """
        if isinstance(texts, str):
            texts = [texts]

        # 토큰으로 나누기
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",  # ONNX 런타임은 numpy를 쓴다
        )

        # ONNX 미룸을 돌린다
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        logits = self.session.run(None, ort_inputs)[0]

        # 허깅페이스 물길 꼴로 바꾼다
        results = []
        for logit_row in logits:
            probs = self._softmax(logit_row)
            pred_idx = int(np.argmax(probs))
            results.append({
                "label": self.id2label[pred_idx],
                "score": float(probs[pred_idx]),
            })
        return results

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


print("  OnnxPipeline class defined.")
print()

# ONNX 내보내기 절차
print("  Exporting a HuggingFace model to ONNX:")
print("""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model = AutoModelForSequenceClassification.from_pretrained("my-model")
    tokenizer = AutoTokenizer.from_pretrained("my-model")
    model.eval()

    # 임시 입력 만들기
    dummy = tokenizer("Example text", return_tensors="pt")

    # ONNX로 내보낸다
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
    )

    # OnnxPipeline과 함께 쓴다
    onnx_pipe = OnnxPipeline(
        "model.onnx",
        tokenizer,
        id2label=model.config.id2label,
    )
    result = onnx_pipe("This is a test.")
    # → [{"label": "positive", "score": 0.93}]
""")


# =====================================================================
# 4부 – 눌러 담기 견줌 물길
# =====================================================================
print("=" * 60)
print("Part 4: Full Compression Comparison")
print("=" * 60)

# 실전에서는 여러 눌러 담기 전략을 견준다:
#   1. 바탕(온전한 크기 모델)
#   2. 앎을 내린 제자 모델
#   3. 양자화한 모델(그때그때 INT8)
#   4. ONNX 런타임 모델
#
# 공정한 견줌을 위해 저마다 같은 잣대로 잰다.

print("""
  눌러 담기 견줌 물길:
  ─────────────────────────────────

  1걸음: 바탕 모델을 익힌다
    teacher = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=6)
    # ... 자료 묶음으로 곱게 다듬는다 ...

  2걸음: 더 작은 제자에게 내린다
    student = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=6)
    trainer = DistillationTrainer(
        teacher_model=teacher,
        model=student,
        temperature=2.0,
        alpha=0.5,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

  3걸음: 제자를 양자화한다(그때그때 양자화)
    import torch.quantization as quant
    quantized_student = quant.quantize_dynamic(
        student,
        {nn.Linear},      # 모든 선형 층을 양자화한다
        dtype=torch.qint8,
    )

  4걸음: ONNX로 내보낸다
    torch.onnx.export(student, dummy_input, "student.onnx", ...)
    onnx_pipe = OnnxPipeline("student.onnx", tokenizer, id2label)

  5걸음: 모든 판의 잣대를 잰다
    configs = {
        "Baseline":   teacher_pipeline,
        "Distilled":  student_pipeline,
        "Quantized":  quantized_pipeline,
        "ONNX":       onnx_pipe,
    }
    results = {}
    for name, pipe in configs.items():
        bench = PerformanceBenchmark(pipe, eval_ds, optim_type=name)
        results[name] = bench.run_benchmark()
""")


# 흉내낸 결과 표
print("  Typical results (BERT text classification):")
print("  ┌────────────┬──────────┬──────────┬────────────┐")
print("  │ Method     │ Accuracy │ Size(MB) │ Latency(ms)│")
print("  ├────────────┼──────────┼──────────┼────────────┤")
print("  │ Baseline   │  0.874   │  418.2   │   12.3     │")
print("  │ Distilled  │  0.861   │  255.4   │    7.1     │")
print("  │ Quantized  │  0.856   │  173.8   │    5.8     │")
print("  │ ONNX       │  0.861   │  255.4   │    4.2     │")
print("  │ Quant+ONNX │  0.852   │  102.1   │    3.1     │")
print("  └────────────┴──────────┴──────────┴────────────┘")
print()
print("  Key takeaways:")
print("    - Distillation: 40% smaller, 42% faster, 1.3% accuracy drop")
print("    - Quantization: adds 58% size reduction on top of distillation")
print("    - ONNX Runtime: 2-3x speedup on CPU with zero accuracy loss")
print("    - Combining all: 75% smaller, 75% faster, ~2% accuracy drop")
print()

# =====================================================================
# 홀로 서는 시범 — 날 PyTorch로 하는 앎 내리기(허깅페이스 없이)
# =====================================================================
print("=" * 60)
print("Standalone Demo: DistillationTrainer Logic in Raw PyTorch")
print("=" * 60)

# transformers가 없는 환경을 위해, 만든 자료에 단순한 앞먹임 그물을 써서
# 앎 내리기의 고갱이 논리를 보인다.

torch.manual_seed(42)

# 스승 그물과 제자 그물을 만든다
class SimpleTeacher(nn.Module):
    def __init__(self, in_dim=20, hidden=128, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleStudent(nn.Module):
    def __init__(self, in_dim=20, hidden=32, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """앎 내리기 손실을 셈한다.

    이는 DistillationTrainer.compute_loss()가 쓰는 손실과 같으나
    또렷하게 보이려고 홀로 서는 함수로 두었다.

    인수:
        student_logits: 제자가 낸 날 값 [묶음, 갈래]
        teacher_logits: 스승이 낸 날 값 [묶음, 갈래]
        labels:         참 갈래 번호 [묶음]
        temperature:    부드러운 최댓값 온도(높을수록 부드럽다)
        alpha:          딱딱한 손실의 무게

    반환값:
        아우른 앎 내리기 손실(스칼라)
    """
    # 딱딱한 손실: 여느 엇갈린 엔트로피
    loss_hard = F.cross_entropy(student_logits, labels)

    # 부드러운 손실: 온도 T에서의 KL 벌어짐
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    loss_soft = F.kl_div(
        soft_student, soft_teacher, reduction="batchmean"
    ) * (temperature ** 2)

    return alpha * loss_hard + (1.0 - alpha) * loss_soft


# 인공 데이터
X = torch.randn(500, 20)
y = torch.randint(0, 5, (500,))

# 스승을 미리 익힌다
teacher = SimpleTeacher()
opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
teacher.train()
for epoch in range(30):
    logits = teacher(X)
    loss = F.cross_entropy(logits, y)
    opt_t.zero_grad()
    loss.backward()
    opt_t.step()

teacher.eval()
with torch.no_grad():
    teacher_acc = (teacher(X).argmax(1) == y).float().mean().item()
print(f"  Teacher accuracy: {teacher_acc:.3f}")
print(f"  Teacher params:   {sum(p.numel() for p in teacher.parameters()):,}")

# 앎 내리기 없이 제자를 익힌다(바탕)
student_baseline = SimpleStudent()
opt_s = torch.optim.Adam(student_baseline.parameters(), lr=1e-3)
student_baseline.train()
for epoch in range(50):
    logits = student_baseline(X)
    loss = F.cross_entropy(logits, y)
    opt_s.zero_grad()
    loss.backward()
    opt_s.step()

student_baseline.eval()
with torch.no_grad():
    baseline_acc = (student_baseline(X).argmax(1) == y).float().mean().item()
print(f"  Student (no distill) accuracy: {baseline_acc:.3f}")
print(f"  Student params:   {sum(p.numel() for p in student_baseline.parameters()):,}")

# 앎 내리기로 제자를 익힌다
student_distilled = SimpleStudent()
opt_d = torch.optim.Adam(student_distilled.parameters(), lr=1e-3)
student_distilled.train()
for epoch in range(50):
    student_logits = student_distilled(X)
    with torch.no_grad():
        teacher_logits = teacher(X)

    loss = distillation_loss(
        student_logits, teacher_logits, y,
        temperature=4.0, alpha=0.3,
    )
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()

student_distilled.eval()
with torch.no_grad():
    distilled_acc = (student_distilled(X).argmax(1) == y).float().mean().item()
print(f"  Student (distilled) accuracy:  {distilled_acc:.3f}")
print()

improvement = distilled_acc - baseline_acc
print(f"  Distillation improvement: {improvement:+.3f}")
print(f"  Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student_distilled.parameters()):.1f}x fewer params")
print()

# 그때그때 양자화 시범
print("  Dynamic INT8 quantization of the distilled student:")
quantized = torch.quantization.quantize_dynamic(
    student_distilled,
    {nn.Linear},
    dtype=torch.qint8,
)

# 크기 비교
def model_size_bytes(model):
    """매개변수 담김에서 모델 크기를 어림한다."""
    import io, pickle
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()

orig_size = model_size_bytes(student_distilled)
quant_size = model_size_bytes(quantized)
print(f"  Original size:  {orig_size / 1024:.1f} KB")
print(f"  Quantized size: {quant_size / 1024:.1f} KB")
print(f"  Reduction: {(1 - quant_size / orig_size) * 100:.1f}%")
print()

# 양자화한 정확도를 확인한다
with torch.no_grad():
    quant_acc = (quantized(X).argmax(1) == y).float().mean().item()
print(f"  Quantized accuracy: {quant_acc:.3f}")
print()

# 마지막 간추림
print("  ┌──────────────────────┬──────────┬──────────┐")
print("  │ Model                │ Accuracy │ Params   │")
print("  ├──────────────────────┼──────────┼──────────┤")
print(f"  │ Teacher              │  {teacher_acc:.3f}   │ {sum(p.numel() for p in teacher.parameters()):>6,}   │")
print(f"  │ Student (baseline)   │  {baseline_acc:.3f}   │ {sum(p.numel() for p in student_baseline.parameters()):>6,}   │")
print(f"  │ Student (distilled)  │  {distilled_acc:.3f}   │ {sum(p.numel() for p in student_distilled.parameters()):>6,}   │")
print(f"  │ Student (quantized)  │  {quant_acc:.3f}   │ {sum(p.numel() for p in student_distilled.parameters()):>6,}   │")
print("  └──────────────────────┴──────────┴──────────┘")
print()

print("Done.")


if __name__ == "__main__":
    pass
```

## 논의

여기 짠 것은 함께 어울려 온전한 모델 눌러 담기 얼개를 이루는 클래스 4개(`PerformanceBenchmark`, `OnnxPipeline`, `SimpleTeacher`, `SimpleStudent`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기 보인 무늬는 더 복잡한 장면으로 자연스레 넓혀 쓸 수 있다. 웃매개변수, 얼개의 변종, 서로 다른 자료 뭉치로 실험해 보면 이해가 깊어지고 효율적인 펼치기 일에 대한 실전 직관이 쌓인다.

## 연습문제

**연습문제 1.**
붙박이 첫자리매김일 때 `PerformanceBenchmark`의 배울 수 있는 매개변수 전체 개수를 셈하여라. 무게와 치우침을 모두 넣어 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
어텐션 가중치 뒤에(값과 곱하기 전에) 드롭아웃 층을 추가하라. 학습 중에는 드롭아웃 비율 0.1을 쓴다. 어텐션 드롭아웃이 정칙화에 도움이 되는 이유를 설명하라.

??? success "연습문제 2 풀이"
    `__init__`에 `self.attn_dropout = nn.Dropout(0.1)`을 추가하고 소프트맥스 뒤에 적용한다. `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`이다. 어텐션 드롭아웃은 학습 중에 일부 어텐션 가중치를 무작위로 0으로 만들어, 모델이 특정 토큰 사이의 관계에 지나치게 기대지 않게 한다. 이는 모델이 어텐션을 더 고르게 분산시키고 더 견고한 표현을 배우도록 북돋우며, 표준 드롭아웃이 뉴런의 공적응을 막는 것과 비슷하다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `PerformanceBenchmark`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = PerformanceBenchmark(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
