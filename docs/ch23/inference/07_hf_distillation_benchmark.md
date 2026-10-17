# Hugging Face 지식 증류와 성능 벤치마크

이 절은 분류 모델의 증류 로스를 Hugging Face `Trainer`에 연결하는 방법과 정확도·저장 크기·지연 시간의 측정을 살펴본다. 코드 앞부분은 클래스 정의와 사용 예시 문자열이며, 마지막 부분에는 합성 데이터로 작은 뉴럴 넷트웍을 학습하는 독립 데모가 있다. 출력문에 미리 적힌 BERT 성능 표와 실제 데모에서 계산하는 값은 구분해야 한다.

## 1. 예제 코드

    ```python
    """HuggingFace 앎 내리기 잣대."""
    # ---
    # title: "허깅페이스 앎 내리기와 미룸 잣대 재기"
    # description: "맞춤 로스를 쓴 DistillationTrainer, 정확도/크기/늦음을 재는
    #               PerformanceBenchmark, ONNX 런타임 내보내기, 눌러 담기 견줌"
    # ---
    #
    # 앞선 각본(01~06)은 모델 눌러 담기에 날 PyTorch를 쓴다.
    # 실전에서는 허깅페이스 Trainer가 맞춤 익히기 되풀이를 위한
    # 힘 있는 추상을 준다. 이 각본은 다음을 보인다:
    #
    #   1부 – DistillationTrainer: 맞춤 로스로 허깅페이스 Trainer 넓히기
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
    # 1부 – DistillationTrainer: 허깅페이스 Trainer에 맞춤 로스 쓰기
    # =====================================================================
    print("=" * 60)
    print("Part 1: DistillationTrainer — Extending HF Trainer")
    print("=" * 60)

    # 핵심 결: 허깅페이스 Trainer의 compute_loss()를 덮어써서
    # 여느 엇갈린 엔트로피와 스승에 대한 KL 벌어짐을 아우른다.
    #
    #   L = α × CE(제자 로짓, 이름표) + (1-α) × T² × KL(부드러운 제자, 부드러운 스승)
    #
    # 이러면 Trainer의 모든 기능(러닝 빠르기 차례 짜기, 기록,
    # 그래디언트 쌓기, 되짚기)을 쓰면서 앎 내리기를 더할 수 있다.

    try:
        from transformers import Trainer, TrainingArguments

        class DistillationTrainer(Trainer):
            """앎 내리기 로스로 넓힌 허깅페이스 Trainer.

            스승 모델은 밸류 방식으로 돈다(그래디언트 없음). 제자는
            다음을 무게 붙여 아우른 것으로 익힌다:
              - 딱딱한 로스: 참 이름표에 대한 엇갈린 엔트로피
              - 부드러운 로스: 스승의 부드러운 어림에 대한 KL 벌어짐

            인수:
                teacher_model: 미리 익힌 스승(익히는 동안 얼림)
                temperature:   부드러운 목표를 위한 부드러운 최댓값 온도(붙박이: 2.0)
                alpha:         딱딱한 로스의 무게. 부드러운 로스는 (1-alpha)
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
                """Trainer의 로스 셈하기를 앎 내리기 로스로 덮어쓴다.

                스승이 온도 T에서 부드러운 목표를 낸다.
                제자는 딱딱한 이름표와 부드러운 목표 둘 다에서 배운다.

                수식:
                    soft_teacher = softmax(teacher_logits / T)
                    soft_student = log_softmax(student_logits / T)
                    L_soft = KL(p_teacher ‖ p_student) × T²
                             = F.kl_div(soft_student, soft_teacher) × T²
                             (KL은 비대칭이다. 수식에서는 교사가 앞이고,
                              F.kl_div 에서는 학생이 첫 인자다 — 자리가 반대다)
                    L_hard = CE(student_logits, labels)
                    L = α × L_hard + (1 - α) × L_soft
                """
                # 제자의 피드포워드
                labels = inputs.pop("labels", None)
                outputs_student = model(**inputs)
                student_logits = outputs_student.logits

                # 딱딱한 로스(여느 엇갈린 엔트로피)
                loss_hard = F.cross_entropy(student_logits, labels)

                if self.teacher is not None:
                    # 스승의 피드포워드(그래디언트 없음)
                    with torch.no_grad():
                        outputs_teacher = self.teacher(**inputs)
                        teacher_logits = outputs_teacher.logits

                    # 부드러운 로스(온도 T에서의 KL 벌어짐)
                    soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                    soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

                    loss_soft = F.kl_div(
                        soft_student,
                        soft_teacher,
                        reduction="batchmean",
                    ) * (self.temperature ** 2)

                    # 결합된 로스
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

        이 클래스는 여러 모델 눌러 담기 재주를 같은 잣대로 견주는
        표준화된 길을 준다.

        인수:
            pipeline:    허깅페이스 물길(예: text-classification)
            dataset:     'text'와 'label' 세로줄을 갖춘 밸류 자료 묶음
            optim_type:  이 자리매김의 이름표(예: 'Baseline', 'Distilled')
        """

        def __init__(self, pipeline, dataset, optim_type: str = "Baseline"):
            self.pipeline = pipeline
            self.dataset = dataset
            self.optim_type = optim_type

        def compute_accuracy(self) -> float:
            """밸류 자료 묶음에서의 정확도를 셈한다."""
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
            id2label:    클래스 번호를 이름표 글로 대응시킨 것
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

      스텝1: 바탕 모델을 익힌다
        teacher = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=6)
        # ... 자료 묶음으로 곱게 다듬는다 ...

      스텝2: 더 작은 제자에게 내린다
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

      스텝3: 제자를 양자화한다(그때그때 양자화)
        import torch.quantization as quant
        quantized_student = quant.quantize_dynamic(
            student,
            {nn.Linear},      # 모든 선형 레이어를 양자화한다
            dtype=torch.qint8,
        )

      스텝4: ONNX로 내보낸다
        torch.onnx.export(student, dummy_input, "student.onnx", ...)
        onnx_pipe = OnnxPipeline("student.onnx", tokenizer, id2label)

      스텝5: 모든 판의 잣대를 잰다
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

    # transformers가 없는 환경을 위해, 만든 자료에 단순한 피드포워드 넷트웍을 써서
    # 앎 내리기의 핵심 논리를 보인다.

    torch.manual_seed(42)

    # 스승 넷트웍과 제자 넷트웍을 만든다
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
        """앎 내리기 로스를 셈한다.

        이는 DistillationTrainer.compute_loss()가 쓰는 로스와 같으나
        또렷하게 보이려고 홀로 서는 함수로 두었다.

        인수:
            student_logits: 제자가 낸 날 값 [배치, 클래스]
            teacher_logits: 스승이 낸 날 값 [배치, 클래스]
            labels:         참 클래스 번호 [배치]
            temperature:    부드러운 최댓값 온도(높을수록 부드럽다)
            alpha:          딱딱한 로스의 무게

        반환값:
            아우른 앎 내리기 로스(스칼라)
        """
        # 딱딱한 로스: 여느 엇갈린 엔트로피
        loss_hard = F.cross_entropy(student_logits, labels)

        # 부드러운 로스: 온도 T에서의 KL 벌어짐
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
        """파라메터 담김에서 모델 크기를 어림한다."""
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

## 2. Trainer에 증류 로스 연결하기

`DistillationTrainer.compute_loss`는 단일 레이블 분류의 B×C 로짓과 길이 B 정답을 전제로 한다. 정답 교차 엔트로피와 온도 T에서의 교사→학생 KL 발산을 결합한다. 교사와 학생의 클래스 수뿐 아니라 클래스 인덱스의 의미도 같아야 한다. T>0, 0≤α≤1을 사용한다.

교사는 `eval()`과 `requires_grad=False`로 설정되고 순전파가 `no_grad()` 안에 있다. 그러나 주석과 달리 교사를 학생과 같은 장치로 이동하는 코드는 없다. 교사의 장치·dtype과 입력을 맞추는 책임은 별도로 남는다. 두 모델이 동일한 입력 키와 토큰화를 지원하는지도 확인해야 한다. 예를 들어 한 모델만 `token_type_ids`를 받는다면 같은 입력 사전을 그대로 전달할 수 없다.

`labels`가 없으면 현재 교차 엔트로피 경로는 유효하지 않다. 토큰 분류나 언어 모델의 B×T×V 출력을 그대로 넣어도 이 로스와 마스킹 규칙이 자동으로 맞지는 않는다. 출력되는 Trainer 사용법은 실제 학습 호출이 아닌 문자열이고 데이터셋·평가 함수도 정의되어 있지 않다. 설치된 라이브러리 버전에 맞는 인수와 체크포인트를 확인해야 한다.

## 3. 벤치마크 클래스가 측정하는 범위

`PerformanceBenchmark`는 뉴럴 넷트웍이 아니라 파이프라인과 데이터셋을 보관하는 측정용 클래스다. 자체 학습 파라미터가 없다.

정확도는 문자열 레이블을 직접 비교하거나 `label2id`로 변환해 계산한다. 데이터셋의 레이블 의미와 모델의 매핑이 다르면 숫자 범위가 같아도 틀린 비교가 된다. 빈 데이터셋, 여러 레이블 반환 형태와 예측 개수의 일치도 별도로 처리해야 한다. F1을 계산하는 코드는 없다.

크기는 `save_pretrained`가 만든 최상위 `.bin`·`.safetensors` 파일의 합이다. 토크나이저와 설정 파일, 하위 디렉터리와 실행 메모리를 포함하지 않으며 단위는 MiB다. 양자화 모델의 저장 형식이 이 호출과 호환되는지도 확인한다.

지연 시간은 단일 문장을 워밍업 후 100번 호출한 평균과 표준편차다. 토큰화와 후처리를 포함한 파이프라인 호출 시간이며 전체 데이터셋 처리량이 아니다. GPU를 비교하려면 비동기 실행의 동기화와 같은 측정 경계를 확인하고 입력 길이·배치·장치를 고정한다. 상위 백분위 지연 시간도 별도로 보고할 수 있다.

## 4. ONNX 파이프라인의 호환성

`OnnxPipeline`은 토크나이저가 만든 `input_ids`와 `attention_mask` 두 배열을 ONNX Runtime에 전달한다. 내보낸 그래프의 실제 입력 이름, dtype, 필요한 추가 입력과 일치해야 한다. 첫 번째 출력이 B×C 로짓이라는 가정과 `id2label` 매핑도 확인한다. 길이는 최대 512로 잘리므로 비교할 PyTorch 파이프라인의 전처리도 같아야 한다.

이 래퍼에는 `.model`이 없어서 `PerformanceBenchmark.compute_size`나 정수 레이블 경로를 그대로 사용할 수 없다. ONNX 파일과 외부 웨이트 파일의 크기를 별도로 세고 레이블 매핑을 독립적으로 제공하는 측정 어댑터가 필요하다. 내보내기와 ONNX 추론 역시 코드 속 사용 안내 문자열이며 이 데모에서 실제 수행되는 과정은 아니다.

ONNX 변환은 항상 동일한 정확도나 일정 배수의 가속을 보장하지 않는다. 같은 입력의 로짓·예측과 태스크 지표를 비교하고, 사용할 실행 공급자와 그래프 최적화 설정에서 속도를 측정한다. 출력된 “Typical results” 표는 하드코딩된 예시 수치이며 이 스크립트가 BERT를 학습해 얻은 결과가 아니다.

## 5. 독립 PyTorch 데모의 해석

마지막 데모는 입력 20차원, 클래스 5개인 표본 500개를 무작위로 만든다. 레이블도 입력과 독립적으로 무작위 생성하므로 학습 정확도가 높아져도 새로운 데이터에서의 일반화 근거가 되지 않는다. 교사, 기준 학생, 증류 학생 모두 학습에 사용한 같은 표본에서 평가한다.

교사 파라미터는 19,845개, 학생은 837개로 약 23.71배 차이다. 기준 학생과 증류 학생의 학습 횟수는 같지만 초기 웨이트는 서로 다르다. 증류의 효과를 비교하려면 동일한 초기값과 여러 시드, 별도 평가 데이터가 필요하다.

이 부분의 `quantize_dynamic`은 실제로 학생의 선형 계층에 동적 INT8 양자화를 적용한다. 앞 절의 크기를 4로 나누는 모사와 다르다. 다만 바이어스와 저장 부가 정보가 있어 전체 체크포인트가 정확히 1/4이 되는 것은 아니다. 표의 양자화 모델 파라미터 수는 원래 학생의 개수를 재사용하며 저비트 패킹 상태의 크기를 나타내지 않는다.

`model_size_bytes`는 `torch.save(state_dict())`의 직렬화 크기를 측정하고, 출력의 KB는 실제로 KiB다. 정확도는 여전히 같은 학습 데이터의 비율이다. 마지막 데모에는 추론 지연 시간 측정이 없으므로 앞의 예시 표와 합쳐 속도 향상을 주장할 수 없다. 또한 주요 실행문이 최상위에 있어 파일을 import해도 실행되며 마지막 `__main__` 블록의 `pass`가 이를 막지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
PerformanceBenchmark와 SimpleTeacher·SimpleStudent의 역할을 구분하고 두 뉴럴 넷트웍의 파라미터 수를 구하라.

</div>

??? success "풀이"

    PerformanceBenchmark는 측정용 일반 클래스라 자체 학습 파라미터가 없다. 교사는 (20×128+128)+(128×128+128)+(128×5+5)=19845개다. 학생은 (20×32+32)+(32×5+5)=837개다. 개수 비율은 약 23.71배이며 실제 저장 바이트 비율과는 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
교사와 학생의 클래스 수가 같아도 증류가 잘못될 수 있는 입력·출력 조건을 두 가지 이상 설명하라.

</div>

??? success "풀이"

    클래스 인덱스 의미가 다르면 서로 다른 레이블의 확률을 맞추게 된다. 토크나이저나 입력 키가 호환되지 않으면 같은 사전을 두 모델에 전달할 수 없다. 장치가 다르면 순전파도 실패할 수 있다. 교사·학생이 같은 태스크의 같은 표본과 클래스 순서를 처리하는지 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
OnnxPipeline을 PerformanceBenchmark에 그대로 넣을 때 발생할 호환성 문제를 설명하라.

</div>

??? success "풀이"

    래퍼에 model이 없어 크기 측정의 model.save_pretrained 호출이 실패한다. 정수 레이블 정확도 경로의 model.config.label2id도 사용할 수 없다. ONNX 아티팩트 크기를 따로 측정하고 레이블 매핑을 공통 인터페이스로 제공해야 한다. tokenizer 설정과 ONNX 입력 이름·dtype도 동일한 전처리 조건에 맞춘다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
여러 깊이의 학생 모델에서 증류 효과를 비교하는 실험을 설계하라. 아래 코드 조각만으로 벤치마크의 깊이를 바꿀 수 있는가?

</div>

??? success "풀이"

    아래 ModuleList는 뉴럴 넷트웍 모듈을 등록하는 골격이며 YourBlock이 정의되지 않았다. PerformanceBenchmark 자체에는 깊이가 없다. 깊이가 다른 학생 nn.Module을 별도로 만들고 같은 초기값 조건의 비증류·증류 쌍을 비교한다. 학습·검증·테스트 데이터를 나누고 여러 시드, 학습 예산, 동일 전처리와 장치 조건에서 품질·실제 파일 크기·지연 시간을 보고한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

증류 학습, 모델 변환과 성능 측정은 서로 다른 절차다. 이 예제의 Trainer·ONNX 사용 안내와 고정 성능 표를 마지막 합성 데이터 데모의 측정 결과로 혼동하지 않는다. 클래스와 입력의 호환성, 평가 데이터의 독립성, 저장·메모리·지연 시간의 측정 범위를 맞추어야 압축 효과를 비교할 수 있다.
