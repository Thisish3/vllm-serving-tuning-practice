# n-gram speculative decoding 켜기/끄기 (RTX 3060 Ti)

vLLM 오프라인 `LLM()`에서 n-gram(prompt lookup) speculative decoding을 켰을 때와 껐을 때를, 배치 1·greedy·출력 길이 고정으로 비교했습니다.

- 코드: [`spec.py`](spec.py) (`python spec.py off|ngram`), 결과: [`results/spec_off.json`](results/spec_off.json), [`results/spec_ngram.json`](results/spec_ngram.json)
- 환경: RTX 3060 Ti 8GB, 드라이버 581.80(CUDA 13.0), WSL2 Ubuntu 22.04, vLLM 0.11.2 + torch 2.9.0+cu128 (pip venv)
- 모델: `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, `dtype="float16"`, `max_model_len=4096`, `gpu_memory_utilization=0.8`
- 샘플링: `temperature=0`, `max_tokens=min_tokens=256`, 배치 1(프롬프트 하나씩 `generate`), 채팅 템플릿 적용. 프롬프트마다 5회 반복.
- spec 설정: `speculative_config={"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}`

## 설정 키와 지표 이름 (설치된 vLLM 0.11.2 소스에서 확인)

| 무엇 | 파일 (site-packages 기준) |
|---|---|
| `method="ngram"`, `num_speculative_tokens`, `prompt_lookup_max`, `prompt_lookup_min` | `vllm/config/speculative.py` (44, 61, 67, 107, 110행) |
| 수락률 카운터 `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens` | `vllm/v1/spec_decode/metrics.py` (154, 163, 172행) |
| `LLM()`은 `disable_log_stats`를 지정하지 않으면 `True`로 둠 → 카운터를 읽으려고 `False`로 지정 | `vllm/entrypoints/llm.py` (228~229행) |
| `LLM.get_metrics()` → Counter 값 | `vllm/entrypoints/llm.py` (1540행), `vllm/v1/metrics/reader.py` |

수락률 = 5회 동안 늘어난 `num_accepted_tokens` ÷ `num_draft_tokens`입니다.

## 결과

tok/s = 256 ÷ `generate` 한 번의 벽시계 시간(prefill 포함), ms/token = 그 시간 ÷ 256입니다.

| 워크로드 | 프롬프트 토큰 | off tok/s (중앙값) | off 5회 | ngram tok/s (중앙값) | ngram 5회 | 배율 | ms/token off → ngram | 수락률 (수락/제안, draft 수) | greedy 출력 동일 |
|---|---|---|---|---|---|---|---|---|---|
| 복사 (코드 파일 그대로 다시 쓰기) | 838 | 98.3 | 95.6, 98.1, 98.3, 98.7, 98.7 | 396.5 | 354.7, 396.5, 399.8, 392.9, 404.9 | 4.03배 | 10.17 → 2.52 | 96.2% (1,010/1,050, 265) | 예 |
| 대화 1 (공부 집중 습관) | 43 | 98.8 | 98.8, 99.2, 98.7, 98.7, 98.9 | 93.8 | 93.8, 93.1, 93.8, 93.9, 93.6 | 0.95배 | 10.13 → 10.67 | 14.1% (45/320, 80) | 예 |
| 대화 2 (냉장고 원리) | 42 | 98.7 | 98.7, 98.8, 98.7, 98.7, 98.8 | 103.3 | 103.3, 103.0, 103.4, 103.4, 103.1 | 1.05배 | 10.13 → 9.68 | 21.5% (155/720, 180) | 예 |
| 대화 3 (등산 준비물) | 44 | 98.6 | 98.5, 98.6, 98.7, 98.6, 98.8 | 95.4 | 95.4, 95.5, 95.3, 95.3, 95.4 | 0.97배 | 10.14 → 10.49 | 15.5% (65/420, 105) | 예 |

- 복사 작업은 4.03배 빨라졌고, 대화 3개는 0.95~1.05배였습니다.
- 네 프롬프트 모두 off와 ngram의 256토큰 출력 토큰 ID가 같았습니다. 5회 반복 사이에도 출력이 바뀌지 않았습니다(코드에서 assert).

## 측정하면서 확인한 것

1. **`VLLM_ENABLE_V1_MULTIPROCESSING=0`에서는 draft가 쓰이지 않았습니다.** 처음에는 실험 1 노트북처럼 이 값을 0으로 두고 쟀습니다.
   그때 ngram이 off보다 약 9% 느렸고(복사 99.8 → 90.4 tok/s), 수락 카운터가 모두 0이었습니다.
   [`debug_metrics.py`](debug_metrics.py)로 `NgramProposer.propose`를 감싸 세어 보니, 256토큰 생성 동안 256번 호출돼 매번 draft 4개(총 1,024개)를 돌려줬습니다.
   그런데 `vllm:spec_decode_num_drafts`는 0이었습니다. 같은 스크립트를 기본값(1)으로 돌리면 draft 210개 중 202개가 수락됐습니다.
   원인은 확인하지 못했습니다. 위 표는 off/ngram 모두 기본값(1)으로 잰 값입니다.
2. spec decode를 켜면 vLLM이 `min_p, logit_bias, and min_tokens parameters won't currently work with speculative decoding enabled.` 경고를 냅니다
   (`vllm/v1/sample/logits_processor/__init__.py` 205행, spec 설정이 있으면 내장 logits processor를 만들지 않음).
   이번 측정에서는 ngram 쪽 출력도 매번 256토큰이었고 off와 토큰이 같았습니다.

## 프로파일러로 본 디코드 스텝 (복사 프롬프트, 원인 확인용)

Nsight Systems가 이 WSL2 환경에서 GPU 커널을 기록하지 않아(상위 README 실험 1 참고), vLLM 내장 torch.profiler로 측정한 `generate` 한 번을 잡았습니다(`VLLM_TORCH_PROFILER_DIR=… python spec.py off|ngram prof`).
스텝 경계는 `Memcpy DtoH`이고, DtoH 수는 off 256개·ngram 54개로 스텝 수와 같았습니다([`../prof_steps.py`](../prof_steps.py), [`results/prof_steps_spec_off.json`](results/prof_steps_spec_off.json), [`results/prof_steps_spec_ngram.json`](results/prof_steps_spec_ngram.json)).

| | off | ngram |
|---|---|---|
| 스텝 수 (첫 스텝 prefill 포함) | 256 | 54 |
| 디코드 스텝당 커널 수 | 405 (255스텝 모두) | 300 (53스텝 모두) |
| 디코드 스텝 GPU busy (중앙값) | 8.52ms | 9.85ms |
| 스텝 안 GPU 유휴 (중앙값) | 0.54ms | 0.22ms |
| 스텝 사이 GPU 유휴 (중앙값) | 1.58ms | 2.15ms |
| 프로파일러 켠 상태의 generate 시간 | 2.775s | 0.673s |

커널 시간 상위 5개 (커널 시간 합 대비)
- off: `gemv2T_kernel_val<…__half…, 128, 16, 4, 4, false, false…>` 56.0%, `cutlass … wmma_tensorop_f16 … 16x16_128x2` 27.7%, `gemv2T_kernel_val<…, false, true…>` 5.8%, `flash_fwd_splitkv_kernel` 3.1%, `flash_fwd_splitkv_combine_kernel` 2.5%
- ngram: `cutlass … tensorop_f16_s16816gemm_relu_f16_64x64_32x10` 40.6%, `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn` 21.5%, `flash_fwd_splitkv_kernel` 13.0%, `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x6_tn` 11.2%(53회), `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_relu_f2f_stages_64x6_tn` 5.7%
